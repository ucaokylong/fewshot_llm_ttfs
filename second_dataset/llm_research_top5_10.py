import os
import re
import json
import torch
import certifi
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 0. SYSTEM INITIALIZATION (SSL & ENV)
# ==========================================
load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
class ExperimentConfig:
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    
    # Path mappings for the pre-balanced Framingham files
    CANDIDATE_POOL_PATH = "framingham_candidate_balanced.csv"
    TEST_SET_PATH = "framingham_test_fixed.csv"
    INDICES_PATH = "master_granular_indices_framingham.json"
    
    ITERATIONS = 20
    BATCH_SIZE = 32  # Optimized based on VRAM capacity
    K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
    
    # Feature subsets derived directly from your TabPFN-KernelSHAP ranking
    TOP_5_FEATURES = ["BMI", "sysBP", "age", "totChol", "diaBP"]
    TOP_10_FEATURES = ["BMI", "sysBP", "age", "totChol", "diaBP", "glucose", "heartRate", "cigsPerDay", "prevalentHyp", "education"]
    
    FEATURE_SETS = {
        "top5": TOP_5_FEATURES,
        "top10": TOP_10_FEATURES
    }
    
    TARGET = 'TenYearCHD'

# ==========================================
# 2. MODEL MANAGEMENT
# ==========================================
def load_llm_pipeline():
    print(f"[*] Initializing model architecture: {ExperimentConfig.MODEL_ID}")
    hf_token = os.getenv("HF_TOKEN")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        ExperimentConfig.MODEL_ID, 
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        ExperimentConfig.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )
    return tokenizer, model

# ==========================================
# 3. PROMPT GENERATION & PREDICTION PARSING
# ==========================================
def create_medical_prompt(support_df, test_batch, feature_list):
    # Construct historical reference context (Few-shot samples)
    support_lines = []
    for _, row in support_df.iterrows():
        feat_strs = []
        for f in feature_list:
            val = row[f]
            # Defensive clinical float formatting: preserve decimals for continuous vitals
            val_str = f"{int(val)}" if val == int(val) else f"{val:.2f}"
            feat_strs.append(f"{f}:{val_str}")
        
        feat_str = ", ".join(feat_strs)
        support_lines.append(f"Input: [{feat_str}] -> Result: {int(row[ExperimentConfig.TARGET])}")
    
    # Construct current test instances to evaluate within the batch window
    test_lines = []
    for idx, row in test_batch.iterrows():
        feat_strs = []
        for f in feature_list:
            val = row[f]
            val_str = f"{int(val)}" if val == int(val) else f"{val:.2f}"
            feat_strs.append(f"{f}:{val_str}")
            
        feat_str = ", ".join(feat_strs)
        test_lines.append(f"ID:{idx} - [{feat_str}]")
        
    prompt = f"""<|im_start|>system
You are a medical expert system. Analyze the provided health records and predict the 10-year risk of coronary heart disease (CHD).
Output MUST be a strictly formatted JSON array of objects: [{{"id": <ID>, "result": 0_or_1}}, ...].
No conversational text, no explanations.<|im_end|>
<|im_start|>user
Historical Cases:
{chr(10).join(support_lines)}

Current Patients to Evaluate:
{chr(10).join(test_lines)}<|im_end|>
<|im_start|>assistant
["""
    return prompt

def extract_predictions(raw_output, test_ids):
    full_text = "[" + raw_output
    predictions = {}
    
    try:
        match = re.search(r'\[.*\]', full_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            for item in data:
                clean_id = str(int(float(item['id'])))
                predictions[clean_id] = int(item['result'])
    except:
        for tid in test_ids:
            pattern = rf"{tid}\D*?(\d)"
            m = re.search(pattern, full_text)
            if m:
                predictions[str(tid)] = int(m.group(1))
                
    return predictions

# ==========================================
# 4. MAIN EXPERIMENTAL EXECUTION LOOP
# ==========================================
def run_experiment():
    # 4.1 Load separate pre-balanced dataset pools directly
    print("[*] Accessing separate pre-balanced dataset files...")
    train_pool = pd.read_csv(ExperimentConfig.CANDIDATE_POOL_PATH)
    test_fixed = pd.read_csv(ExperimentConfig.TEST_SET_PATH)
    
    total_test_samples = len(test_fixed)
    print(f"[INFO] Candidate Resource Pool Size: {train_pool.shape[0]}")
    print(f"[INFO] Total Fixed Evaluation Target Samples: {total_test_samples}")
    
    # 4.2 Load master synchronization index registry
    if not os.path.exists(ExperimentConfig.INDICES_PATH):
        raise FileNotFoundError(f"[ERROR] Master index path not found at: {ExperimentConfig.INDICES_PATH}. Please run your step3 indexing script first.")
        
    with open(ExperimentConfig.INDICES_PATH, 'r') as f:
        indices_map = json.load(f)

    # 4.3 Initialize Model pipeline
    tokenizer, model = load_llm_pipeline()

    # 4.4 Feature Subset Slicing Loop (Top 5 -> Top 10)
    for config_name, feature_list in ExperimentConfig.FEATURE_SETS.items():
        print(f"\n{'='*60}")
        # Dynamic display based on actual feature subsets length
        print(f">>> LAUNCHING EVALUATION PIPELINE: {config_name.upper()} ({len(feature_list)} Features)")
        print(f"{'='*60}")
        
        output_dir = f"framingham_llm_baselines_{config_name}"
        os.makedirs(output_dir, exist_ok=True)

        # 4.5 Parametric scaling execution loop (K & Iterations)
        for k in ExperimentConfig.K_VALUES:
            results_file = os.path.join(output_dir, f'qwen_results_granular_k{k}.csv')
            start_iter = 0
            
            # Checkpoint Management: Resume if matching execution logging exists
            if os.path.exists(results_file):
                try:
                    existing_df = pd.read_csv(results_file, header=None)
                    completed = existing_df[0].value_counts()
                    # Automatically matches the true size of the fixed test set (400 samples)
                    valid_iters = completed[completed >= total_test_samples].index.tolist()
                    if valid_iters:
                        start_iter = max(valid_iters) + 1
                        print(f"[#] K={k} ({config_name}) | Resuming execution from Iteration {start_iter}")
                except: 
                    pass

            for i in range(start_iter, ExperimentConfig.ITERATIONS):
                print(f"\n[EXEC] Qwen ({config_name}) | K={k} | Iteration {i+1}/{ExperimentConfig.ITERATIONS}")
                support_indices = indices_map[str(k)][i]
                support_df = train_pool.loc[support_indices]
                
                iteration_logs = []
                
                # Execute batch evaluation step over the isolated test boundaries
                for b_start in tqdm(range(0, total_test_samples, ExperimentConfig.BATCH_SIZE), desc=f"K={k} {config_name} Iter {i}"):
                    batch_df = test_fixed.iloc[b_start : b_start + ExperimentConfig.BATCH_SIZE]
                    
                    # Pass the sliced feature subset list to the compiler
                    prompt = create_medical_prompt(support_df, batch_df, feature_list)
                    
                    # Context inference
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1500,
                            temperature=0.01,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    raw_gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    preds = extract_predictions(raw_gen, batch_df.index.astype(str).tolist())
                    
                    for idx, row in batch_df.iterrows():
                        p = preds.get(str(idx), -1)  # Log -1 if a generation structure failure is encountered
                        iteration_logs.append([i, idx, int(row[ExperimentConfig.TARGET]), p])
                
                # Flush iteration metrics to disk safely to secure intermediate progress
                pd.DataFrame(iteration_logs).to_csv(results_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    try:
        run_experiment()
        print("\n" + "="*50 + "\n[SUCCESS] Feature subset evaluation experiment completed successfully.\n" + "="*50)
    except Exception as e:
        print(f"\n[FATAL SYSTEM ERROR] {str(e)}")