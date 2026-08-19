import os
import re
import json
import torch
import certifi
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 0. SYSTEM INITIALIZATION (SSL & ENV)
# ==========================================
load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

class ExperimentConfig:
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    
    # Path mappings synchronized with the new Cardiovascular data pipelines
    CANDIDATE_POOL_PATH = "cardio_candidate_balanced.csv"
    TEST_SET_PATH = "cardio_test_fixed.csv"
    INDICES_PATH = "master_granular_indices_cardio.json"
    
    # Folders containing the pruned TTFS-TabPFN trajectory logs for CVD
    TTFS_FOLDERS = {
        "top5": "cardio_ttfs_tabpfn_top5",
        "top10": "cardio_ttfs_tabpfn_top10"
    }
    
    ITERATIONS = 20
    BATCH_SIZE = 32  # Optimized based on VRAM capacity
    K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
    TARGET = 'cardio'

# ==========================================
# 1. LLM PIPELINE MANAGEMENT
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
    
    # Mandatory padding_side='left' for stable batched text generation
    tokenizer = AutoTokenizer.from_pretrained(ExperimentConfig.MODEL_ID, token=hf_token, padding_side='left')
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
# 2. PROMPT LOGIC (INDIVIDUALLY PERSONALIZED)
# ==========================================
def create_single_patient_prompt(support_df, target_id, target_row, feature_list):
    """
    Builds a surgically precise prompt based on patient-specific retained features:
    - Support data: Contextual features mapped to empirical targets.
    - Test data: Isolated query features stripped of target leaks.
    """
    # 1. REFERENCE CONTEXT (Includes Inputs -> Target Mapping)
    support_lines = []
    for _, row in support_df.iterrows():
        feat_strs = []
        for f in feature_list:
            val = row[f]
            # Defensive clinical float formatting: preserve decimals for continuous vitals
            val_str = f"{int(val)}" if val == int(val) else f"{val:.2f}"
            feat_strs.append(f"{f}:{val_str}")
            
        target_val = int(row[ExperimentConfig.TARGET])
        support_lines.append(f"Input: [{', '.join(feat_strs)}] -> Result: {target_val}")
    
    # 2. CURRENT TEST PATIENT (Isolated Input Sequence)
    target_feat_strs = []
    for f in feature_list:
        val = target_row[f]
        val_str = f"{int(val)}" if val == int(val) else f"{val:.2f}"
        target_feat_strs.append(f"{f}:{val_str}")
        
    target_line = f"ID:{target_id} - [{', '.join(target_feat_strs)}]"
        
    prompt = f"""<|im_start|>system
You are a medical expert system. Analyze the provided health records and predict the presence or absence of cardiovascular disease.
Output MUST be strictly JSON: {{"id": <ID>, "result": 0_or_1}}. No other text.<|im_end|>
<|im_start|>user
Historical Cases:
{chr(10).join(support_lines)}

Current Patient to Evaluate:
{target_line}<|im_end|>
<|im_start|>assistant
{{"""
    return prompt

def extract_single_prediction(raw_output, target_id):
    """Robust JSON extraction mechanism for single-target responses."""
    full_text = "{" + raw_output
    try:
        match = re.search(r'\{.*?\}', full_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return int(data.get('result', -1))
    except: 
        pass
    
    # Fallback regex parsing if LLM generation formatting structurally fails
    m = re.search(r'result"\s*:\s*(\d)', full_text, re.IGNORECASE)
    if m: 
        return int(m.group(1))
    return -1

# ==========================================
# 3. MAIN EXECUTION CONTROLLER
# ==========================================
def run_sniper_experiment():
    print("[*] Accessing separate pre-balanced dataset files...")
    train_pool = pd.read_csv(ExperimentConfig.CANDIDATE_POOL_PATH)
    test_fixed = pd.read_csv(ExperimentConfig.TEST_SET_PATH)
    
    total_test_samples = len(test_fixed)
    print(f"[INFO] Total Fixed Evaluation Target Samples: {total_test_samples}")
    
    if not os.path.exists(ExperimentConfig.INDICES_PATH):
        raise FileNotFoundError(f"[ERROR] Master index path not found at: {ExperimentConfig.INDICES_PATH}")
        
    with open(ExperimentConfig.INDICES_PATH, 'r') as f:
        indices_map = json.load(f)

    tokenizer, model = load_llm_pipeline()

    for config_name, ttfs_folder in ExperimentConfig.TTFS_FOLDERS.items():
        print(f"\n{'='*70}\n🎯 LAUNCHING LLM SNIPER PROTOCOL: {config_name.upper()}\n{'='*70}")
        
        output_dir = f"cardio_llm_sniper_{config_name}"
        os.makedirs(output_dir, exist_ok=True)

        for k in ExperimentConfig.K_VALUES:
            ttfs_log_file = os.path.join(ttfs_folder, f'cardio_dynamic_results_k{k}.csv')
            results_file = os.path.join(output_dir, f'cardio_qwen_sniper_k{k}.csv')
            
            if not os.path.exists(ttfs_log_file): 
                print(f"[!] Bypassing K={k}: Required trajectory log not found at {ttfs_log_file}")
                continue
                
            # Parse localized TTFS-TabPFN trajectory map
            ttfs_df = pd.read_csv(ttfs_log_file, header=None, 
                                 names=['iter', 'id', 'gt', 'pred_tab', 'f_count', 'f_list'],
                                 dtype={'f_list': str})
            
            # Intelligent Checkpoint Resume Logic
            start_iter = 0
            if os.path.exists(results_file):
                try:
                    existing_df = pd.read_csv(results_file, header=None)
                    completed = existing_df[0].value_counts()
                    # Verify completeness against true test set boundary
                    valid_iters = completed[completed >= total_test_samples].index.tolist()
                    if valid_iters: 
                        start_iter = max(valid_iters) + 1
                        print(f"[#] K={k} | Automatically resuming execution from Iteration {start_iter}")
                except: 
                    pass

            for i in range(start_iter, ExperimentConfig.ITERATIONS):
                support_indices = indices_map[str(k)][i]
                support_df = train_pool.loc[support_indices]
                iter_ttfs_df = ttfs_df[ttfs_df['iter'] == i].reset_index(drop=True)
                
                iteration_logs = []
                
                # Iterate through dynamic execution batches
                for b_start in tqdm(range(0, len(iter_ttfs_df), ExperimentConfig.BATCH_SIZE), desc=f"K={k} Iter {i}"):
                    batch_ttfs_data = iter_ttfs_df.iloc[b_start : b_start + ExperimentConfig.BATCH_SIZE]
                    
                    batch_prompts = []
                    batch_metadata = []
                    
                    # Step 1: Construct personalized prompt array
                    for _, row in batch_ttfs_data.iterrows():
                        target_id = int(row['id'])
                        feature_list = row['f_list'].split(";")
                        target_row = test_fixed.loc[target_id]
                        
                        prompt = create_single_patient_prompt(support_df, target_id, target_row, feature_list)
                        batch_prompts.append(prompt)
                        batch_metadata.append({'id': target_id, 'gt': row['gt'], 'f_count': row['f_count'], 'f_list': row['f_list']})
                    
                    # Step 2: Dispatch tensor arrays to LLM
                    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,  # Cap generation limits heavily to maximize processing throughput
                            temperature=0.01,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Step 3: Decode text and extract categorical outputs
                    input_length = inputs.input_ids.shape[1]
                    for idx, output in enumerate(outputs):
                        raw_gen = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                        meta = batch_metadata[idx]
                        
                        pred = extract_single_prediction(raw_gen, meta['id'])
                        
                        # Archive log matching standardized 6-column metric schema
                        iteration_logs.append([i, meta['id'], meta['gt'], pred, meta['f_count'], meta['f_list']])
                    
                    # Aggressive VRAM Sanitization
                    del inputs
                    del outputs
                    torch.cuda.empty_cache()
                
                # Safely flush iteration logs to target tracking file
                pd.DataFrame(iteration_logs).to_csv(results_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    try:
        run_sniper_experiment()
        print("\n" + "="*70 + "\n[SUCCESS] CARDIOVASCULAR LLM SNIPER EVALUATION FULLY FINALIZED!\n" + "="*70)
    except Exception as e:
        print(f"\n[FATAL SYSTEM ERROR] {str(e)}")