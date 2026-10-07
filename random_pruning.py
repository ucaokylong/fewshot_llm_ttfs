import os
import re
import json
import torch
import random
import certifi
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

class ExperimentConfig:
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
    INDICES_PATH = "master_granular_indices.json"
    
    ITERATIONS = 20
    BATCH_SIZE = 16  # Chạy 16 prompt đơn song song
    K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
    
    # Tập ứng viên chuẩn bị cho cắt ngẫu nhiên
    TOP_5_FEATURES = ["Age", "BMI", "Income", "HighBP", "HighChol"]
    TOP_10_FEATURES = ["Age", "BMI", "Income", "HighBP", "HighChol", "GenHlth", "PhysHlth", "Sex", "PhysActivity", "MentHlth"]
    
    FEATURE_SETS = {
        "top5": TOP_5_FEATURES,
        "top10": TOP_10_FEATURES
    }
    
    TARGET = 'Diabetes_binary'

def load_llm_pipeline():
    hf_token = os.getenv("HF_TOKEN")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(ExperimentConfig.MODEL_ID, token=hf_token)
    
    # Bắt buộc padding trái cho LLM batch inference
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        ExperimentConfig.MODEL_ID, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, token=hf_token
    )
    return tokenizer, model

def create_random_pruning_prompt(support_df, target_id, target_row, candidate_features, target_col):
    """Cắt ngẫu nhiên 1-2 đặc trưng từ tập candidate và tạo prompt."""
    num_to_drop = random.choice([1, 2])
    if num_to_drop >= len(candidate_features):
        num_to_drop = 1
        
    retained_features = random.sample(candidate_features, len(candidate_features) - num_to_drop)
    
    support_lines = []
    for _, row in support_df.iterrows():
        feat_strs = [f"{f}:{int(row[f]) if row[f]==int(row[f]) else f'{row[f]:.2f}'}" for f in retained_features]
        support_lines.append(f"Input: [{', '.join(feat_strs)}] -> Result: {int(row[target_col])}")
    support_context = chr(10).join(support_lines)

    target_feat_strs = [f"{f}:{int(target_row[f]) if target_row[f]==int(target_row[f]) else f'{target_row[f]:.2f}'}" for f in retained_features]
    target_line = f"ID:{target_id} - [{', '.join(target_feat_strs)}]"
    
    prompt = f"""<|im_start|>system
You are a medical expert system. Analyze the provided health indicators and predict diabetes status.
Output MUST be strictly JSON: {{"id": "{target_id}", "result": 0_or_1}}. No other text.<|im_end|>
<|im_start|>user
Historical Cases:
{support_context}

Current Patient to Evaluate:
{target_line}<|im_end|>
<|im_start|>assistant
{{"""
    return prompt, len(retained_features), ";".join(retained_features)

def extract_single_prediction(raw_output):
    full_text = "{" + raw_output
    try:
        match = re.search(r'\{.*?\}', full_text, re.DOTALL)
        if match:
            return int(json.loads(match.group()).get('result', -1))
    except: pass
    
    m = re.search(r'result"\s*:\s*(\d)', full_text, re.IGNORECASE)
    return int(m.group(1)) if m else -1

def run_experiment():
    df = pd.read_csv(ExperimentConfig.DATA_PATH)
    train_pool, test_fixed = train_test_split(df, test_size=1000, stratify=df[ExperimentConfig.TARGET], random_state=42)
    
    with open(ExperimentConfig.INDICES_PATH, 'r') as f:
        indices_map = json.load(f)

    tokenizer, model = load_llm_pipeline()

    for config_name, feature_list in ExperimentConfig.FEATURE_SETS.items():
        print(f"\n{'='*60}\n>>> RANDOM PRUNING BASELINE: {config_name.upper()}\n{'='*60}")
        output_dir = f"diabetes_random_pruning_{config_name}"
        os.makedirs(output_dir, exist_ok=True)

        for k in ExperimentConfig.K_VALUES:
            results_file = os.path.join(output_dir, f'qwen_random_k{k}.csv')
            start_iter = 0
            
            if os.path.exists(results_file):
                try:
                    existing_df = pd.read_csv(results_file, header=None)
                    completed = existing_df[0].value_counts()
                    valid_iters = completed[completed >= 1000].index.tolist()
                    if valid_iters:
                        start_iter = max(valid_iters) + 1
                except: pass

            for i in range(start_iter, ExperimentConfig.ITERATIONS):
                print(f"[EXEC] Diabetes Random ({config_name}) | K={k} | Iteration {i+1}/{ExperimentConfig.ITERATIONS}")
                support_indices = indices_map[str(k)][i]
                support_df = train_pool.loc[support_indices]
                iteration_logs = []
                
                for b_start in tqdm(range(0, 1000, ExperimentConfig.BATCH_SIZE), desc=f"K={k} Iter {i}"):
                    batch_df = test_fixed.iloc[b_start : b_start + ExperimentConfig.BATCH_SIZE]
                    
                    batch_prompts = []
                    batch_metadata = []
                    
                    for idx, row in batch_df.iterrows():
                        prompt, f_count, f_list_str = create_random_pruning_prompt(
                            support_df, idx, row, feature_list, ExperimentConfig.TARGET
                        )
                        batch_prompts.append(prompt)
                        batch_metadata.append((idx, int(row[ExperimentConfig.TARGET]), f_count, f_list_str))
                    
                    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_new_tokens=20, temperature=0.01, do_sample=False, pad_token_id=tokenizer.eos_token_id
                        )
                    
                    input_length = inputs.input_ids.shape[1]
                    for p_idx, output in enumerate(outputs):
                        raw_gen = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                        t_id, gt, f_c, f_l = batch_metadata[p_idx]
                        pred = extract_single_prediction(raw_gen)
                        
                        # Ghi log giống hệt TTFS: iter, id, gt, pred, f_count, f_list
                        iteration_logs.append([i, t_id, gt, pred, f_c, f_l])
                    
                    del inputs, outputs
                    torch.cuda.empty_cache()
                
                pd.DataFrame(iteration_logs).to_csv(results_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    run_experiment()