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

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
class ExperimentConfig:
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    
    CANDIDATE_POOL_PATH = "cardio_candidate_balanced.csv"
    TEST_SET_PATH = "cardio_test_fixed.csv"
    INDICES_PATH = "master_granular_indices_cardio.json"
    
    ITERATIONS = 20
    BATCH_SIZE = 16  # Chạy 16 prompt đơn song song trên GPU (có thể tăng lên 32 nếu VRAM đủ)
    K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
    
    FEATURES = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 
        'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'
    ]
    TARGET = 'cardio'

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
    # Bắt buộc padding_side='left' khi chạy batch generation trên Causal LM
    tokenizer.padding_side = "left"
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
# 3. PROMPT GENERATION (1 PROMPT / 1 PATIENT) & PARSING
# ==========================================
def create_single_patient_prompt(support_df, target_id, target_row):
    """Xây dựng đúng 1 prompt y tế cho duy nhất 1 bệnh nhân cần test."""
    support_lines = []
    for _, row in support_df.iterrows():
        feat_strs = []
        for f in ExperimentConfig.FEATURES:
            val = row[f]
            val_str = f"{int(val)}" if val == int(val) else f"{val:.2f}"
            feat_strs.append(f"{f}:{val_str}")
        
        feat_str = ", ".join(feat_strs)
        support_lines.append(f"Input: [{feat_str}] -> Result: {int(row[ExperimentConfig.TARGET])}")
    support_context = chr(10).join(support_lines)

    target_feat_strs = []
    for f in ExperimentConfig.FEATURES:
        val = target_row[f]
        val_str = f"{int(val)}" if val == int(val) else f"{val:.2f}"
        target_feat_strs.append(f"{f}:{val_str}")
    
    target_feat_str = ", ".join(target_feat_strs)
    target_line = f"ID:{target_id} - [{target_feat_str}]"
    
    prompt = f"""<|im_start|>system
You are a medical expert system. Analyze the provided health parameters and evaluation records to predict the presence or absence of cardiovascular disease.
Output MUST be strictly JSON: {{"id": "{target_id}", "result": 0_or_1}}. No other text.<|im_end|>
<|im_start|>user
Historical Cases:
{support_context}

Current Patient to Evaluate:
{target_line}<|im_end|>
<|im_start|>assistant
{{"""
    return prompt

def extract_single_prediction(raw_output, target_id):
    """Bóc tách dự đoán dạng số (0 hoặc 1) từ JSON đơn."""
    full_text = "{" + raw_output
    try:
        match = re.search(r'\{.*?\}', full_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return int(data.get('result', -1))
    except:
        pass
    
    m = re.search(r'result"\s*:\s*(\d)', full_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return -1

# ==========================================
# 4. MAIN EXPERIMENTAL EXECUTION LOOP
# ==========================================
def run_experiment():
    print("[*] Accessing separate pre-balanced dataset files...")
    train_pool = pd.read_csv(ExperimentConfig.CANDIDATE_POOL_PATH)
    test_fixed = pd.read_csv(ExperimentConfig.TEST_SET_PATH)
    
    total_test_samples = len(test_fixed)
    print(f"[INFO] Candidate Resource Pool Size: {train_pool.shape[0]}")
    print(f"[INFO] Total Fixed Evaluation Target Samples: {total_test_samples}")
    
    if not os.path.exists(ExperimentConfig.INDICES_PATH):
        raise FileNotFoundError(f"[ERROR] Master index path not found at: {ExperimentConfig.INDICES_PATH}")
        
    with open(ExperimentConfig.INDICES_PATH, 'r') as f:
        indices_map = json.load(f)

    tokenizer, model = load_llm_pipeline()

    for k in ExperimentConfig.K_VALUES:
        results_file = f'cardio_results_granular_k{k}.csv'
        start_iter = 0
        
        if os.path.exists(results_file):
            try:
                existing_df = pd.read_csv(results_file, header=None)
                completed = existing_df[0].value_counts()
                valid_iters = completed[completed >= total_test_samples].index.tolist()
                if valid_iters:
                    start_iter = max(valid_iters) + 1
                    print(f"[#] K={k} | Resuming execution from Iteration {start_iter}")
            except: 
                pass

        for i in range(start_iter, ExperimentConfig.ITERATIONS):
            print(f"\n[EXEC] Qwen Cardio Baseline | K={k} | Iteration {i+1}/{ExperimentConfig.ITERATIONS}")
            support_indices = indices_map[str(k)][i]
            support_df = train_pool.loc[support_indices]
            
            iteration_logs = []
            
            for b_start in tqdm(range(0, total_test_samples, ExperimentConfig.BATCH_SIZE), desc=f"K={k} Iter {i}"):
                batch_df = test_fixed.iloc[b_start : b_start + ExperimentConfig.BATCH_SIZE]
                
                batch_prompts = []
                batch_metadata = []
                
                # 1. Tạo danh sách các prompt riêng biệt
                for idx, row in batch_df.iterrows():
                    prompt = create_single_patient_prompt(support_df, idx, row)
                    batch_prompts.append(prompt)
                    batch_metadata.append((idx, int(row[ExperimentConfig.TARGET])))
                
                # 2. Tokenize song song có left-padding
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
                
                # 3. Model sinh token đồng loạt
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,  # Chỉ cần output JSON ngắn
                        temperature=0.01,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # 4. Trích xuất kết quả từng prompt
                input_length = inputs.input_ids.shape[1]
                for p_idx, output in enumerate(outputs):
                    raw_gen = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                    target_id, ground_truth = batch_metadata[p_idx]
                    pred = extract_single_prediction(raw_gen, target_id)
                    iteration_logs.append([i, target_id, ground_truth, pred])
                
                # Dọn cache VRAM định kỳ
                del inputs, outputs
                torch.cuda.empty_cache()
            
            pd.DataFrame(iteration_logs).to_csv(results_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    try:
        run_experiment()
        print("\n" + "="*50 + "\n[SUCCESS] Cardiovascular dataset execution finalized.\n" + "="*50)
    except Exception as e:
        print(f"\n[FATAL SYSTEM ERROR] {str(e)}")