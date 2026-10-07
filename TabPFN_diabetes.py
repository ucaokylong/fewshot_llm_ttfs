import os
import re
import json
import torch
import certifi
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 0. KHỞI TẠO HỆ THỐNG (SSL & ENV)
# ==========================================
load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# ==========================================
# 1. CẤU HÌNH THÍ NGHIỆM
# ==========================================
class ExperimentConfig:
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
    INDICES_PATH = "master_granular_indices.json"
    
    ITERATIONS = 20
    BATCH_SIZE = 16  # Gom 16 prompt đơn chạy song song trên GPU (có thể tăng lên 32 nếu VRAM dư)
    K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
    
    FEATURES = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    TARGET = 'Diabetes_binary'

# ==========================================
# 2. QUẢN LÝ MODEL
# ==========================================
def load_llm_pipeline():
    print(f"[*] Đang khởi tạo Model: {ExperimentConfig.MODEL_ID}")
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
# 3. LOGIC TẠO PROMPT ĐƠN & BÓC TÁCH KẾT QUẢ
# ==========================================
def create_single_patient_prompt(support_df, target_id, target_row):
    """
    Tạo 1 prompt chuẩn y tế duy nhất cho đúng 1 bệnh nhân cần test.
    """
    support_lines = []
    for _, row in support_df.iterrows():
        feat_str = ", ".join([f"{f}:{int(row[f])}" for f in ExperimentConfig.FEATURES])
        support_lines.append(f"Input: [{feat_str}] -> Result: {int(row[ExperimentConfig.TARGET])}")
    support_context = chr(10).join(support_lines)

    target_feat_str = ", ".join([f"{f}:{int(target_row[f])}" for f in ExperimentConfig.FEATURES])
    target_line = f"ID:{target_id} - [{target_feat_str}]"
    
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
    return prompt

def extract_single_prediction(raw_output, target_id):
    """Bóc tách JSON cho câu trả lời của 1 bệnh nhân."""
    full_text = "{" + raw_output
    try:
        match = re.search(r'\{.*?\}', full_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return int(data.get('result', -1))
    except:
        pass
    
    # Fallback regex nếu JSON bị lỗi nhẹ
    m = re.search(r'result"\s*:\s*(\d)', full_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return -1

# ==========================================
# 4. TIẾN TRÌNH THỰC NGHIỆM CHÍNH
# ==========================================
def run_experiment():
    df = pd.read_csv(ExperimentConfig.DATA_PATH)
    train_pool, test_fixed = train_test_split(
        df, test_size=1000, stratify=df[ExperimentConfig.TARGET], random_state=42
    )
    
    if not os.path.exists(ExperimentConfig.INDICES_PATH):
        print(f"[!] Tạo mới file Index đồng bộ...")
        indices_map = {str(k): [] for k in ExperimentConfig.K_VALUES}
        for k in ExperimentConfig.K_VALUES:
            for i in range(ExperimentConfig.ITERATIONS):
                s0 = train_pool[train_pool[ExperimentConfig.TARGET]==0].sample(k//2, random_state=i).index.tolist()
                s1 = train_pool[train_pool[ExperimentConfig.TARGET]==1].sample(k//2, random_state=i).index.tolist()
                indices_map[str(k)].append(s0 + s1)
        with open(ExperimentConfig.INDICES_PATH, 'w') as f:
            json.dump(indices_map, f)
    else:
        with open(ExperimentConfig.INDICES_PATH, 'r') as f:
            indices_map = json.load(f)

    tokenizer, model = load_llm_pipeline()

    for k in ExperimentConfig.K_VALUES:
        results_file = f'qwen_results_granular_k{k}.csv'
        start_iter = 0
        
        # Checkpoint: Resume nếu trước đó bị ngắt giữa chừng
        if os.path.exists(results_file):
            try:
                existing_df = pd.read_csv(results_file, header=None)
                completed = existing_df[0].value_counts()
                valid_iters = completed[completed >= 1000].index.tolist()
                if valid_iters:
                    start_iter = max(valid_iters) + 1
                    print(f"[#] K={k} | Tự động resume từ Iteration {start_iter}")
            except:
                pass

        for i in range(start_iter, ExperimentConfig.ITERATIONS):
            print(f"\n[EXEC] Qwen Baseline | K={k} | Iteration {i+1}/{ExperimentConfig.ITERATIONS}")
            support_indices = indices_map[str(k)][i]
            support_df = train_pool.loc[support_indices]
            
            iteration_logs = []
            
            # Quét test set theo từng BATCH_SIZE
            for b_start in tqdm(range(0, 1000, ExperimentConfig.BATCH_SIZE), desc=f"K={k} Iter {i}"):
                batch_df = test_fixed.iloc[b_start : b_start + ExperimentConfig.BATCH_SIZE]
                
                batch_prompts = []
                batch_metadata = []
                
                # 1. Tạo batch gồm các prompt riêng biệt
                for idx, row in batch_df.iterrows():
                    prompt = create_single_patient_prompt(support_df, idx, row)
                    batch_prompts.append(prompt)
                    batch_metadata.append((idx, int(row[ExperimentConfig.TARGET])))
                
                # 2. Tokenize song song với padding bên trái
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
                
                # 3. Sinh token đồng loạt
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,  # Chỉ cần output 1 JSON nhỏ, cắt giảm token để tăng tốc gấp nhiều lần
                        temperature=0.01,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # 4. Bóc tách kết quả cho từng prompt trong batch
                input_length = inputs.input_ids.shape[1]
                for p_idx, output in enumerate(outputs):
                    raw_gen = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                    target_id, ground_truth = batch_metadata[p_idx]
                    pred = extract_single_prediction(raw_gen, target_id)
                    iteration_logs.append([i, target_id, ground_truth, pred])
                
                # Giải phóng bộ nhớ đệm GPU tránh tràn VRAM
                del inputs, outputs
                torch.cuda.empty_cache()
            
            # Lưu an toàn sau mỗi iteration
            pd.DataFrame(iteration_logs).to_csv(results_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    try:
        run_experiment()
        print("\n" + "="*50 + "\n[SUCCESS] Toàn bộ thí nghiệm Diabetes Baseline đã hoàn thành.\n" + "="*50)
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")