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
    BATCH_SIZE = 32  # Điều chỉnh dựa trên VRAM (A40 46GB có thể lên 40-50)
    K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
    
    # [ĐÃ SỬA]: Chia tách thành Top 5 và Top 10 từ kết quả SHAP của sếp
    TOP_5_FEATURES = ["Age", "BMI", "Income", "HighBP", "HighChol"]
    TOP_10_FEATURES = ["Age", "BMI", "Income", "HighBP", "HighChol", "GenHlth", "PhysHlth", "Sex", "PhysActivity", "MentHlth"]
    
    FEATURE_SETS = {
        "top5": TOP_5_FEATURES,
        "top10": TOP_10_FEATURES
    }
    
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
# 3. LOGIC PROMPT & PARSING
# ==========================================
def create_medical_prompt(support_df, test_batch, feature_list):
    """
    [ĐÃ SỬA]: Thêm tham số feature_list để Prompt tự động co dãn số lượng biến
    """
    # Xây dựng phần Examples (Few-shot)
    support_lines = []
    for _, row in support_df.iterrows():
        feat_str = ", ".join([f"{f}:{int(row[f])}" for f in feature_list])
        support_lines.append(f"Input: [{feat_str}] -> Result: {int(row[ExperimentConfig.TARGET])}")
    
    # Xây dựng phần Test cases
    test_lines = []
    for idx, row in test_batch.iterrows():
        feat_str = ", ".join([f"{f}:{int(row[f])}" for f in feature_list])
        test_lines.append(f"ID:{idx} - [{feat_str}]")
        
    prompt = f"""<|im_start|>system
You are a medical expert system. Analyze the provided health indicators and predict diabetes status.
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
    # Thêm lại dấu ngoặc vuông bị LLM bỏ quên ở đầu do Prompt ép buộc
    full_text = "[" + raw_output
    predictions = {}
    
    # Cách 1: Thử parse JSON chuẩn
    try:
        # Regex tìm khối [...] cuối cùng hoặc duy nhất
        match = re.search(r'\[.*\]', full_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            for item in data:
                predictions[str(int(float(item['id'])))] = int(item['result'])
    except:
        # Cách 2: Regex cứu cánh nếu JSON bị hỏng format
        for tid in test_ids:
            # Tìm pattern ID:XXX ... Result: 0/1
            pattern = rf"{tid}\D*?(\d)"
            m = re.search(pattern, full_text)
            if m:
                predictions[str(tid)] = int(m.group(1))
                
    return predictions

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
def run_experiment():
    # 4.1 Chuẩn bị dữ liệu
    df = pd.read_csv(ExperimentConfig.DATA_PATH)
    train_pool, test_fixed = train_test_split(
        df, test_size=1000, stratify=df[ExperimentConfig.TARGET], random_state=42
    )
    
    # 4.2 Đồng bộ Index (Cực kỳ quan trọng để so sánh với NumLog)
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

    # 4.3 Khởi tạo Model
    tokenizer, model = load_llm_pipeline()

    # 4.4 Vòng lặp Feature (Top 5 -> Top 10)
    for config_name, feature_list in ExperimentConfig.FEATURE_SETS.items():
        print(f"\n{'='*50}")
        print(f">>> BẮT ĐẦU CHẠY PHIÊN BẢN: {config_name.upper()} ({len(feature_list)} Features)")
        print(f"{'='*50}")
        
        # Tạo folder tự động
        output_dir = f"llm_baselines_{config_name}"
        os.makedirs(output_dir, exist_ok=True)

        # 4.5 Vòng lặp thí nghiệm (K & Iterations)
        for k in ExperimentConfig.K_VALUES:
            # Sửa đường dẫn lưu file vào đúng folder
            results_file = os.path.join(output_dir, f'qwen_results_granular_k{k}.csv')
            start_iter = 0
            
            # Checkpoint: Resume nếu đã chạy một phần
            if os.path.exists(results_file):
                try:
                    existing_df = pd.read_csv(results_file, header=None)
                    completed = existing_df[0].value_counts()
                    # Mỗi iteration phải có đúng 1000 mẫu test
                    valid_iters = completed[completed >= 1000].index.tolist()
                    if valid_iters:
                        start_iter = max(valid_iters) + 1
                        print(f"[#] K={k} ({config_name}) | Resume từ Iteration {start_iter}")
                except: pass

            for i in range(start_iter, ExperimentConfig.ITERATIONS):
                print(f"\n[EXEC] Qwen ({config_name}) | K={k} | Iteration {i+1}/{ExperimentConfig.ITERATIONS}")
                support_indices = indices_map[str(k)][i]
                support_df = train_pool.loc[support_indices]
                
                iteration_logs = []
                
                # Batching để tăng tốc
                for b_start in tqdm(range(0, 1000, ExperimentConfig.BATCH_SIZE), desc=f"K={k} Iter {i}"):
                    batch_df = test_fixed.iloc[b_start : b_start + ExperimentConfig.BATCH_SIZE]
                    
                    # Truyền feature_list vào hàm tạo prompt
                    prompt = create_medical_prompt(support_df, batch_df, feature_list)
                    
                    # Inference
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
                        p = preds.get(str(idx), -1) # -1 nếu fail hoàn toàn
                        iteration_logs.append([i, idx, int(row[ExperimentConfig.TARGET]), p])
                
                # Lưu sau mỗi Iteration
                pd.DataFrame(iteration_logs).to_csv(results_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    try:
        run_experiment()
        print("\n" + "="*50 + "\n[SUCCESS] Toàn bộ thí nghiệm đã hoàn thành.\n" + "="*50)
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")