import os
import re
import json
import torch
import certifi
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 0. KHỞI TẠO HỆ THỐNG
# ==========================================
load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

class ExperimentConfig:
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
    INDICES_PATH = "master_granular_indices.json"
    
    # Folder chứa kết quả đã gọt tỉa từ TabPFN
    TTFS_FOLDERS = {
        "top5": "ttfs_tabpfn_top5",
        "top10": "ttfs_tabpfn_top10"
    }
    
    ITERATIONS = 20
    BATCH_SIZE = 32  # Tối ưu cho VRAM A40
    K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
    TARGET = 'Diabetes_binary'

# ==========================================
# 1. QUẢN LÝ LLM (SÚNG NGẮM)
# ==========================================
def load_llm_pipeline():
    print(f"[*] Đang nạp Model: {ExperimentConfig.MODEL_ID}")
    hf_token = os.getenv("HF_TOKEN")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Bắt buộc padding_side='left' để sinh text chuẩn trong Batching
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
# 2. LOGIC PROMPT (CÁ NHÂN HÓA CHO TỪNG BỆNH NHÂN)
# ==========================================
def create_single_patient_prompt(support_df, target_id, target_row, feature_list):
    """
    Build prompt: 
    - Support data: CÓ CHỨA TARGET.
    - Test data: CHỈ CÓ INPUT.
    """
    # 1. DỮ LIỆU MỒI (Có Input -> Result cụ thể)
    support_lines = []
    for _, row in support_df.iterrows():
        feat_str = ", ".join([f"{f}:{int(row[f])}" for f in feature_list])
        target_val = int(row[ExperimentConfig.TARGET])
        support_lines.append(f"Input: [{feat_str}] -> Result: {target_val}")
    
    # 2. DỮ LIỆU TEST CỦA BỆNH NHÂN HIỆN TẠI (Chỉ có Input)
    target_feat_str = ", ".join([f"{f}:{int(target_row[f])}" for f in feature_list])
    target_line = f"ID:{target_id} - [{target_feat_str}]"
        
    prompt = f"""<|im_start|>system
You are a medical expert system. Analyze the provided health indicators and predict diabetes status.
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
    """Bóc tách JSON an toàn cho 1 câu trả lời duy nhất"""
    full_text = "{" + raw_output
    try:
        match = re.search(r'\{.*?\}', full_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return int(data.get('result', -1))
    except: pass
    
    # Regex dự phòng nếu LLM sinh lỗi format JSON
    m = re.search(r'result"\s*:\s*(\d)', full_text, re.IGNORECASE)
    if m: return int(m.group(1))
    return -1

# ==========================================
# 3. TRÌNH ĐIỀU KHIỂN CHÍNH
# ==========================================
def run_sniper_experiment():
    df = pd.read_csv(ExperimentConfig.DATA_PATH)
    train_pool, test_fixed = train_test_split(
        df, test_size=1000, stratify=df[ExperimentConfig.TARGET], random_state=42
    )
    
    with open(ExperimentConfig.INDICES_PATH, 'r') as f:
        indices_map = json.load(f)

    tokenizer, model = load_llm_pipeline()

    for config_name, ttfs_folder in ExperimentConfig.TTFS_FOLDERS.items():
        print(f"\n{'='*60}\n🎯 CHẠY LLM SNIPER: {config_name.upper()}\n{'='*60}")
        
        output_dir = f"llm_sniper_{config_name}"
        os.makedirs(output_dir, exist_ok=True)

        for k in ExperimentConfig.K_VALUES:
            ttfs_log_file = os.path.join(ttfs_folder, f'dynamic_results_k{k}.csv')
            results_file = os.path.join(output_dir, f'qwen_sniper_k{k}.csv')
            
            if not os.path.exists(ttfs_log_file): 
                print(f"[!] Bỏ qua K={k}: Không tìm thấy file {ttfs_log_file}")
                continue
                
            # Đọc file Log của TTFS-TabPFN
            ttfs_df = pd.read_csv(ttfs_log_file, header=None, 
                                  names=['iter', 'id', 'gt', 'pred_tab', 'f_count', 'f_list'],
                                  dtype={'f_list': str})
            
            # Logic Resume thông minh
            start_iter = 0
            if os.path.exists(results_file):
                try:
                    existing_df = pd.read_csv(results_file, header=None)
                    completed = existing_df[0].value_counts()
                    valid_iters = completed[completed >= 1000].index.tolist()
                    if valid_iters: 
                        start_iter = max(valid_iters) + 1
                        print(f"[#] K={k} | Tự động Resume từ Iteration {start_iter}")
                except: pass

            for i in range(start_iter, ExperimentConfig.ITERATIONS):
                support_indices = indices_map[str(k)][i]
                support_df = train_pool.loc[support_indices]
                iter_ttfs_df = ttfs_df[ttfs_df['iter'] == i].reset_index(drop=True)
                
                iteration_logs = []
                
                # Duyệt lô Batch 32
                for b_start in tqdm(range(0, len(iter_ttfs_df), ExperimentConfig.BATCH_SIZE), desc=f"K={k} Iter {i}"):
                    batch_ttfs_data = iter_ttfs_df.iloc[b_start : b_start + ExperimentConfig.BATCH_SIZE]
                    
                    batch_prompts = []
                    batch_metadata = []
                    
                    # Bước 1: Build danh sách Prompt cho Batch
                    for _, row in batch_ttfs_data.iterrows():
                        target_id = int(row['id'])
                        feature_list = row['f_list'].split(";")
                        target_row = test_fixed.loc[target_id]
                        
                        prompt = create_single_patient_prompt(support_df, target_id, target_row, feature_list)
                        batch_prompts.append(prompt)
                        batch_metadata.append({'id': target_id, 'gt': row['gt'], 'f_count': row['f_count'], 'f_list': row['f_list']})
                    
                    # Bước 2: Bắn cả cụm string vào LLM
                    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20, # Giới hạn token sinh ra cực ngắn để tăng tốc
                            temperature=0.01,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Bước 3: Giải mã và bóc tách từng kết quả
                    input_length = inputs.input_ids.shape[1]
                    for idx, output in enumerate(outputs):
                        raw_gen = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                        meta = batch_metadata[idx]
                        
                        pred = extract_single_prediction(raw_gen, meta['id'])
                        
                        # Ghi log chuẩn định dạng 6 cột để tính Metric
                        iteration_logs.append([i, meta['id'], meta['gt'], pred, meta['f_count'], meta['f_list']])
                    
                    # Dọn dẹp VRAM sau mỗi Batch
                    del inputs
                    del outputs
                    torch.cuda.empty_cache()
                
                # Lưu file an toàn sau mỗi Iteration
                pd.DataFrame(iteration_logs).to_csv(results_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    try:
        run_sniper_experiment()
        print("\n" + "="*60 + "\n[SUCCESS] THÍ NGHIỆM LLM SNIPER ĐÃ XONG TOÀN BỘ!\n" + "="*60)
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")