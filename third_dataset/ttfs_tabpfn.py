import os
import json
import torch
import pandas as pd
import numpy as np
from scipy.stats import entropy
from tabpfn import TabPFNClassifier
from tqdm import tqdm

# =====================================================================
# 1. SYSTEM CONFIGURATION & FEATURE SPACE
# =====================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Đồng bộ hóa đường dẫn tệp tin với pipeline dữ liệu Heart Attack mới
CANDIDATE_POOL_PATH = "heart_attack_candidate_balanced.csv"
TEST_SET_PATH = "heart_attack_test_fixed.csv"
INDEX_FILE_PATH = "master_granular_indices_heart_attack.json"
TARGET = "Heart Attack Risk"

FEW_SHOT_K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
TOTAL_ITERATIONS = 20
MINIMUM_CONFIDENCE_THRESHOLD = 0.60

# Đồng bộ hóa không gian phân tách KernelSHAP thực tế từ dữ liệu quét thực nghiệm
TOP_5_BASE_FEATURES = ["Stress Level", "Country", "BMI", "Continent", "Diastolic BP"]
TOP_10_BASE_FEATURES = [
    "Stress Level", "Country", "BMI", "Continent", "Diastolic BP", 
    "Cholesterol", "Exercise Hours Per Week", "Systolic BP", "Obesity", "Sedentary Hours Per Day"
]

CONFIGURATIONS_TO_RUN = {
    "top5": TOP_5_BASE_FEATURES,
    "top10": TOP_10_BASE_FEATURES
}

# =====================================================================
# 2. CORE ALGORITHM: BATCHED FLEXIBLE TTFS
# =====================================================================
def execute_batched_flexible_ttfs(tabpfn_model, full_support_x, full_support_y, test_patient_x_all, starting_feature_list):
    """
    Thuật toán TTFS nâng cấp: Tự động đánh giá và gọt bỏ các nhiễu toán học trên từng thực thể.
    Áp dụng gom cụm trạng thái hình học (State Clustering) để tăng tốc độ suy luận song song trên GPU.
    """
    num_patients = len(test_patient_x_all)
    
    # Khởi tạo trạng thái ban đầu chứa toàn bộ không gian đặc trưng tĩnh cho mọi bệnh nhân
    current_states = [tuple(starting_feature_list) for _ in range(num_patients)]
    
    best_confidences = np.zeros(num_patients)
    best_probabilities = np.zeros((num_patients, 2))
    
    # ---------------------------------------------------------
    # BƯỚC 1: Thiết lập điểm neo tự tin cơ sở (Full Feature Anchor)
    # ---------------------------------------------------------
    base_features = list(starting_feature_list)
    tabpfn_model.fit(full_support_x[base_features], full_support_y)
    
    base_probs = tabpfn_model.predict_proba(test_patient_x_all[base_features])
    base_confs = 1.0 - entropy(base_probs, base=2, axis=1)
    
    best_confidences[:] = base_confs
    best_probabilities[:] = base_probs
    
    # ---------------------------------------------------------
    # BƯỚC 2: Duyệt tham lam loại bỏ đặc trưng từ dưới lên (Bottom-Up)
    # ---------------------------------------------------------
    for feature_to_test in reversed(starting_feature_list):
        
        # Gom cụm các bệnh nhân có cùng tập đặc trưng đang hoạt động tại vòng lặp hiện tại
        state_to_patient_indices = {}
        for i, state in enumerate(current_states):
            # Chỉ thử nghiệm loại bỏ nếu đặc trưng tồn tại và đảm bảo tính toàn vẹn ngữ cảnh (>2 biến)
            if feature_to_test in state and len(state) > 2:
                if state not in state_to_patient_indices:
                    state_to_patient_indices[state] = []
                state_to_patient_indices[state].append(i)
                
        # Xử lý các cụm trạng thái song song tối ưu hóa tài nguyên GPU
        for state, patient_indices in state_to_patient_indices.items():
            
            # Tạo lập trạng thái ứng viên thử nghiệm loại bỏ biến nhiễu mục tiêu
            new_state = tuple(f for f in state if f != feature_to_test)
            state_list = list(new_state)
            
            subset_test_x = test_patient_x_all.iloc[patient_indices]
            
            # Đẩy khối ma trận thu gọn xuống GPU để TabPFN đánh giá In-Context Learning
            tabpfn_model.fit(full_support_x[state_list], full_support_y)
            new_probs = tabpfn_model.predict_proba(subset_test_x[state_list])
            new_confs = 1.0 - entropy(new_probs, base=2, axis=1)
            
            # Đánh giá độ lệch Entropy tự tin cho từng thực thể cục bộ trong cụm
            for local_idx, global_idx in enumerate(patient_indices):
                curr_conf = new_confs[local_idx]
                
                # QUY TẮC KIỂM SOÁT (GATING RULE): Nếu việc gọt bỏ biến làm tăng/giữ vững độ tự tin và vượt ngưỡng an toàn
                if curr_conf >= best_confidences[global_idx] and curr_conf >= MINIMUM_CONFIDENCE_THRESHOLD:
                    current_states[global_idx] = new_state  # Chấp nhận vĩnh viễn việc hạ số chiều cho thực thể này
                    best_confidences[global_idx] = curr_conf
                    best_probabilities[global_idx] = new_probs[local_idx]
                else:
                    # TỰ ĐỘNG QUAY LUI (AUTOMATIC BACKTRACK): Từ chối loại bỏ, giữ nguyên không gian chiều cũ
                    pass
                    
    return current_states, best_probabilities

# =====================================================================
# 3. MAIN EXECUTION PROCESSOR
# =====================================================================
if __name__ == "__main__":
    print(f"[*] Initializing Batched-TTFS Acceleration Framework on: {DEVICE.upper()}")
    
    # Tải trực tiếp các phân vùng dữ liệu đối chứng của bộ dữ liệu Heart Attack
    training_pool = pd.read_csv(CANDIDATE_POOL_PATH)
    test_patient_pool = pd.read_csv(TEST_SET_PATH)
    
    if not os.path.exists(INDEX_FILE_PATH):
        raise FileNotFoundError(f"[ERROR] Master index map missing at: {INDEX_FILE_PATH}")
        
    with open(INDEX_FILE_PATH, 'r') as index_file:
        few_shot_indices_map = json.load(index_file)
        
    global_tabpfn_model = TabPFNClassifier(device=DEVICE)
    
    # Chạy thực nghiệm song song hai cấu hình Top-5 và Top-10 bẫy nhiễu địa lý
    for config_name, starting_features in CONFIGURATIONS_TO_RUN.items():
        output_directory = f"heart_attack_ttfs_tabpfn_{config_name}"
        os.makedirs(output_directory, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🚀 LAUNCHING FLEXIBLE-BATCHED TTFS: {config_name.upper()}")
        print(f"{'='*60}")
        
        for current_k in FEW_SHOT_K_VALUES:
            results_csv_path = os.path.join(output_directory, f'heart_attack_dynamic_results_k{current_k}.csv')
            
            if os.path.exists(results_csv_path):
                os.remove(results_csv_path)
                
            iteration_logs = []
            iter_progress = tqdm(range(TOTAL_ITERATIONS), desc=f"Evaluating Sequence K={current_k}")
            
            for current_iter in iter_progress:
                support_indices = few_shot_indices_map[str(current_k)][current_iter]
                support_x = training_pool.loc[support_indices]
                support_y = training_pool.loc[support_indices, TARGET]
                
                # Thực thi kiến trúc lõi TTFS duyệt Entropy động trên từng hàng dữ liệu bệnh nhân
                final_states, final_probs = execute_batched_flexible_ttfs(
                    tabpfn_model=global_tabpfn_model,
                    full_support_x=support_x,
                    full_support_y=support_y,
                    test_patient_x_all=test_patient_pool,
                    starting_feature_list=starting_features
                )
                
                # Tổng hợp cấu trúc nhật ký chi tiết và lưu danh sách các biến được giữ lại
                for global_idx in range(len(test_patient_pool)):
                    ground_truth_label = int(test_patient_pool.iloc[global_idx][TARGET])
                    predicted_label = 1 if final_probs[global_idx][1] > 0.5 else 0
                    retained_features_count = len(final_states[global_idx])
                    
                    retained_features_list_str = ";".join(final_states[global_idx])
                    real_test_idx = test_patient_pool.index[global_idx]
                    
                    iteration_logs.append([
                        current_iter, 
                        real_test_idx, 
                        ground_truth_label, 
                        predicted_label, 
                        retained_features_count,
                        retained_features_list_str
                    ])
                    
            pd.DataFrame(iteration_logs).to_csv(results_csv_path, index=False, header=False)
            print(f"✅ Sequence logging finalized for K={current_k} under configuration {config_name.upper()}.")