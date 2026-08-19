import os
import json
import torch
import pandas as pd
import numpy as np
from scipy.stats import entropy
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =====================================================================
# 1. CẤU HÌNH HỆ THỐNG VÀ KHÔNG GIAN ĐẶC TRƯNG
# =====================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
INDEX_FILE_PATH = "master_granular_indices.json"

FEW_SHOT_K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
TOTAL_ITERATIONS = 20
MINIMUM_CONFIDENCE_THRESHOLD = 0.60

# Xếp hạng SHAP từ thực nghiệm
SHAP_RANKING = ["Age", "BMI", "Income", "HighBP", "HighChol", "GenHlth", "PhysHlth", "Sex", "PhysActivity", "MentHlth"]

TOP_5_BASE_FEATURES = SHAP_RANKING[:5]
TOP_10_BASE_FEATURES = SHAP_RANKING[:10]

CONFIGURATIONS_TO_RUN = {
    "top5": TOP_5_BASE_FEATURES,
    "top10": TOP_10_BASE_FEATURES
}

# =====================================================================
# 2. LÕI THUẬT TOÁN: BATCHED FLEXIBLE TTFS (CHO PHÉP TỔ HỢP TỰ DO)
# =====================================================================
def execute_batched_flexible_ttfs(tabpfn_model, full_support_x, full_support_y, test_patient_x_all, starting_feature_list):
    """
    Thuật toán TTFS nâng cấp: Cho phép loại bỏ bất kỳ tính năng nhiễu nào ở giữa danh sách.
    Tận dụng Batching để gom nhóm các bệnh nhân có cùng trạng thái hình học đặc trưng.
    """
    num_patients = len(test_patient_x_all)
    
    # Trạng thái ban đầu của từng bệnh nhân chứa đầy đủ danh sách đặc trưng xuất phát
    current_states = [tuple(starting_feature_list) for _ in range(num_patients)]
    
    best_confidences = np.zeros(num_patients)
    best_probabilities = np.zeros((num_patients, 2))
    
    # ---------------------------------------------------------
    # BƯỚC 1: Thiết lập mốc điểm tự tin cơ sở ban đầu (Full Features)
    # ---------------------------------------------------------
    base_features = list(starting_feature_list)
    tabpfn_model.fit(full_support_x[base_features], full_support_y)
    
    base_probs = tabpfn_model.predict_proba(test_patient_x_all[base_features])
    base_confs = 1.0 - entropy(base_probs, base=2, axis=1)
    
    best_confidences[:] = base_confs
    best_probabilities[:] = base_probs
    
    # ---------------------------------------------------------
    # BƯỚC 2: Duyệt tham lam từng đặc trưng đơn lẻ từ dưới lên trên
    # ---------------------------------------------------------
    for feature_to_test in reversed(starting_feature_list):
        
        # Gom cụm các bệnh nhân đang có chung không gian đặc trưng tại vòng lặp này
        state_to_patient_indices = {}
        for i, state in enumerate(current_states):
            # Chỉ thử nghiệm gọt tỉa nếu đặc trưng này tồn tại trong bộ nhớ và số lượng còn lại > 2
            if feature_to_test in state and len(state) > 2:
                if state not in state_to_patient_indices:
                    state_to_patient_indices[state] = []
                state_to_patient_indices[state].append(i)
                
        # Duyệt song song qua từng nhóm trạng thái
        for state, patient_indices in state_to_patient_indices.items():
            
            # Tạo trạng thái mới: Loại bỏ ĐÚNG đặc trưng đang xét (bảo toàn các lỗ thủng đã có trước đó)
            new_state = tuple(f for f in state if f != feature_to_test)
            state_list = list(new_state)
            
            subset_test_x = test_patient_x_all.iloc[patient_indices]
            
            # Đẩy nguyên ma trận nhóm lên GPU xử lý 1 lệnh duy nhất
            tabpfn_model.fit(full_support_x[state_list], full_support_y)
            new_probs = tabpfn_model.predict_proba(subset_test_x[state_list])
            new_confs = 1.0 - entropy(new_probs, base=2, axis=1)
            
            # Đánh giá độc lập cho từng thực thể trong nhóm
            for local_idx, global_idx in enumerate(patient_indices):
                curr_conf = new_confs[local_idx]
                
                # LUẬT CHỐT CHẶN: Nếu việc gọt bỏ giúp tăng hoặc giữ nguyên độ tự tin, và đạt ngưỡng an toàn
                if curr_conf >= best_confidences[global_idx] and curr_conf >= MINIMUM_CONFIDENCE_THRESHOLD:
                    current_states[global_idx] = new_state  # Chấp nhận gọt bỏ vĩnh viễn đặc trưng này
                    best_confidences[global_idx] = curr_conf
                    best_probabilities[global_idx] = new_probs[local_idx]
                else:
                    # LỆNH BACKTRACK TỰ ĐỘNG: Giữ nguyên current_states[global_idx]
                    pass
                    
    return current_states, best_probabilities

# =====================================================================
# 3. TRÌNH ĐIỀU KHIỂN CHÍNH (MAIN PROCESSOR)
# =====================================================================
if __name__ == "__main__":
    print(f"[*] Khởi động Hệ thống Tổ hợp siêu tốc Batched-TTFS trên: {DEVICE.upper()}")
    
    master_dataframe = pd.read_csv(DATA_FILE_PATH)
    training_pool, test_patient_pool = train_test_split(
        master_dataframe, test_size=1000, stratify=master_dataframe['Diabetes_binary'], random_state=42
    )
    
    with open(INDEX_FILE_PATH, 'r') as index_file:
        few_shot_indices_map = json.load(index_file)
        
    global_tabpfn_model = TabPFNClassifier(device=DEVICE)
    
    for config_name, starting_features in CONFIGURATIONS_TO_RUN.items():
        output_directory = f"ttfs_tabpfn_{config_name}"
        os.makedirs(output_directory, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🚀 RUNNING FLEXIBLE-BATCHED TTFS: {config_name.upper()}")
        print(f"{'='*60}")
        
        for current_k in FEW_SHOT_K_VALUES:
            results_csv_path = os.path.join(output_directory, f'dynamic_results_k{current_k}.csv')
            
            if os.path.exists(results_csv_path):
                os.remove(results_csv_path)
                
            iteration_logs = []
            iter_progress = tqdm(range(TOTAL_ITERATIONS), desc=f"Tiến độ K={current_k}")
            
            for current_iter in iter_progress:
                support_indices = few_shot_indices_map[str(current_k)][current_iter]
                support_x = training_pool.loc[support_indices]
                support_y = training_pool.loc[support_indices, 'Diabetes_binary']
                
                # ĐÃ SỬA: Gọi đúng tên hàm execute_batched_flexible_ttfs đồng bộ với Mục 2
                final_states, final_probs = execute_batched_flexible_ttfs(
                    tabpfn_model=global_tabpfn_model,
                    full_support_x=support_x,
                    full_support_y=support_y,
                    test_patient_x_all=test_patient_pool,
                    starting_feature_list=starting_features
                )
                
                # Đóng gói nhật ký dữ liệu sạch
                for global_idx in range(len(test_patient_pool)):
                    ground_truth_label = int(test_patient_pool.iloc[global_idx]['Diabetes_binary'])
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
            print(f"✅ Đã kết xuất dữ liệu sạch mốc K={current_k} cho cấu hình {config_name.upper()}.")