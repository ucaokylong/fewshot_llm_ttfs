import os
import json
import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
# Tự động nhận diện thiết bị
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
INDICES_PATH = "master_granular_indices.json" 
ITERATIONS = 20
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16] 

# Phân tách cấu hình Feature theo chuẩn SHAP
TOP_5_FEATURES = ["Age", "BMI", "Income", "HighBP", "HighChol"]
TOP_10_FEATURES = ["Age", "BMI", "Income", "HighBP", "HighChol", "GenHlth", "PhysHlth", "Sex", "PhysActivity", "MentHlth"]

FEATURE_SETS = {
    "top5": TOP_5_FEATURES,
    "top10": TOP_10_FEATURES
}

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU
# ==========================================
print(f"[*] Running XGBoost Baselines on: {DEVICE.upper()}")
df = pd.read_csv(DATA_PATH)

train_pool, test_fixed = train_test_split(
    df, test_size=1000, stratify=df['Diabetes_binary'], random_state=42
)

if not os.path.exists(INDICES_PATH):
    print(f"[ERROR] Không tìm thấy {INDICES_PATH}. Sếp chạy script tạo index trước nhé!")
    exit()

with open(INDICES_PATH, 'r') as f:
    master_indices = json.load(f)

# ==========================================
# 3. THỰC THI THÍ NGHIỆM (VÒNG LẶP KÉP)
# ==========================================
for config_name, feature_list in FEATURE_SETS.items():
    print(f"\n{'='*50}")
    print(f">>> BẮT ĐẦU CHẠY XGBOOST PHIÊN BẢN: {config_name.upper()} ({len(feature_list)} Features)")
    print(f"{'='*50}")
    
    # Tạo thư mục xuất kết quả độc lập
    output_dir = f"xgboost_baselines_{config_name}"
    os.makedirs(output_dir, exist_ok=True)

    for k in K_VALUES:
        res_file = os.path.join(output_dir, f'granular_xgboost_results_k{k}.csv')
        print(f"\n>>> [XGBoost - {config_name}] Đang chạy mốc K={k}")
        
        # Xóa file cũ nếu chạy lại từ đầu để tránh ghi đè dữ liệu rác
        if os.path.exists(res_file):
            os.remove(res_file)
        
        for i in range(ITERATIONS):
            # Lấy Support Set và chỉ cắt đúng những feature được yêu cầu
            support_idx = master_indices[str(k)][i]
            X_train = train_pool.loc[support_idx, feature_list]
            y_train = train_pool.loc[support_idx, 'Diabetes_binary']
            
            # Cấu hình XGBoost
            model = xgb.XGBClassifier(
                tree_method='hist',
                device=DEVICE,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=i,
                verbosity=0,
                n_jobs=16 
            )
            
            # Train trực tiếp trên tập feature đã bị cắt xén
            model.fit(X_train, y_train)
            
            # Dự đoán trên tập Test cũng phải cắt theo feature list
            X_test = test_fixed[feature_list]
            predictions = model.predict(X_test)
            
            # Lưu log ngay lập tức sau mỗi Iteration
            iter_logs = []
            for idx, pred in zip(test_fixed.index, predictions):
                iter_logs.append([i, idx, int(test_fixed.loc[idx, 'Diabetes_binary']), int(pred)])
            
            pd.DataFrame(iter_logs).to_csv(res_file, mode='a', index=False, header=False)
            print(f"K={k} | Iteration {i+1}/{ITERATIONS} hoàn tất.", end='\r')
        
        print(f"\n[OK] Đã hoàn thành mốc K={k} cho {config_name}")

print("\n" + "="*50)
print("[SUCCESS] Hoàn thành toàn bộ thí nghiệm XGBoost! Data đã nằm gọn trong các thư mục.")
print("="*50)