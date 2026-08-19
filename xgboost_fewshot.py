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

ALL_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU
# ==========================================
print(f"[*] Running XGBoost on: {DEVICE.upper()}")
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
# 3. THỰC THI THÍ NGHIỆM
# ==========================================
for k in K_VALUES:
    res_file = f'granular_xgboost_results_k{k}.csv'
    print(f"\n>>> [XGBoost] Đang chạy mốc K={k}")
    
    # Xóa file cũ nếu chạy lại từ đầu để tránh ghi đè dữ liệu rác
    if os.path.exists(res_file):
        os.remove(res_file)
    
    for i in range(ITERATIONS):
        # Lấy Support Set
        support_idx = master_indices[str(k)][i]
        X_train = train_pool.loc[support_idx, ALL_FEATURES]
        y_train = train_pool.loc[support_idx, 'Diabetes_binary']
        
        # Cấu hình XGBoost linh hoạt
        # tree_method='hist' là lựa chọn tối ưu cho cả CPU và GPU hiện nay
        model = xgb.XGBClassifier(
            tree_method='hist',
            device=DEVICE,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=i,
            verbosity=0,
            n_jobs=16 # Tận dụng 16 cores CPU sếp request trong file sh
        )
        
        # Train
        model.fit(X_train, y_train)
        
        # Dự đoán
        X_test = test_fixed[ALL_FEATURES]
        predictions = model.predict(X_test)
        
        # Lưu log ngay lập tức sau mỗi Iteration
        iter_logs = []
        for idx, pred in zip(test_fixed.index, predictions):
            iter_logs.append([i, idx, int(test_fixed.loc[idx, 'Diabetes_binary']), int(pred)])
        
        pd.DataFrame(iter_logs).to_csv(res_file, mode='a', index=False, header=False)
        print(f"K={k} | Iteration {i+1}/{ITERATIONS} hoàn tất.", end='\r')
    
    print(f"\n[OK] Đã hoàn thành mốc K={k}")

print("\n[SUCCESS] Hoàn thành toàn bộ thí nghiệm XGBoost.")