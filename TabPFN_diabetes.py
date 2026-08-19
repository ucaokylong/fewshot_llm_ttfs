import os
import json
import torch
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
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
# 2. CHUẨN BỊ DỮ LIỆU & ĐỒNG BỘ INDEX
# ==========================================
print(f"[*] Đang chạy trên thiết bị: {DEVICE.upper()}")
df = pd.read_csv(DATA_PATH)

# Bước này cực kỳ quan trọng: Luôn fix random_state=42 để 1000 mẫu test là bất biến
train_pool, test_fixed = train_test_split(
    df, test_size=1000, stratify=df['Diabetes_binary'], random_state=42
)

# --- BỘ TẠO INDEX ĐỒNG BỘ (CẤY TỪ CODE QWEN) ---
if not os.path.exists(INDICES_PATH):
    print(f"[!] Không tìm thấy {INDICES_PATH}. Đang khởi tạo bộ Index đồng bộ...")
    master_indices = {str(k): [] for k in K_VALUES}
    for k in K_VALUES:
        for i in range(ITERATIONS):
            # Bốc mẫu cân bằng 50/50 cho Support Set (Few-shot context)
            # Dùng random_state=i để mỗi Iteration là duy nhất nhưng có thể tái hiện
            idx0 = train_pool[train_pool['Diabetes_binary']==0].sample(k//2, random_state=i).index.tolist()
            idx1 = train_pool[train_pool['Diabetes_binary']==1].sample(k//2, random_state=i).index.tolist()
            master_indices[str(k)].append(idx0 + idx1)
            
    with open(INDICES_PATH, 'w') as f:
        json.dump(master_indices, f)
    print(f"[SUCCESS] Đã tạo xong file index: {INDICES_PATH}")
else:
    print(f"[*] Đã nhận diện file index có sẵn: {INDICES_PATH}")
    with open(INDICES_PATH, 'r') as f:
        master_indices = json.load(f)

# ==========================================
# 3. KHỞI TẠO MÔ HÌNH TABPFN
# ==========================================
# TabPFN v2.x tối ưu hóa rất tốt trên GPU
clf = TabPFNClassifier(device=DEVICE)

# ==========================================
# 4. CHẠY THÍ NGHIỆM
# ==========================================
for k in K_VALUES:
    res_file = f'granular_tabpfn_results_k{k}.csv'
    print(f"\n>>> [TabPFN] Đang xử lý mốc K={k}")
    
    all_logs = []
    for i in range(ITERATIONS):
        # Lấy đúng index từ bộ não chung
        support_idx = master_indices[str(k)][i]
        X_train = train_pool.loc[support_idx, ALL_FEATURES]
        y_train = train_pool.loc[support_idx, 'Diabetes_binary']
        
        X_test = test_fixed[ALL_FEATURES]
        
        # In-context Learning: TabPFN không train, nó chỉ hấp thụ context
        clf.fit(X_train, y_train)
        
        # Dự đoán cho 1000 mẫu test
        predictions = clf.predict(X_test)
        
        for idx, pred in zip(test_fixed.index, predictions):
            all_logs.append([i, idx, int(test_fixed.loc[idx, 'Diabetes_binary']), int(pred)])
            
        print(f"Iteration {i+1}/20 hoàn tất.", end='\r')
    
    pd.DataFrame(all_logs).to_csv(res_file, index=False, header=False)
    print(f"\n[OK] Kết quả K={k} đã được lưu.")

print("\n[FINISH] Toàn bộ dải Granular cho TabPFN đã hoàn tất.")