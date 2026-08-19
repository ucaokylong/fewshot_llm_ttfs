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

# Phân tách cấu hình Feature theo chuẩn SHAP của sếp
TOP_5_FEATURES = ["Age", "BMI", "Income", "HighBP", "HighChol"]
TOP_10_FEATURES = ["Age", "BMI", "Income", "HighBP", "HighChol", "GenHlth", "PhysHlth", "Sex", "PhysActivity", "MentHlth"]

FEATURE_SETS = {
    "top5": TOP_5_FEATURES,
    "top10": TOP_10_FEATURES
}

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU & ĐỒNG BỘ INDEX
# ==========================================
print(f"[*] Đang chạy TabPFN Baselines trên thiết bị: {DEVICE.upper()}")
df = pd.read_csv(DATA_PATH)

# Bước này cực kỳ quan trọng: Luôn fix random_state=42 để 1000 mẫu test là bất biến
train_pool, test_fixed = train_test_split(
    df, test_size=1000, stratify=df['Diabetes_binary'], random_state=42
)

# --- BỘ TẠO INDEX ĐỒNG BỘ ---
if not os.path.exists(INDICES_PATH):
    print(f"[!] Không tìm thấy {INDICES_PATH}. Đang khởi tạo bộ Index đồng bộ...")
    master_indices = {str(k): [] for k in K_VALUES}
    for k in K_VALUES:
        for i in range(ITERATIONS):
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
clf = TabPFNClassifier(device=DEVICE)

# ==========================================
# 4. CHẠY THÍ NGHIỆM (VÒNG LẶP KÉP CẮT FEATURE)
# ==========================================
for config_name, feature_list in FEATURE_SETS.items():
    print(f"\n{'='*50}")
    print(f">>> BẮT ĐẦU CHẠY TABPFN PHIÊN BẢN: {config_name.upper()} ({len(feature_list)} Features)")
    print(f"{'='*50}")
    
    # Tạo thư mục xuất kết quả độc lập
    output_dir = f"tabpfn_baselines_{config_name}"
    os.makedirs(output_dir, exist_ok=True)

    for k in K_VALUES:
        res_file = os.path.join(output_dir, f'granular_tabpfn_results_k{k}.csv')
        print(f"\n>>> [TabPFN - {config_name}] Đang xử lý mốc K={k}")
        
        # Xóa file cũ nếu chạy lại từ đầu để tránh ghi đè dữ liệu rác
        if os.path.exists(res_file):
            os.remove(res_file)
            
        for i in range(ITERATIONS):
            # Lấy đúng index từ bộ não chung và ép số chiều theo config
            support_idx = master_indices[str(k)][i]
            X_train = train_pool.loc[support_idx, feature_list]
            y_train = train_pool.loc[support_idx, 'Diabetes_binary']
            
            X_test = test_fixed[feature_list]
            
            # In-context Learning: TabPFN không train, nó chỉ hấp thụ context
            clf.fit(X_train, y_train)
            
            # Dự đoán cho 1000 mẫu test
            predictions = clf.predict(X_test)
            
            # Ghi log ngay lập tức sau mỗi Iteration
            iter_logs = []
            for idx, pred in zip(test_fixed.index, predictions):
                iter_logs.append([i, idx, int(test_fixed.loc[idx, 'Diabetes_binary']), int(pred)])
                
            pd.DataFrame(iter_logs).to_csv(res_file, mode='a', index=False, header=False)
            print(f"K={k} | Iteration {i+1}/{ITERATIONS} hoàn tất.", end='\r')
        
        print(f"\n[OK] Kết quả K={k} cho {config_name} đã được lưu an toàn.")

print("\n" + "="*50)
print("[FINISH] Toàn bộ dải Granular tĩnh cho TabPFN đã hoàn tất.")
print("="*50)