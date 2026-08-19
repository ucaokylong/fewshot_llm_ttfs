import pandas as pd
import json
import torch
import shap
import numpy as np
import os
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDICES_PATH = "master_granular_indices.json"
DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
K_TARGET = "16" 
ITERATIONS = 20

# Tạo thư mục lưu tạm để chống mất dữ liệu khi crash
TEMP_DIR = f"xai_temp_k{K_TARGET}"
os.makedirs(TEMP_DIR, exist_ok=True)

ALL_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# ==========================================
# 2. LOAD DỮ LIỆU
# ==========================================
print(f"[*] Khởi động trích xuất XAI Global Ranking trên K={K_TARGET}...")
df = pd.read_csv(DATA_PATH)
train_pool, test_fixed = train_test_split(
    df, test_size=1000, stratify=df['Diabetes_binary'], random_state=42
)

with open(INDICES_PATH, 'r') as f:
    master_indices = json.load(f)

all_importance_results = []
clf = TabPFNClassifier(device=DEVICE)

# ==========================================
# 3. QUÉT SHAP (ĐÃ FIX LỖI & THÊM RESUME)
# ==========================================
for i in range(ITERATIONS):
    temp_file = os.path.join(TEMP_DIR, f"iter_{i}.csv")
    
    # Nếu iteration này đã chạy xong trước đó thì load lại, không chạy lại
    if os.path.exists(temp_file):
        print(f"[*] Đã tìm thấy kết quả Iteration {i+1}/{ITERATIONS}. Bỏ qua chạy lại...")
        all_importance_results.append(pd.read_csv(temp_file))
        continue

    print(f"--- Đang xử lý Iteration {i+1}/{ITERATIONS} ---")
    support_idx = master_indices[K_TARGET][i]
    X_train = train_pool.loc[support_idx, ALL_FEATURES]
    y_train = train_pool.loc[support_idx, 'Diabetes_binary']

    clf.fit(X_train, y_train)

    explainer = shap.KernelExplainer(clf.predict_proba, X_train)
    shap_values = explainer.shap_values(X_train)

    # ---------- FIX LỖI SHAPE Ở ĐÂY ----------
    # Bắt mọi trường hợp SHAP trả về List hoặc Mảng Numpy
    if isinstance(shap_values, list):
        importance_matrix = np.abs(shap_values[1])
    else:
        if len(shap_values.shape) == 3:
            importance_matrix = np.abs(shap_values[:, :, 1])
        else:
            importance_matrix = np.abs(shap_values)

    # Tính điểm trung bình dọc theo trục 0 (các bệnh nhân) để lấy điểm cho 21 features
    importance_scores = importance_matrix.mean(axis=0)

    # Đảm bảo chắc chắn là mảng 1D độ dài 21
    if importance_scores.ndim == 0:
        importance_scores = np.full(len(ALL_FEATURES), importance_scores)
    elif len(importance_scores) != len(ALL_FEATURES):
        print(f"[!] Cảnh báo: Kích thước sai {importance_scores.shape}")
    # ------------------------------------------

    iter_df = pd.DataFrame({
        'feature': ALL_FEATURES,
        f'iter_{i}': importance_scores
    })
    
    # Lưu tạm kết quả xuống ổ cứng ngay lập tức
    iter_df.to_csv(temp_file, index=False)
    all_importance_results.append(iter_df)

# ==========================================
# 4. TỔNG HỢP VÀ XẾP HẠNG TOÀN CỤC
# ==========================================
print("\n[*] Đang tổng hợp và tính điểm...")
final_df = all_importance_results[0]
for next_df in all_importance_results[1:]:
    final_df = final_df.merge(next_df, on='feature')

iter_cols = [f'iter_{i}' for i in range(ITERATIONS)]
final_df['mean_importance'] = final_df[iter_cols].mean(axis=1)
final_df['std_importance'] = final_df[iter_cols].std(axis=1)

ranking_df = final_df[['feature', 'mean_importance', 'std_importance']].sort_values(
    by='mean_importance', ascending=False
).reset_index(drop=True)

ranking_df.insert(0, 'rank', ranking_df.index + 1)

print("\n\n" + "="*60)
print(f"BẢNG XẾP HẠNG ĐẶC TRƯNG TOÀN CỤC (K={K_TARGET} | 20 ITERS)")
print("="*60)
print(ranking_df.to_string(index=False))
print("="*60)

# ==========================================
# 5. LƯU KẾT QUẢ
# ==========================================
ranking_df.to_csv(f"xai_global_ranking_k{K_TARGET}.csv", index=False)

top_5 = ranking_df.head(5)['feature'].tolist()
top_10 = ranking_df.head(10)['feature'].tolist()

with open(f"xai_selected_features_k{K_TARGET}.json", "w") as f:
    json.dump({
        "full_ranking": ranking_df.to_dict('records'),
        "top_5": top_5,
        "top_10": top_10
    }, f, indent=4)

print(f"\n[SUCCESS] Đã lưu bảng xếp hạng vào xai_global_ranking_k{K_TARGET}.csv")
print(f"[*] Top 5 đề xuất cho NumLog: {top_5}")