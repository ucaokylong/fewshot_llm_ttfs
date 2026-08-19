import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
ITERATIONS = 20
COL_NAMES = ['iteration', 'test_id', 'gt', 'pred']
OUTPUT_SUMMARY = "cardio_tabpfn_metrics_summary_all_configs.csv"

# Cập nhật đường dẫn thư mục chứa tệp kết quả theo đúng cấu trúc của sếp
CONFIG_FILE_PATTERNS = {
    "TabPFN (Full)": os.path.join("tabPFN_full_features", "cardio_tabpfn_results_granular_k{k}.csv"),
    "TabPFN (Top 5)": os.path.join("cardio_tabpfn_baselines_top5", "cardio_tabpfn_results_granular_k{k}.csv"),
    "TabPFN (Top 10)": os.path.join("cardio_tabpfn_baselines_top10", "cardio_tabpfn_results_granular_k{k}.csv")
}

# ==========================================
# 2. HÀM TÍNH TOÁN METRICS (METRICS ENGINE)
# ==========================================
def calculate_metrics(file_path):
    if not os.path.exists(file_path):
        return None
    
    try:
        # Đọc dữ liệu log của TabPFN (không có header)
        df = pd.read_csv(file_path, header=None, names=COL_NAMES)
    except Exception:
        return None
    
    metrics_per_iter = []
    for i in range(ITERATIONS):
        sub = df[df['iteration'] == i]
        if len(sub) == 0: 
            continue
        
        y_true = sub['gt'].values
        y_pred = sub['pred'].values
        
        # Lọc bỏ các mẫu lỗi chưa dự đoán (-1) nếu có
        valid_mask = y_pred != -1
        if not np.any(valid_mask):
            continue
            
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        # Tính toán các chỉ số cho từng vòng lặp (Iteration)
        acc = (y_pred_valid == y_true_valid).mean()
        prec = precision_score(y_true_valid, y_pred_valid, zero_division=0)
        rec = recall_score(y_true_valid, y_pred_valid, zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0)
        
        metrics_per_iter.append([acc, prec, rec, f1])
    
    if not metrics_per_iter: 
        return None
    
    # Tính Mean và Std qua 20 vòng lặp
    metrics_array = np.array(metrics_per_iter)
    means = np.mean(metrics_array, axis=0)
    stds = np.std(metrics_array, axis=0)
    
    return {
        'acc': (means[0], stds[0]),
        'prec': (means[1], stds[1]),
        'rec': (means[2], stds[2]),
        'f1': (means[3], stds[3])
    }

# ==========================================
# 3. THỰC THI & TỔNG HỢP (EXECUTION)
# ==========================================
summary_results = []

print(f"{'Config':<16} | {'K':<3} | {'Acc (±Std)':<16} | {'F1 (±Std)':<16} | {'Recall':<8}")
print("-" * 72)

for config_name, file_pattern in CONFIG_FILE_PATTERNS.items():
    for k in K_VALUES:
        file_path = file_pattern.format(k=k)
        res = calculate_metrics(file_path)
        
        if res:
            # In nhanh ra Terminal để theo dõi
            print(f"{config_name:<16} | {k:<3} | {res['acc'][0]:.3f}±{res['acc'][1]:.3f} | "
                  f"{res['f1'][0]:.3f}±{res['f1'][1]:.3f} | {res['rec'][0]:.3f}")
            
            # Lưu vào danh sách tổng hợp
            summary_results.append({
                'Configuration': config_name,
                'K': k,
                'Acc_Mean': res['acc'][0], 
                'Acc_Std': res['acc'][1],
                'F1_Mean': res['f1'][0], 
                'F1_Std': res['f1'][1],
                'Precision_Mean': res['prec'][0],
                'Precision_Std': res['prec'][1],
                'Recall_Mean': res['rec'][0],
                'Recall_Std': res['rec'][1]
            })
        else:
            print(f"{config_name:<16} | {k:<3} | Missing file: {file_path}")

# Xuất bảng tổng hợp ra file CSV
if summary_results:
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"\n[SUCCESS] Aggregated metrics safely saved to: {OUTPUT_SUMMARY}")
else:
    print("\n[WARNING] No valid log files found to calculate metrics.")