import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================
# 1. CẤU HÌNH (Configuration)
# ==========================================
# Khớp với dải K granular sếp vừa chạy trên HAKUSAN
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
ITERATIONS = 20
COL_NAMES = ['iteration', 'test_id', 'gt', 'pred']
OUTPUT_SUMMARY = "tabpfn_granular_all_features_metrics_summary.csv"

def calculate_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"[!] Canh bao: Thieu file {file_path}")
        return None
    
    # Doc du lieu (TabPFN thuong khong co header trong file log cua minh)
    df = pd.read_csv(file_path, header=None, names=COL_NAMES)
    
    metrics_per_iter = []
    for i in range(ITERATIONS):
        sub = df[df['iteration'] == i]
        if len(sub) == 0: continue
        
        y_true = sub['gt'].values
        y_pred = sub['pred'].values
        
        # Tinh toan metrics cho tung Iteration
        acc = (y_pred == y_true).mean()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics_per_iter.append([acc, prec, rec, f1])
    
    if not metrics_per_iter: return None
    
    # Chuyen sang numpy de tinh Mean va Std
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
# 2. THỰC THI & HIỂN THỊ
# ==========================================
summary_results = []

print(f"{'K':<3} | {'Method':<10} | {'Acc (±Std)':<16} | {'F1 (±Std)':<16} | {'Recall':<8}")
print("-" * 70)

for k in K_VALUES:
    file_name = f'framingham_tabpfn_results_granular_k{k}.csv'
    res = calculate_metrics(file_name)
    
    if res:
        # In ra Terminal de sếp xem nhanh
        print(f"{k:<3} | {'TabPFN':<10} | {res['acc'][0]:.3f}±{res['acc'][1]:.3f} | "
              f"{res['f1'][0]:.3f}±{res['f1'][1]:.3f} | {res['rec'][0]:.3f}")
        
        # Luu vao list de xuat CSV
        summary_results.append({
            'K': k,
            'Method': 'TabPFN',
            'Acc_Mean': res['acc'][0], 'Acc_Std': res['acc'][1],
            'F1_Mean': res['f1'][0], 'F1_Std': res['f1'][1],
            'Precision_Mean': res['prec'][0],
            'Recall_Mean': res['rec'][0]
        })

# Xuất file tổng hợp
pd.DataFrame(summary_results).to_csv(OUTPUT_SUMMARY, index=False)
print(f"\n[SUCCESS] Da tong hop xong. Ket qua luu tai: {OUTPUT_SUMMARY}")