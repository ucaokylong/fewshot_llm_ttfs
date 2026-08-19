import pd as pd
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================
# 1. CẤU HÌNH ĐỒNG BỘ GRANULAR
# ==========================================
# Cập nhật dải K mới
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16] 
ITERATIONS = 20
COL_NAMES = ['iteration', 'test_id', 'gt', 'pred']
OUTPUT_FILE = "final_granular_comparison_results.csv"

def calculate_metrics(file_path, is_llm=False):
    if not os.path.exists(file_path):
        # In thông báo để sếp biết file nào còn thiếu
        # print(f"[!] Warning: Missing {file_path}")
        return None
    
    df = pd.read_csv(file_path, header=None, names=COL_NAMES)
    metrics_list = []
    
    for i in range(ITERATIONS):
        sub = df[df['iteration'] == i]
        if len(sub) == 0: continue
        
        y_true = sub['gt'].tolist()
        y_pred = sub['pred'].tolist()
        
        # Phạt lỗi Parsing của LLM: -1 trở thành dự đoán sai
        y_pred_fixed = [p if p != -1 else (1 - t) for p, t in zip(y_pred, y_true)]
        
        acc = (np.array(y_pred_fixed) == np.array(y_true)).mean()
        prec = precision_score(y_true, y_pred_fixed, zero_division=0)
        rec = recall_score(y_true, y_pred_fixed, zero_division=0)
        f1 = f1_score(y_true, y_pred_fixed, zero_division=0)
        
        invalid_rate = (sub['pred'] == -1).sum() / len(sub) if is_llm else 0
        metrics_list.append([acc, prec, rec, f1, invalid_rate])
    
    if not metrics_list: return None
    
    metrics_array = np.array(metrics_list)
    means = np.mean(metrics_array, axis=0)
    stds = np.std(metrics_array, axis=0)
    
    return {
        'acc': (means[0], stds[0]), 'prec': (means[1], stds[1]),
        'rec': (means[2], stds[2]), 'f1': (means[3], stds[3]),
        'invalid': means[4]
    }

# ==========================================
# 2. THỰC THI (ĐÃ ĐỔI TÊN FILE KHỚP VỚI CÁC SCRIPT TRƯỚC)
# ==========================================
summary_data = []

print(f"{'K':<3} | {'Method':<10} | {'Acc (±Std)':<16} | {'F1 (±Std)':<16} | {'Recall':<8} | {'Invalid':<8}")
print("-" * 85)

for k in K_VALUES:
    # TÊN FILE PHẢI KHỚP VỚI GRANULAR PREFIX
    models = [
        ('NumLog', f'granular_numlog_test_k{k}.csv', False), # Sếp cần script test cho NumLog (xem bên dưới)
        ('TabPFN', f'granular_tabpfn_results_k{k}.csv', False),
        ('XGBoost', f'granular_xgboost_results_k{k}.csv', False),
        ('Qwen2.5', f'granular_qwen_results_k{k}.csv', True)
    ]
    
    for name, path, is_llm in models:
        res = calculate_metrics(path, is_llm)
        if res:
            print(f"{k:<3} | {name:<10} | {res['acc'][0]:.3f}±{res['acc'][1]:.3f} | "
                  f"{res['f1'][0]:.3f}±{res['f1'][1]:.3f} | {res['rec'][0]:.3f} | {res['invalid']*100:.1f}%")
            
            summary_data.append({
                'K': k, 'Method': name,
                'Acc_Mean': res['acc'][0], 'Acc_Std': res['acc'][1],
                'F1_Mean': res['f1'][0], 'F1_Std': res['f1'][1],
                'Invalid_Rate': res['invalid'] * 100
            })

pd.DataFrame(summary_data).to_csv(OUTPUT_FILE, index=False)
print(f"\n[SUCCESS] Kết quả cuối cùng đã lưu tại: {OUTPUT_FILE}")