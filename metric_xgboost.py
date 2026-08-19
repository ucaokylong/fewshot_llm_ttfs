import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# CẤU HÌNH (CONFIGURATION)
# ==========================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
OUTPUT_SUMMARY = "xgboost_granular_top10_summary.csv"

def calculate_metrics():
    summary_data = []

    print(f"{'K':<5} | {'Accuracy':<12} | {'F1-Score':<12} | {'Precision':<12}")
    print("-" * 50)

    for k in K_VALUES:
        file_path = f'granular_xgboost_results_k{k}.csv'
        
        if not os.path.exists(file_path):
            print(f"[!] Warning: Không tìm thấy file cho K={k}")
            continue
            
        # Load dữ liệu thô (không có header)
        # Format: [iteration, test_id, ground_truth, prediction]
        df = pd.read_csv(file_path, header=None, names=['iter', 'id', 'gt', 'pred'])
        
        # XGBoost có thể dự đoán ra -1 nếu bị lỗi (hiếm), ta nên lọc bỏ nếu có
        df = df[df['pred'] != -1]

        iter_accuracies = []
        iter_f1s = []
        iter_precisions = []

        # Tính toán metric cho từng Iteration để lấy độ lệch chuẩn
        for i in df['iter'].unique():
            subset = df[df['iter'] == i]
            y_true = subset['gt']
            y_pred = subset['pred']
            
            iter_accuracies.append(accuracy_score(y_true, y_pred))
            iter_f1s.append(f1_score(y_true, y_pred, zero_division=0))
            iter_precisions.append(precision_score(y_true, y_pred, zero_division=0))

        # Tính giá trị trung bình và độ lệch chuẩn (Standard Deviation)
        mean_acc = np.mean(iter_accuracies)
        std_acc = np.std(iter_accuracies)
        mean_f1 = np.mean(iter_f1s)
        std_f1 = np.std(iter_f1s)
        
        print(f"{k:<5} | {mean_acc:.4f} ± {std_acc:.4f} | {mean_f1:.4f} ± {std_f1:.4f} | {np.mean(iter_precisions):.4f}")

        summary_data.append({
            'K': k,
            'Accuracy_Mean': mean_acc,
            'Accuracy_Std': std_acc,
            'F1_Mean': mean_f1,
            'F1_Std': std_f1
        })

    # Lưu lại file tổng hợp để sếp vẽ biểu đồ
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"\n[OK] Đã lưu bảng tổng hợp vào: {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    calculate_metrics()