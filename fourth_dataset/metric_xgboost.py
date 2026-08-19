import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
ITERATIONS = 20
COL_NAMES = ['iter', 'id', 'gt', 'pred']
OUTPUT_SUMMARY = "cardio_xgboost_metrics_summary_all_configs.csv"

# Đường dẫn thư mục chứa tệp kết quả theo từng cấu hình của XGBoost
CONFIG_FILE_PATTERNS = {
    "XGBoost (Full)": os.path.join("xgboost_full_features", "cardio_xgboost_results_granular_k{k}.csv"),
    "XGBoost (Top 5)": os.path.join("cardio_xgboost_baselines_top5", "cardio_xgboost_results_granular_k{k}.csv"),
    "XGBoost (Top 10)": os.path.join("cardio_xgboost_baselines_top10", "cardio_xgboost_results_granular_k{k}.csv")
}

# ==========================================
# 2. HÀM TÍNH TOÁN METRICS (METRICS ENGINE)
# ==========================================
def calculate_metrics_for_config(file_path):
    if not os.path.exists(file_path):
        return None

    try:
        # Load dữ liệu thô (format: [iter, id, gt, pred])
        df = pd.read_csv(file_path, header=None, names=COL_NAMES)
    except Exception:
        return None

    # Lọc bỏ các mẫu dự đoán bị lỗi (-1) nếu có
    df = df[df['pred'] != -1]
    if len(df) == 0:
        return None

    iter_accs = []
    iter_f1s = []
    iter_precs = []
    iter_recs = []

    # Tính toán chỉ số cho từng vòng lặp
    for i in range(ITERATIONS):
        subset = df[df['iter'] == i]
        if len(subset) == 0:
            continue

        y_true = subset['gt'].values
        y_pred = subset['pred'].values

        iter_accs.append(accuracy_score(y_true, y_pred))
        iter_f1s.append(f1_score(y_true, y_pred, zero_division=0))
        iter_precs.append(precision_score(y_true, y_pred, zero_division=0))
        iter_recs.append(recall_score(y_true, y_pred, zero_division=0))

    if not iter_accs:
        return None

    return {
        'acc': (np.mean(iter_accs), np.std(iter_accs)),
        'f1': (np.mean(iter_f1s), np.std(iter_f1s)),
        'prec': (np.mean(iter_precs), np.std(iter_precs)),
        'rec': (np.mean(iter_recs), np.std(iter_recs))
    }

# ==========================================
# 3. THỰC THI & TỔNG HỢP (EXECUTION)
# ==========================================
def main():
    summary_data = []

    print(f"{'Config':<16} | {'K':<3} | {'Acc (±Std)':<16} | {'F1 (±Std)':<16} | {'Recall':<8}")
    print("-" * 72)

    for config_name, file_pattern in CONFIG_FILE_PATTERNS.items():
        for k in K_VALUES:
            file_path = file_pattern.format(k=k)
            res = calculate_metrics_for_config(file_path)

            if res:
                # In ra màn hình để kiểm tra nhanh
                print(f"{config_name:<16} | {k:<3} | {res['acc'][0]:.4f} ± {res['acc'][1]:.4f} | "
                      f"{res['f1'][0]:.4f} ± {res['f1'][1]:.4f} | {res['rec'][0]:.4f}")

                summary_data.append({
                    'Configuration': config_name,
                    'K': k,
                    'Accuracy_Mean': res['acc'][0],
                    'Accuracy_Std': res['acc'][1],
                    'F1_Mean': res['f1'][0],
                    'F1_Std': res['f1'][1],
                    'Precision_Mean': res['prec'][0],
                    'Precision_Std': res['prec'][1],
                    'Recall_Mean': res['rec'][0],
                    'Recall_Std': res['rec'][1]
                })
            else:
                print(f"{config_name:<16} | {k:<3} | Missing or incomplete file: {file_path}")

    # Xuất tệp CSV tổng hợp
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(OUTPUT_SUMMARY, index=False)
        print(f"\n[OK] Đã lưu bảng tổng hợp XGBoost vào: {OUTPUT_SUMMARY}")
    else:
        print("\n[!] Không tìm thấy dữ liệu hợp lệ để tổng hợp.")

if __name__ == "__main__":
    main()