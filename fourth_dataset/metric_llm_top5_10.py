import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]

FOLDERS_CONFIG = {
    "cardio_llm_baselines_top5": {
        "output_summary": "cardio_llm_baselines_top5_summary.csv",
        "method_name": "Qwen-7B (Static Top 5)"
    },
    "cardio_llm_baselines_top10": {
        "output_summary": "cardio_llm_baselines_top10_summary.csv",
        "method_name": "Qwen-7B (Static Top 10)"
    }
}

# ==========================================
# 2. HÀM TÍNH TOÁN METRICS (METRICS ENGINE)
# ==========================================
def calculate_static_baseline_metrics():
    for folder_name, config in FOLDERS_CONFIG.items():
        summary_data = []
        output_summary = config["output_summary"]
        method_name = config["method_name"]

        print(f"\n{'='*85}")
        print(f"📊 Calculating Metrics for Folder: {folder_name.upper()}")
        print(f"{'='*85}")
        print(f"{'K':<5} | {'Accuracy Mean':<18} | {'F1 Mean':<18} | {'Recall Mean':<12} | {'Precision Mean'}")
        print("-" * 85)

        for k in K_VALUES:
            file_path = os.path.join(folder_name, f"qwen_results_granular_k{k}.csv")
            
            if not os.path.exists(file_path):
                print(f"[!] Warning: Không tìm thấy tệp cho K={k} tại: {file_path}")
                continue
                
            try:
                # Format dữ liệu từ LLM Baselines: [iter, id, ground_truth, prediction]
                df = pd.read_csv(file_path, header=None, names=['iter', 'id', 'gt', 'pred'])
            except Exception as e:
                print(f"[ERROR] Lỗi đọc file K={k} ({file_path}): {e}")
                continue

            iter_accuracies = []
            iter_f1s = []
            iter_precisions = []
            iter_recalls = []

            iterations = df['iter'].unique()
            
            for i in iterations:
                subset = df[df['iter'] == i]
                if len(subset) == 0:
                    continue
                
                # Ép kiểu dữ liệu về numeric an toàn
                y_true = pd.to_numeric(subset['gt'], errors='coerce').fillna(0).astype(int)
                y_pred_raw = pd.to_numeric(subset['pred'], errors='coerce').fillna(-1).astype(int)
                
                # Xử lý lỗi sinh token (-1): gán nhãn dự đoán ngược lại với ground truth
                y_pred_cleaned = np.where(
                    (y_pred_raw != 0) & (y_pred_raw != 1), 
                    1 - y_true, 
                    y_pred_raw
                )
                
                # Tính các chỉ số đánh giá cho từng iteration
                acc = accuracy_score(y_true, y_pred_cleaned)
                f1 = f1_score(y_true, y_pred_cleaned, average='binary', zero_division=0)
                prec = precision_score(y_true, y_pred_cleaned, average='binary', zero_division=0)
                rec = recall_score(y_true, y_pred_cleaned, average='binary', zero_division=0)
                
                iter_accuracies.append(acc)
                iter_f1s.append(f1)
                iter_precisions.append(prec)
                iter_recalls.append(rec)

            if not iter_accuracies:
                continue

            # Tính giá trị Trung bình (Mean) và Độ lệch chuẩn (Std) qua các iterations
            m_acc, s_acc = np.mean(iter_accuracies), np.std(iter_accuracies)
            m_f1, s_f1 = np.mean(iter_f1s), np.std(iter_f1s)
            m_prec, s_prec = np.mean(iter_precisions), np.std(iter_precisions)
            m_rec, s_rec = np.mean(iter_recalls), np.std(iter_recalls)

            print(f"{k:<5} | {m_acc:.4f} ± {s_acc:.4f} | {m_f1:.4f} ± {s_f1:.4f} | {m_rec:.4f}       | {m_prec:.4f}")

            summary_data.append({
                'K': k,
                'Method': method_name,
                'Acc_Mean': m_acc,
                'Acc_Std': s_acc,
                'F1_Mean': m_f1,
                'F1_Std': s_f1,
                'Precision_Mean': m_prec,
                'Precision_Std': s_prec,
                'Recall_Mean': m_rec,
                'Recall_Std': s_rec
            })

        # Xuất file CSV kết quả cho từng cấu hình Top
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_summary, index=False)
            print(f"\n[SUCCESS] Đã lưu tổng hợp vào: {output_summary}")
        else:
            print(f"\n[!] Không có dữ liệu hợp lệ được xử lý trong folder {folder_name}.")

if __name__ == "__main__":
    calculate_static_baseline_metrics()