import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
ITERATIONS = 20
OUTPUT_SUMMARY = "cardio_llm_full_features_metrics_summary.csv"
INPUT_FOLDER = "llm_full_features"
FILE_PATTERN = os.path.join(INPUT_FOLDER, "cardio_results_granular_k{k}.csv")

# ==========================================
# 2. HÀM TÍNH TOÁN METRICS (METRICS ENGINE)
# ==========================================
def calculate_qwen_metrics():
    summary_data = []

    print(f"{'K':<5} | {'Accuracy Mean':<18} | {'F1 Mean':<18} | {'Recall Mean':<12} | {'Precision Mean'}")
    print("-" * 80)

    for k in K_VALUES:
        file_path = FILE_PATTERN.format(k=k)
        
        if not os.path.exists(file_path):
            print(f"[!] Warning: Không tìm thấy tệp cho K={k} tại đường dẫn: {file_path}")
            continue
            
        try:
            # Format dữ liệu ghi nhận từ LLM Full: [iter, id, ground_truth, prediction]
            df = pd.read_csv(file_path, header=None, names=['iter', 'id', 'gt', 'pred'])
        except Exception as e:
            print(f"[ERROR] Lỗi đọc file K={k}: {e}")
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
            
            # Ép kiểu dữ liệu về dạng số nguyên an toàn
            y_true = pd.to_numeric(subset['gt'], errors='coerce').fillna(0).astype(int)
            y_pred_raw = pd.to_numeric(subset['pred'], errors='coerce').fillna(-1).astype(int)
            
            # Xử lý trường hợp LLM sinh lỗi (-1): ép thành nhãn dự đoán "SAI" so với Ground Truth
            y_pred_cleaned = np.where(
                (y_pred_raw != 0) & (y_pred_raw != 1), 
                1 - y_true, 
                y_pred_raw
            )
            
            # Tính toán chỉ số cho từng vòng lặp (Iteration)
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

        # Tính toán giá trị Trung bình (Mean) và Độ lệch chuẩn (Std) qua 20 iterations
        m_acc, s_acc = np.mean(iter_accuracies), np.std(iter_accuracies)
        m_f1, s_f1 = np.mean(iter_f1s), np.std(iter_f1s)
        m_prec, s_prec = np.mean(iter_precisions), np.std(iter_precisions)
        m_rec, s_rec = np.mean(iter_recalls), np.std(iter_recalls)

        print(f"{k:<5} | {m_acc:.4f} ± {s_acc:.4f} | {m_f1:.4f} ± {s_f1:.4f} | {m_rec:.4f}       | {m_prec:.4f}")

        summary_data.append({
            'K': k,
            'Method': 'Qwen-7B (Full Features)',
            'Acc_Mean': m_acc,
            'Acc_Std': s_acc,
            'F1_Mean': m_f1,
            'F1_Std': s_f1,
            'Precision_Mean': m_prec,
            'Precision_Std': s_prec,
            'Recall_Mean': m_rec,
            'Recall_Std': s_rec
        })

    # Xuất file CSV tổng hợp
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(OUTPUT_SUMMARY, index=False)
        print(f"\n[SUCCESS] Đã lưu kết quả chỉ số LLM Full Features vào: {OUTPUT_SUMMARY}")
    else:
        print("\n[!] Không tìm thấy dữ liệu hợp lệ để tính toán.")

if __name__ == "__main__":
    calculate_qwen_metrics()