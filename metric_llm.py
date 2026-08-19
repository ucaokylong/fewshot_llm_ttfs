import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score

# ==========================================
# CẤU HÌNH (CONFIGURATION)
# ==========================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16] # Qwen sếp mới chạy đến 10
OUTPUT_SUMMARY = "qwen_granular_top10_summary.csv"

def calculate_qwen_metrics():
    summary_data = []

    print(f"{'K':<5} | {'Accuracy Mean':<15} | {'F1 Mean':<15} | {'Precision'}")
    print("-" * 65)

    for k in K_VALUES:
        file_path = f'qwen_results_granular_k{k}.csv'
        
        if not os.path.exists(file_path):
            print(f"[!] Warning: Không tìm thấy file cho K={k}")
            continue
            
        # Load dữ liệu (Format: iter, id, ground_truth, prediction)
        # Giả sử file không có header như sếp mô tả
        try:
            df = pd.read_csv(file_path, header=None, names=['iter', 'id', 'gt', 'pred'])
        except Exception as e:
            print(f"[ERROR] Lỗi đọc file K={k}: {e}")
            continue

        iter_accuracies = []
        iter_f1s = []
        iter_precisions = []

        # Tính toán metric cho từng Iteration (thường là 0 đến 19)
        iterations = df['iter'].unique()
        # Sửa lại đoạn tính toán trong vòng lặp Iteration
# Trong vòng lặp Iteration của sếp:
        for i in iterations:
            subset = df[df['iter'] == i]
            
            # Ép kiểu về numeric để tránh lỗi so sánh string/object
            y_true = pd.to_numeric(subset['gt'], errors='coerce').fillna(0).astype(int)
            y_pred_raw = pd.to_numeric(subset['pred'], errors='coerce').fillna(-1).astype(int)
            
            # MẸO QUAN TRỌNG: Biến đổi tất cả những gì không phải 0 hoặc 1 thành nhãn "SAI"
            # Nếu y_pred là -1, ta ép nó thành (1 - y_true). 
            # Ví dụ: GT=1 mà Pred=-1 -> Biến Pred thành 0 (Sai). 
            # Ví dụ: GT=0 mà Pred=-1 -> Biến Pred thành 1 (Sai).
            y_pred_cleaned = np.where(
                (y_pred_raw != 0) & (y_pred_raw != 1), 
                1 - y_true, 
                y_pred_raw
            )
            
            # Bây giờ y_pred_cleaned chỉ chứa 0 và 1, sếp chạy tẹt ga không lo lỗi
            acc = accuracy_score(y_true, y_pred_cleaned)
            f1 = f1_score(y_true, y_pred_cleaned, average='binary', zero_division=0)
            prec = precision_score(y_true, y_pred_cleaned, average='binary', zero_division=0)
            
            iter_accuracies.append(acc)
            iter_f1s.append(f1)
            iter_precisions.append(prec)

        # Tính Mean và Std
        m_acc, s_acc = np.mean(iter_accuracies), np.std(iter_accuracies)
        m_f1, s_f1 = np.mean(iter_f1s), np.std(iter_f1s)
        m_prec = np.mean(iter_precisions)

        print(f"{k:<5} | {m_acc:.4f} ± {s_acc:.4f} | {m_f1:.4f} ± {s_f1:.4f} | {m_prec:.4f}")

        summary_data.append({
            'K': k,
            'Method': 'Qwen-7B-Instruct', # Hoặc model sếp dùng
            'Acc_Mean': m_acc,
            'Acc_Std': s_acc,
            'F1_Mean': m_f1,
            'F1_Std': s_f1,
            'Precision_Mean': m_prec
        })

    # Lưu bảng tổng hợp
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"\n[SUCCESS] Đã lưu metric Qwen vào: {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    calculate_qwen_metrics()