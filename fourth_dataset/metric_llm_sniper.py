import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# CẤU HÌNH (CONFIGURATION)
# ==========================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]

FOLDERS_CONFIG = {
    "cardio_llm_sniper_top5": "cardio_qwen_sniper_top5_summary.csv",
    "cardio_llm_sniper_top10": "cardio_qwen_sniper_top10_summary.csv"
}

def calculate_sniper_metrics():
    for folder_name, summary_output in FOLDERS_CONFIG.items():
        summary_data = []

        print(f"\n{'='*75}")
        print(f"📊 Calculating metrics for folder: {folder_name.upper()}")
        print(f"{'='*75}")
        print(f"{'K':<5} | {'Accuracy Mean':<18} | {'F1 Mean':<18} | {'Recall':<8} | {'Avg_Feats'}")
        print("-" * 75)

        for k in K_VALUES:
            file_path = os.path.join(folder_name, f'cardio_qwen_sniper_k{k}.csv')
            
            if not os.path.exists(file_path):
                print(f"[!] Warning: File not found: {file_path}")
                continue
                
            try:
                # Format: [iter, id, gt, pred, f_count, f_list]
                df = pd.read_csv(
                    file_path, 
                    header=None, 
                    names=['iter', 'id', 'gt', 'pred', 'f_count', 'f_list'],
                    dtype={'f_list': str}
                )
            except Exception as e:
                print(f"[ERROR] Error reading file K={k} at {file_path}: {e}")
                continue

            iter_accuracies = []
            iter_f1s = []
            iter_precisions = []
            iter_recalls = []
            iter_avg_feats = []

            iterations = df['iter'].unique()
            
            for i in iterations:
                subset = df[df['iter'] == i]
                if len(subset) == 0:
                    continue
                
                # Ép kiểu dữ liệu về dạng số nguyên an toàn
                y_true = pd.to_numeric(subset['gt'], errors='coerce').fillna(0).astype(int)
                y_pred_raw = pd.to_numeric(subset['pred'], errors='coerce').fillna(-1).astype(int)
                
                # Biến đổi tất cả giá trị lỗi (-1) thành nhãn SAI
                y_pred_cleaned = np.where(
                    (y_pred_raw != 0) & (y_pred_raw != 1), 
                    1 - y_true, 
                    y_pred_raw
                )
                
                acc = accuracy_score(y_true, y_pred_cleaned)
                f1 = f1_score(y_true, y_pred_cleaned, average='binary', zero_division=0)
                prec = precision_score(y_true, y_pred_cleaned, average='binary', zero_division=0)
                rec = recall_score(y_true, y_pred_cleaned, average='binary', zero_division=0)
                
                # Tính số đặc trưng trung bình thực tế được dùng trong prompt
                avg_f = subset['f_count'].mean() if 'f_count' in subset.columns else np.nan
                
                iter_accuracies.append(acc)
                iter_f1s.append(f1)
                iter_precisions.append(prec)
                iter_recalls.append(rec)
                if not np.isnan(avg_f):
                    iter_avg_feats.append(avg_f)

            if len(iter_accuracies) == 0:
                continue

            m_acc, s_acc = np.mean(iter_accuracies), np.std(iter_accuracies)
            m_f1, s_f1 = np.mean(iter_f1s), np.std(iter_f1s)
            m_prec = np.mean(iter_precisions)
            m_rec = np.mean(iter_recalls)
            m_feats = np.mean(iter_avg_feats) if iter_avg_feats else 0.0

            print(f"{k:<5} | {m_acc:.4f} ± {s_acc:.4f} | {m_f1:.4f} ± {s_f1:.4f} | {m_rec:.4f} | {m_feats:.2f}")

            summary_data.append({
                'K': k,
                'Method': f'Qwen2.5-7B (Sniper-{folder_name.split("_")[-1].upper()})',
                'Acc_Mean': m_acc,
                'Acc_Std': s_acc,
                'F1_Mean': m_f1,
                'F1_Std': s_f1,
                'Precision_Mean': m_prec,
                'Recall_Mean': m_rec,
                'Avg_Features': m_feats
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_output, index=False)
            print(f"\n[SUCCESS] Saved in: {summary_output}")

if __name__ == "__main__":
    calculate_sniper_metrics()