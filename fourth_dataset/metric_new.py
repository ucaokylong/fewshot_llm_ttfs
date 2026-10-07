import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================================
# 1. CẤU HÌNH THƯ MỤC VÀ TẬP TIN
# =====================================================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
ITERATIONS = 20

# Định nghĩa các thư mục cần xử lý cùng cấu trúc cột tương ứng
# Định nghĩa các thư mục cần xử lý cùng cấu trúc cột tương ứng
CONFIGS = [
    {
        "name": "Full Features",
        "folder": "llm_full_features_batchrun",  # Đã xóa 'diabetes_project/'
        "file_pattern": "cardio_results_granular_k{}.csv",
        "output_csv": "summary_cardio_llm_full_features_batchrun.csv",
        "columns": ['iter', 'id', 'gt', 'pred']
    },
    {
        "name": "Random Pruning Top-10",
        "folder": "cardio_random_pruning_top10", # Đã xóa 'diabetes_project/'
        "file_pattern": "qwen_random_k{}.csv",
        "output_csv": "summary_cardio_random_top10.csv",
        "columns": ['iter', 'id', 'gt', 'pred', 'f_count', 'f_list']
    },
    {
        "name": "Random Pruning Top-5",
        "folder": "cardio_random_pruning_top5",  # Đã xóa 'diabetes_project/'
        "file_pattern": "qwen_random_k{}.csv",
        "output_csv": "summary_cardio_random_top5.csv",
        "columns": ['iter', 'id', 'gt', 'pred', 'f_count', 'f_list']
    }
]

# =====================================================================
# 2. HÀM TÍNH TOÁN METRICS CHI TIẾT
# =====================================================================
def process_file_metrics(file_path, col_names):
    if not os.path.exists(file_path):
        print(f"[!] Bỏ qua: Không tìm thấy {file_path}")
        return None
        
    try:
        if len(col_names) == 6:
            df = pd.read_csv(file_path, header=None, names=col_names, dtype={'f_list': str})
        else:
            df = pd.read_csv(file_path, header=None, names=col_names)
    except Exception as e:
        print(f"[ERROR] Lỗi đọc file {file_path}: {e}")
        return None

    metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    has_f_count = 'f_count' in col_names
    if has_f_count:
        metrics['avg_feat'] = []

    for i in range(ITERATIONS):
        sub = df[df['iter'] == i]
        if len(sub) == 0:
            continue
            
        y_true = pd.to_numeric(sub['gt'], errors='coerce').fillna(0).astype(int)
        y_pred_raw = pd.to_numeric(sub['pred'], errors='coerce').fillna(-1).astype(int)
        
        # Xử lý các ca LLM trả về format sai (-1) thành dự đoán sai
        y_pred_cleaned = np.where((y_pred_raw != 0) & (y_pred_raw != 1), 1 - y_true, y_pred_raw)
        
        metrics['acc'].append(accuracy_score(y_true, y_pred_cleaned))
        metrics['prec'].append(precision_score(y_true, y_pred_cleaned, zero_division=0))
        metrics['rec'].append(recall_score(y_true, y_pred_cleaned, zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred_cleaned, zero_division=0))
        
        if has_f_count:
            feat_counts = pd.to_numeric(sub['f_count'], errors='coerce').fillna(0).values
            metrics['avg_feat'].append(feat_counts.mean())

    # Trả về None nếu không có iteration nào hợp lệ
    if not metrics['acc']:
        return None
        
    result = {
        'Acc_Mean': np.mean(metrics['acc']), 'Acc_Std': np.std(metrics['acc']),
        'F1_Mean': np.mean(metrics['f1']), 'F1_Std': np.std(metrics['f1']),
        'Prec_Mean': np.mean(metrics['prec']), 'Prec_Std': np.std(metrics['prec']),
        'Rec_Mean': np.mean(metrics['rec']), 'Rec_Std': np.std(metrics['rec'])
    }
    
    if has_f_count:
        result['Avg_Retained_Features'] = np.mean(metrics['avg_feat'])
        
    return result

# =====================================================================
# 3. TRÌNH ĐIỀU KHIỂN CHÍNH
# =====================================================================
def run_all_metrics():
    print(f"{'='*80}")
    print(f"{'TỔNG HỢP METRICS CHO BỘ DIABETES DATASET':^80}")
    print(f"{'='*80}")

    for config in CONFIGS:
        print(f"\n>>> ĐANG XỬ LÝ: {config['name'].upper()}")
        print(f"Thư mục nguồn: {config['folder']}")
        
        summary_data = []
        has_feats_col = 'f_count' in config['columns']
        
        # Tiêu đề bảng in ra terminal
        if has_feats_col:
            print(f"{'K':<4} | {'Acc (±Std)':<15} | {'F1 (±Std)':<15} | {'Precision':<10} | {'Recall':<8} | {'Avg Feats'}")
        else:
            print(f"{'K':<4} | {'Acc (±Std)':<15} | {'F1 (±Std)':<15} | {'Precision':<10} | {'Recall':<8}")
        print("-" * 80)

        for k in K_VALUES:
            file_path = os.path.join(config['folder'], config['file_pattern'].format(k))
            res = process_file_metrics(file_path, config['columns'])
            
            if res:
                row = {
                    'K': k,
                    'Method': config['name'],
                    'Acc_Mean': res['Acc_Mean'], 'Acc_Std': res['Acc_Std'],
                    'F1_Mean': res['F1_Mean'], 'F1_Std': res['F1_Std'],
                    'Precision_Mean': res['Prec_Mean'], 'Precision_Std': res['Prec_Std'],
                    'Recall_Mean': res['Rec_Mean'], 'Recall_Std': res['Rec_Std']
                }
                
                if has_feats_col:
                    row['Avg_Retained_Features'] = res['Avg_Retained_Features']
                    print(f"{k:<4} | {res['Acc_Mean']:.4f}±{res['Acc_Std']:.4f} | {res['F1_Mean']:.4f}±{res['F1_Std']:.4f} | {res['Prec_Mean']:.4f} | {res['Rec_Mean']:.4f} | {res['Avg_Retained_Features']:.2f}")
                else:
                    print(f"{k:<4} | {res['Acc_Mean']:.4f}±{res['Acc_Std']:.4f} | {res['F1_Mean']:.4f}±{res['F1_Std']:.4f} | {res['Prec_Mean']:.4f} | {res['Rec_Mean']:.4f}")
                
                summary_data.append(row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(config['output_csv'], index=False)
            print(f"✅ Đã lưu báo cáo thành công tại: {config['output_csv']}")

if __name__ == "__main__":
    run_all_metrics()