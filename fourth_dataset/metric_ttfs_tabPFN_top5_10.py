import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# =====================================================================
# 1. RUNTIME CONFIGURATION AND DIRECTORY MAPPINGS
# =====================================================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
ITERATIONS = 20

# Cấu trúc 6 cột dữ liệu chuẩn xuất ra từ TTFS Engine
COL_NAMES = ['iteration', 'test_id', 'gt', 'pred', 'feature_count', 'feature_list']

# Đường dẫn thư mục chứa kết quả gọt tỉa động của TTFS-TabPFN cho bộ Cardiovascular
TARGET_FOLDERS = {
    "top5": "cardio_ttfs_tabpfn_top5",
    "top10": "cardio_ttfs_tabpfn_top10"
}

# =====================================================================
# 2. GRANULAR METRICS COMPUTATION ENGINE
# =====================================================================
def calculate_ttfs_folder_metrics(file_path):
    if not os.path.exists(file_path):
        return None
    
    try:
        # Ép kiểu chuỗi cho cột feature_list để tránh lỗi định dạng dấu chấm phẩy
        df = pd.read_csv(file_path, header=None, names=COL_NAMES, dtype={'feature_list': str})
    except Exception:
        return None
    
    # Lọc bỏ các mẫu dự đoán bị lỗi (-1) nếu có
    df = df[df['pred'] != -1]
    if len(df) == 0:
        return None
    
    metrics_per_iter = []
    features_count_per_iter = []
    
    for i in range(ITERATIONS):
        sub = df[df['iteration'] == i]
        if len(sub) == 0: 
            continue
        
        y_true = sub['gt'].values
        y_pred = sub['pred'].values
        feat_counts = sub['feature_count'].values
        
        # 1. Tính toán các chỉ số đánh giá phân loại cơ bản
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 2. Tính số lượng đặc trưng trung bình còn giữ lại sau gọt tỉa ở iteration này
        avg_feats_in_iter = feat_counts.mean()
        
        metrics_per_iter.append([acc, prec, rec, f1])
        features_count_per_iter.append(avg_feats_in_iter)
    
    if not metrics_per_iter: 
        return None
    
    # Tính trung bình và độ lệch chuẩn qua 20 vòng lặp thực nghiệm
    metrics_array = np.array(metrics_per_iter)
    means = np.mean(metrics_array, axis=0)
    stds = np.std(metrics_array, axis=0)
    
    avg_retained_features = np.mean(features_count_per_iter)
    
    return {
        'acc': (means[0], stds[0]),
        'prec': (means[1], stds[1]),
        'rec': (means[2], stds[2]),
        'f1': (means[3], stds[3]),
        'avg_features': avg_retained_features
    }

# =====================================================================
# 3. MAIN RUNTIME CONTROLLER & REPORT GENERATOR
# =====================================================================
if __name__ == "__main__":
    print("[*] Initiating parametric parsing suite for Cardiovascular TTFS-TabPFN logs...")
    
    for config_label, folder_name in TARGET_FOLDERS.items():
        summary_results = []
        output_summary_csv = f"cardio_ttfs_tabpfn_{config_label}_metrics_summary.csv"
        
        print(f"\n{'-'*95}")
        print(f"📊 EVALUATION REPORT CONFIGURATION: TTFS-{config_label.upper()} (Source: {folder_name})")
        print(f"{'-'*95}")
        print(f"{'K':<3} | {'Method':<12} | {'Acc (±Std)':<15} | {'F1 (±Std)':<15} | {'Precision':<10} | {'Recall':<8} | {'Avg_Feats':<10}")
        print(f"{'-'*95}")
        
        for k in K_VALUES:
            file_path = os.path.join(folder_name, f'cardio_dynamic_results_k{k}.csv')
            res = calculate_ttfs_folder_metrics(file_path)
            
            if res:
                method_name = f"TTFS-{config_label.upper()}"
                
                # In định dạng rõ ràng ra Terminal
                print(f"{k:<3} | {method_name:<12} | "
                      f"{res['acc'][0]:.4f}±{res['acc'][1]:.4f} | "
                      f"{res['f1'][0]:.4f}±{res['f1'][1]:.4f} | "
                      f"{res['prec'][0]:.4f} | "
                      f"{res['rec'][0]:.4f} | "
                      f"{res['avg_features']:.2f}")
                
                # Lưu thông số chi tiết
                summary_results.append({
                    'K': k,
                    'Method': method_name,
                    'Acc_Mean': res['acc'][0], 'Acc_Std': res['acc'][1],
                    'F1_Mean': res['f1'][0], 'F1_Std': res['f1'][1],
                    'Precision_Mean': res['prec'][0], 'Precision_Std': res['prec'][1],
                    'Recall_Mean': res['rec'][0], 'Recall_Std': res['rec'][1],
                    'Avg_Retained_Features': res['avg_features']
                })
            else:
                print(f"{k:<3} | TTFS-{config_label.upper():<7} | File missing or incomplete: {file_path}")
        
        # Lưu file CSV tổng hợp
        if summary_results:
            pd.DataFrame(summary_results).to_csv(output_summary_csv, index=False)
            print(f"{'-'*95}")
            print(f"[SUCCESS] Performance summary log successfully compiled at: {output_summary_csv}")