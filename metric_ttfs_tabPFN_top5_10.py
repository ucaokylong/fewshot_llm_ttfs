import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# =====================================================================
# 1. CẤU HÌNH HỆ THỐNG VÀ ĐƯỜNG DẪN
# =====================================================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
ITERATIONS = 20

# Khai báo chuẩn 6 cột dựa trên dữ liệu thực tế sếp vừa cung cấp
COL_NAMES = ['iteration', 'test_id', 'gt', 'pred', 'feature_count', 'feature_list']

TARGET_FOLDERS = {
    "top5": "ttfs_tabpfn_top5",
    "top10": "ttfs_tabpfn_top10"
}

# =====================================================================
# 2. HÀM TÍNH METRICS CHI TIẾT
# =====================================================================
def calculate_ttfs_folder_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"[!] Cảnh báo: Thiếu file kết quả {file_path}")
        return None
    
    # Đọc file (Cột số 5 là chuỗi string nên ép kiểu str để tránh lỗi định dạng)
    df = pd.read_csv(file_path, header=None, names=COL_NAMES, dtype={'feature_list': str})
    
    metrics_per_iter = []
    features_count_per_iter = []
    
    for i in range(ITERATIONS):
        sub = df[df['iteration'] == i]
        if len(sub) == 0: 
            continue
        
        y_true = sub['gt'].values
        y_pred = sub['pred'].values
        feat_counts = sub['feature_count'].values
        
        # 1. Tính toán các chỉ số phân loại truyền thống
        acc = (y_pred == y_true).mean()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 2. Tính số lượng feature trung bình được giữ lại trong iteration này
        avg_feats_in_iter = feat_counts.mean()
        
        metrics_per_iter.append([acc, prec, rec, f1])
        features_count_per_iter.append(avg_feats_in_iter)
    
    if not metrics_per_iter: 
        return None
    
    # Chuyển đổi tính Mean và Std qua Numpy
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
# 3. TRÌNH ĐIỀU KHIỂN CHÍNH & XUẤT BÁO CÁO
# =====================================================================
if __name__ == "__main__":
    print("[*] Đang tiến hành bóc tách kết quả thí nghiệm TTFS-TabPFN...")
    
    for config_label, folder_name in TARGET_FOLDERS.items():
        summary_results = []
        output_summary_csv = f"ttfs_tabpfn_{config_label}_metrics_summary.csv"
        
        print(f"\n{'-'*85}")
        print(f"📊 BẢNG METRICS CHO CẤU HÌNH: {config_label.upper()} (Nguồn: {folder_name})")
        print(f"{'-'*85}")
        print(f"{'K':<3} | {'Method':<12} | {'Acc (±Std)':<15} | {'F1 (±Std)':<15} | {'Precision':<10} | {'Recall':<8} | {'Avg_Feats':<10}")
        print(f"{'-'*85}")
        
        for k in K_VALUES:
            file_path = os.path.join(folder_name, f'dynamic_results_k{k}.csv')
            res = calculate_ttfs_folder_metrics(file_path)
            
            if res:
                method_name = f"TTFS-{config_label.upper()}"
                
                # In đẹp đẽ ra terminal để sếp tiện tay chụp màn hình báo cáo hoặc đối chiếu nhanh
                print(f"{k:<3} | {method_name:<12} | "
                      f"{res['acc'][0]:.4f}±{res['acc'][1]:.4f} | "
                      f"{res['f1'][0]:.4f}±{res['f1'][1]:.4f} | "
                      f"{res['prec'][0]:.4f} | "
                      f"{res['rec'][0]:.4f} | "
                      f"{res['avg_features']:.2f}")
                
                # Nạp vào bộ nhớ để chuẩn bị ghi file CSV tổng hợp
                summary_results.append({
                    'K': k,
                    'Method': method_name,
                    'Acc_Mean': res['acc'][0], 'Acc_Std': res['acc'][1],
                    'F1_Mean': res['f1'][0], 'F1_Std': res['f1'][1],
                    'Precision_Mean': res['prec'][0], 'Precision_Std': res['prec'][1],
                    'Recall_Mean': res['rec'][0], 'Recall_Std': res['rec'][1],
                    'Avg_Retained_Features': res['avg_features']
                })
        
        # Lưu file tổng hợp riêng cho từng cấu hình
        if summary_results:
            pd.DataFrame(summary_results).to_csv(output_summary_csv, index=False)
            print(f"{'-'*85}")
            print(f"✅ Đã kết xuất báo cáo thành công tại: {output_summary_csv}")