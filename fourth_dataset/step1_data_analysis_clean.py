import pandas as pd
import numpy as np

def analyze_and_clean_data(file_path="cardio_train.csv"):
    print("=== STEP 1: LOADING & CLEANING CARDIOVASCULAR DATASET ===")
    
    # Đọc dữ liệu raw với phân cách dấu chấm phẩy ';'
    df = pd.read_csv(file_path, sep=';')
    print(f"[INFO] Initial raw dataset dimensions: {df.shape}")
    
    # 1. Loại bỏ cột định danh 'id' nếu tồn tại
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        print("[INFO] Dropped administrative identifier field: 'id'")
        
    # 2. Quy đổi cột 'age' từ đơn vị ngày sang tuổi (năm)
    if 'age' in df.columns:
        df['age'] = (df['age'] / 365.25).apply(np.floor).astype(int)
        print("[INFO] Converted 'age' attribute from days to integer years")
        
    # Kiếm tra phân phối giá trị thiếu (Null/NaN)
    missing_counts = df.isnull().sum()
    print("\n[ANALYSIS] Missing value counts per attribute:")
    print(missing_counts[missing_counts > 0])
    
    # Thực hiện listwise deletion để đảm bảo tính toàn vẹn mẫu
    df_clean = df.dropna().reset_index(drop=True)
    print(f"\n[INFO] Cleaned dataset dimensions (Complete Cases): {df_clean.shape}")
    
    # Kiểm tra phân phối nhãn mục tiêu 'cardio'
    TARGET = 'cardio'
    print(f"\n[ANALYSIS] Class distribution of target variable ({TARGET}):")
    print(df_clean[TARGET].value_counts(normalize=True))
    print(df_clean[TARGET].value_counts())
    
    # Lưu file sạch
    output_cleaned = "cardio_cleaned.csv"
    df_clean.to_csv(output_cleaned, index=False)
    print(f"\n[SUCCESS] Cleaned dataset successfully saved to: {output_cleaned}\n")

if __name__ == "__main__":
    analyze_and_clean_data()