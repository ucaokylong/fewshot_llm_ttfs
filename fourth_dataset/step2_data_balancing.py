import pandas as pd

def execute_data_balancing(file_path="cardio_cleaned.csv"):
    print("=== STEP 2: STRATIFIED SPLITTING & 50:50 BALANCING ===")
    df = pd.read_csv(file_path)
    TARGET = 'cardio'
    
    # Phân tách dữ liệu theo nhãn mục tiêu
    df_pos = df[df[TARGET] == 1]
    df_neg = df[df[TARGET] == 0]
    
    print(f"[INFO] Available clean samples - Class 0: {len(df_neg)}, Class 1: {len(df_pos)}")
    
    # Nâng kích thước tập Test lên chính xác 1,000 mẫu (500 mẫu cho mỗi lớp)
    test_samples_per_class = 500
    
    if len(df_pos) < test_samples_per_class or len(df_neg) < test_samples_per_class:
        raise ValueError(f"Not enough samples to create a {test_samples_per_class}-sample per class test split.")
        
    # Trích xuất chính xác 500 mẫu ngẫu nhiên từ mỗi nhãn để làm tập Test cố định
    test_pos = df_pos.sample(n=test_samples_per_class, random_state=42)
    test_neg = df_neg.sample(n=test_samples_per_class, random_state=42)
    
    df_test_fixed = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n[INFO] Fixed invariant Test Set dimensions: {df_test_fixed.shape}")
    print(df_test_fixed[TARGET].value_counts())
    df_test_fixed.to_csv("cardio_test_fixed.csv", index=False)
    
    # Loại bỏ các mẫu đã chọn làm Test khỏi tập tài nguyên
    remaining_pos = df_pos.drop(test_pos.index)
    remaining_neg = df_neg.drop(test_neg.index)
    
    # Xác định quy mô tối đa cho Candidate Pool cân bằng 50:50
    min_remaining_size = min(len(remaining_pos), len(remaining_neg))
    
    candidate_pos = remaining_pos.sample(n=min_remaining_size, random_state=42)
    candidate_neg = remaining_neg.sample(n=min_remaining_size, random_state=42)
    
    df_candidate_balanced = pd.concat([candidate_pos, candidate_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n[INFO] Balanced Candidate Pool dimensions: {df_candidate_balanced.shape}")
    print(df_candidate_balanced[TARGET].value_counts())
    df_candidate_balanced.to_csv("cardio_candidate_balanced.csv", index=False)
    
    print("\n[SUCCESS] Dataset splitting and double-balancing completed successfully.\n")

if __name__ == "__main__":
    execute_data_balancing()