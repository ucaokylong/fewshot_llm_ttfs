import pandas as pd

def execute_data_balancing(file_path="framingham_cleaned.csv"):
    print("=== STEP 2: STRATIFIED SPLITTING & 50:50 BALANCING ===")
    df = pd.read_csv(file_path)
    TARGET = 'TenYearCHD'
    
    # Separate the cleaned dataset by class
    df_pos = df[df[TARGET] == 1]
    df_neg = df[df[TARGET] == 0]
    
    print(f"[INFO] Available clean samples - Class 0: {len(df_neg)}, Class 1: {len(df_pos)}")
    
    # Strategy: Reduce test size to 400 (200 per class) to preserve more positive samples for the candidate pool
    test_samples_per_class = 200
    
    if len(df_pos) < test_samples_per_class:
        raise ValueError(f"Not enough positive samples to create a {test_samples_per_class}-sample positive test split.")
        
    # Extract exactly 200 samples from each class to construct the invariant Test Set
    test_pos = df_pos.sample(n=test_samples_per_class, random_state=42)
    test_neg = df_neg.sample(n=test_samples_per_class, random_state=42)
    
    df_test_fixed = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n[INFO] Fixed invariant Test Set dimensions: {df_test_fixed.shape}")
    print(df_test_fixed[TARGET].value_counts())
    df_test_fixed.to_csv("framingham_test_fixed.csv", index=False)
    
    # Remove the selected test samples to form the remaining pool
    remaining_pos = df_pos.drop(test_pos.index)
    remaining_neg = df_neg.drop(test_neg.index)
    
    # Determine the maximum possible size for a balanced 50:50 Candidate Pool
    min_remaining_size = min(len(remaining_pos), len(remaining_neg))
    
    # Downsample the majority remaining class to match the minority remaining class
    candidate_pos = remaining_pos.sample(n=min_remaining_size, random_state=42)
    candidate_neg = remaining_neg.sample(n=min_remaining_size, random_state=42)
    
    df_candidate_balanced = pd.concat([candidate_pos, candidate_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n[INFO] Balanced Candidate Pool dimensions: {df_candidate_balanced.shape}")
    print(df_candidate_balanced[TARGET].value_counts())
    df_candidate_balanced.to_csv("framingham_candidate_balanced.csv", index=False)
    
    print("\n[SUCCESS] Dataset splitting and double-balancing completed successfully with optimal sample distribution.\n")

if __name__ == "__main__":
    execute_data_balancing()