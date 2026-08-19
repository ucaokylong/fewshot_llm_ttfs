import pandas as pd

def analyze_and_clean_data(file_path="framingham.csv"):
    print("=== STEP 1: LOADING & CLEANING FRAMINGHAM DATASET ===")
    
    # Load the raw dataset downloaded from Kaggle
    df = pd.read_csv(file_path)
    print(f"[INFO] Initial raw dataset dimensions: {df.shape}")
    
    # Detect and display the missing values distribution
    missing_counts = df.isnull().sum()
    print("\n[ANALYSIS] Missing value counts per attribute:")
    print(missing_counts[missing_counts > 0])
    
    # Execute listwise deletion (dropna) to secure complete medical cases
    # This reduces variance inflation caused by arbitrary imputation methods
    df_clean = df.dropna().reset_index(drop=True)
    print(f"\n[INFO] Cleaned dataset dimensions (Complete Cases): {df_clean.shape}")
    
    # Verify the baseline class distribution of the predictive target
    TARGET = 'TenYearCHD'
    print("\n[ANALYSIS] Class distribution of the target variable (TenYearCHD):")
    print(df_clean[TARGET].value_counts(normalize=True))
    
    # Serialize the cleaned complete-case dataframe
    output_cleaned = "framingham_cleaned.csv"
    df_clean.to_csv(output_cleaned, index=False)
    print(f"\n[SUCCESS] Cleaned dataset successfully saved to: {output_cleaned}\n")

if __name__ == "__main__":
    analyze_and_clean_data()