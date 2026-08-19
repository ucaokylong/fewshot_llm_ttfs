import pandas as pd
import numpy as np

def inspect_cvd_dataset(file_path="cardio_train.csv"):
    print("==========================================================")
    print("=== STEP 0: EXPLORATORY DATA ANALYSIS (CVD DATASET) ===")
    print("==========================================================")
    
    # 1. Load Dataset (Dataset uses semicolon ';' delimiter)
    try:
        df = pd.read_csv(file_path, sep=';')
        print(f"[*] Successfully loaded dataset from: {file_path}")
    except Exception as e:
        print(f"[FATAL] Failed to read CSV file: {str(e)}")
        return

    print(f"[*] Raw Dataset Matrix Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # 2. Display Feature Types and Null Distribution
    print("--- 1. FEATURE SCHEMA & NULL VALUE DISTRIBUTION ---")
    info_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Null_Count': df.isnull().sum(),
        'Unique_Values': df.nunique()
    })
    print(info_df)
    print("-" * 55)
    
    # 3. Target Distribution Check
    TARGET = 'cardio'
    if TARGET in df.columns:
        print(f"\n--- 2. TARGET CLASS DISTRIBUTION ({TARGET}) ---")
        class_counts = df[TARGET].value_counts()
        class_props = df[TARGET].value_counts(normalize=True) * 100
        target_summary = pd.DataFrame({'Count': class_counts, 'Percentage (%)': class_props})
        print(target_summary)
        print("-" * 55)
    else:
        print(f"[WARNING] Target column '{TARGET}' not found in raw index.")
        return

    # 4. Preliminary Feature Engineering for Signal Verification
    # Convert age from days to years and drop 'id'
    df_temp = df.copy()
    if 'id' in df_temp.columns:
        df_temp = df_temp.drop(columns=['id'])
    if 'age' in df_temp.columns:
        df_temp['age'] = (df_temp['age'] / 365.25).apply(np.floor).astype(int)

    # Clean extreme blood pressure outliers to prevent correlation distortion
    df_temp = df_temp[(df_temp['ap_hi'] >= 60) & (df_temp['ap_hi'] <= 250)]
    df_temp = df_temp[(df_temp['ap_lo'] >= 40) & (df_temp['ap_lo'] <= 150)]
    df_temp = df_temp[df_temp['ap_hi'] >= df_temp['ap_lo']]

    print(f"\n[*] Cleaned Sub-sample Volume for Signal Check: {len(df_temp)} records")

    # 5. Feature Signal Strength (Correlation to Target)
    print("\n--- 3. FEATURE-TO-TARGET PEARSON CORRELATION (PREDICTIVE SIGNAL) ---")
    correlations = df_temp.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    corr_df = pd.DataFrame({'Pearson_Correlation': correlations})
    print(corr_df)
    print("-" * 55)
    
    print("\n[SUMMARY] Data inspection finalized successfully. Signal verification complete.")

if __name__ == "__main__":
    inspect_cvd_dataset()