import pandas as pd

def analyze_and_clean_data(file_path="/home/s2410433/diabetes_project/third_dataset/heart_attack_prediction_dataset.csv"):
    print("=== STEP 1: LOADING & CLEANING HEART ATTACK DATASET ===")
    
    # Load the raw dataset
    df = pd.read_csv(file_path)
    print(f"[INFO] Initial raw dataset dimensions: {df.shape}")
    
    # Drop non-predictive administrative unique identifiers
    if 'Patient ID' in df.columns:
        df = df.drop(columns=['Patient ID'])
        print("[INFO] Dropped administrative field: 'Patient ID'")
        
    # Programmatically extract Systolic and Diastolic values out of the combined text string
    if 'Blood Pressure' in df.columns:
        print("[INFO] Splitting 'Blood Pressure' into separate numeric parameters...")
        df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)
        df['Systolic BP'] = pd.to_numeric(df['Systolic BP'], errors='coerce')
        df['Diastolic BP'] = pd.to_numeric(df['Diastolic BP'], errors='coerce')
        df = df.drop(columns=['Blood Pressure'])
    
    # Detect and display missing values distribution
    missing_counts = df.isnull().sum()
    print("\n[ANALYSIS] Missing value counts per attribute:")
    print(missing_counts[missing_counts > 0])
    
    # Execute listwise deletion to secure complete clinical cases
    df_clean = df.dropna().reset_index(drop=True)
    print(f"\n[INFO] Cleaned dataset dimensions (Complete Cases): {df_clean.shape}")
    
    # Verify the baseline class distribution of the target variable
    TARGET = 'Heart Attack Risk'
    print(f"\n[ANALYSIS] Class distribution of the target variable ({TARGET}):")
    print(df_clean[TARGET].value_counts(normalize=True))
    print(df_clean[TARGET].value_counts())
    
    # Serialize the cleaned complete-case dataframe
    output_cleaned = "heart_attack_cleaned.csv"
    df_clean.to_csv(output_cleaned, index=False)
    print(f"\n[SUCCESS] Cleaned dataset successfully saved to: {output_cleaned}\n")

if __name__ == "__main__":
    analyze_and_clean_data()