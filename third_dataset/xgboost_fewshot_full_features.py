import os
import json
import torch
import pandas as pd
import numpy as np
import xgboost as xgb

# ==========================================
# 1. HARDWARE & RUNTIME CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ExperimentConfig:
    # Path mappings synchronized with the new Heart Attack data pipelines
    CANDIDATE_POOL_PATH = "heart_attack_candidate_balanced.csv"
    TEST_SET_PATH = "heart_attack_test_fixed.csv"
    INDICES_PATH = "master_granular_indices_heart_attack.json"
    
    ITERATIONS = 20
    K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16] 
    
    # Complete 26 features matching the Heart Attack Risk Prediction schema
    FEATURES = [
        'Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 
        'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 
        'Exercise Hours Per Week', 'Diet', 'Previous Heart Problems', 
        'Medication Use', 'Stress Level', 'Sedentary Hours Per Day', 
        'Income', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week', 
        'Sleep Hours Per Day', 'Country', 'Continent', 'Hemisphere',
        'Systolic BP', 'Diastolic BP'
    ]
    
    # Text/Object columns that need to be parsed explicitly as categorical types
    CATEGORICAL_FEATURES = ['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere']
    
    TARGET = 'Heart Attack Risk'

# ==========================================
# 2. DATA LOADING & RESOURCE INITIALIZATION
# ==========================================
print(f"[*] Execution runtime device engine: {DEVICE.upper()}")

# Load separate pre-balanced dataset pools directly
print("[*] Accessing pre-balanced dataset partitions...")
train_pool = pd.read_csv(ExperimentConfig.CANDIDATE_POOL_PATH)
test_fixed = pd.read_csv(ExperimentConfig.TEST_SET_PATH)

# Fix the object dtype fault: Enforce categorical data arrays
print("[*] Enforcing category types on string-based object features...")
for col in ExperimentConfig.CATEGORICAL_FEATURES:
    if col in train_pool.columns:
        train_pool[col] = train_pool[col].astype('category')
    if col in test_fixed.columns:
        test_fixed[col] = test_fixed[col].astype('category')

total_test_samples = len(test_fixed)
print(f"[INFO] Candidate Resource Pool Size: {train_pool.shape[0]}")
print(f"[INFO] Total Fixed Evaluation Target Samples: {total_test_samples}")

# Load the master synchronization index registry
if not os.path.exists(ExperimentConfig.INDICES_PATH):
    raise FileNotFoundError(
        f"[ERROR] Master index map not found at {ExperimentConfig.INDICES_PATH}. "
        f"Please run your step3 indexing script first."
    )

print(f"[*] Parsing unified evaluation matrix: {ExperimentConfig.INDICES_PATH}")
with open(ExperimentConfig.INDICES_PATH, 'r') as f:
    master_indices = json.load(f)

# ==========================================
# 3. MAIN EXPERIMENTAL EXECUTION LOOP
# ==========================================
for k in ExperimentConfig.K_VALUES:
    res_file = f'heart_attack_xgboost_results_granular_k{k}.csv'
    print(f"\n>>> [XGBoost] Evaluating scaling boundary: K={k}")
    
    # Reset existing tracking metrics to prevent append accumulation faults
    if os.path.exists(res_file):
        os.remove(res_file)
    
    for i in range(ExperimentConfig.ITERATIONS):
        # Extract the exact cross-paradigm synchronized row pointers
        support_idx = master_indices[str(k)][i]
        
        # Isolate support context training vectors
        X_train = train_pool.loc[support_idx, ExperimentConfig.FEATURES]
        y_train = train_pool.loc[support_idx, ExperimentConfig.TARGET]
        
        # Instantiate optimized XGBoost model structure
        model = xgb.XGBClassifier(
            tree_method='hist',
            device=DEVICE,
            enable_categorical=True,  # Natively processes designated Pandas category types
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,              # Prevent severe over-fitting on small support sets
            random_state=i,
            verbosity=0,
            n_jobs=16                 # Fully utilizes allocated CPU processing cores
        )
        
        # Train on the low-resource support frame
        model.fit(X_train, y_train)
        
        # Perform inference on the frozen evaluation target space
        X_test = test_fixed[ExperimentConfig.FEATURES]
        predictions = model.predict(X_test)
        
        # Immediately append tracking logs to preserve data security after each run
        iter_logs = []
        for idx, pred in zip(test_fixed.index, predictions):
            ground_truth = int(test_fixed.loc[idx, ExperimentConfig.TARGET])
            iter_logs.append([i, idx, ground_truth, int(pred)])
        
        pd.DataFrame(iter_logs).to_csv(res_file, mode='a', index=False, header=False)
        print(f"    K={k} | Iteration {i+1}/{ExperimentConfig.ITERATIONS} finalized.", end='\r')
    
    print(f"\n[SUCCESS] Parametric tracking logs compiled at: {res_file}")

print("\n[FINISH] Full feature array evaluation pipeline completed for XGBoost.")