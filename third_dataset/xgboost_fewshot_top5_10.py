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
    
    # Static Baseline Subsets matching your exact raw KernelSHAP rankings
    TOP_5_FEATURES = ["Stress Level", "Country", "BMI", "Continent", "Diastolic BP"]
    TOP_10_FEATURES = [
        "Stress Level", "Country", "BMI", "Continent", "Diastolic BP", 
        "Cholesterol", "Exercise Hours Per Week", "Systolic BP", "Obesity", "Sedentary Hours Per Day"
    ]
    
    FEATURE_SETS = {
        "top5": TOP_5_FEATURES,
        "top10": TOP_10_FEATURES
    }
    
    TARGET = 'Heart Attack Risk'

# ==========================================
# 2. DATA LOADING & RESOURCE INITIALIZATION
# ==========================================
print(f"[*] Execution runtime device engine: {DEVICE.upper()}")

# Load separate pre-balanced dataset pools directly
print("[*] Slicing explicit validation data targets...")
train_pool = pd.read_csv(ExperimentConfig.CANDIDATE_POOL_PATH)
test_fixed = pd.read_csv(ExperimentConfig.TEST_SET_PATH)

# Chuẩn hóa kỹ thuật: Ép kiểu toàn bộ các cột dạng chuỗi (object) sang category cho XGBoost nhận diện
for col in train_pool.columns:
    if train_pool[col].dtype == 'object':
        train_pool[col] = train_pool[col].astype('category')
        test_fixed[col] = test_fixed[col].astype('category')

total_test_samples = len(test_fixed)
print(f"[INFO] Candidate Resource Pool Size: {train_pool.shape[0]}")
print(f"[INFO] Total Fixed Evaluation Target Samples: {total_test_samples}")

# Load the master cross-paradigm synchronization index registry
if not os.path.exists(ExperimentConfig.INDICES_PATH):
    raise FileNotFoundError(
        f"[ERROR] Master index path not found at: {ExperimentConfig.INDICES_PATH}. "
        f"Please execute your baseline generation indexing framework script first."
    )

print(f"[*] Synchronizing evaluation tracking space via: {ExperimentConfig.INDICES_PATH}")
with open(ExperimentConfig.INDICES_PATH, 'r') as f:
    master_indices = json.load(f)

# ==========================================
# 3. EXPERIMENTAL EXECUTION LOOP
# ==========================================
# Outer loop: Slicing feature spaces (Top-5 vs Top-10)
for config_name, feature_list in ExperimentConfig.FEATURE_SETS.items():
    print(f"\n{'='*60}")
    print(f">>> LAUNCHING XGBOOST PIPELINE: {config_name.upper()} ({len(feature_list)} Features)")
    print(f"{'='*60}")
    
    # Organize outputs into isolated subdirectories
    output_dir = f"heart_attack_xgboost_baselines_{config_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Inner loops: Few-shot constraints and iterations
    for k in ExperimentConfig.K_VALUES:
        res_file = os.path.join(output_dir, f'heart_attack_xgboost_results_granular_k{k}.csv')
        print(f"\n>>> [XGBoost - {config_name}] Evaluating scaling boundary: K={k}")
        
        # Reset existing logging files to guarantee clean programmatic tracking
        if os.path.exists(res_file):
            os.remove(res_file)
        
        for i in range(ExperimentConfig.ITERATIONS):
            # Extract the exact row pointers from the common index registry
            support_idx = master_indices[str(k)][i]
            
            # Sub-slice the support and training sets using the current active feature configuration
            X_train = train_pool.loc[support_idx, feature_list]
            y_train = train_pool.loc[support_idx, ExperimentConfig.TARGET]
            
            # Isolate matching evaluation target matrices
            X_test = test_fixed[feature_list]
            
            # Instantiate model with parameters adapted for categorical features
            model = xgb.XGBClassifier(
                tree_method='hist',
                device=DEVICE,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,         # Softly penalized depth to prevent over-fitting on small support sizes
                enable_categorical=True,  # KHÁO SÁT CHÍNH XÁC: Kích hoạt cơ chế xử lý biến định danh của XGBoost
                random_state=i,
                verbosity=0,
                n_jobs=16 
            )
            
            # Train the localized tree assembly directly on the extracted feature slice
            model.fit(X_train, y_train)
            
            # Perform prediction over the frozen evaluation data
            predictions = model.predict(X_test)
            
            # Accumulate metrics and execute persistent disk synchronization
            iter_logs = []
            for idx, pred in zip(test_fixed.index, predictions):
                ground_truth = int(test_fixed.loc[idx, ExperimentConfig.TARGET])
                iter_logs.append([i, idx, ground_truth, int(pred)])
            
            pd.DataFrame(iter_logs).to_csv(res_file, mode='a', index=False, header=False)
            print(f"    K={k} | Iteration {i+1}/{ExperimentConfig.ITERATIONS} finalized.", end='\r')
        
        print(f"\n[SUCCESS] Parametric logging matrix compiled at: {res_file}")

print("\n" + "="*60)
print("[FINISH] All feature-stratified baseline simulations completed for XGBoost.")
print("="*60)