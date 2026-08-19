import os
import json
import torch
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier

# ==========================================
# 1. HARDWARE & RUNTIME CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ExperimentConfig:
    # Path mappings for the pre-balanced Framingham files
    CANDIDATE_POOL_PATH = "framingham_candidate_balanced.csv"
    TEST_SET_PATH = "framingham_test_fixed.csv"
    INDICES_PATH = "master_granular_indices_framingham.json"
    
    ITERATIONS = 20
    K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16] 
    
    # Feature subsets derived directly from your TabPFN-KernelSHAP ranking json
    TOP_5_FEATURES = ["BMI", "sysBP", "age", "totChol", "diaBP"]
    TOP_10_FEATURES = [
        "BMI", "sysBP", "age", "totChol", "diaBP", 
        "glucose", "heartRate", "cigsPerDay", "prevalentHyp", "education"
    ]
    
    FEATURE_SETS = {
        "top5": TOP_5_FEATURES,
        "top10": TOP_10_FEATURES
    }
    
    TARGET = 'TenYearCHD'

# ==========================================
# 2. DATA LOADING & RESOURCE INITIALIZATION
# ==========================================
print(f"[*] Execution runtime device: {DEVICE.upper()}")

# Load separate pre-balanced dataset pools directly
print("[*] Accessing pre-balanced dataset partitions...")
train_pool = pd.read_csv(ExperimentConfig.CANDIDATE_POOL_PATH)
test_fixed = pd.read_csv(ExperimentConfig.TEST_SET_PATH)

total_test_samples = len(test_fixed)
print(f"[INFO] Candidate Resource Pool Size: {train_pool.shape[0]}")
print(f"[INFO] Total Fixed Evaluation Target Samples: {total_test_samples}")

# Load the master cross-paradigm synchronization index registry
if not os.path.exists(ExperimentConfig.INDICES_PATH):
    raise FileNotFoundError(
        f"[ERROR] Master index path not found at: {ExperimentConfig.INDICES_PATH}. "
        f"Please run your step3 indexing script first."
    )

print(f"[*] Parsing unified evaluation matrix: {ExperimentConfig.INDICES_PATH}")
with open(ExperimentConfig.INDICES_PATH, 'r') as f:
    master_indices = json.load(f)

# ==========================================
# 3. INITIALIZE TABPFN ARCHITECTURE
# ==========================================
print("[*] Instantiating TabPFN classifier architecture...")
clf = TabPFNClassifier(device=DEVICE)

# ==========================================
# 4. MAIN EXPERIMENTAL EXECUTION LOOP
# ==========================================
# Outer loop: Slicing feature spaces (Top-5 vs Top-10)
for config_name, feature_list in ExperimentConfig.FEATURE_SETS.items():
    print(f"\n{'='*60}")
    print(f">>> LAUNCHING TABPFN BASELINE: {config_name.upper()} ({len(feature_list)} Features)")
    print(f"{'='*60}")
    
    # Organize outputs into isolated subdirectories
    output_dir = f"framingham_tabpfn_baselines_{config_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Inner loops: Few-shot constraints and iterations
    for k in ExperimentConfig.K_VALUES:
        res_file = os.path.join(output_dir, f'framingham_tabpfn_results_granular_k{k}.csv')
        print(f"\n>>> [TabPFN - {config_name}] Evaluating scaling boundary: K={k}")
        
        # Reset existing tracking files to prevent dirty writes or appending duplication
        if os.path.exists(res_file):
            os.remove(res_file)
            
        for i in range(ExperimentConfig.ITERATIONS):
            # Extract the exact cross-paradigm synchronized row pointers
            support_idx = master_indices[str(k)][i]
            
            # Isolate support context tensors bound to the current feature slice subset
            X_train = train_pool.loc[support_idx, feature_list]
            y_train = train_pool.loc[support_idx, ExperimentConfig.TARGET]
            
            # Isolate the query validation matrix slice
            X_test = test_fixed[feature_list]
            
            # Execute In-Context Learning conditioning step
            clf.fit(X_train, y_train)
            
            # Perform localized inference over the frozen target partition
            predictions = clf.predict(X_test)
            
            # Consolidate and flush records immediately after each iteration step
            iter_logs = []
            for idx, pred in zip(test_fixed.index, predictions):
                ground_truth = int(test_fixed.loc[idx, ExperimentConfig.TARGET])
                iter_logs.append([i, idx, ground_truth, int(pred)])
                
            pd.DataFrame(iter_logs).to_csv(res_file, mode='a', index=False, header=False)
            print(f"    K={k} | Iteration {i+1}/{ExperimentConfig.ITERATIONS} finalized.", end='\r')
        
        print(f"\n[SUCCESS] Parametric logging matrix compiled at: {res_file}")

print("\n" + "="*60)
print("[FINISH] All feature-stratified baseline simulations completed for TabPFN.")
print("="*60)