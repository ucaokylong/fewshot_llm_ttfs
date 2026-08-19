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
    
    # Complete 15 features matching the Framingham Heart Study data space
    FEATURES = [
        'male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 
        'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
        'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
    ]
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
# 3. INITIALIZE TABPFN ARCHITECTURE
# ==========================================
# TabPFN evaluates in-context without formal backpropagation weights updates
print("[*] Instantiating TabPFN classifier architecture...")
clf = TabPFNClassifier(device=DEVICE)

# ==========================================
# 4. MAIN EXPERIMENTAL EXECUTION LOOP
# ==========================================
for k in ExperimentConfig.K_VALUES:
    res_file = f'framingham_tabpfn_results_granular_k{k}.csv'
    print(f"\n>>> [TabPFN] Evaluating scaling boundary: K={k}")
    
    all_logs = []
    for i in range(ExperimentConfig.ITERATIONS):
        # Extract the exact cross-paradigm synchronized row pointers
        support_idx = master_indices[str(k)][i]
        
        # Isolate support context tensors
        X_train = train_pool.loc[support_idx, ExperimentConfig.FEATURES]
        y_train = train_pool.loc[support_idx, ExperimentConfig.TARGET]
        
        # Isolate the evaluation target space
        X_test = test_fixed[ExperimentConfig.FEATURES]
        
        # Execute In-Context Learning conditioning step
        clf.fit(X_train, y_train)
        
        # Perform localized query set inference
        predictions = clf.predict(X_test)
        
        # Log granular record outputs matching the exact sequence framework
        for idx, pred in zip(test_fixed.index, predictions):
            ground_truth = int(test_fixed.loc[idx, ExperimentConfig.TARGET])
            all_logs.append([i, idx, ground_truth, int(pred)])
            
        print(f"    Iteration {i+1}/{ExperimentConfig.ITERATIONS} finalized.", end='\r')
    
    # Flush evaluation dataframe directly to storage disk
    pd.DataFrame(all_logs).to_csv(res_file, index=False, header=False)
    print(f"\n[SUCCESS] Parametric tracking logs compiled at: {res_file}")

print("\n[FINISH] Full feature array evaluation pipeline completed for TabPFN.")