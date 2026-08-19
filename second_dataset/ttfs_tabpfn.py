import os
import json
import torch
import pandas as pd
import numpy as np
from scipy.stats import entropy
from tabpfn import TabPFNClassifier
from tqdm import tqdm

# =====================================================================
# 1. SYSTEM CONFIGURATION & FEATURE SPACE
# =====================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Explicit paths to the pre-balanced Framingham partitions
CANDIDATE_POOL_PATH = "framingham_candidate_balanced.csv"
TEST_SET_PATH = "framingham_test_fixed.csv"
INDEX_FILE_PATH = "master_granular_indices_framingham.json"
TARGET = "TenYearCHD"

FEW_SHOT_K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
TOTAL_ITERATIONS = 20
MINIMUM_CONFIDENCE_THRESHOLD = 0.60

# Empirical KernelSHAP feature rankings established for the Framingham dataset
TOP_5_BASE_FEATURES = ["BMI", "sysBP", "age", "totChol", "diaBP"]
TOP_10_BASE_FEATURES = [
    "BMI", "sysBP", "age", "totChol", "diaBP", 
    "glucose", "heartRate", "cigsPerDay", "prevalentHyp", "education"
]

CONFIGURATIONS_TO_RUN = {
    "top5": TOP_5_BASE_FEATURES,
    "top10": TOP_10_BASE_FEATURES
}

# =====================================================================
# 2. CORE ALGORITHM: BATCHED FLEXIBLE TTFS
# =====================================================================
def execute_batched_flexible_ttfs(tabpfn_model, full_support_x, full_support_y, test_patient_x_all, starting_feature_list):
    """
    Upgraded TTFS Algorithm: Allows the elimination of any noisy features arbitrarily.
    Leverages dynamic batching to group patients sharing identical feature geometric states.
    """
    num_patients = len(test_patient_x_all)
    
    # Initial state matrices contain the complete starting feature list for all target patients
    current_states = [tuple(starting_feature_list) for _ in range(num_patients)]
    
    best_confidences = np.zeros(num_patients)
    best_probabilities = np.zeros((num_patients, 2))
    
    # ---------------------------------------------------------
    # STEP 1: Establish baseline confidence metrics (Full Feature Anchor)
    # ---------------------------------------------------------
    base_features = list(starting_feature_list)
    tabpfn_model.fit(full_support_x[base_features], full_support_y)
    
    base_probs = tabpfn_model.predict_proba(test_patient_x_all[base_features])
    base_confs = 1.0 - entropy(base_probs, base=2, axis=1)
    
    best_confidences[:] = base_confs
    best_probabilities[:] = base_probs
    
    # ---------------------------------------------------------
    # STEP 2: Greedy single-feature pruning traversal (Bottom-Up)
    # ---------------------------------------------------------
    for feature_to_test in reversed(starting_feature_list):
        
        # Cluster patients sharing identical feature dimensional spaces in the current loop
        state_to_patient_indices = {}
        for i, state in enumerate(current_states):
            # Only attempt pruning if the feature exists and dimensional integrity is maintained (>2 features)
            if feature_to_test in state and len(state) > 2:
                if state not in state_to_patient_indices:
                    state_to_patient_indices[state] = []
                state_to_patient_indices[state].append(i)
                
        # Process clustered state tensors in parallel validation batches
        for state, patient_indices in state_to_patient_indices.items():
            
            # Generate candidate state: Exclude the target feature while preserving prior pruning holes
            new_state = tuple(f for f in state if f != feature_to_test)
            state_list = list(new_state)
            
            subset_test_x = test_patient_x_all.iloc[patient_indices]
            
            # Dispatch the isolated matrix chunk to the GPU for inference conditioning
            tabpfn_model.fit(full_support_x[state_list], full_support_y)
            new_probs = tabpfn_model.predict_proba(subset_test_x[state_list])
            new_confs = 1.0 - entropy(new_probs, base=2, axis=1)
            
            # Evaluate the confidence delta for each localized entity
            for local_idx, global_idx in enumerate(patient_indices):
                curr_conf = new_confs[local_idx]
                
                # GATING RULE: If pruning improves or sustains confidence, and exceeds the safety boundary
                if curr_conf >= best_confidences[global_idx] and curr_conf >= MINIMUM_CONFIDENCE_THRESHOLD:
                    current_states[global_idx] = new_state  # Permanently accept the dimensionality reduction
                    best_confidences[global_idx] = curr_conf
                    best_probabilities[global_idx] = new_probs[local_idx]
                else:
                    # AUTOMATIC BACKTRACK: Reject the pruning attempt and preserve previous dimensional state
                    pass
                    
    return current_states, best_probabilities

# =====================================================================
# 3. MAIN EXECUTION PROCESSOR
# =====================================================================
if __name__ == "__main__":
    print(f"[*] Initializing Batched-TTFS Acceleration Framework on: {DEVICE.upper()}")
    
    # Directly slice pre-balanced candidate validation pools
    training_pool = pd.read_csv(CANDIDATE_POOL_PATH)
    test_patient_pool = pd.read_csv(TEST_SET_PATH)
    
    if not os.path.exists(INDEX_FILE_PATH):
        raise FileNotFoundError(f"[ERROR] Master index map missing at: {INDEX_FILE_PATH}")
        
    with open(INDEX_FILE_PATH, 'r') as index_file:
        few_shot_indices_map = json.load(index_file)
        
    global_tabpfn_model = TabPFNClassifier(device=DEVICE)
    
    for config_name, starting_features in CONFIGURATIONS_TO_RUN.items():
        output_directory = f"framingham_ttfs_tabpfn_{config_name}"
        os.makedirs(output_directory, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🚀 LAUNCHING FLEXIBLE-BATCHED TTFS: {config_name.upper()}")
        print(f"{'='*60}")
        
        for current_k in FEW_SHOT_K_VALUES:
            results_csv_path = os.path.join(output_directory, f'framingham_dynamic_results_k{current_k}.csv')
            
            if os.path.exists(results_csv_path):
                os.remove(results_csv_path)
                
            iteration_logs = []
            iter_progress = tqdm(range(TOTAL_ITERATIONS), desc=f"Evaluating Sequence K={current_k}")
            
            for current_iter in iter_progress:
                support_indices = few_shot_indices_map[str(current_k)][current_iter]
                support_x = training_pool.loc[support_indices]
                support_y = training_pool.loc[support_indices, TARGET]
                
                # Execute core algorithmic block
                final_states, final_probs = execute_batched_flexible_ttfs(
                    tabpfn_model=global_tabpfn_model,
                    full_support_x=support_x,
                    full_support_y=support_y,
                    test_patient_x_all=test_patient_pool,
                    starting_feature_list=starting_features
                )
                
                # Consolidate pristine structured tracking metrics
                for global_idx in range(len(test_patient_pool)):
                    ground_truth_label = int(test_patient_pool.iloc[global_idx][TARGET])
                    predicted_label = 1 if final_probs[global_idx][1] > 0.5 else 0
                    retained_features_count = len(final_states[global_idx])
                    
                    retained_features_list_str = ";".join(final_states[global_idx])
                    real_test_idx = test_patient_pool.index[global_idx]
                    
                    iteration_logs.append([
                        current_iter, 
                        real_test_idx, 
                        ground_truth_label, 
                        predicted_label, 
                        retained_features_count,
                        retained_features_list_str
                    ])
                    
            pd.DataFrame(iteration_logs).to_csv(results_csv_path, index=False, header=False)
            print(f"✅ Sequence logging finalized for K={current_k} under configuration {config_name.upper()}.")