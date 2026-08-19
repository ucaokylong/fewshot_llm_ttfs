import os
import json
import torch
import shap
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier

# ==========================================
# 1. RUNTIME CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ExperimentConfig:
    # Path mappings for the pre-balanced Framingham files
    CANDIDATE_POOL_PATH = "framingham_candidate_balanced.csv"
    INDICES_PATH = "master_granular_indices_framingham.json"
    
    # Target few-shot shot-size string key to compute the explanation space
    K_TARGET = "16" 
    ITERATIONS = 20
    
    # Complete 15 features matching the Framingham Heart Study data space
    FEATURES = [
        'male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 
        'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
        'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
    ]
    TARGET = 'TenYearCHD'

# Configure isolated scratch directory to safely store progress checkpoints
TEMP_DIR = f"framingham_xai_temp_k{ExperimentConfig.K_TARGET}"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 2. RESOURCE & DATA LOADING
# ==========================================
print(f"[*] Initializing Global XAI Ranking Routine for K={ExperimentConfig.K_TARGET} on {DEVICE.upper()}")

# Load separate pre-balanced candidate training pool directly
print("[*] Accessing candidate dataset partition...")
train_pool = pd.read_csv(ExperimentConfig.CANDIDATE_POOL_PATH)
print(f"[INFO] Candidate Resource Pool Size: {train_pool.shape[0]}")

# Load the master cross-paradigm synchronization index matrix
with open(ExperimentConfig.INDICES_PATH, 'r') as f:
    master_indices = json.load(f)

all_importance_results = []

# Instantiate the specialized tabular transformer architecture
clf = TabPFNClassifier(device=DEVICE)

# ==========================================
# 3. GRANULAR SHAP EXECUTION LOOP (WITH CHECKPOINT RESUME)
# ==========================================
for i in range(ExperimentConfig.ITERATIONS):
    temp_file = os.path.join(TEMP_DIR, f"iter_{i}.csv")
    
    # Checkpoint Validation: Bypass iteration if matching logs exist on disk
    if os.path.exists(temp_file):
        print(f"[*] Found pre-computed logs for Iteration {i+1}/{ExperimentConfig.ITERATIONS}. Skipping execution...")
        all_importance_results.append(pd.read_csv(temp_file))
        continue

    print(f"--- Processing Explanation Sequence: Iteration {i+1}/{ExperimentConfig.ITERATIONS} ---")
    
    # Extract the exact cross-paradigm synchronized row pointers
    support_idx = master_indices[ExperimentConfig.K_TARGET][i]
    X_train = train_pool.loc[support_idx, ExperimentConfig.FEATURES]
    y_train = train_pool.loc[support_idx, ExperimentConfig.TARGET]

    # Condition model on the isolated reference context
    clf.fit(X_train, y_train)

    # Initialize KernelSHAP to estimate game-theoretic feature attributions
    explainer = shap.KernelExplainer(clf.predict_proba, X_train)
    shap_values = explainer.shap_values(X_train)

    # Shape Normalization Handling: Resolve multi-class list vs array outputs
    if isinstance(shap_values, list):
        # Extract attribution scores corresponding to the positive class (Index 1)
        importance_matrix = np.abs(shap_values[1])
    else:
        if len(shap_values.shape) == 3:
            importance_matrix = np.abs(shap_values[:, :, 1])
        else:
            importance_matrix = np.abs(shap_values)

    # Compute mean absolute attribution across the active context instances
    importance_scores = importance_matrix.mean(axis=0)

    # Strict structural defensive parsing: Force 1D array matching feature dimensions
    if importance_scores.ndim == 0:
        importance_scores = np.full(len(ExperimentConfig.FEATURES), importance_scores)
    elif len(importance_scores) != len(ExperimentConfig.FEATURES):
        print(f"[WARNING] Dimension mismatch detected: {importance_scores.shape}")
    
    # Construct localized iteration metrics data structures
    iter_df = pd.DataFrame({
        'feature': ExperimentConfig.FEATURES,
        f'iter_{i}': importance_scores
    })
    
    # Secure tracking records to storage disk immediately
    iter_df.to_csv(temp_file, index=False)
    all_importance_results.append(iter_df)

# ==========================================
# 4. GLOBAL MATRICES CONSOLIDATION & AGGREGATION
# ==========================================
print("\n[*] Consolidating cross-iteration metric spaces...")
final_df = all_importance_results[0]
for next_df in all_importance_results[1:]:
    final_df = final_df.merge(next_df, on='feature')

# Calculate mathematical statistical parameters across the 20 trials
iter_cols = [f'iter_{i}' for i in range(ExperimentConfig.ITERATIONS)]
final_df['mean_importance'] = final_df[iter_cols].mean(axis=1)
final_df['std_importance'] = final_df[iter_cols].std(axis=1)

# Sort feature tracking indicators based on aggregate mean attribution scores
ranking_df = final_df[['feature', 'mean_importance', 'std_importance']].sort_values(
    by='mean_importance', ascending=False
).reset_index(drop=True)

# Map human-readable ranking index pointers
ranking_df.insert(0, 'rank', ranking_df.index + 1)

print("\n\n" + "="*60)
print(f"GLOBAL FEATURE IMPORTANCE RANKING (K={ExperimentConfig.K_TARGET} | {ExperimentConfig.ITERATIONS} ITERS)")
print("="*60)
print(ranking_df.to_string(index=False))
print("="*60)

# ==========================================
# 5. METRIC SERIALIZATION
# ==========================================
output_csv = f"framingham_xai_global_ranking_k{ExperimentConfig.K_TARGET}.csv"
output_json = f"framingham_xai_selected_features_k{ExperimentConfig.K_TARGET}.json"

ranking_df.to_csv(output_csv, index=False)

top_5 = ranking_df.head(5)['feature'].tolist()
top_10 = ranking_df.head(10)['feature'].tolist()

with open(output_json, "w") as f:
    json.dump({
        "full_ranking": ranking_df.to_dict('records'),
        "top_5": top_5,
        "top_10": top_10
    }, f, indent=4)

print(f"\n[SUCCESS] Global ranking metrics saved to: {output_csv}")
print(f"[INFO] Top 5 Identified Risk Factors: {top_5}")
print(f"[INFO] Top 10 Identified Risk Factors: {top_10}")
print(f"[SUCCESS] Target subsets logged to: {output_json}\n")