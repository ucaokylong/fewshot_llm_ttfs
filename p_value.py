import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

# =====================================================================
# EXPERIMENTAL CONFIGURATION (DIABETES DATASET)
# =====================================================================
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
CONFIGS = ["top5", "top10"]
OUTPUT_CSV = "statistical_test_diabetes.csv"

def get_paths(config, k):
    return {
        "ttfs": f"llm_sniper_{config}/qwen_sniper_k{k}.csv",
        "static": f"llm_baselines_{config}/qwen_results_granular_k{k}.csv"
    }

def extract_iteration_f1(file_path):
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3], names=['iter', 'id', 'gt', 'pred'])
    
    iter_f1s = []
    for i in range(20):
        sub = df[df['iter'] == i]
        if len(sub) == 0: 
            continue
            
        y_true = pd.to_numeric(sub['gt'], errors='coerce').fillna(0).astype(int)
        y_pred_raw = pd.to_numeric(sub['pred'], errors='coerce').fillna(-1).astype(int)
        y_pred_cleaned = np.where((y_pred_raw != 0) & (y_pred_raw != 1), 1 - y_true, y_pred_raw)
        
        iter_f1s.append(f1_score(y_true, y_pred_cleaned, zero_division=0))
        
    return np.array(iter_f1s) if len(iter_f1s) == 20 else None

def run_paired_t_test():
    print(f"\n{'='*90}")
    print(f"{'PAIRED STUDENT T-TEST EVALUATION (F1-SCORE): LLM-TTFS vs. STATIC LLM':^90}")
    print(f"{'='*90}")

    summary_data = []

    for config in CONFIGS:
        print(f"\n>>> CANDIDATE FEATURE POOL: {config.upper()}")
        print(f"{'-'*90}")
        print(f"{'K':<4} | {'F1 TTFS (Mean)':<16} | {'F1 Static (Mean)':<17} | {'p-value':<12} | {'Statistical Verdict'}")
        print(f"{'-'*90}")

        for k in K_VALUES:
            paths = get_paths(config, k)
            ttfs_f1 = extract_iteration_f1(paths['ttfs'])
            static_f1 = extract_iteration_f1(paths['static'])
            
            if ttfs_f1 is None or static_f1 is None:
                print(f"{k:<4} | {'[Missing Data]':<16} | {'[Missing Data]':<17} | {'N/A':<12} | -")
                continue
                
            t_stat, p_val = stats.ttest_rel(ttfs_f1, static_f1)
            mean_ttfs = np.mean(ttfs_f1)
            mean_static = np.mean(static_f1)
            
            if p_val < 0.05 and mean_ttfs > mean_static:
                verdict = "TTFS Superior (p < 0.05) *"
            elif p_val < 0.05 and mean_ttfs < mean_static:
                verdict = "Static Superior (p < 0.05) *"
            else:
                verdict = "Not Significant (p >= 0.05)"
                
            print(f"{k:<4} | {mean_ttfs:<16.4f} | {mean_static:<17.4f} | {p_val:<12.4e} | {verdict}")
            
            summary_data.append({
                'Feature_Space': config.upper(),
                'K': k,
                'F1_TTFS_Mean': mean_ttfs,
                'F1_Static_Mean': mean_static,
                'p_value': p_val,
                'Verdict': verdict
            })
            
        print(f"{'-'*90}")

    if summary_data:
        pd.DataFrame(summary_data).to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Successfully exported test results to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_paired_t_test()