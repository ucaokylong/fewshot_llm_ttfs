import pandas as pd
import numpy as np
import json

def generate_master_indices(candidate_file="heart_attack_candidate_balanced.csv"):
    print("=== STEP 3: GENERATING MASTER GRANULAR INDICES MATRIX ===")
    df = pd.read_csv(candidate_file)
    TARGET = 'Heart Attack Risk'
    
    shot_sizes = [2, 4, 6, 8, 10, 12, 14, 16]
    num_iterations = 20
    master_indices = {}
    
    # Isolate row index arrays for each class partition within the balanced candidate pool
    idx_class_0 = df[df[TARGET] == 0].index.tolist()
    idx_class_1 = df[df[TARGET] == 1].index.tolist()
    
    # Use Generator instance for uniform matrix permutation
    rng = np.random.default_rng(seed=42)
    
    for K in shot_sizes:
        shots_per_class = K // 2
        K_loops = []
        
        for loop in range(num_iterations):
            # Draw random indices without replacement from each respective class pool
            samples_0 = rng.choice(idx_class_0, size=shots_per_class, replace=False).tolist()
            samples_1 = rng.choice(idx_class_1, size=shots_per_class, replace=False).tolist()
            
            # Combine and shuffle the index selections to prevent ordering bias inside the prompt
            combined_shot_indices = samples_0 + samples_1
            rng.shuffle(combined_shot_indices)
            
            K_loops.append(combined_shot_indices)
            
        master_indices[str(K)] = K_loops
        
    # Serialize the generated evaluation matrix into the unified JSON schema
    output_json_path = "master_granular_indices_heart_attack.json"
    with open(output_json_path, "w") as f:
        json.dump(master_indices, f)
        
    print(f"\n[SUCCESS] Master cross-paradigm JSON index map created at: {output_json_path}\n")

if __name__ == "__main__":
    generate_master_indices()