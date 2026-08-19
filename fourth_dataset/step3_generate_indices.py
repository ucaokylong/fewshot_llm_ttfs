import pandas as pd
import numpy as np
import json

def generate_master_indices(candidate_file="cardio_candidate_balanced.csv"):
    print("=== STEP 3: GENERATING MASTER GRANULAR INDICES MATRIX ===")
    df = pd.read_csv(candidate_file)
    TARGET = 'cardio'
    
    shot_sizes = [2, 4, 6, 8, 10, 12, 14, 16]
    num_iterations = 20
    master_indices = {}
    
    # Tách chỉ mục dòng cho từng lớp nhãn
    idx_class_0 = df[df[TARGET] == 0].index.tolist()
    idx_class_1 = df[df[TARGET] == 1].index.tolist()
    
    rng = np.random.default_rng(seed=42)
    
    for K in shot_sizes:
        shots_per_class = K // 2
        K_loops = []
        
        for loop in range(num_iterations):
            # Rút mẫu ngẫu nhiên không thay thế từ từng phân vùng lớp
            samples_0 = rng.choice(idx_class_0, size=shots_per_class, replace=False).tolist()
            samples_1 = rng.choice(idx_class_1, size=shots_per_class, replace=False).tolist()
            
            # Trộn lẫn chỉ mục để tránh độ lệch thứ tự mẫu trong prompt
            combined_shot_indices = samples_0 + samples_1
            rng.shuffle(combined_shot_indices)
            
            K_loops.append(combined_shot_indices)
            
        master_indices[str(K)] = K_loops
        
    # Serialize ma trận lưu trữ chỉ mục
    output_json_path = "master_granular_indices_cardio.json"
    with open(output_json_path, "w") as f:
        json.dump(master_indices, f)
        
    print(f"\n[SUCCESS] Master JSON index map created at: {output_json_path}\n")

if __name__ == "__main__":
    generate_master_indices()