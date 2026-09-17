#!/bin/bash
#SBATCH -p GPU-1A
#SBATCH -J Qwen_Framingham_Full_Inference
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH -o Qwen_Inference_Full_Output-%j.log

# Navigate to the job submission working directory
cd $SLURM_SUBMIT_DIR

# ==========================================
# STEP 0: ENVIRONMENT SANITIZATION
# ==========================================
unset PYTHONPATH
unset PYTHONHOME

# ==========================================
# STEP 1: CONDA ENVIRONMENT ACTIVATION
# ==========================================
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate research_venv

# ==========================================
# STEP 2: HARDWARE & SSL RUNTIME SETUPS
# ==========================================
# Load the verified CUDA toolkit module
module load cuda/12.1 

# Bind production-grade SSL verification paths via certifi at the shell layer
export SSL_CERT_FILE=$(python -m certifi)
export REQUESTS_CA_BUNDLE=$(python -m certifi)

# Redirect HuggingFace caches to storage targets to prevent $HOME quota overflows
export HF_HOME=$SLURM_SUBMIT_DIR/.cache
mkdir -p $HF_HOME

# Output execution environment metadata logs for systematic auditing
echo "[*] Job Execution Started At: $(date)"
echo "[*] Python Runtime Path: $(which python)"
echo "[*] Bound SSL Certificate Path: $SSL_CERT_FILE"
nvidia-smi

# ==========================================
# STEP 3: PIPELINE EXECUTION
# ==========================================
echo "=========================================================="
echo "[RUNNING] Qwen2.5-7B Full Feature Space Context Inference..."
echo "=========================================================="

# Disable memory allocator telemetry to bypass potential bitsandbytes GPU faults
export BN_DISABLE_MALLOC_STATS=1

# Trigger full feature space evaluation script with unbuffered logging flag
# python -u llm_fewshot_full_features.py
python -u llm_research_top5_10.py

echo "=========================================================="
echo "[SUCCESS] Experimental evaluation sequence finalized at: $(date)"