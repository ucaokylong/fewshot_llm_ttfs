#!/bin/bash
#SBATCH -p GPU-1
#SBATCH -J TabPFN_Diabetes_Top5_10_Inference
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH -o TabPFN_Inference_Top5_10_Output-%j.log

# Di chuyển vào thư mục submit job
cd $SLURM_SUBMIT_DIR

# ==========================================
# BƯỚC 0: DỌN DẸP MÔI TRƯỜNG
# ==========================================
unset PYTHONPATH
unset PYTHONHOME

# ==========================================
# BƯỚC 1: KÍCH HOẠT CONDA
# ==========================================
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate research_venv

# ==========================================
# BƯỚC 2: THIẾT LẬP CUDA & SSL (QUAN TRỌNG)
# ==========================================
# Sếp check bản cuda cao nhất trên máy (ví dụ 12.1 hoặc 12.2)
module load cuda/12.1 

# Fix triệt để lỗi SSL từ tầng Shell
export SSL_CERT_FILE=$(python -m certifi)
export REQUESTS_CA_BUNDLE=$(python -m certifi)

# Chuyển cache HF sang thư mục làm việc (tránh đầy 20GB quota thư mục Home)
export HF_HOME=$SLURM_SUBMIT_DIR/.cache
mkdir -p $HF_HOME

# Log thông tin để kiểm tra sau này
echo "[*] Thoi gian bat dau: $(date)"
echo "[*] Python dang dung: $(which python)"
echo "[*] SSL Cert dang dung: $SSL_CERT_FILE"
nvidia-smi

# ==========================================
# BƯỚC 3: CHẠY INFERENCE
# ==========================================
echo "=========================================================="
echo "[RUNNING] TabPFN Granular Evaluation..."
echo "=========================================================="

# Thêm biến flag để bitsandbytes không bị lỗi trên một số dòng GPU cũ
export BN_DISABLE_MALLOC_STATS=1

# Chạy script với cờ -u (unbuffered) để log in ra file .log ngay lập tức
python -u TabPFN_diabetes_top5_10.py

echo "=========================================================="
echo "[SUCCESS] Toan bo thi nghiem hoan thanh luc: $(date)"