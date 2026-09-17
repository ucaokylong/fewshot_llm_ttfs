#!/bin/bash
#SBATCH -p SINGLE        # Sử dụng phân vùng CPU đơn node trên HAKUSAN 
#SBATCH -J TabPFN_Diabetes
#SBATCH -n 1             # Số lượng task [cite: 590]
#SBATCH -c 16            # Yêu cầu 16 lõi CPU cho task này [cite: 590]
#SBATCH --mem=32G        # Yêu cầu 32GB RAM [cite: 590]
#SBATCH --time=24:00:00  # Giới hạn thời gian 24 giờ 
#SBATCH -o TabPFN_Output-%j.log

# 1. Di chuyển vào thư mục làm việc [cite: 549]
cd ${SLURM_SUBMIT_DIR}

# ==========================================
# BƯỚC 0: DỌN DẸP MÔI TRƯỜNG
# ==========================================
unset PYTHONPATH
unset PYTHONHOME

# ==========================================
# BƯỚC 1: KÍCH HOẠT MÔI TRƯỜNG
# ==========================================
# HAKUSAN chạy Ubuntu 24.04 trên các compute node [cite: 156]
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate research_venv

# Đảm bảo dùng đúng Python trong môi trường ảo
export PATH=$HOME/miniconda3/envs/research_venv/bin:$PATH

# ==========================================
# BƯỚC 2: CẤU HÌNH CACHE
# ==========================================
# Lưu ý: HAKUSAN có hệ thống lưu trữ /home dùng chung [cite: 89, 318]
export HF_HOME=${SLURM_SUBMIT_DIR}/.cache
mkdir -p .cache

# ==========================================
# BƯỚC 3: CHẠY TABPFN (CHỈ DÙNG CPU)
# ==========================================
echo "[*] Bat dau chay TabPFN tren HAKUSAN luc: $(date)"
echo "[*] Node dang chay: $(hostname)" # Sẽ là các node lcpcc-xxx [cite: 141]

# Chạy script với cờ -u để log được cập nhật liên tục
# python -u TabPFN_diabetes.py
python -u xai_feature_extraction.py

echo "[*] Hoan thanh luc: $(date)"