#!/bin/bash
#PBS -q SINGLE
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N Diabetes_TabPFN_LLM_metrics

cd $PBS_O_WORKDIR

# 1. Kích hoạt môi trường
source $HOME/miniconda3/bin/activate research_venv

# 2. Cài đặt TabPFN ngay trong script (Sử dụng --quiet để log sạch hơn)
# TabPFN là mô hình Foundation có khả năng vượt trội trên các tập dữ liệu dưới 10,000 mẫu
# echo "[*] Kiểm tra và cập nhật thư viện TabPFN..."
# pip install --upgrade tabpfn --quiet

# 3. Cấu hình Cache cho Hugging Face (TabPFN sẽ tải pre-trained weights về đây)



# 4. Chạy Baseline với TabPFN
# TabPFN có thể xử lý dữ liệu bảng với tối đa 500 đặc trưng[cite: 31, 759].
echo "--- STARTING TABPFN BASELINE ---"
python calculate_metric.py