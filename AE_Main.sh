#!/bin/bash

. D:/Coding/miniconda/etc/profile.d/conda.sh  # add conda command
conda activate RUL_Benchmark    # activate your conda env in Windows


# Run Python script
python run.py \
  --kernel_size 3 \
  --block_num 1 \
  --model_name AE_Main_k3_blk1 >> ./log/AE_Main_k3_blk1.log


python run.py \
  --kernel_size 3 \
  --block_num 2 \
  --model_name AE_Main_k3_blk2 >> ./log/AE_Main_k3_blk2.log


##################################################################
python run.py \
  --kernel_size 5 \
  --block_num 1 \
  --model_name AE_Main_k5_blk1 >> ./log/AE_Main_k5_blk1.log


python run.py \
  --kernel_size 5 \
  --block_num 2 \
  --model_name AE_Main_k5_blk2 >> ./log/AE_Main_k5_blk2.log


##################################################################
python run.py \
  --kernel_size 9 \
  --block_num 1 \
  --model_name AE_Main_k9_blk1 >> ./log/AE_Main_k9_blk1.log


python run.py \
  --kernel_size 9 \
  --block_num 2 \
  --model_name AE_Main_k9_blk2 >> ./log/AE_Main_k9_blk2.log
