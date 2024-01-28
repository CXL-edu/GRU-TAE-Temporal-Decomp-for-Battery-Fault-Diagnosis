#!/bin/bash

. D:/Coding/miniconda/etc/profile.d/conda.sh  # 添加 conda 命令
conda activate RUL_Benchmark    # 激活 conda 环境




# 运行 Python 脚本
python run.py \
  --seq_len 96 \
  --hid_len 64 \
  --input_size 7 \
  --hid_size 96 \
  --model_name AE_CrossAttn_hl64_hs96 >> ./log/AE_CrossAttn_hl64_hs96.log


python run.py \
  --seq_len 96 \
  --hid_len 64 \
  --input_size 7 \
  --hid_size 64 \
  --model_name AE_CrossAttn_hl64_hs64 >> ./log/AE_CrossAttn_hl64_hs64.log


python run.py \
  --seq_len 96 \
  --hid_len 64 \
  --input_size 7 \
  --hid_size 28 \
  --model_name AE_CrossAttn_hl64_hs28 >> ./log/AE_CrossAttn_hl64_hs28.log

##############################################################################################################

python run.py \
  --seq_len 96 \
  --hid_len 28 \
  --input_size 7 \
  --hid_size 28 \
  --model_name AE_CrossAttn_hl28_hs28 >> ./log/AE_CrossAttn_hl28_hs28.log


python run.py \
  --seq_len 96 \
  --hid_len 28 \
  --input_size 7 \
  --hid_size 64 \
  --model_name AE_CrossAttn_hl28_hs64 >> ./log/AE_CrossAttn_hl28_hs64.log


python run.py \
  --seq_len 96 \
  --hid_len 28 \
  --input_size 7 \
  --hid_size 96 \
  --model_name AE_CrossAttn_hl28_hs96 >> ./log/AE_CrossAttn_hl28_hs96.log

##############################################################################################################

python -u run.py \
  --seq_len 96 \
  --hid_len 96 \
  --input_size 7 \
  --hid_size 28 \
  --model_name AE_CrossAttn_hl96_hs28 >> ./log/AE_CrossAttn_hl96_hs28.log


python run.py \
  --seq_len 96 \
  --hid_len 96 \
  --input_size 7 \
  --hid_size 64 \
  --model_name AE_CrossAttn_hl96_hs64 >> ./log/AE_CrossAttn_hl96_hs64.log


python run.py \
  --seq_len 96 \
  --hid_len 96 \
  --input_size 7 \
  --hid_size 96 \
  --model_name AE_CrossAttn_hl96_hs96 >> ./log/AE_CrossAttn_hl96_hs96.log