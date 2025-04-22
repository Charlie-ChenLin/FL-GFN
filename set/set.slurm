#!/bin/bash

#SBATCH --job-name=FLGFN-set
#SBATCH --partition=RTX3090
#SBATCH --nodes=1               # 单节点
#SBATCH --ntasks-per-node=1     # 每个节点上的任务数，2张GPU对应2个任务
#SBATCH --cpus-per-task=8    # 每个任务分配的CPU核数
#SBATCH --gres=gpu:1            
#SBATCH --mail-type=all         # 邮件通知类型
#SBATCH --mail-user=chenlin114514@sjtu.edu.cn # 邮件地址
#SBATCH --output=/home/lchen/home/FL-GFN/log/%j.out
#SBATCH --error=/home/lchen/home/FL-GFN/log/%j.err

# 加载 Conda 环境
source /home/lchen/miniconda3/etc/profile.d/conda.sh
conda activate flgfn-set

# 切换到项目目录
cd /home/lchen/home/FL-GFN/set

# 设置随机 MASTER_PORT 避免端口冲突
export WANDB_API_KEY="2e3f0cdfb6089a1e52ee117856ca6dfcf331858b"
export MASTER_PORT=$((20000 + RANDOM % 10000))
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# baselines
# python gflownet.py --method db_gfn --size small --seed 1 --alpha 0.5 # finished
# python gflownet.py --method tb_gfn --size small --seed 1 --alpha 0.5 # finished
# python gflownet.py --method db_gfn --fl 1 --size small --seed 1 --alpha 0.5 # finished

# python gflownet.py --method db_gfn --size medium --seed 1 --alpha 0.5 --mode_threshold 700000 --num_threads 4 # finished
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1 --alpha 0.5 --mode_threshold 700000 --num_threads 4 # RTX3090 PD
# python gflownet.py --method tb_gfn --size medium --seed 1 --alpha 0.5 --mode_threshold 700000 --num_threads 8 # finished

# python gflownet.py --method db_gfn --size large --seed 1 --alpha 0.5 --mode_threshold 700000 --num_threads 4 # finished
# python gflownet.py --method tb_gfn --size large --seed 1 --alpha 0.5 --mode_threshold 700000 --num_threads 4 # ADA6000 PD
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1 --alpha 0.5 --mode_threshold 700000 --num_threads 4 # finished

# biased GFN, small
# vanilla weight

# RTX3090 79644 79645 79646
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.2
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.2
# python gflownet.py --method db_gfn --fl 1 --size small --seed 1 --mode_threshold 0.25 --num_threads 8 --alpha 0.2

# RTX3090 79588 finished
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.8
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.8
# python gflownet.py --method db_gfn --fl 1 --size small --seed 1 --mode_threshold 0.25 --num_threads 8 --alpha 0.8

# RTX3090 79649 79650 ADA6000 79651
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.7
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.7
# python gflownet.py --method db_gfn --fl 1 --size small --seed 1 --mode_threshold 0.25 --num_threads 8 --alpha 0.7

# RTX3090 79652 79654 79655
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.6
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.6
# python gflownet.py --method db_gfn --fl 1 --size small --seed 1 --mode_threshold 0.25 --num_threads 8 --alpha 0.6

# RTX3090 79656 79657 79658
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.9
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.9
# python gflownet.py --method db_gfn --fl 1 --size small --seed 1 --mode_threshold 0.25 --num_threads 8 --alpha 0.9

# ppl-like weight, reverse weight~=0.5764
# RTX3090 79668 79669 79670
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.5764
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.5764
# python gflownet.py --method db_gfn --fl 1 --size small --seed 1 --mode_threshold 0.25 --num_threads 8 --alpha 0.5764

# ppl-like weight, weight~=0.4236
# RTX3090 79695 79696 79697
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4236
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4236
# python gflownet.py --method db_gfn --fl 1 --size small --seed 1 --mode_threshold 0.25 --num_threads 8 --alpha 0.4236

# TODO: rerun biased TB, there are some bugs
# RTX3090 79710 79712 79414 79715 79716 79717 79718
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.2
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4236
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.5764
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.6
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.7
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.8
# python gflownet.py --method tb_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.9

# rerun biased DB, deleted by mistake
# RTX3090 79724
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.2
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4236
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.5764
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.6
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.7
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.8
# python gflownet.py --method db_gfn --size small --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.9


# biased GFN, medium
# RTX3090 79735
# python gflownet.py --method db_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.2
# python gflownet.py --method db_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.3
# python gflownet.py --method db_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4
# python gflownet.py --method db_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4236
# python gflownet.py --method db_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.55
# python gflownet.py --method db_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.5764
# python gflownet.py --method db_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.6
# python gflownet.py --method db_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.7
# python gflownet.py --method db_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.8
# python gflownet.py --method db_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.9

# RTX3090 79736
# python gflownet.py --method tb_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.2
# python gflownet.py --method tb_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.3
# python gflownet.py --method tb_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4
# python gflownet.py --method tb_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4236
# python gflownet.py --method tb_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.55
# python gflownet.py --method tb_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.5764
# python gflownet.py --method tb_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.6
# python gflownet.py --method tb_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.7
# python gflownet.py --method tb_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.8
# python gflownet.py --method tb_gfn --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.9

# RTX3090 79738
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.2
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.3
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4236
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.55
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.5764
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.6
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.7
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.8
# python gflownet.py --method db_gfn --fl 1 --size medium --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.9

# biased GFN, large
# RTX3090 79776
# python gflownet.py --method db_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.2
# python gflownet.py --method db_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.3
# python gflownet.py --method db_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4
# python gflownet.py --method db_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4236
# python gflownet.py --method db_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.55
# python gflownet.py --method db_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.5764
# python gflownet.py --method db_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.6
# python gflownet.py --method db_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.7
# python gflownet.py --method db_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.8
# python gflownet.py --method db_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.9

# RTX3090 79779
# python gflownet.py --method tb_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.2
# python gflownet.py --method tb_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.3
# python gflownet.py --method tb_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4
# python gflownet.py --method tb_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4236
# python gflownet.py --method tb_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.55
# python gflownet.py --method tb_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.5764
# python gflownet.py --method tb_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.6
# python gflownet.py --method tb_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.7
# python gflownet.py --method tb_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.8
# python gflownet.py --method tb_gfn --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.9

# RTX3090 79780
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.2
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.3
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.4236
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.55
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.5764
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.6
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.7
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.8
# python gflownet.py --method db_gfn --fl 1 --size large --seed 1  --mode_threshold 0.25 --num_threads 8 --alpha 0.9
