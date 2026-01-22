#!/bin/bash
GPU_IDS=${1:-"0,1"}
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# -----------------------------
# 关键：彻底禁止分布式（否则会碰 c10d/IPv6 然后卡死）
# -----------------------------
unset RANK WORLD_SIZE LOCAL_RANK NODE_RANK GROUP_RANK ROLE_RANK
unset MASTER_ADDR MASTER_PORT
unset TORCHELASTIC_RUN_ID TORCHELASTIC_USE_AGENT

# 有些库内部会尝试初始化通信；强制只用 IPv4（保险）
export NCCL_SOCKET_FAMILY="AF_INET"
export GLOO_USE_IPV6="0"
export TORCH_DISTRIBUTED_USE_IPV6="0"
export HOSTNAME="localhost"

# 你原来的禁用项（如果你想让跨卡更快，可以把 P2P_DISABLE=0；先保守不动）
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

export NCCL_DEBUG="INFO"

echo "Running on GPUs (visible): ${GPU_IDS}"
echo "Model parallel mode: single process + infer_auto_device_map"

# -----------------------------
# 单进程运行：依赖模型侧 infer_auto_device_map 做 device_map
# -----------------------------
python -m lmms_eval \
  --model bagel \
  --model_args pretrained=./BAGEL-7B-MoT,mode=generation,infer_auto_device_map=true,output_image_dir=/mnt/data/bagel_maze/images \
  --tasks uni_mmmu_maze100_visual_cot \
  --batch_size 1 \
  --output_path /mnt/data/bagel_maze \
  --log_samples \
  --verbosity INFO
