#!/bin/bash
# Complete MIO Visual CoT evaluation script for all tasks
# Each task runs on a different GPU with a unique port
#
# Usage:
#   bash complete_mio_cot.sh
#
# Tasks:
#   - chartqa100_visual_cot: Chart question answering with visual CoT
#   - mathvista_visual_cot: Mathematical visual reasoning
#   - uni_mmmu_jigsaw100_visual_cot: Jigsaw puzzle solving
#   - Add more visual CoT tasks as needed

echo "======================================"
echo "MIO Visual CoT - Complete Evaluation"
echo "======================================"
echo "Starting evaluation on all Visual CoT tasks..."
echo ""

# ChartQA Visual CoT - GPU 0
echo "Task 1/4: ChartQA Visual CoT"
bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "2" "chartqa100_visual_cot" "./logs/mio_cot/chartqa" "m-a-p/MIO-7B-Instruct" "29602"

# MathVista Visual CoT - GPU 1
echo "Task 2/4: MathVista Visual CoT"
bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "1" "mathvista_visual_cot" "./logs/mio_cot/mathvista" "m-a-p/MIO-7B-Instruct" "29603"

# Uni-MMMU Jigsaw Visual CoT - GPU 2
echo "Task 3/4: Uni-MMMU Jigsaw Visual CoT"
bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "2" "uni_mmmu_jigsaw100_visual_cot" "./logs/mio_cot/jigsaw" "m-a-p/MIO-7B-Instruct" "29604"

# Add more visual CoT tasks here as needed
# Example:
# echo "Task 4/4: Custom Visual CoT"
# bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "3" "custom_visual_cot" "./logs/mio_cot/custom" "m-a-p/MIO-7B-Instruct" "29605"

echo ""
echo "======================================"
echo "All evaluations completed!"
echo "Results saved to: ./logs/mio_cot/"
echo "======================================"
