#!/bin/bash

#SBATCH --job-name=my_rafl_mnist
#SBATCH --time=0-40:30:00
#SBATCH --gres=gpu:nvidia-a100:1
#SBATCH --mem-per-cpu=32G
#SBATCH --output="/data/bio-eng-llm/Robust_AFL_non_convex/logs/%x-%j.log"

# Change directory within the script


# Load any necessary modules, if require

source /data/bio-eng-llm/virtual_envs/dnn_env/bin/activate

# Execute the Python script
python /data/bio-eng-llm/Robust_AFL_non_convex/src/robust_async_fdl_mnist_delay_track_lr_non_iid.py

