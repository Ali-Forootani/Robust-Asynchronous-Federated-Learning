#!/bin/bash -l
#SBATCH -J cifar_RAFL_parallel
#SBATCH -p p.geany
#SBATCH --nodelist=geanyg101
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=168:00:00
#SBATCH --array=0-11%4

#SBATCH -D /u/alfo/RAFL_revision
#SBATCH -o /u/alfo/RAFL_revision/logs/%x-%A_%a.out
#SBATCH -e /u/alfo/RAFL_revision/logs/%x-%A_%a.err

set -euo pipefail

module purge
module load cuda/12.2 || module load cuda || true
module load gcc/14 || true
source /u/alfo/dnn_env/bin/activate

CMDS=(
"python rafl_cifar10_experiment_suite.py --suite smoke"
"python rafl_cifar10_experiment_suite.py --suite attacks"
"python rafl_cifar10_experiment_suite.py --suite baselines"
"python rafl_cifar10_experiment_suite.py --suite sensitivity"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 70.0"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 80.0"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 90.0"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 100.0"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 110.0"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 120.0"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 130.0"
)

echo "Task ${SLURM_ARRAY_TASK_ID} on ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Running: ${CMDS[$SLURM_ARRAY_TASK_ID]}"

srun ${CMDS[$SLURM_ARRAY_TASK_ID]}
