#!/bin/bash -l
#SBATCH -J cifar_RAFL_fullCPU
#SBATCH -p p.geany
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --array=0-11%74

#SBATCH -D /u/alfo/RAFL_revision
#SBATCH -o /u/alfo/RAFL_revision/logs/%x-%A_%a.out
#SBATCH -e /u/alfo/RAFL_revision/logs/%x-%A_%a.err

set -euo pipefail

module purge
module load gcc/14 || true
source /u/alfo/dnn_env/bin/activate

export CUDA_VISIBLE_DEVICES=""

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p /u/alfo/RAFL_revision/logs

CMDS=(
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
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Running: ${CMDS[$SLURM_ARRAY_TASK_ID]}"

srun --cpu-bind=cores ${CMDS[$SLURM_ARRAY_TASK_ID]}
