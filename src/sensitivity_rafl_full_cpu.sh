#!/bin/bash -l
#SBATCH -J sensitivity_RAFL_fullCPU
#SBATCH -p p.geany
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --array=0-50%74

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
# MNIST sensitivity experiments
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --alpha_stale 0.0"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --alpha_stale 0.1"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --alpha_stale 0.2"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --alpha_stale 0.5"

#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.0"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.005"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.01"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.05"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.1"

#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --byz_frac 0.0"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --byz_frac 0.1"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --byz_frac 0.2"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --byz_frac 0.3"
#"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --byz_frac 0.4"

"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --tau_max_rounds 0"
"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --tau_max_rounds 3"
"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --tau_max_rounds 5"
"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --tau_max_rounds 10"
"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --tau_max_rounds 20"

"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --num_clients 10 --num_clients_per_round 5 --max_parallel_clients 5"
"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --num_clients 20 --num_clients_per_round 5 --max_parallel_clients 5"
"python rafl_mnist_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --num_clients 50 --num_clients_per_round 10 --max_parallel_clients 10"

# CIFAR-10 sensitivity experiments

#"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --alpha_stale 0.0"
#"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --alpha_stale 0.1"
#"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --alpha_stale 0.2"
#"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --alpha_stale 0.5"

#"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.0"
#"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.005"
#"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.01"
#"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.05"
#"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.1"

#"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --byz_frac 0.0"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --byz_frac 0.1"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --byz_frac 0.2"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --byz_frac 0.3"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --byz_frac 0.4"

"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --tau_max_rounds 0"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --tau_max_rounds 3"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --tau_max_rounds 5"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --tau_max_rounds 10"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --tau_max_rounds 20"

"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --num_clients 10 --num_clients_per_round 5 --max_parallel_clients 5"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --num_clients 20 --num_clients_per_round 5 --max_parallel_clients 5"
"python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --num_clients 50 --num_clients_per_round 10 --max_parallel_clients 10"
)

echo "Task ${SLURM_ARRAY_TASK_ID} on ${SLURM_NODELIST}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Running: ${CMDS[$SLURM_ARRAY_TASK_ID]}"

srun --cpu-bind=cores ${CMDS[$SLURM_ARRAY_TASK_ID]}
EOF
