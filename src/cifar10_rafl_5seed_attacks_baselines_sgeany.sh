#!/bin/bash -l
#SBATCH -J cifar_RAFL_5seed
#SBATCH -p s.geany
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=20G
#SBATCH --time=168:00:00
#SBATCH --array=0-64%65

#SBATCH -D /u/alfo/RAFL_revision
#SBATCH -o /u/alfo/RAFL_revision/logs/%x-%A_%a.out
#SBATCH -e /u/alfo/RAFL_revision/logs/%x-%A_%a.err

set -euo pipefail

# ============================================================
# CIFAR-10 RAFL reviewer multiseed experiments
#
# Five fixed seeds:
#   4046, 3076, 9872, 6845, 2534
#
# Unique experiments:
#
#   Attack table:
#       5 attacks/settings x 5 seeds = 25 jobs
#
#   Baseline aggregation table under sign-flip:
#       8 additional baselines x 5 seeds = 40 jobs
#
#   ASB + sign-flip is already computed in the attack table and
#   should be reused as the ASB row of the baseline table.
#
# Total unique jobs:
#
#       25 + 40 = 65
#
# Therefore:
#
#       --array=0-64%65
#
# All 65 jobs are allowed to run simultaneously if Slurm
# resources and scheduler policy permit.
# ============================================================


# ============================================================
# Environment
# ============================================================

module purge
module load gcc/14 || true

source /u/alfo/dnn_env/bin/activate


# ============================================================
# CPU execution
# ============================================================

# Explicitly disable GPU visibility because this job runs on
# the shared CPU partition s.geany.
export CUDA_VISIBLE_DEVICES=""

export PYTHONUNBUFFERED=1


# ============================================================
# CPU threading
# ============================================================
#
# The Python code may train up to 5 clients concurrently using
# asyncio.to_thread().
#
# Slurm allocation:
#
#       30 CPUs / task
#
# Parallel clients:
#
#       5
#
# Therefore:
#
#       30 / 5 = 6 CPU threads per concurrent client
#
# ============================================================

MAX_PARALLEL_CLIENTS=5

THREADS_PER_CLIENT=$(( SLURM_CPUS_PER_TASK / MAX_PARALLEL_CLIENTS ))

if (( THREADS_PER_CLIENT < 1 )); then
    THREADS_PER_CLIENT=1
fi

export OMP_NUM_THREADS=${THREADS_PER_CLIENT}
export MKL_NUM_THREADS=${THREADS_PER_CLIENT}
export OPENBLAS_NUM_THREADS=${THREADS_PER_CLIENT}
export NUMEXPR_NUM_THREADS=${THREADS_PER_CLIENT}
export VECLIB_MAXIMUM_THREADS=${THREADS_PER_CLIENT}

export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE


# ============================================================
# Paths
# ============================================================

mkdir -p /u/alfo/RAFL_revision/logs

PY_SCRIPT="/u/alfo/RAFL_revision/rafl_cifar10_experiment_suite.py"

RESULT_ROOT="/u/alfo/RAFL_revision/results_cifar10_5seed_revision"


# ============================================================
# Fixed seed list requested for statistical reporting
# ============================================================

# Previous seeds:
# SEEDS=(12 17 44 122 1254)

SEEDS=(
    4046
    3076
    9872
    6845
    2534
)


# ============================================================
# Table I:
# Exactly the five attack/settings shown in the manuscript
# ============================================================

ATTACKS=(
    gaussian
    label_flip
    model_replacement
    none
    signflip
)


# ============================================================
# Table IV:
#
# ASB is intentionally omitted here because ASB + signflip
# is already one of the attack-table jobs above and should
# be reused as the ASB row of the baseline table.
# ============================================================

BASELINES=(
    aflguard
    fedasync
    fedbuff
    krum
    mean
    median
    trimmed_mean
    zeno
)


# ============================================================
# Number of experiments
# ============================================================

N_SEEDS=${#SEEDS[@]}
N_ATTACKS=${#ATTACKS[@]}
N_BASELINES=${#BASELINES[@]}

ATTACK_TASKS=$(( N_SEEDS * N_ATTACKS ))
BASELINE_TASKS=$(( N_SEEDS * N_BASELINES ))
TOTAL_TASKS=$(( ATTACK_TASKS + BASELINE_TASKS ))

# With the current configuration:
#
#   N_SEEDS        = 5
#   N_ATTACKS      = 5
#   N_BASELINES    = 8
#
#   ATTACK_TASKS   = 25
#   BASELINE_TASKS = 40
#   TOTAL_TASKS    = 65


# ============================================================
# Current Slurm array task
# ============================================================

TASK_ID=${SLURM_ARRAY_TASK_ID}


# ============================================================
# Safety check
# ============================================================

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "ERROR:"
    echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
    echo "is outside the valid range:"
    echo "0 .. $((TOTAL_TASKS - 1))"
    exit 2
fi


# ============================================================
# Map one Slurm array task to exactly one experiment
# ============================================================

if (( TASK_ID < ATTACK_TASKS )); then

    # --------------------------------------------------------
    # ATTACK TABLE
    #
    # Tasks:
    #
    #       0 ... 24
    #
    # Mapping:
    #
    #       5 seeds x 5 attack/settings
    #
    # --------------------------------------------------------

    SEED_IDX=$(( TASK_ID / N_ATTACKS ))
    CASE_IDX=$(( TASK_ID % N_ATTACKS ))

    SEED=${SEEDS[$SEED_IDX]}
    ATTACK=${ATTACKS[$CASE_IDX]}

    AGGREGATOR="asb"
    GROUP="attacks"

    RUN_BASE="${RESULT_ROOT}/attacks/asb/${ATTACK}/seed_${SEED}"

    BYZ_ARGS=()

    # "none" means no malicious Byzantine corruption.
    if [[ "${ATTACK}" != "none" ]]; then
        BYZ_ARGS+=(--enable_byzantine)
    fi

else

    # --------------------------------------------------------
    # BASELINE TABLE
    #
    # Tasks:
    #
    #       25 ... 64
    #
    # Mapping:
    #
    #       5 seeds x 8 baseline aggregators
    #
    # All baseline experiments use sign-flip.
    #
    # --------------------------------------------------------

    BIDX=$(( TASK_ID - ATTACK_TASKS ))

    SEED_IDX=$(( BIDX / N_BASELINES ))
    CASE_IDX=$(( BIDX % N_BASELINES ))

    SEED=${SEEDS[$SEED_IDX]}
    AGGREGATOR=${BASELINES[$CASE_IDX]}

    ATTACK="signflip"
    GROUP="baselines"

    RUN_BASE="${RESULT_ROOT}/baselines/signflip/${AGGREGATOR}/seed_${SEED}"

    BYZ_ARGS=(--enable_byzantine)
fi


# ============================================================
# Output directory
# ============================================================

mkdir -p "${RUN_BASE}"


# ============================================================
# Common experiment arguments
#
# Keep all principal hyperparameters explicit so that all seeds,
# attacks, and baselines are evaluated under exactly the same
# protocol.
# ============================================================

COMMON_ARGS=(

    --suite single

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    --seed "${SEED}"

    # --------------------------------------------------------
    # Federated-learning configuration
    # --------------------------------------------------------

    --num_clients 10
    --num_clients_per_round 5
    --max_parallel_clients "${MAX_PARALLEL_CLIENTS}"

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    --num_rounds 400
    --local_epochs 10
    --batch_size 128

    # --------------------------------------------------------
    # Non-IID partition
    # --------------------------------------------------------

    --alpha_dirichlet 0.5

    # --------------------------------------------------------
    # Client optimization / server schedule
    # --------------------------------------------------------

    --eta_client 1e-3
    --zeta0 1e-2
    --L_smooth 1.0

    # --------------------------------------------------------
    # Staleness / event triggering
    # --------------------------------------------------------

    --alpha_stale 0.2
    --beta_et 0.5
    --tau_max_rounds 3
    --trigger_eps 0.0

    # --------------------------------------------------------
    # Staleness-weight bounds
    # --------------------------------------------------------

    --w_min 0.5
    --w_max 1.5

    # --------------------------------------------------------
    # Byzantine configuration
    # --------------------------------------------------------

    --byz_frac 0.3
    --byz_scale 10.0

    # --------------------------------------------------------
    # FedBuff configuration
    # --------------------------------------------------------

    --fedbuff_size 10

    # --------------------------------------------------------
    # Optimization/runtime
    # --------------------------------------------------------

    --accumulation_steps 1
    --early_stopping_patience 10

    # --------------------------------------------------------
    # Experiment-specific aggregation / attack
    # --------------------------------------------------------

    --aggregator "${AGGREGATOR}"
    --attack "${ATTACK}"

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    --base_dir "${RUN_BASE}"
)


# ============================================================
# ASB-specific arguments
#
# ASB trimming is already True by default in the current Python
# parser, but pass the flag explicitly for ASB experiments.
# ============================================================

ASB_ARGS=()

if [[ "${AGGREGATOR}" == "asb" ]]; then
    ASB_ARGS+=(--asb_trim)
fi


# ============================================================
# Construct command
# ============================================================

CMD=(
    python -u "${PY_SCRIPT}"
    "${COMMON_ARGS[@]}"
    "${BYZ_ARGS[@]}"
    "${ASB_ARGS[@]}"
)


# ============================================================
# Print experiment configuration
# ============================================================

echo "============================================================"
echo "RAFL CIFAR-10 five-seed reviewer experiment"
echo "============================================================"

echo "Slurm job ID       : ${SLURM_JOB_ID}"
echo "Array task ID      : ${TASK_ID} / $((TOTAL_TASKS - 1))"
echo "Node               : ${SLURM_NODELIST}"
echo "Partition          : s.geany"

echo
echo "Group              : ${GROUP}"
echo "Seed               : ${SEED}"
echo "Aggregator         : ${AGGREGATOR}"
echo "Attack             : ${ATTACK}"

echo
echo "Configured byz_frac: 0.3"
echo "Selected/round J   : 5"

echo
echo "Execution device   : CPU"
echo "CPU cores/task     : ${SLURM_CPUS_PER_TASK}"
echo "Parallel clients   : ${MAX_PARALLEL_CLIENTS}"
echo "OMP threads/client : ${OMP_NUM_THREADS}"

echo
echo "Python script      : ${PY_SCRIPT}"
echo "Results base       : ${RUN_BASE}"

echo
echo "Command:"
printf ' %q' "${CMD[@]}"
echo

echo "============================================================"


# ============================================================
# Run experiment
# ============================================================

srun --cpu-bind=cores "${CMD[@]}"


# ============================================================
# Completion message
# ============================================================

echo
echo "============================================================"
echo "[DONE] CIFAR-10 RAFL reviewer experiment"
echo "============================================================"

echo "Task ID     : ${TASK_ID}"
echo "Group       : ${GROUP}"
echo "Seed        : ${SEED}"
echo "Aggregator  : ${AGGREGATOR}"
echo "Attack      : ${ATTACK}"
echo "Results     : ${RUN_BASE}"
echo "Time        : $(date)"

echo "============================================================"
