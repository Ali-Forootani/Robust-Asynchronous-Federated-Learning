#!/bin/bash -l
#SBATCH -J cifar_RAFL_C6_5seed
#SBATCH -p s.geany
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=20G
#SBATCH --time=168:00:00

# ============================================================
# PARALLEL EXECUTION
#
# Current seed list:
#   SEEDS=(1494)
#
# For each seed:
#
#   5 adaptive attacks
#   +
#   6 aggregators x 5 Byzantine counts = 30
#
#   Total = 35 independent experiments.
#
# Therefore, for ONE seed:
#   array = 0 ... 34
#
# %35 allows all 35 jobs to run simultaneously if Slurm
# resources/QoS permit.
# ============================================================

#SBATCH --array=0-34%35

#SBATCH -D /u/alfo/RAFL_revision
#SBATCH -o /u/alfo/RAFL_revision/logs/%x-%A_%a.out
#SBATCH -e /u/alfo/RAFL_revision/logs/%x-%A_%a.err

set -euo pipefail


# ============================================================
# RAFL CIFAR-10
# Reviewer #2 - Comment 6
#
# PARALLEL VERSION
#
# Each Slurm-array task executes exactly ONE experiment.
#
# Per seed:
#
# PART A:
#   0  norm_evasion
#   1  staleness_exploit
#   2  collusion
#   3  alie
#   4  adaptive_joint
#
# PART B:
#   5 ... 34
#
#   6 aggregators:
#       median
#       trimmed_mean
#       krum
#       aflguard
#       zeno
#       asb
#
#   x
#
#   5 Byzantine counts:
#       B = 0,1,2,3,4
#
# Thus:
#       5 + 6*5 = 35 experiments per seed.
# ============================================================


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

module purge
module load gcc/14 || true

source /u/alfo/dnn_env/bin/activate

# CPU-only execution
export CUDA_VISIBLE_DEVICES=""


# ------------------------------------------------------------
# CPU threading
#
# The Python program may train up to 5 clients concurrently.
# Do NOT give all 64 OpenMP threads to every client, otherwise
# one experiment may create approximately 5 x 64 threads.
#
# Split the allocated CPUs approximately across the five clients.
# ------------------------------------------------------------

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

export PYTHONUNBUFFERED=1


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

WORKDIR="/u/alfo/RAFL_revision"

PY_SCRIPT="${WORKDIR}/rafl_cifar10_experiment_suite_comment6.py"

LOGDIR="${WORKDIR}/logs"

RESULT_ROOT="${WORKDIR}/results_comment6_cifar10"

mkdir -p "${LOGDIR}"
mkdir -p "${RESULT_ROOT}"


# ------------------------------------------------------------
# Seed list
#
# Kept EXACTLY as in your current script.
# ------------------------------------------------------------

SEEDS=(5027)


# ------------------------------------------------------------
# Experiment definitions
# ------------------------------------------------------------

ADAPTIVE_ATTACKS=(
    norm_evasion
    staleness_exploit
    collusion
    alie
    adaptive_joint
)

FAILURE_AGGREGATORS=(
    median
    trimmed_mean
    krum
    aflguard
    zeno
    asb
)

FAILURE_BYZ_COUNTS=(
    0
    1
    2
    3
    4
)


N_ADAPTIVE=${#ADAPTIVE_ATTACKS[@]}
N_AGGREGATORS=${#FAILURE_AGGREGATORS[@]}
N_BYZ_COUNTS=${#FAILURE_BYZ_COUNTS[@]}

N_BOUNDARY=$(( N_AGGREGATORS * N_BYZ_COUNTS ))

EXPERIMENTS_PER_SEED=$(( N_ADAPTIVE + N_BOUNDARY ))

N_SEEDS=${#SEEDS[@]}

TOTAL_TASKS=$(( N_SEEDS * EXPERIMENTS_PER_SEED ))


# ------------------------------------------------------------
# Map Slurm array ID -> seed + experiment
# ------------------------------------------------------------

TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "ERROR:"
    echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
    echo "but valid range is:"
    echo "0 .. $((TOTAL_TASKS - 1))"
    exit 1
fi


SEED_IDX=$(( TASK_ID / EXPERIMENTS_PER_SEED ))

EXPERIMENT_IDX=$(( TASK_ID % EXPERIMENTS_PER_SEED ))

SEED=${SEEDS[$SEED_IDX]}


# ------------------------------------------------------------
# Seed-specific output directories
# ------------------------------------------------------------

SEED_DIR="${RESULT_ROOT}/seed_${SEED}"

ADAPTIVE_DIR="${SEED_DIR}/adaptive_attacks"

BOUNDARY_DIR="${SEED_DIR}/failure_boundary"

mkdir -p "${ADAPTIVE_DIR}"
mkdir -p "${BOUNDARY_DIR}"


# ------------------------------------------------------------
# Common CIFAR-10 configuration
#
# EXACTLY the same experimental parameters as your original
# Comment 6 script.
# ------------------------------------------------------------

COMMON_ARGS=(
    --seed "${SEED}"

    --num_clients 10
    --num_clients_per_round 5
    --max_parallel_clients "${MAX_PARALLEL_CLIENTS}"
    --alpha_dirichlet 0.5

    --num_rounds 200
    --local_epochs 10
    --batch_size 64

    --eta_client 5e-3
    --zeta0 1e-2
    --L_smooth 1.0

    --alpha_stale 0.2
    --beta_et 0.5
    --tau_max_rounds 3
    --trigger_eps 0.0

    --w_min 0.5
    --w_max 1.5

    --accumulation_steps 1
    --early_stopping_patience 10
)


# ============================================================
# PART A
#
# Array IDs 0 ... 4 for each seed
#
# ONE adaptive attack per Slurm task
# ============================================================

if (( EXPERIMENT_IDX < N_ADAPTIVE )); then

    ATTACK_IDX=${EXPERIMENT_IDX}

    ATTACK=${ADAPTIVE_ATTACKS[$ATTACK_IDX]}

    # Separate directory is necessary because these jobs now
    # execute simultaneously. The original root path remains
    # unchanged.
    RUN_DIR="${ADAPTIVE_DIR}/${ATTACK}"

    mkdir -p "${RUN_DIR}"


    echo "============================================================"
    echo "RAFL CIFAR-10 - Reviewer #2 Comment 6"
    echo "PARALLEL ADAPTIVE-ATTACK EXPERIMENT"
    echo "============================================================"
    echo "Slurm job ID       : ${SLURM_JOB_ID}"
    echo "Array task ID      : ${TASK_ID}"
    echo "Experiment index   : ${EXPERIMENT_IDX}"
    echo "Node               : ${SLURM_NODELIST}"
    echo "CPUs per task      : ${SLURM_CPUS_PER_TASK}"
    echo "OMP threads/client : ${OMP_NUM_THREADS}"
    echo "Seed               : ${SEED}"
    echo "Suite              : adaptive_attacks"
    echo "Aggregator         : asb"
    echo "Attack             : ${ATTACK}"
    echo "Byzantine count    : 2 / 5"
    echo "Byzantine fraction : 0.4"
    echo "Python script      : ${PY_SCRIPT}"
    echo "Results directory  : ${RUN_DIR}"
    echo "============================================================"


    srun --cpu-bind=cores \
        python -u "${PY_SCRIPT}" \
        --suite adaptive_attacks \
        --aggregator asb \
        --enable_byzantine \
        --byz_frac 0.4 \
        --byz_count_override 2 \
        --asb_trim \
        --adaptive_attack_list "${ATTACK}" \
        --adaptive_norm_quantile 0.50 \
        --adaptive_norm_ratio 0.95 \
        --adaptive_z 1.5 \
        --byz_scale 10.0 \
        --base_dir "${RUN_DIR}" \
        "${COMMON_ARGS[@]}"


    echo
    echo "============================================================"
    echo "ADAPTIVE ATTACK COMPLETED"
    echo "============================================================"
    echo "Seed   : ${SEED}"
    echo "Attack : ${ATTACK}"
    echo "Output : ${RUN_DIR}"
    echo "Time   : $(date)"
    echo "============================================================"

    exit 0
fi


# ============================================================
# PART B
#
# Remaining 30 array jobs:
#
#       6 aggregators x 5 Byzantine counts
#
# ONE (aggregator, B) pair per Slurm task
# ============================================================

BOUNDARY_IDX=$(( EXPERIMENT_IDX - N_ADAPTIVE ))

AGG_IDX=$(( BOUNDARY_IDX / N_BYZ_COUNTS ))

BYZ_IDX=$(( BOUNDARY_IDX % N_BYZ_COUNTS ))

AGGREGATOR=${FAILURE_AGGREGATORS[$AGG_IDX]}

BYZ_COUNT=${FAILURE_BYZ_COUNTS[$BYZ_IDX]}


# ------------------------------------------------------------
# Corresponding Byzantine fraction, since J = 5.
#
# B=0 -> q=0.0
# B=1 -> q=0.2
# B=2 -> q=0.4
# B=3 -> q=0.6
# B=4 -> q=0.8
# ------------------------------------------------------------

case "${BYZ_COUNT}" in
    0)
        BYZ_FRAC="0.0"
        ;;
    1)
        BYZ_FRAC="0.2"
        ;;
    2)
        BYZ_FRAC="0.4"
        ;;
    3)
        BYZ_FRAC="0.6"
        ;;
    4)
        BYZ_FRAC="0.8"
        ;;
    *)
        echo "ERROR: Unsupported Byzantine count ${BYZ_COUNT}"
        exit 2
        ;;
esac


# ------------------------------------------------------------
# Separate directories prevent simultaneous failure-boundary
# processes from overwriting their CSV/JSON/PNG summary files.
#
# Root remains:
#
# results_comment6_cifar10/
#     seed_1494/
#         failure_boundary/
# ------------------------------------------------------------

RUN_DIR="${BOUNDARY_DIR}/${AGGREGATOR}/B_${BYZ_COUNT}"

mkdir -p "${RUN_DIR}"


echo "============================================================"
echo "RAFL CIFAR-10 - Reviewer #2 Comment 6"
echo "PARALLEL FAILURE-BOUNDARY EXPERIMENT"
echo "============================================================"
echo "Slurm job ID       : ${SLURM_JOB_ID}"
echo "Array task ID      : ${TASK_ID}"
echo "Experiment index   : ${EXPERIMENT_IDX}"
echo "Boundary index     : ${BOUNDARY_IDX}"
echo "Node               : ${SLURM_NODELIST}"
echo "CPUs per task      : ${SLURM_CPUS_PER_TASK}"
echo "OMP threads/client : ${OMP_NUM_THREADS}"
echo "Seed               : ${SEED}"
echo "Suite              : failure_boundary"
echo "Attack             : adaptive_joint"
echo "Aggregator         : ${AGGREGATOR}"
echo "Byzantine count B  : ${BYZ_COUNT}"
echo "Byzantine fraction : ${BYZ_FRAC}"
echo "Participants J     : 5"
echo "Python script      : ${PY_SCRIPT}"
echo "Results directory  : ${RUN_DIR}"
echo "============================================================"


srun --cpu-bind=cores \
    python -u "${PY_SCRIPT}" \
    --suite failure_boundary \
    --failure_attack adaptive_joint \
    --failure_aggregators "${AGGREGATOR}" \
    --failure_byz_counts "${BYZ_COUNT}" \
    --failure_tail_rounds 20 \
    --failure_drop_threshold 0.10 \
    --failure_acc_floor 0.20 \
    --adaptive_norm_quantile 0.50 \
    --adaptive_norm_ratio 0.95 \
    --adaptive_z 1.5 \
    --byz_scale 10.0 \
    --asb_trim \
    --base_dir "${RUN_DIR}" \
    "${COMMON_ARGS[@]}"


echo
echo "============================================================"
echo "FAILURE-BOUNDARY EXPERIMENT COMPLETED"
echo "============================================================"
echo "Seed       : ${SEED}"
echo "Aggregator : ${AGGREGATOR}"
echo "B          : ${BYZ_COUNT}"
echo "q          : ${BYZ_FRAC}"
echo "Output     : ${RUN_DIR}"
echo "Time       : $(date)"
echo "============================================================"
