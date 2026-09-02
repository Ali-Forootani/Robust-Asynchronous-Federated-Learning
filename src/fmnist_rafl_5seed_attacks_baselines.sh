#!/bin/bash -l
#SBATCH -J fmnist_RAFL_5seed
#SBATCH -p p.geany
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --array=0-64%10

#SBATCH -D /u/alfo/RAFL_revision
#SBATCH -o /u/alfo/RAFL_revision/logs/%x-%A_%a.out
#SBATCH -e /u/alfo/RAFL_revision/logs/%x-%A_%a.err

set -euo pipefail

# ============================================================
# Fashion-MNIST RAFL: 5-seed reviewer experiments
#
# Fixed seeds:
#   12, 17, 44, 122, 1254
#
# Unique runs:
#   Attack robustness:
#       5 conditions x 5 seeds = 25 jobs
#
#   Baseline aggregation comparison under sign-flip:
#       8 additional baselines x 5 seeds = 40 jobs
#
#   ASB + sign-flip is already included in the attack runs
#   and is reused as the ASB row in the baseline comparison.
#
# Total unique jobs = 25 + 40 = 65
# Slurm array = 0-64
# ============================================================

module purge
module load gcc/14 || true
source /u/alfo/dnn_env/bin/activate

# CPU only
export CUDA_VISIBLE_DEVICES=""

# Python can train up to 5 clients concurrently.
# Avoid assigning all 64 OpenMP threads to each client.
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

mkdir -p /u/alfo/RAFL_revision/logs

# IMPORTANT:
# The uploaded Python code is Fashion-MNIST and its examples use
# rafl_fmnist_experiment_suite.py.
#
# If your actual filename on the cluster is instead
# rafl_mnist_experiment_suite.py, change ONLY this line.
PY_SCRIPT="/u/alfo/RAFL_revision/rafl_fmnist_experiment_suite.py"

RESULT_ROOT="/u/alfo/RAFL_revision/results_fmnist_5seed_revision"

# ------------------------------------------------------------
# Five independent seeds
# ------------------------------------------------------------
SEEDS=(
    12
    17
    44
    122
    1254
)

# ------------------------------------------------------------
# Attack robustness table
# Same attack settings as the existing attack suite
# ------------------------------------------------------------
ATTACKS=(
    gaussian
    label_flip
    model_replacement
    none
    signflip
)

# ------------------------------------------------------------
# Baseline aggregation table
#
# ASB is deliberately omitted here because ASB+signflip
# is already computed above in ATTACKS and can be reused.
# ------------------------------------------------------------
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

N_SEEDS=${#SEEDS[@]}
N_ATTACKS=${#ATTACKS[@]}
N_BASELINES=${#BASELINES[@]}

ATTACK_TASKS=$(( N_SEEDS * N_ATTACKS ))
BASELINE_TASKS=$(( N_SEEDS * N_BASELINES ))
TOTAL_TASKS=$(( ATTACK_TASKS + BASELINE_TASKS ))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}"
    echo "Expected task ID in [0,$((TOTAL_TASKS - 1))]"
    exit 2
fi

# ============================================================
# Decode the Slurm array index
# ============================================================

if (( TASK_ID < ATTACK_TASKS )); then

    # --------------------------------------------------------
    # Tasks 0..24:
    # 5 seeds x 5 attack conditions using ASB
    # --------------------------------------------------------
    SEED_IDX=$(( TASK_ID / N_ATTACKS ))
    CASE_IDX=$(( TASK_ID % N_ATTACKS ))

    SEED=${SEEDS[$SEED_IDX]}
    ATTACK=${ATTACKS[$CASE_IDX]}
    AGGREGATOR="asb"
    GROUP="attacks"

    RUN_BASE="${RESULT_ROOT}/attacks/asb/${ATTACK}/seed_${SEED}"

    BYZ_ARGS=()
    if [[ "${ATTACK}" != "none" ]]; then
        BYZ_ARGS+=(--enable_byzantine)
    fi

else

    # --------------------------------------------------------
    # Tasks 25..64:
    # 5 seeds x 8 additional baseline aggregators
    # all tested under sign-flip
    # --------------------------------------------------------
    BASE_TASK_ID=$(( TASK_ID - ATTACK_TASKS ))

    SEED_IDX=$(( BASE_TASK_ID / N_BASELINES ))
    CASE_IDX=$(( BASE_TASK_ID % N_BASELINES ))

    SEED=${SEEDS[$SEED_IDX]}
    AGGREGATOR=${BASELINES[$CASE_IDX]}
    ATTACK="signflip"
    GROUP="baselines"

    RUN_BASE="${RESULT_ROOT}/baselines/signflip/${AGGREGATOR}/seed_${SEED}"

    BYZ_ARGS=(--enable_byzantine)
fi

mkdir -p "${RUN_BASE}"

# ============================================================
# Common Fashion-MNIST settings
#
# These explicitly reproduce the current Python CLI defaults:
#   rounds               = 500
#   local epochs         = 3
#   clients              = 10
#   clients per round    = 5
#   batch size           = 128
#   eta_client           = 1e-2
#   zeta0                = 1e-2
#   alpha_stale          = 0.2
#   beta_et              = 0.5
#   tau_max_rounds       = 3
#   trigger_eps          = 0.0
#   Byzantine fraction   = 0.3
#   attack scale         = 10.0
# ============================================================

COMMON_ARGS=(
    --suite single
    --seed "${SEED}"

    --num_clients 10
    --alpha_dirichlet 0.5

    --num_rounds 500
    --local_epochs 3

    --num_clients_per_round 5
    --max_parallel_clients "${MAX_PARALLEL_CLIENTS}"

    --batch_size 128

    --eta_client 1e-2
    --zeta0 1e-2
    --L_smooth 1.0

    --alpha_stale 0.2
    --beta_et 0.5
    --tau_max_rounds 3

    --trigger_eps 0.0

    --w_min 0.5
    --w_max 1.5

    --byz_frac 0.3
    --byz_scale 10.0

    --fedbuff_size 10

    --accumulation_steps 1
    --early_stopping_patience 5

    --aggregator "${AGGREGATOR}"
    --attack "${ATTACK}"

    --base_dir "${RUN_BASE}"
)

# Explicitly request ASB trimming for ASB runs.
# In the current Python parser it is already True by default,
# but keeping this flag makes the Slurm configuration explicit.
ASB_ARGS=()

if [[ "${AGGREGATOR}" == "asb" ]]; then
    ASB_ARGS+=(--asb_trim)
fi

CMD=(
    python -u "${PY_SCRIPT}"
    "${COMMON_ARGS[@]}"
    "${BYZ_ARGS[@]}"
    "${ASB_ARGS[@]}"
)

# ============================================================
# Job information
# ============================================================

echo "============================================================"
echo "RAFL Fashion-MNIST five-seed reviewer experiment"
echo "============================================================"
echo "Slurm job ID        : ${SLURM_JOB_ID}"
echo "Array task ID       : ${TASK_ID} / $((TOTAL_TASKS - 1))"
echo "Node                : ${SLURM_NODELIST}"
echo
echo "Experiment group    : ${GROUP}"
echo "Seed                : ${SEED}"
echo "Aggregator          : ${AGGREGATOR}"
echo "Attack              : ${ATTACK}"
echo
echo "Number clients      : 10"
echo "Clients per round   : 5"
echo "Rounds              : 500"
echo "Local epochs        : 3"
echo "Byzantine fraction  : 0.3"
echo "Attack scale        : 10.0"
echo "Trigger epsilon     : 0.0"
echo
echo "CPU cores/task      : ${SLURM_CPUS_PER_TASK}"
echo "OMP threads         : ${OMP_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo
echo "Results directory   : ${RUN_BASE}"
echo
echo "Command:"
printf ' %q' "${CMD[@]}"
echo
echo "============================================================"

# ============================================================
# Run
# ============================================================

srun --cpu-bind=cores "${CMD[@]}"

echo
echo "============================================================"
echo "[DONE]"
echo "Task       : ${TASK_ID}"
echo "Group      : ${GROUP}"
echo "Seed       : ${SEED}"
echo "Aggregator : ${AGGREGATOR}"
echo "Attack     : ${ATTACK}"
echo "Results    : ${RUN_BASE}"
echo "============================================================"
