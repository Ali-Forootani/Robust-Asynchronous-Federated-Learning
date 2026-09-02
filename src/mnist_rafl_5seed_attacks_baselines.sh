#!/bin/bash -l
#SBATCH -J mnist_RAFL_5seed
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
# MNIST RAFL: 5-seed reviewer experiments
#
# Fixed seeds:
#   12, 17, 44, 122, 1254
#
# Unique experiments:
#
#   1) Attack robustness with ASB:
#      5 attack/settings x 5 seeds = 25 runs
#
#   2) Baseline comparison under sign-flip:
#      8 additional aggregators x 5 seeds = 40 runs
#
# ASB + sign-flip is already included in the attack experiments,
# so it is reused as the ASB row of the baseline table.
#
# Total unique simulations:
#   25 + 40 = 65
#
# Therefore:
#   #SBATCH --array=0-64
# ============================================================


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

module purge
module load gcc/14 || true
source /u/alfo/dnn_env/bin/activate

# Force CPU execution
export CUDA_VISIBLE_DEVICES=""

# The Python program can train up to 5 clients concurrently via
# asyncio.to_thread. Avoid assigning all 64 BLAS/OpenMP threads
# to every concurrent client.
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


# ------------------------------------------------------------
# Python program and result root
# ------------------------------------------------------------

PY_SCRIPT="/u/alfo/RAFL_revision/rafl_mnist_experiment_suite.py"

RESULT_ROOT="/u/alfo/RAFL_revision/results_mnist_5seed_revision"


# ============================================================
# Experiment definitions
# ============================================================

# Five independent random seeds
SEEDS=(
    12
    17
    44
    122
    1254
)


# ------------------------------------------------------------
# Attack robustness experiments
#
# These correspond to the five attack/settings in the
# Python "attacks" suite.
# ------------------------------------------------------------

ATTACKS=(
    gaussian
    label_flip
    model_replacement
    none
    signflip
)


# ------------------------------------------------------------
# Baseline aggregation experiments
#
# All are evaluated under sign-flip.
#
# ASB is intentionally excluded here because ASB + signflip
# is already computed in the ATTACKS block above.
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


# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} is invalid."
    echo "Expected task ID between 0 and $((TOTAL_TASKS - 1))."
    exit 2
fi


# ============================================================
# Decode the array index
# ============================================================

if (( TASK_ID < ATTACK_TASKS )); then

    # --------------------------------------------------------
    # Tasks 0..24
    #
    # 5 seeds x 5 attack/settings
    # Aggregator = ASB
    # --------------------------------------------------------

    SEED_IDX=$(( TASK_ID / N_ATTACKS ))
    CASE_IDX=$(( TASK_ID % N_ATTACKS ))

    SEED=${SEEDS[$SEED_IDX]}
    ATTACK=${ATTACKS[$CASE_IDX]}
    AGGREGATOR="asb"
    GROUP="attacks"

    RUN_BASE="${RESULT_ROOT}/attacks/asb/${ATTACK}/seed_${SEED}"

    BYZ_ARGS=()

    # Clean run: no Byzantine clients.
    # All other attack runs: Byzantine clients enabled.
    if [[ "${ATTACK}" != "none" ]]; then
        BYZ_ARGS+=(--enable_byzantine)
    fi

else

    # --------------------------------------------------------
    # Tasks 25..64
    #
    # 5 seeds x 8 baseline aggregators
    # Attack = sign-flip
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
# Common MNIST experimental settings
#
# These values explicitly match the current CLI defaults in
# rafl_mnist_experiment_suite.py.
# ============================================================

COMMON_ARGS=(

    --suite single

    --seed "${SEED}"

    # Federated system
    --num_clients 10
    --num_clients_per_round 5
    --max_parallel_clients "${MAX_PARALLEL_CLIENTS}"

    # Non-IID partition
    --alpha_dirichlet 0.5

    # Training
    --num_rounds 500
    --local_epochs 3
    --batch_size 128

    # Optimization
    --eta_client 1e-2
    --zeta0 1e-2
    --L_smooth 1.0

    # Staleness / event-trigger
    --alpha_stale 0.2
    --beta_et 0.5
    --tau_max_rounds 3
    --trigger_eps 0.0

    # Staleness-weight limits
    --w_min 0.5
    --w_max 1.5

    # Byzantine settings
    --byz_frac 0.3
    --byz_scale 10.0

    # FedBuff
    --fedbuff_size 10

    # Local optimization
    --accumulation_steps 1
    --early_stopping_patience 5

    # Experiment case
    --aggregator "${AGGREGATOR}"
    --attack "${ATTACK}"

    # Separate output directory for every Slurm task
    --base_dir "${RUN_BASE}"
)


# ------------------------------------------------------------
# ASB trimming
#
# The Python parser already defaults asb_trim=True, but passing
# --asb_trim explicitly makes the experimental configuration
# unambiguous.
# ------------------------------------------------------------

ASB_ARGS=()

if [[ "${AGGREGATOR}" == "asb" ]]; then
    ASB_ARGS+=(--asb_trim)
fi


# ============================================================
# Construct command as a Bash array
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
echo "RAFL MNIST five-seed reviewer experiment"
echo "============================================================"

echo "Slurm job ID         : ${SLURM_JOB_ID}"
echo "Array task ID        : ${TASK_ID} / $((TOTAL_TASKS - 1))"
echo "Node                 : ${SLURM_NODELIST}"

echo
echo "Experiment group     : ${GROUP}"
echo "Seed                 : ${SEED}"
echo "Aggregator           : ${AGGREGATOR}"
echo "Attack               : ${ATTACK}"

echo
echo "Number of clients    : 10"
echo "Clients per round    : 5"
echo "Rounds               : 500"
echo "Local epochs         : 3"
echo "Batch size           : 128"

echo
echo "Byzantine fraction   : 0.3"
echo "Attack scale         : 10.0"
echo "Trigger epsilon      : 0.0"
echo "Max staleness        : 3"

echo
echo "CPU cores/task       : ${SLURM_CPUS_PER_TASK}"
echo "Parallel clients     : ${MAX_PARALLEL_CLIENTS}"
echo "OMP threads/client   : ${OMP_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-unset}"

echo
echo "Results directory:"
echo "${RUN_BASE}"

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
echo "[DONE] MNIST experiment completed"
echo "============================================================"
echo "Task ID     : ${TASK_ID}"
echo "Group       : ${GROUP}"
echo "Seed        : ${SEED}"
echo "Aggregator  : ${AGGREGATOR}"
echo "Attack      : ${ATTACK}"
echo "Results     : ${RUN_BASE}"
echo "============================================================"
