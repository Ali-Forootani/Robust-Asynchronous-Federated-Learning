#!/bin/bash -l
#SBATCH -J fmnist_R3C3_step
#SBATCH -p s.geany
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=20G
#SBATCH --time=168:00:00
#SBATCH --array=0-19%20

#SBATCH -D /u/alfo/RAFL_revision
#SBATCH -o /u/alfo/RAFL_revision/logs/%x-%A_%a.out
#SBATCH -e /u/alfo/RAFL_revision/logs/%x-%A_%a.err

set -euo pipefail

# ============================================================
# Fashion-MNIST RAFL
# Reviewer #3, Comment 3
# Dedicated server step-size ablation
#
# Four variants:
#
#   1) fixed
#   2) staleness_only
#   3) trigger_only
#   4) full
#
# Five independent seeds:
#
#   12, 17, 44, 122, 1254
#
# Total experiments:
#
#   4 variants x 5 seeds = 20
#
# IMPORTANT:
#   Event-trigger threshold epsilon = 170
#
# Table XX shows that epsilon = 170 gives:
#
#   Accuracy                = 0.9178
#   Loss                    = 0.2373
#   Communication reduction = 0.0512
#
# Thus the trigger is ACTIVE, while predictive performance
# remains strong.
#
# The ONLY quantity varied between the four experiments for
# a given seed is the server step-size rule.
# ============================================================


# ============================================================
# Environment
# ============================================================

module purge
module load gcc/14 || true

source /u/alfo/dnn_env/bin/activate


# ============================================================
# CPU-only execution
# ============================================================

export CUDA_VISIBLE_DEVICES=""


# ============================================================
# Parallel client execution
#
# Five clients participate per round, so allow all five client
# computations to execute concurrently.
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
export PYTHONUNBUFFERED=1


# ============================================================
# Logs
# ============================================================

mkdir -p /u/alfo/RAFL_revision/logs


# ============================================================
# Python experiment script
# ============================================================

PY_SCRIPT="/u/alfo/RAFL_revision/rafl_fmnist_comment3_step_ablation.py"


# ============================================================
# Scratch results
#
# IMPORTANT:
# Keep this rerun separate from the previous epsilon=1 run.
# ============================================================

SCRATCH_ROOT="/geany/fs/scratch/alfo"

RESULT_ROOT="${SCRATCH_ROOT}/RAFL_revision_reviewer3_comment3_fmnist_step_ablation_5seed_eps170"

mkdir -p "${RESULT_ROOT}"


# ============================================================
# Five independent random seeds
# ============================================================

SEEDS=(
    35
    36
    356
    3546
    3395
)


# ============================================================
# Four server step-size variants requested by Reviewer #3
# ============================================================

STEP_MODES=(
    fixed
    staleness_only
    trigger_only
    full
)


# ============================================================
# Number of experiments
# ============================================================

N_SEEDS=${#SEEDS[@]}
N_MODES=${#STEP_MODES[@]}

TOTAL_TASKS=$(( N_SEEDS * N_MODES ))

TASK_ID=${SLURM_ARRAY_TASK_ID}


# ============================================================
# Safety check
# ============================================================

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then

    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}"
    echo "Expected task ID in [0,$((TOTAL_TASKS - 1))]"

    exit 2

fi


# ============================================================
# Decode Slurm-array index
#
# Mapping:
#
# Task  0 : seed 12   fixed
# Task  1 : seed 12   staleness_only
# Task  2 : seed 12   trigger_only
# Task  3 : seed 12   full
#
# Task  4 : seed 17   fixed
# Task  5 : seed 17   staleness_only
# Task  6 : seed 17   trigger_only
# Task  7 : seed 17   full
#
# ...
#
# Task 16 : seed 1254 fixed
# Task 17 : seed 1254 staleness_only
# Task 18 : seed 1254 trigger_only
# Task 19 : seed 1254 full
# ============================================================

SEED_IDX=$(( TASK_ID / N_MODES ))
MODE_IDX=$(( TASK_ID % N_MODES ))

SEED=${SEEDS[$SEED_IDX]}
STEP_MODE=${STEP_MODES[$MODE_IDX]}


# ============================================================
# Event-trigger threshold
#
# IMPORTANT:
#
# Fashion-MNIST sensitivity analysis indicates that epsilon=170
# gives meaningful event-trigger activity without excessively
# degrading predictive performance.
#
# This value is used IDENTICALLY for all four variants.
#
# It may still be overridden during submission:
#
# sbatch --export=ALL,TRIGGER_EPS=180.0 \
#        fmnist_comment3_step_ablation_5seed.sh
#
# but the default reviewer experiment is epsilon = 170.
# ============================================================

TRIGGER_EPS="${TRIGGER_EPS:-170.0}"


# ============================================================
# Validate epsilon
# ============================================================

python - "${TRIGGER_EPS}" <<'PY'

import sys

eps = float(sys.argv[1])

if eps <= 0.0:
    raise SystemExit(
        "ERROR: TRIGGER_EPS must be > 0 for "
        "Reviewer #3 Comment 3."
    )

print(f"[CHECK] Using event-trigger threshold epsilon = {eps}")

PY


# ============================================================
# Per-run result directory
#
# Python creates its timestamped experiment folder underneath
# this deterministic directory.
# ============================================================

RUN_BASE="${RESULT_ROOT}/${STEP_MODE}/seed_${SEED}"

mkdir -p "${RUN_BASE}"


# ============================================================
# Common Fashion-MNIST configuration
#
# ALL quantities below are identical across the four variants,
# except:
#
#       --server_step_mode
#
# Therefore, this experiment directly isolates the contribution
# of the SERVER STEP-SIZE RULE.
#
# Configuration:
#
#   Dataset              : Fashion-MNIST
#   Clients              : 10
#   Clients/round        : 5
#   Rounds               : 500
#   Local epochs         : 3
#   Batch size           : 128
#
#   Client optimizer     : Adam
#   eta_client           : 1e-2
#   zeta0                : 1e-2
#
#   alpha_stale          : 0.2
#   beta_et              : 0.5
#   tau_max_rounds       : 3
#
#   Event trigger eps    : 170
#
#   Byzantine fraction   : 0.3
#   Attack               : signflip
#   Attack scale         : 10
#
#   Aggregator           : ASB
#   ASB trimming         : enabled
# ============================================================

COMMON_ARGS=(

    --suite single

    --seed "${SEED}"

    # --------------------------------------------------------
    # Client/data configuration
    # --------------------------------------------------------

    --num_clients 10

    --alpha_dirichlet 0.5


    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    --num_rounds 500

    --local_epochs 3

    --num_clients_per_round 5

    --max_parallel_clients "${MAX_PARALLEL_CLIENTS}"

    --batch_size 128


    # --------------------------------------------------------
    # Client/server optimization
    # --------------------------------------------------------

    --eta_client 1e-2

    --zeta0 1e-2

    --L_smooth 1.0


    # --------------------------------------------------------
    # Staleness / adaptive server step size
    # --------------------------------------------------------

    --alpha_stale 0.2

    --beta_et 0.5

    --server_step_mode "${STEP_MODE}"

    --tau_max_rounds 3


    # --------------------------------------------------------
    # Event-trigger threshold
    # --------------------------------------------------------

    --trigger_eps "${TRIGGER_EPS}"


    # --------------------------------------------------------
    # Staleness-aware weight bounds
    # --------------------------------------------------------

    --w_min 0.5

    --w_max 1.5


    # --------------------------------------------------------
    # Byzantine configuration
    # --------------------------------------------------------

    --enable_byzantine

    --byz_frac 0.3

    --attack signflip

    --byz_scale 10.0


    # --------------------------------------------------------
    # Robust aggregation
    # --------------------------------------------------------

    --aggregator asb

    --asb_trim


    # --------------------------------------------------------
    # Other controlled parameters
    # --------------------------------------------------------

    --fedbuff_size 10

    --accumulation_steps 1

    --early_stopping_patience 5


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    --base_dir "${RUN_BASE}"
)


# ============================================================
# Complete Python command
# ============================================================

CMD=(

    python -u "${PY_SCRIPT}"

    "${COMMON_ARGS[@]}"

)


# ============================================================
# Print experiment information
# ============================================================

echo
echo "============================================================"
echo "RAFL Fashion-MNIST"
echo "Reviewer #3 - Comment 3"
echo "Server Step-Size Ablation"
echo "============================================================"
echo
echo "Slurm job ID         : ${SLURM_JOB_ID}"
echo "Array task ID        : ${TASK_ID}"
echo "Total tasks          : ${TOTAL_TASKS}"
echo "Node                 : ${SLURM_NODELIST}"
echo
echo "------------------------------------------------------------"
echo "EXPERIMENT"
echo "------------------------------------------------------------"
echo
echo "Seed                 : ${SEED}"
echo "Server step mode     : ${STEP_MODE}"
echo
echo "------------------------------------------------------------"
echo "MODEL / TRAINING"
echo "------------------------------------------------------------"
echo
echo "Number clients       : 10"
echo "Clients per round    : 5"
echo "Rounds               : 500"
echo "Local epochs         : 3"
echo "Batch size           : 128"
echo
echo "Client optimizer     : Adam"
echo "eta_client           : 1e-2"
echo "zeta0                : 1e-2"
echo
echo "------------------------------------------------------------"
echo "ASYNCHRONY / EVENT TRIGGER"
echo "------------------------------------------------------------"
echo
echo "alpha_stale          : 0.2"
echo "beta_et              : 0.5"
echo "tau_max_rounds       : 3"
echo "Trigger epsilon      : ${TRIGGER_EPS}"
echo
echo "------------------------------------------------------------"
echo "BYZANTINE / AGGREGATION"
echo "------------------------------------------------------------"
echo
echo "Aggregator           : ASB"
echo "ASB trimming         : enabled"
echo "Attack               : signflip"
echo "Byzantine fraction   : 0.3"
echo "Attack scale         : 10.0"
echo
echo "------------------------------------------------------------"
echo "COMPUTE"
echo "------------------------------------------------------------"
echo
echo "CPU cores/task       : ${SLURM_CPUS_PER_TASK}"
echo "Parallel clients     : ${MAX_PARALLEL_CLIENTS}"
echo "Threads/client       : ${THREADS_PER_CLIENT}"
echo "OMP_NUM_THREADS      : ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS      : ${MKL_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-unset}"
echo
echo "------------------------------------------------------------"
echo "OUTPUT"
echo "------------------------------------------------------------"
echo
echo "Scratch root:"
echo "${SCRATCH_ROOT}"
echo
echo "Experiment root:"
echo "${RESULT_ROOT}"
echo
echo "Run base:"
echo "${RUN_BASE}"
echo
echo "------------------------------------------------------------"
echo "COMMAND"
echo "------------------------------------------------------------"
echo

printf ' %q' "${CMD[@]}"

echo
echo
echo "============================================================"


# ============================================================
# Execute experiment
# ============================================================

srun --cpu-bind=cores "${CMD[@]}"


# ============================================================
# Finished
# ============================================================

echo
echo "============================================================"
echo "[DONE] Reviewer #3 Comment 3 ablation"
echo "============================================================"
echo
echo "Task             : ${TASK_ID}"
echo "Seed             : ${SEED}"
echo "Server step mode : ${STEP_MODE}"
echo "Trigger epsilon  : ${TRIGGER_EPS}"
echo
echo "Results:"
echo "${RUN_BASE}"
echo
echo "============================================================"
