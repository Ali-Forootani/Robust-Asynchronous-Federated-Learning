#!/bin/bash -l
#SBATCH -J cifar_RAFL_5seed
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
# CIFAR-10 RAFL reviewer multiseed experiments
#
# Five fixed seeds:
#   12, 17, 44, 122, 1254
#
# Unique experiments:
#   Attack table:
#       5 attacks/settings x 5 seeds = 25 jobs
#
#   Baseline aggregation table under sign-flip:
#       8 additional baselines x 5 seeds = 40 jobs
#
#   ASB + sign-flip is already computed in the attack table and
#   should be reused as the ASB row of the baseline table.
#
# Total unique jobs = 25 + 40 = 65, hence --array=0-64.
# ============================================================

module purge
module load gcc/14 || true
source /u/alfo/dnn_env/bin/activate

# Force CPU execution.
export CUDA_VISIBLE_DEVICES=""

# The Python code may train up to 5 clients concurrently using
# asyncio.to_thread. Giving every client all 64 OpenMP threads
# would oversubscribe the node, so split CPU threads approximately
# across the 5 concurrent clients.
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

PY_SCRIPT="/u/alfo/RAFL_revision/rafl_cifar10_experiment_suite.py"
RESULT_ROOT="/u/alfo/RAFL_revision/results_cifar10_5seed_revision"

# Fixed seed list requested for statistical reporting.
SEEDS=(12 17 44 122 1254)

# Table I: exactly the five attack/settings shown in the manuscript.
ATTACKS=(
    gaussian
    label_flip
    model_replacement
    none
    signflip
)

# Table IV: ASB is intentionally omitted here because ASB + signflip
# is already one of the attack-table jobs above and should be reused.
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

ATTACK_TASKS=$(( N_SEEDS * N_ATTACKS ))       # 25
BASELINE_TASKS=$(( N_SEEDS * N_BASELINES ))  # 40
TOTAL_TASKS=$(( ATTACK_TASKS + BASELINE_TASKS ))  # 65

TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} is outside [0,$((TOTAL_TASKS-1))]."
    exit 2
fi

# ------------------------------------------------------------
# Map one Slurm array task to exactly one experiment.
# ------------------------------------------------------------
if (( TASK_ID < ATTACK_TASKS )); then
    # 0..24 -> 5 seeds x 5 attack/settings
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
    # 25..64 -> 5 seeds x 8 additional baselines, all under sign-flip
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

mkdir -p "${RUN_BASE}"

# ------------------------------------------------------------
# Keep all principal hyperparameters explicit so that all seeds,
# attacks, and baselines are run under the same protocol.
# These values match the current Python defaults.
# ------------------------------------------------------------
COMMON_ARGS=(
    --suite single
    --seed "${SEED}"
    --num_clients 10
    --num_clients_per_round 5
    --max_parallel_clients "${MAX_PARALLEL_CLIENTS}"
    --num_rounds 400
    --local_epochs 10
    --batch_size 128
    --alpha_dirichlet 0.5
    --eta_client 1e-3
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
    --early_stopping_patience 10
    --aggregator "${AGGREGATOR}"
    --attack "${ATTACK}"
    --base_dir "${RUN_BASE}"
)

# ASB trimming is already True by default in the current Python parser,
# but pass the flag explicitly for ASB runs to make the configuration clear.
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

echo "============================================================"
echo "RAFL CIFAR-10 five-seed reviewer experiment"
echo "============================================================"
echo "Slurm job ID       : ${SLURM_JOB_ID}"
echo "Array task ID      : ${TASK_ID} / $((TOTAL_TASKS-1))"
echo "Node               : ${SLURM_NODELIST}"
echo "Group              : ${GROUP}"
echo "Seed               : ${SEED}"
echo "Aggregator         : ${AGGREGATOR}"
echo "Attack             : ${ATTACK}"
echo "Configured byz_frac: 0.3"
echo "Selected/round J   : 5"
echo "CPU cores/task     : ${SLURM_CPUS_PER_TASK}"
echo "OMP threads/client : ${OMP_NUM_THREADS}"
echo "Results base       : ${RUN_BASE}"
echo "Command            : ${CMD[*]}"
echo "============================================================"

srun --cpu-bind=cores "${CMD[@]}"

echo
echo "[DONE] task ${TASK_ID}: group=${GROUP}, seed=${SEED}, aggregator=${AGGREGATOR}, attack=${ATTACK}"
