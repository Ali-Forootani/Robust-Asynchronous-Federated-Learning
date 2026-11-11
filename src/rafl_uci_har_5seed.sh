#!/bin/bash -l
#SBATCH -J har_RAFL_5seed
#SBATCH --partition=s.geany
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=168:00:00
#SBATCH --array=0-25%26
#SBATCH -D /u/alfo/RAFL_revision
#SBATCH -o /u/alfo/RAFL_revision/logs/%x-%A_%a.out
#SBATCH -e /u/alfo/RAFL_revision/logs/%x-%A_%a.err

set -euo pipefail

# UCI HAR inertial-sensor classification.  Same 65 unique simulations as MNIST:
# 5 attack settings x 5 seeds (ASB) + 8 baseline methods x 5 seeds (signflip).
# The ASB/signflip case is reused from the attack block.
# The optimization and attack parameters remain the same as the MNIST protocol.

module purge
module load gcc/14 || true
source /u/alfo/dnn_env/bin/activate
export CUDA_VISIBLE_DEVICES=""

MAX_PARALLEL_CLIENTS=5
THREADS_PER_CLIENT=$(( SLURM_CPUS_PER_TASK / MAX_PARALLEL_CLIENTS ))
if (( THREADS_PER_CLIENT < 1 )); then THREADS_PER_CLIENT=1; fi
export OMP_NUM_THREADS=${THREADS_PER_CLIENT}
export MKL_NUM_THREADS=${THREADS_PER_CLIENT}
export OPENBLAS_NUM_THREADS=${THREADS_PER_CLIENT}
export NUMEXPR_NUM_THREADS=${THREADS_PER_CLIENT}
export VECLIB_MAXIMUM_THREADS=${THREADS_PER_CLIENT}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE
export PYTHONUNBUFFERED=1

PY_SCRIPT="/u/alfo/RAFL_revision/rafl_uci_har_experiment_suite.py"
DATA_ROOT="/u/alfo/RAFL_revision/data/uci_har"
RESULT_ROOT="/u/alfo/RAFL_revision/results_uci_har_5seed_revision"

# Important: stage the official UCI HAR files in DATA_ROOT before sbatch.
if [[ ! -f "${DATA_ROOT}/UCI HAR Dataset/train/X_train.txt" && \
      ! -f "${DATA_ROOT}/train/X_train.txt" ]]; then
    echo "ERROR: UCI HAR dataset is missing. Extract UCI HAR Dataset.zip at ${DATA_ROOT} before submitting." >&2
    exit 2
fi

SEEDS=(12 17 44 122 1254)
ATTACKS=(gaussian label_flip model_replacement none signflip)
BASELINES=(aflguard fedasync fedbuff krum mean median trimmed_mean zeno)

N_SEEDS=${#SEEDS[@]}
N_ATTACKS=${#ATTACKS[@]}
N_BASELINES=${#BASELINES[@]}
ATTACK_TASKS=$((N_SEEDS * N_ATTACKS))
TOTAL_TASKS=$((ATTACK_TASKS + N_SEEDS * N_BASELINES))
TASK_ID=${SLURM_ARRAY_TASK_ID}

if ((TASK_ID < 0 || TASK_ID >= TOTAL_TASKS)); then
  echo "Invalid array index: ${TASK_ID}" >&2
  exit 2
fi

if ((TASK_ID < ATTACK_TASKS)); then
  SEED=${SEEDS[$((TASK_ID / N_ATTACKS))]}
  ATTACK=${ATTACKS[$((TASK_ID % N_ATTACKS))]}
  AGGREGATOR=asb
  GROUP=attacks
  RUN_BASE="${RESULT_ROOT}/attacks/asb/${ATTACK}/seed_${SEED}"
  BYZ_ARGS=()
  if [[ "$ATTACK" != none ]]; then BYZ_ARGS+=(--enable_byzantine); fi
else
  B=$((TASK_ID - ATTACK_TASKS))
  SEED=${SEEDS[$((B / N_BASELINES))]}
  AGGREGATOR=${BASELINES[$((B % N_BASELINES))]}
  ATTACK=signflip
  GROUP=baselines
  RUN_BASE="${RESULT_ROOT}/baselines/signflip/${AGGREGATOR}/seed_${SEED}"
  BYZ_ARGS=(--enable_byzantine)
fi

mkdir -p /u/alfo/RAFL_revision/logs "${RUN_BASE}"

COMMON_ARGS=(
  --suite single
  --seed "${SEED}"
  --data_root "${DATA_ROOT}"
  --partition_mode dirichlet
  --num_clients 10
  --num_clients_per_round 5
  --max_parallel_clients "${MAX_PARALLEL_CLIENTS}"
  --alpha_dirichlet 0.5
  --num_rounds 500
  --local_epochs 3
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
ASB_ARGS=()
if [[ "${AGGREGATOR}" == asb ]]; then ASB_ARGS+=(--asb_trim); fi
CMD=(python -u "${PY_SCRIPT}" "${COMMON_ARGS[@]}" "${BYZ_ARGS[@]}" "${ASB_ARGS[@]}")

echo "============================================================"
echo "UCI HAR RAFL five-seed experiment"
echo "Job ${SLURM_JOB_ID}, task ${TASK_ID}/${TOTAL_TASKS}, node ${SLURM_NODELIST}"
echo "Group ${GROUP}, seed ${SEED}, method ${AGGREGATOR}, attack ${ATTACK}"
echo "Dataset ${DATA_ROOT}, partition Dirichlet(alpha=0.5)"
echo "Clients 10, selected 5, rounds 500, local epochs 3, batch 128"
echo "Byzantine fraction 0.3, attack strength 10, trigger 0, maximum delay 3"
echo "Cores ${SLURM_CPUS_PER_TASK}, parallel clients ${MAX_PARALLEL_CLIENTS}, OMP ${OMP_NUM_THREADS}"
echo "Results ${RUN_BASE}"
printf ' %q' "${CMD[@]}"; echo

srun --cpu-bind=cores "${CMD[@]}"
echo "[DONE] ${GROUP} ${AGGREGATOR}/${ATTACK}, seed ${SEED}"
