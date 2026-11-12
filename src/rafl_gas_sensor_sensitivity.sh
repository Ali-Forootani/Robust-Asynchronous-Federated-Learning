#!/bin/bash -l
#SBATCH -J sensitivity_RAFL_GAS_fullCPU
#SBATCH --partition=s.geany
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=168:00:00
#SBATCH --array=0-21%22
#SBATCH -D /u/alfo/RAFL_revision
#SBATCH -o /u/alfo/RAFL_revision/logs/%x-%A_%a.out
#SBATCH -e /u/alfo/RAFL_revision/logs/%x-%A_%a.err

set -euo pipefail

module purge
module load gcc/14 || true
source /u/alfo/dnn_env/bin/activate

# Force CPU execution, as in the existing sensitivity campaign.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"

SCRIPT="/u/alfo/RAFL_revision/rafl_gas_sensor_experiment_suite.py"
DATA_ROOT="/u/alfo/RAFL_revision/gas_sensor"
RESULTS_ROOT="/u/alfo/RAFL_revision/results_rafl_gas_sensor_sensitivity"

[[ -f "$SCRIPT" ]] || { echo "Missing Python script: $SCRIPT" >&2; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "Missing dataset directory: $DATA_ROOT" >&2; exit 1; }

# One job = one configuration. Preserves the original single-suite defaults,
# except the named sensitivity variable (and explicitly selected client counts).
# Values: 4 alpha_stale + 5 trigger_eps + 5 byz_frac + 5 tau_max_rounds
#         + 3 num_clients = 22 array tasks.
PARAM_SETS=(
  "alpha_stale 0.0"
  "alpha_stale 0.1"
  "alpha_stale 0.2"
  "alpha_stale 0.5"

  "trigger_eps 0.0"
  "trigger_eps 0.005"
  "trigger_eps 0.01"
  "trigger_eps 0.05"
  "trigger_eps 0.1"

  "byz_frac 0.0"
  "byz_frac 0.1"
  "byz_frac 0.2"
  "byz_frac 0.3"
  "byz_frac 0.4"

  "tau_max_rounds 0"
  "tau_max_rounds 3"
  "tau_max_rounds 5"
  "tau_max_rounds 10"
  "tau_max_rounds 20"

  "num_clients 10"
  "num_clients 20"
  "num_clients 50"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch to supply array ID}"
if (( TASK_ID < 0 || TASK_ID >= ${#PARAM_SETS[@]} )); then
  echo "Invalid task ID: $TASK_ID" >&2
  exit 2
fi

read -r PARAM VALUE <<< "${PARAM_SETS[$TASK_ID]}"
EXTRA_ARGS=("--${PARAM}" "$VALUE")

# Same client-participation configurations as the reference SLURM script.
if [[ "$PARAM" == "num_clients" ]]; then
  case "$VALUE" in
    10|20) EXTRA_ARGS+=(--num_clients_per_round 5 --max_parallel_clients 5) ;;
    50)    EXTRA_ARGS+=(--num_clients_per_round 10 --max_parallel_clients 10) ;;
  esac
fi

# Independent path per task prevents summary CSV/JSON overwrites.
RUN_DIR="${RESULTS_ROOT}/task_${TASK_ID}_${PARAM}_${VALUE}"
mkdir -p "$RUN_DIR"

CMD=(
  python "$SCRIPT"
  --suite single
  --aggregator asb
  --attack signflip
  --enable_byzantine
  --data_root "$DATA_ROOT"
  --base_dir "$RUN_DIR"
  "${EXTRA_ARGS[@]}"
)

printf 'Task ID: %s / %s\nNode(s): %s\nCPUs: %s\nDataset: %s\nOutput: %s\n' \
  "$TASK_ID" "${#PARAM_SETS[@]}" "${SLURM_NODELIST:-unknown}" \
  "$SLURM_CPUS_PER_TASK" "$DATA_ROOT" "$RUN_DIR"
printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'

srun --cpu-bind=cores "${CMD[@]}"
