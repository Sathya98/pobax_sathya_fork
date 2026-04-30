#!/usr/bin/env bash
#SBATCH --partition=cees6000
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn031
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#
# --job-name, --time, --output, --error are set per-combo by submit_pobax.sh
# on the sbatch CLI (NOT #SBATCH'd here).
#
# Runs ONE (method, env) combo on one exclusive 3090, with N_SEEDS vmapped
# inside a single JAX process. Required env vars, set by submit_pobax.sh via
# sbatch --export:
#
#   METHOD         urnn_standard | urnn_legacy | gru | ppo_ld | eunn_<L> | qurnn_standard | qurnn_legacy | qeunn_<L>
#   ENV_NAME       pobax env id
#   HIDDEN_SIZE    int
#   NUM_ENVS       int
#   NUM_STEPS      int
#   LR             float
#   LAMBDA0        float
#   ENTROPY_COEFF  float
#   TOTAL_STEPS    int
#   LAMBDA1        float, only for ppo_ld
#   LD_WEIGHT      float, only for ppo_ld
# Optional:
#   COMPLEX_LR=8e-5   SEED_BASE=2025   N_SEEDS=5
#   STEPS_LOG_FREQ=1   UPDATE_LOG_FREQ=1   (craftax uses 16/16)

set -eo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"

# slurm does not source ~/.bashrc; activate qrl_env explicitly (CLAUDE.md).
# shellcheck disable=SC1091
source ~/qrl_env/bin/activate

: "${METHOD:?}"; : "${ENV_NAME:?}"; : "${HIDDEN_SIZE:?}"; : "${NUM_ENVS:?}"
: "${NUM_STEPS:?}"; : "${LR:?}"; : "${LAMBDA0:?}"; : "${ENTROPY_COEFF:?}"
: "${TOTAL_STEPS:?}"
COMPLEX_LR="${COMPLEX_LR:-8e-5}"
SEED_BASE="${SEED_BASE:-2025}"
N_SEEDS="${N_SEEDS:-5}"
STEPS_LOG_FREQ="${STEPS_LOG_FREQ:-1}"
UPDATE_LOG_FREQ="${UPDATE_LOG_FREQ:-1}"

# Override env vars from sweep-best YAML if it exists.
SWEEP_BEST="$REPO_DIR/scripts/hyperparams/sweep_best/${METHOD}_${ENV_NAME}.yaml"
if [[ -f "$SWEEP_BEST" ]]; then
    echo "Loading sweep-best overrides from $SWEEP_BEST"
    while IFS=': ' read -r key val; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        export "$key=$val"
    done < "$SWEEP_BEST"
fi

# Build method-specific CLI args (array -> safe word-splitting).
case "$METHOD" in
    urnn_standard) METHOD_FLAGS=(--memory_type urnn --urnn_variant standard --complex_lr "$COMPLEX_LR") ;;
    urnn_legacy)   METHOD_FLAGS=(--memory_type urnn --urnn_variant legacy   --complex_lr "$COMPLEX_LR") ;;
    eunn_*)
        EUNN_CAPACITY="${METHOD#eunn_}"
        [[ "$EUNN_CAPACITY" =~ ^[0-9]+$ ]] || {
            echo "ERROR: bad METHOD=$METHOD (expected eunn_<int>, got suffix '$EUNN_CAPACITY')" >&2
            exit 2
        }
        METHOD_FLAGS=(--memory_type eunn --eunn_capacity "$EUNN_CAPACITY" --complex_lr "$COMPLEX_LR")
        ;;
    qurnn_standard) METHOD_FLAGS=(--memory_type urnn --urnn_variant standard --complex_lr "$COMPLEX_LR" --policy_head born) ;;
    qurnn_legacy)   METHOD_FLAGS=(--memory_type urnn --urnn_variant legacy   --complex_lr "$COMPLEX_LR" --policy_head born) ;;
    qeunn_*)
        EUNN_CAPACITY="${METHOD#qeunn_}"
        [[ "$EUNN_CAPACITY" =~ ^[0-9]+$ ]] || {
            echo "ERROR: bad METHOD=$METHOD (expected qeunn_<int>, got suffix '$EUNN_CAPACITY')" >&2
            exit 2
        }
        METHOD_FLAGS=(--memory_type eunn --eunn_capacity "$EUNN_CAPACITY" --complex_lr "$COMPLEX_LR" --policy_head born)
        ;;
    gru)           METHOD_FLAGS=() ;;
    ppo_ld)        METHOD_FLAGS=(--double_critic) ;;
    *) echo "ERROR: unknown METHOD=$METHOD" >&2; exit 2 ;;
esac

# PPO-LD also needs lambda1 and ld_weight; non-LD runs skip them.
LD_FLAGS=()
if [[ "$METHOD" == "ppo_ld" ]]; then
    : "${LAMBDA1:?ppo_ld requires LAMBDA1}"; : "${LD_WEIGHT:?ppo_ld requires LD_WEIGHT}"
    LD_FLAGS=(--lambda1 "$LAMBDA1" --ld_weight "$LD_WEIGHT")
fi

STUDY="${STUDY_NAME:-${METHOD}_${ENV_NAME}_paper}"

echo "==== ${STUDY} ===="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  nodelist=${SLURM_JOB_NODELIST:-?}"
echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
python -c "import pobax, jax; print(f'pobax ok; jax={jax.__version__}; devices={jax.devices()}')"
echo "hparams: hidden=$HIDDEN_SIZE num_envs=$NUM_ENVS num_steps=$NUM_STEPS"
echo "         lr=$LR  lambda0=$LAMBDA0  ent=$ENTROPY_COEFF  total=$TOTAL_STEPS"
echo "         seeds=${N_SEEDS}@${SEED_BASE}  method=$METHOD  complex_lr=$COMPLEX_LR"
[[ "$METHOD" == eunn_* || "$METHOD" == qeunn_* ]] && echo "         eunn_capacity=$EUNN_CAPACITY"
[[ "$METHOD" == q* ]] && echo "         policy_head=born"
[[ "$METHOD" == "ppo_ld" ]] && echo "         lambda1=$LAMBDA1  ld_weight=$LD_WEIGHT"
echo "         steps_log_freq=$STEPS_LOG_FREQ  update_log_freq=$UPDATE_LOG_FREQ"
echo

# One job = one whole 3090; no MEM_FRAC cap.
python -m pobax.algos.ppo \
    --env "$ENV_NAME" \
    --action_concat \
    --hidden_size "$HIDDEN_SIZE" \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --lr "$LR" \
    --lambda0 "$LAMBDA0" \
    --entropy_coeff "$ENTROPY_COEFF" \
    --total_steps "$TOTAL_STEPS" \
    --seed "$SEED_BASE" \
    --n_seeds "$N_SEEDS" \
    --platform gpu \
    --study_name "$STUDY" \
    --steps_log_freq "$STEPS_LOG_FREQ" \
    --update_log_freq "$UPDATE_LOG_FREQ" \
    "${METHOD_FLAGS[@]}" \
    "${LD_FLAGS[@]}"

echo
echo "==== ${STUDY} done (exit=$?) ===="
