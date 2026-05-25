#!/usr/bin/env bash
# Submit one sbatch HP-sweep job per (method, env) combo.
#
# Unlike submit_pobax.sh (which delegates to run_pobax.sbatch with scalar env
# vars), sweeps pass list-valued TAP args directly to python, so we use
# sbatch --wrap with the full inline command.
#
# Usage:
#   ./submit_sweep.sh                                         # full matrix
#   DRY_RUN=1 ./submit_sweep.sh                               # print commands only
#   FORCE=1 ./submit_sweep.sh                                 # rerun even if results exist
#   METHODS="urnn_legacy" ENVS="tmaze_10" ./submit_sweep.sh   # subset
#   SWEEP_ENTROPY="0.0 0.001" ENVS="craftax" ./submit_sweep.sh  # override grid for one env
#   STUDY_SUFFIX="_entropy_low" ./submit_sweep.sh             # disambiguate reruns w/ different grids

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
mkdir -p logs/slurm

# ═══════════════════════════════════════════════════════════════════════════
# SLURM configuration — edit when switching nodes
# ═══════════════════════════════════════════════════════════════════════════
PARTITION="${PARTITION:-gpu_a100}"
GPUS="gpu:1"
CPUS=12
MEM="64G"

# ═══════════════════════════════════════════════════════════════════════════
# Sweep grid — space-separated values; single value = not swept.
# Each grid is shared across all (method, env) pairs in one invocation —
# scope ENVS=... when grids should differ per env.
# ═══════════════════════════════════════════════════════════════════════════
SWEEP_LR="${SWEEP_LR:-2.5e-4}"
SWEEP_COMPLEX_LR="${SWEEP_COMPLEX_LR:-8e-5}"
SWEEP_ENTROPY="${SWEEP_ENTROPY:-0.01 0.05}"
SWEEP_LAMBDA0="${SWEEP_LAMBDA0:-0.5 0.7}"

# ═══════════════════════════════════════════════════════════════════════════
# Run configuration
# ═══════════════════════════════════════════════════════════════════════════
N_SEEDS="${N_SEEDS:-3}"
SEED_BASE="${SEED_BASE:-2026}"
STEPS_LOG_FREQ="${STEPS_LOG_FREQ:-1}"
UPDATE_LOG_FREQ="${UPDATE_LOG_FREQ:-1}"
STUDY_SUFFIX="${STUDY_SUFFIX:-}"

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

METHODS="${METHODS:-urnn_standard urnn_legacy}"
ENVS="${ENVS:-tmaze_10 rocksample_11_11}"

# ═══════════════════════════════════════════════════════════════════════════
# Per-env base config — fields: hidden num_envs num_steps total_steps time
# Times are padded for vmap over (n_configs × n_seeds).
# ═══════════════════════════════════════════════════════════════════════════
declare -A BASE
BASE[tmaze_10]="32  4   256  5000000    04:00:00"
BASE[rocksample_11_11]="256 8   256  3000000    16:00:00"
BASE[rocksample_15_15]="512 16  256  6000000   24:00:00"
BASE[battleship_10]="512 32  256  6000000   24:00:00"
BASE[Navix-DMLab-Maze-01-v0]="512 256 128  4000000   24:00:00"
BASE[Walker-V-v0]="256 4   256  25000000   48:00:00"
BASE[HalfCheetah-V-v0]="256 4   256  25000000   48:00:00"
BASE[craftax]="512 256 128   40000000  48:00:00"
BASE[craftax_pixels]="512 256 128   40000000  48:00:00"

# ═══════════════════════════════════════════════════════════════════════════
# Method prefix mapping (matches plot_hp_sweep_curves.py METHOD_PREFIXES)
# ═══════════════════════════════════════════════════════════════════════════
declare -A PREFIX
PREFIX[urnn_standard]="urnn_standard"
PREFIX[urnn_legacy]="urnn_legacy"
PREFIX[gru]="gru"
PREFIX[ppo_ld]="ppo_ld"
PREFIX[eunn_2]="eunn_2"
PREFIX[eunn_4]="eunn_4"
PREFIX[eunn_8]="eunn_8"
PREFIX[eunn_32]="eunn_32"
PREFIX[qurnn_standard]="qurnn_standard"
PREFIX[qurnn_legacy]="qurnn_legacy"
PREFIX[qeunn_2]="qeunn_2"
PREFIX[qeunn_4]="qeunn_4"

# ═══════════════════════════════════════════════════════════════════════════

echo "Submitting HP sweep (method, env) matrix."
echo "  methods : $METHODS"
echo "  envs    : $ENVS"
echo "  grid    : lr=[$SWEEP_LR]  clr=[$SWEEP_COMPLEX_LR]  ent=[$SWEEP_ENTROPY]  λ0=[$SWEEP_LAMBDA0]"
echo "  N_SEEDS=$N_SEEDS  SEED=$SEED_BASE  update_log_freq=$UPDATE_LOG_FREQ"
echo "  DRY_RUN=$DRY_RUN  FORCE=$FORCE"
echo

total=0
skipped=0
for method in $METHODS; do
    if [[ -z "${PREFIX[$method]:-}" ]]; then
        echo "  [skip]    unknown method $method (no PREFIX entry)"
        continue
    fi

    # Method-specific flags (complex_lr is in the sweep grid, not here).
    case "$method" in
        urnn_standard)  method_flags="--memory_type urnn --urnn_variant standard" ;;
        urnn_legacy)    method_flags="--memory_type urnn --urnn_variant legacy" ;;
        eunn_*)
            cap="${method#eunn_}"
            method_flags="--memory_type eunn --eunn_capacity $cap"
            ;;
        qurnn_standard) method_flags="--memory_type urnn --urnn_variant standard --policy_head born" ;;
        qurnn_legacy)   method_flags="--memory_type urnn --urnn_variant legacy --policy_head born" ;;
        qeunn_*)
            cap="${method#qeunn_}"
            method_flags="--memory_type eunn --eunn_capacity $cap --policy_head born"
            ;;
        gru)            method_flags="" ;;
        ppo_ld)         method_flags="--double_critic" ;;
        *)
            echo "  [skip]    unknown method $method in case block"
            continue
            ;;
    esac

    for env in $ENVS; do
        # tmaze_<len> envs reuse tmaze_10 settings — ppo accepts any length.
        base_key="$env"
        if [[ -z "${BASE[$base_key]:-}" && "$env" == tmaze_* ]]; then
            base_key="tmaze_10"
        fi
        if [[ -z "${BASE[$base_key]:-}" ]]; then
            echo "  [skip]    unknown env $env (no BASE entry)"
            continue
        fi

        read -r hidden nenvs nsteps tsteps jobtime <<< "${BASE[$base_key]}"
        study="${PREFIX[$method]}_${env}_hpsweep${STUDY_SUFFIX}"

        # Auto-skip if results already exist.
        if [[ "$FORCE" != "1" ]] && compgen -G "$REPO_DIR/results/$study/*/" > /dev/null; then
            printf '  [done]    %-48s (FORCE=1 to rerun)\n' "$study"
            ((skipped+=1))
            continue
        fi

        pycmd="python -m pobax.algos.ppo \
--env $env --action_concat \
--hidden_size $hidden --num_envs $nenvs --num_steps $nsteps \
--lr $SWEEP_LR \
--complex_lr $SWEEP_COMPLEX_LR \
--entropy_coeff $SWEEP_ENTROPY \
--lambda0 $SWEEP_LAMBDA0 \
--total_steps $tsteps \
--seed $SEED_BASE --n_seeds $N_SEEDS \
--platform gpu \
--steps_log_freq $STEPS_LOG_FREQ \
--update_log_freq $UPDATE_LOG_FREQ \
--study_name $study \
$method_flags"

        # Pixel-obs envs route observations through a CNN embedding. On craftax
        # the cuDNN conv autotuner probes algorithms needing 78-158 GB workspace,
        # which OOM the GPU and crash the job (CUDA_ERROR_ILLEGAL_ADDRESS).
        # Disabling conv autotuning sidesteps the bad probes; the wider memory
        # arena helps the genuine CNN activations. No-op for vector-obs envs.
        xla_env=""
        if [[ "$env" == *pixels* ]]; then
            xla_env="XLA_FLAGS='--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0' XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 "
        fi

        cmd=(
            sbatch
            --partition="$PARTITION"
            --gres="$GPUS"
            --cpus-per-task="$CPUS"
            --mem="$MEM"
            --job-name="${method}_${env}_sweep"
            --time="$jobtime"
            --output="logs/slurm/${study}_%j.out"
            --error="logs/slurm/${study}_%j.err"
            --wrap="source /projects/prjs2050/qrl_env/bin/activate && cd $REPO_DIR && ${xla_env}$pycmd"
        )

        if [[ "$DRY_RUN" == "1" ]]; then
            printf '  [dry-run] %s\n' "$study"
            printf '            '; printf '%q ' "${cmd[@]}"; echo
            echo
        else
            printf '  [submit]  %-48s time=%s\n' "$study" "$jobtime"
            "${cmd[@]}"
        fi
        ((total+=1))
    done
done

echo
echo "Dispatched $total jobs, skipped $skipped already-done combos."
if [[ "$DRY_RUN" != "1" && "$total" -gt 0 ]]; then
    echo "Queue:   squeue -u \"$USER\""
    echo "Logs:    $REPO_DIR/logs/slurm/"
    echo "Results: $REPO_DIR/results/<prefix>_<env>_hpsweep/"
fi
