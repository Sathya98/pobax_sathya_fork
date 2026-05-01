#!/usr/bin/env bash
# Submit one sbatch per (method, env) combo. ivi-cn014 has 3 x RTX 3090 with
# no MPS/shard (verified via `scontrol show node ivi-cn014`), so each job
# reserves a whole GPU and slurm co-schedules up to 3 concurrently.
#
# Scope (2026-04-14):
#   7 envs  x 4 methods = 28 jobs. Pixel envs (halfcheetah_pixels, ant_pixels)
#   and Navix-02 + craftax are deferred: madrona_mjx is not yet installed in
#   ~/qrl_env, and the longer envs are too heavy for the current window.
#
# Usage (from the LOGIN node):
#   ./submit_pobax.sh
#   DRY_RUN=1 ./submit_pobax.sh                # print sbatch commands only
#   SMOKE=1 ./submit_pobax.sh                  # 1e5-step sanity runs
#   FORCE=1 ./submit_pobax.sh                  # submit even if results/<study>/ exists
#   METHODS="urnn_standard urnn_legacy" ./submit_pobax.sh    # subset of methods
#   METHODS="eunn_2 eunn_8" ./submit_pobax.sh                 # EUNN runs at L=2 and L=8
#   METHODS="qurnn_standard qurnn_legacy" ./submit_pobax.sh   # QuRNN (Born-rule) runs
#   METHODS="qeunn_2 qeunn_8" ./submit_pobax.sh               # QuEUNN runs at L=2 and L=8
#   ENVS="tmaze_10 battleship_10" ./submit_pobax.sh           # subset of envs
#   N_SEEDS=3 ./submit_pobax.sh                # fewer vmapped seeds per run
#
# Auto-skip: by default, any (method, env) whose results/<method>_<env>_paper/
# directory already contains a run subdir is skipped; set FORCE=1 to override.
#
# Watch:
#   squeue -u "$USER"
#   tail -f logs/slurm/<method>_<env>_paper_<jobid>.out
#
# Outputs:
#   Results : results/<method>_<env>_paper/ (Orbax; one run = one timestamped subdir)
#   Logs    : logs/slurm/<method>_<env>_paper_<jobid>.out/.err

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
mkdir -p logs/slurm

SBATCH_FILE="$REPO_DIR/run_pobax.sbatch"
[[ -f "$SBATCH_FILE" ]] || { echo "ERROR: $SBATCH_FILE missing" >&2; exit 1; }
chmod +x "$SBATCH_FILE" 2>/dev/null || true

SEED_BASE="${SEED_BASE:-2025}"
N_SEEDS="${N_SEEDS:-5}"
COMPLEX_LR="${COMPLEX_LR:-8e-5}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"
METHODS="${METHODS:-urnn_standard urnn_legacy gru ppo_ld}"

# Base config (shared by urnn_standard / urnn_legacy / gru). Sourced from
# scripts/hyperparams/<env>/best/<env>_ppo_best.py. Times are padded for the
# 5-seed-vmap run on an uncontended 3090.
#
# Fields: env hidden num_envs num_steps lr lambda0 entropy total_steps time
declare -A BASE
BASE[tmaze_10]="32  4   128  2.5e-3  0.7   0.01  3000000    01:00:00"
BASE[rocksample_11_11]="256 8   128  2.5e-3  0.7   0.2   5000000    04:00:00"
BASE[rocksample_15_15]="512 16  128  2.5e-3  0.7   0.2   10000000   08:00:00"
BASE[battleship_10]="512 32  128  2.5e-3  0.7   0.05  10000000   10:00:00"
BASE[Navix-DMLab-Maze-01-v0]="512 256 128  2.5e-4  0.9   0.01  10000000   15:00:00"
BASE[Walker-V-v0]="256 4   128  2.5e-4  0.95  0.01  50000000   36:00:00"
BASE[HalfCheetah-V-v0]="256 4   128  2.5e-4  0.9   0.01  50000000   36:00:00"
BASE[craftax]="512 256 64   2.5e-4  0.5   0.01  100000000  52:00:00"
BASE[craftax_pixels]="512 256 64   2.5e-4  0.5   0.01  100000000  52:00:00"

# PPO-LD overrides (lr, lambda0, lambda1, ld_weight). Other fields inherit
# from BASE. Note the tmaze_10 and HalfCheetah-V-v0 lr's differ from BASE
# by 10x each - confirmed with user as intentional paper choice.
#
# Fields: lr lambda0 lambda1 ld_weight
declare -A LD
LD[tmaze_10]="2.5e-4  0.95  0.95  0.25"
LD[rocksample_11_11]="2.5e-3  0.5   0.5   0.25"
LD[rocksample_15_15]="2.5e-3  0.1   0.95  0.5"
LD[battleship_10]="2.5e-3  0.1   0.95  0.5"
LD[Navix-DMLab-Maze-01-v0]="2.5e-4  0.95  0.5   0.25"
LD[Walker-V-v0]="2.5e-4  0.95  0.95  0.5"
LD[HalfCheetah-V-v0]="2.5e-5  0.95  0.7   0.25"
LD[craftax]="2.5e-4  0.1   0.95  0.25"
LD[craftax_pixels]="2.5e-4  0.1   0.95  0.25"

ALL_ENVS=(tmaze_10 rocksample_11_11 rocksample_15_15 battleship_10
          Navix-DMLab-Maze-01-v0 Walker-V-v0 HalfCheetah-V-v0
          craftax craftax_pixels)
ENVS="${ENVS:-${ALL_ENVS[*]}}"

echo "Submitting (method, env) matrix to ivi-cn014."
echo "  methods: $METHODS"
echo "  envs   : $ENVS"
echo "  N_SEEDS=$N_SEEDS  SEED_BASE=$SEED_BASE  COMPLEX_LR=$COMPLEX_LR"
echo "  SMOKE=$SMOKE  DRY_RUN=$DRY_RUN  FORCE=$FORCE"
echo

total=0
skipped=0
for method in $METHODS; do
    for env in $ENVS; do
        # Any tmaze_{length} reuses tmaze_10 settings when no explicit entry exists.
        if [[ -z "${BASE[$env]:-}" && "$env" =~ ^tmaze_[0-9]+$ ]]; then
            BASE[$env]="${BASE[tmaze_10]}"
            LD[$env]="${LD[tmaze_10]}"
        fi

        if [[ -z "${BASE[$env]:-}" ]]; then
            echo "  [skip]    unknown env $env (no BASE entry)"
            continue
        fi

        read -r hidden nenvs nsteps base_lr base_lam0 ent tsteps jobtime <<< "${BASE[$env]}"

        # Method-specific overrides.
        lr="$base_lr"; lam0="$base_lam0"; lam1=""; ld_w=""
        if [[ "$method" == "ppo_ld" ]]; then
            if [[ -z "${LD[$env]:-}" ]]; then
                echo "  [skip]    no LD config for $env (method=$method)"
                continue
            fi
            read -r lr lam0 lam1 ld_w <<< "${LD[$env]}"
        fi

        if [[ "$SMOKE" == "1" ]]; then
            tsteps=100000
            jobtime=00:20:00
        fi

        study="${method}_${env}_paper"
        [[ "$SMOKE" == "1" ]] && study="smoke_${study}"

        # Auto-skip if a run subdir already exists under results/<study>/.
        # FORCE=1 overrides (new run creates a fresh timestamped subdir next
        # to the old one; existing checkpoint is preserved).
        if [[ "$FORCE" != "1" ]] && compgen -G "$REPO_DIR/results/$study/*/" > /dev/null; then
            printf '  [done]    %-48s (FORCE=1 to rerun)\n' "$study"
            ((skipped+=1))
            continue
        fi

        steps_lf=1; update_lf=1
        if [[ "$env" == craftax* ]]; then
            steps_lf=16; update_lf=16
        fi

        exports="ALL"
        exports+=",METHOD=${method},ENV_NAME=${env}"
        exports+=",HIDDEN_SIZE=${hidden},NUM_ENVS=${nenvs},NUM_STEPS=${nsteps}"
        exports+=",LR=${lr},LAMBDA0=${lam0}"
        exports+=",ENTROPY_COEFF=${ent},TOTAL_STEPS=${tsteps}"
        exports+=",COMPLEX_LR=${COMPLEX_LR},SEED_BASE=${SEED_BASE},N_SEEDS=${N_SEEDS}"
        exports+=",STEPS_LOG_FREQ=${steps_lf},UPDATE_LOG_FREQ=${update_lf}"
        [[ -n "$lam1" ]] && exports+=",LAMBDA1=${lam1}"
        [[ -n "$ld_w" ]] && exports+=",LD_WEIGHT=${ld_w}"

        cmd=(
            sbatch
            --job-name="$study"
            --time="$jobtime"
            --output="logs/slurm/${study}_%j.out"
            --error="logs/slurm/${study}_%j.err"
            --export="$exports"
            "$SBATCH_FILE"
        )

        if [[ "$DRY_RUN" == "1" ]]; then
            printf '  [dry-run] '; printf '%q ' "${cmd[@]}"; echo
        else
            printf '  [submit]  %-48s time=%s\n' "$study" "$jobtime"
            "${cmd[@]}"
        fi
        ((total+=1))
    done
done

echo
echo "Dispatched $total jobs, skipped $skipped already-done combos."
if [[ "$DRY_RUN" != "1" ]]; then
    echo "Queue:   squeue -u \"$USER\""
    echo "         squeue --nodelist=ivi-cn022"
    echo "Logs:    $REPO_DIR/logs/slurm/"
    echo "Results: $REPO_DIR/results/<method>_<env>_paper/"
fi
