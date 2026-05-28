#!/usr/bin/env bash
# Loop plot_hp_sweep_curves over (env x method) and write each best-config YAML.
# Usage: ./scripts/visualizations/save_best_rocksample.sh
set -uo pipefail

source ~/qrl_env/bin/activate 2>/dev/null || source /projects/prjs2050/qrl_env/bin/activate

ENVS=(rocksample_15_15)
METHODS=(leg_qurnn qurnn eunn_2 qeunn_2)
SMOOTH="${SMOOTH:-savgol}"

for env in "${ENVS[@]}"; do
    for method in "${METHODS[@]}"; do
        echo "==== ${method} x ${env} ===="
        python -m pobax.utils.plot_hp_sweep_curves \
            --method="$method" --env="$env" --smooth="$SMOOTH" --save_best \
            || echo "  FAILED: ${method} x ${env} (continuing)"
        echo
    done
done
