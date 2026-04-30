"""Plot reward curves for every config on a vmap hyperparameter sweep grid.

Loads a sweep checkpoint (study dir convention: {method_prefix}_{env}_hpsweep),
overlays one mean +/- 95% CI curve per grid point, and prints the best config.
"""
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
from tap import Tap

from pobax.definitions import PROJECT_ROOT_DIR, IVI_STORAGE_DIR
from pobax.utils.plot import mean_confidence_interval
from pobax.utils.plot_reward_curves import _x_axis, _apply_smooth


METHOD_PREFIXES = {
    'urnn':       'urnn_standard',
    'legurnn':    'urnn_legacy',
    'gru_ppo':    'gru',
    'ppo_ld':     'ppo_ld',
    'eunn_2':     'eunn_2',
    'eunn_4':     'eunn_4',
    'eunn_8':     'eunn_8',
    'eunn_32':    'eunn_32',
    'leg_qurnn':  'qurnn_legacy',
    'qurnn':      'qurnn_standard',
    'qeunn_2':    'qeunn_2',
    'qeunn_4':    'qeunn_4',
}

SHORT_KEYS = {
    'lr': 'lr',
    'complex_lr': 'clr',
    'vf_coeff': 'vf',
    'entropy_coeff': 'ent',
    'lambda0': 'λ0',
    'lambda1': 'λ1',
    'ld_weight': 'ld',
}


HPARAM_TO_ENV_VAR = {
    'lr': 'LR',
    'complex_lr': 'COMPLEX_LR',
    'entropy_coeff': 'ENTROPY_COEFF',
    'lambda0': 'LAMBDA0',
    'lambda1': 'LAMBDA1',
    'ld_weight': 'LD_WEIGHT',
    'vf_coeff': 'VF_COEFF',
}


class SweepPlotArgs(Tap):
    env: str
    method: str
    discounted: bool = False
    smooth: Literal['none', 'ema', 'savgol'] = 'none'
    ema_weight: float = 0.9
    out_dir: str = None
    save_best: bool = False                                   # write best config YAML for run_pobax.sbatch


def _study_dir(method: str, env: str) -> Path:
    if method not in METHOD_PREFIXES:
        raise ValueError(f'unknown method {method!r}; known: {sorted(METHOD_PREFIXES)}')
    return Path(IVI_STORAGE_DIR) / f'{METHOD_PREFIXES[method]}_{env}_hpsweep'


def _load_sweep(study_dir: Path, discounted: bool = False):
    """Load latest checkpoint; return (returns[n_hp, n_seeds, n_updates], swept_hparams, args)."""
    run_dirs = sorted(p for p in study_dir.iterdir() if p.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f'no run subdirs under {study_dir}')
    restored = orbax.checkpoint.PyTreeCheckpointer().restore(str(run_dirs[-1]))
    key = 'returned_discounted_episode_returns' if discounted else 'returned_episode_returns'
    arr = np.asarray(restored['out']['metric'][key]).mean(axis=(-2, -1))
    return arr, restored['swept_hparams'], restored['args']


def _build_labels(swept_hparams: dict, n_hparams: int) -> list[str]:
    """Build a short label per config, only including keys that actually vary."""
    varying = {}
    for key, vals in swept_hparams.items():
        vals = np.asarray(vals)
        if vals.size == n_hparams and not np.all(vals == vals[0]):
            varying[key] = vals

    if not varying:
        return [f'config_{i}' for i in range(n_hparams)]

    labels = []
    for i in range(n_hparams):
        parts = []
        for key, vals in varying.items():
            short = SHORT_KEYS.get(key, key)
            parts.append(f'{short}={vals[i]:.2g}')
        labels.append('_'.join(parts))
    return labels


def _save_best_yaml(method: str, env: str, best_idx: int, label: str,
                    final_return: float, swept_hparams: dict, study_name: str):
    """Write best sweep config as YAML for run_pobax.sbatch to consume."""
    yaml_dir = Path(PROJECT_ROOT_DIR, 'scripts', 'hyperparams', 'sweep_best')
    yaml_dir.mkdir(parents=True, exist_ok=True)
    prefix = METHOD_PREFIXES[method]
    yaml_path = yaml_dir / f'{prefix}_{env}.yaml'

    lines = [
        f'# Best HPs from sweep: {study_name}',
        f'# Config [{best_idx}]: {label} (mean final return = {final_return:.3f})',
    ]
    for key, vals in swept_hparams.items():
        env_var = HPARAM_TO_ENV_VAR.get(key)
        if env_var is None:
            continue
        val = float(np.asarray(vals)[best_idx])
        lines.append(f'{env_var}: {val:g}')

    yaml_path.write_text('\n'.join(lines) + '\n')
    print(f'wrote best config to {yaml_path}')


def plot_sweep(args: SweepPlotArgs, out_dir: Path):
    study_dir = _study_dir(args.method, args.env)
    returns, swept_hparams, run_args = _load_sweep(study_dir, discounted=args.discounted)
    n_hparams, n_seeds, n_updates = returns.shape
    labels = _build_labels(swept_hparams, n_hparams)
    x = _x_axis(run_args, n_updates)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_hparams):
        mu, h = mean_confidence_interval(returns[i], axis=0)
        mu = _apply_smooth(mu, args)
        color = f'C{i % 10}'
        ax.plot(x, mu, color=color, label=labels[i])
        ax.fill_between(x, mu - h, mu + h, color=color, alpha=0.15)

    ax.set_xlabel('environment steps')
    ax.set_ylabel('episode return' + (' (discounted)' if args.discounted else ''))
    ax.set_title(f'{args.method} — {args.env}')
    ax.legend(fontsize=7)
    fig.tight_layout()

    out = out_dir / f'{args.method}_{args.env}_hpsweep_{n_hparams}configs.pdf'
    fig.savefig(out)
    print(f'wrote {out}')

    # Print best config by mean final return (last 10% of updates).
    tail = max(1, n_updates // 10)
    final_mean = returns[:, :, -tail:].mean(axis=(1, 2))
    best = int(np.argmax(final_mean))
    print(f'\nbest config [{best}]: {labels[best]}  (mean final return = {final_mean[best]:.3f})')
    for i in np.argsort(-final_mean):
        print(f'  [{i}] {labels[i]:>40s}  {final_mean[i]:.3f}')

    if args.save_best:
        study_name = f'{METHOD_PREFIXES[args.method]}_{args.env}_hpsweep'
        _save_best_yaml(args.method, args.env, best, labels[best],
                        final_mean[best], swept_hparams, study_name)


def main(args: SweepPlotArgs):
    out_dir = Path(args.out_dir) if args.out_dir else Path(PROJECT_ROOT_DIR, 'images', 'urnn_plots')
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_sweep(args, out_dir)


if __name__ == '__main__':
    main(SweepPlotArgs().parse_args())
