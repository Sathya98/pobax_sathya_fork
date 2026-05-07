"""Plot reward curves for every config across all matching hpsweep checkpoints.

Globs all study dirs `{method_prefix}_{env}_hpsweep*` under IVI_STORAGE_DIR,
merges them into one per-config view (concatenating seeds when the same
config tuple appears in multiple studies), then overlays one mean +/- 95%
CI curve per config and prints the global best.

Two configs are "the same" iff they match on every swept hparam (across
any study) plus the non-swept hparams in DEDUP_EXTRA_KEYS. Anything else
is assumed equal across launches by convention; we only hard-error on
`update_log_freq`, `steps_log_freq`, and `n_updates` because those break
the shared x-axis.
"""
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
from tap import Tap

from pobax.definitions import PROJECT_ROOT_DIR, IVI_STORAGE_DIR
from pobax.utils.plot import mean_confidence_interval
from pobax.utils.plot_reward_curves import _apply_smooth


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
    'num_steps': 'ns',
    'num_envs': 'ne',
    'hidden_size': 'h',
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


# Non-swept hparams that should still distinguish configs in the merged
# leaderboard. Anything else is assumed identical across all loaded studies.
DEDUP_EXTRA_KEYS = ('num_steps', 'num_envs', 'hidden_size')


class SweepPlotArgs(Tap):
    env: str
    method: str
    discounted: bool = False
    smooth: Literal['none', 'ema', 'savgol'] = 'none'
    ema_weight: float = 0.9
    out_dir: str = None
    save_best: bool = False                                   # write best config YAML for run_pobax.sbatch
    top_k: int = 10                                           # only plot top-K configs by mean final return (<=0 = all)


def _scalarize(v):
    """Coerce list/array of size 1 to a python scalar; pass through otherwise."""
    if isinstance(v, (list, tuple)):
        return v[0] if len(v) == 1 else tuple(v)
    if isinstance(v, np.ndarray):
        return v.item() if v.size == 1 else tuple(v.tolist())
    return v


def _study_dirs(method: str, env: str) -> list[Path]:
    """All result dirs matching {prefix}_{env}_hpsweep*, sorted by name."""
    if method not in METHOD_PREFIXES:
        raise ValueError(f'unknown method {method!r}; known: {sorted(METHOD_PREFIXES)}')
    root = Path(IVI_STORAGE_DIR)
    pattern = f'{METHOD_PREFIXES[method]}_{env}_hpsweep*'
    dirs = sorted(p for p in root.glob(pattern) if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f'no studies match {root}/{pattern}')
    return dirs


def _load_one_study(study_dir: Path, discounted: bool):
    """Latest checkpoint under study_dir → (returns[n_hp,n_seeds,n_updates], swept, args)."""
    run_dirs = sorted(p for p in study_dir.iterdir() if p.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f'no run subdirs under {study_dir}')
    restored = orbax.checkpoint.PyTreeCheckpointer().restore(str(run_dirs[-1]))
    key = 'returned_discounted_episode_returns' if discounted else 'returned_episode_returns'
    arr = np.asarray(restored['out']['metric'][key]).mean(axis=(-2, -1))
    return arr, restored['swept_hparams'], restored['args']


def _validate_studies(loaded):
    """Hard-error on x-axis-breaking mismatches (matches user's equal-budget assumption)."""
    ref_returns, _, ref_args = loaded[0]
    ref_n_updates = ref_returns.shape[-1]
    refs = {k: _scalarize(ref_args.get(k)) for k in ('update_log_freq', 'steps_log_freq')}
    for returns, _, args in loaded[1:]:
        if returns.shape[-1] != ref_n_updates:
            raise ValueError(
                f'n_updates mismatch across studies (got {returns.shape[-1]} vs {ref_n_updates}); '
                f'all studies must run to the same budget'
            )
        for k, ref_v in refs.items():
            v = _scalarize(args.get(k))
            if v != ref_v:
                raise ValueError(f'{k} mismatch across studies (got {v} vs {ref_v})')


def _combine_studies(loaded):
    """Merge per-study (returns, swept, args) into one per-config view.

    loaded: list of (returns[n_hp, n_seeds, n_up], swept_hparams, args).
    Returns: (returns_per_cfg, hparams_per_cfg, dedup_keys, varying_keys).

    Two (study, cfg_idx) entries collapse iff they match on every key in
    `dedup_keys` = (union of swept keys) + DEDUP_EXTRA_KEYS. Their seed
    arrays are concatenated along axis 0.
    """
    _validate_studies(loaded)

    swept_keys = sorted({k for _, sw, _ in loaded for k in sw})
    dedup_keys = tuple(swept_keys) + DEDUP_EXTRA_KEYS

    grouped: dict[tuple, dict] = {}
    for returns, swept, args in loaded:
        for i in range(returns.shape[0]):
            row = {}
            for k in dedup_keys:
                if k in swept:
                    row[k] = _scalarize(np.asarray(swept[k])[i])
                else:
                    row[k] = _scalarize(args.get(k))
            canonical = tuple((k, row[k]) for k in dedup_keys)
            if canonical not in grouped:
                grouped[canonical] = {'returns': [], 'hparams': row}
            grouped[canonical]['returns'].append(returns[i])  # (n_seeds, n_up)

    returns_per_cfg = [np.concatenate(g['returns'], axis=0) for g in grouped.values()]
    hparams_per_cfg = [g['hparams'] for g in grouped.values()]
    varying_keys = tuple(
        k for k in dedup_keys
        if len({cfg[k] for cfg in hparams_per_cfg}) > 1
    )
    return returns_per_cfg, hparams_per_cfg, dedup_keys, varying_keys


def _build_labels(hparams_per_cfg: list[dict], varying_keys: tuple[str, ...]) -> list[str]:
    """Label each config with just the keys whose value differs across configs."""
    if not varying_keys:
        return [f'config_{i}' for i in range(len(hparams_per_cfg))]
    labels = []
    for cfg in hparams_per_cfg:
        parts = []
        for k in varying_keys:
            short = SHORT_KEYS.get(k, k)
            v = cfg[k]
            if isinstance(v, bool):
                parts.append(f'{short}={v}')
            elif isinstance(v, (int, float)):
                parts.append(f'{short}={v:.2g}')
            else:
                parts.append(f'{short}={v}')
        labels.append('_'.join(parts))
    return labels


def _config_x_axis(cfg: dict, n_updates: int, update_log_freq) -> np.ndarray:
    """Per-config env-step x-axis: matches _x_axis() but uses cfg['num_steps','num_envs']."""
    return np.arange(n_updates) * int(update_log_freq) * int(cfg['num_steps']) * int(cfg['num_envs'])


def _save_best_yaml(method: str, env: str, best_idx: int, label: str,
                    final_return: float, best_hparams: dict, study_label: str,
                    varying_keys: tuple[str, ...]):
    """Write best sweep config as YAML for run_pobax.sbatch to consume."""
    yaml_dir = Path(PROJECT_ROOT_DIR, 'scripts', 'hyperparams', 'sweep_best')
    yaml_dir.mkdir(parents=True, exist_ok=True)
    prefix = METHOD_PREFIXES[method]
    yaml_path = yaml_dir / f'{prefix}_{env}.yaml'

    lines = [
        f'# Best HPs from combined sweeps: {study_label}',
        f'# Config [{best_idx}]: {label} (mean final return = {final_return:.3f})',
    ]
    for key, val in best_hparams.items():
        env_var = HPARAM_TO_ENV_VAR.get(key)
        if env_var is None or val is None:
            continue
        lines.append(f'{env_var}: {float(val):g}')

    yaml_path.write_text('\n'.join(lines) + '\n')
    print(f'wrote best config to {yaml_path}')

    unmapped = [k for k in varying_keys if k not in HPARAM_TO_ENV_VAR]
    if unmapped:
        print(
            f'  warning: best config also varies on {unmapped}, which has no '
            f'HPARAM_TO_ENV_VAR mapping — YAML alone will not fully reproduce the run'
        )


def plot_sweep(args: SweepPlotArgs, out_dir: Path):
    study_dirs = _study_dirs(args.method, args.env)
    print(f'loaded {len(study_dirs)} study dirs:')
    loaded = []
    for d in study_dirs:
        returns, swept, run_args = _load_one_study(d, discounted=args.discounted)
        loaded.append((returns, swept, run_args))
        print(f'  {d.name}  shape={returns.shape}')

    returns_per_cfg, hparams_per_cfg, dedup_keys, varying_keys = _combine_studies(loaded)
    n_cfgs = len(returns_per_cfg)
    n_updates = returns_per_cfg[0].shape[-1]
    update_log_freq = _scalarize(loaded[0][2].get('update_log_freq', 1))
    labels = _build_labels(hparams_per_cfg, varying_keys)

    total_entries = sum(r.shape[0] for r, _, _ in loaded)
    print(f'\ndedup keys: {dedup_keys}')
    print(f'varying keys (used for legend labels): {varying_keys}')
    print(f'merged into {n_cfgs} unique configs (from {total_entries} (study, cfg_idx) entries)')
    seed_totals = [r.shape[0] for r in returns_per_cfg]
    print(f'per-config seed counts: min={min(seed_totals)} max={max(seed_totals)} '
          f'(>min indicates seeds concatenated across studies)')

    # Rank configs by mean final return (last 10% of updates).
    tail = max(1, n_updates // 10)
    final_mean = np.array([r[:, -tail:].mean() for r in returns_per_cfg])
    ranked = np.argsort(-final_mean)
    if args.top_k > 0 and args.top_k < n_cfgs:
        plot_idx = ranked[:args.top_k]
    else:
        plot_idx = ranked

    fig, ax = plt.subplots(figsize=(8, 5))
    for plot_pos, i in enumerate(plot_idx):
        mu, h = mean_confidence_interval(returns_per_cfg[i], axis=0)
        mu = _apply_smooth(mu, args)
        x = _config_x_axis(hparams_per_cfg[i], n_updates, update_log_freq)
        color = f'C{plot_pos % 10}'
        ax.plot(x, mu, color=color, label=f'[{i}] {labels[i]}')
        ax.fill_between(x, mu - h, mu + h, color=color, alpha=0.15)

    ax.set_xlabel('environment steps')
    ax.set_ylabel('episode return' + (' (discounted)' if args.discounted else ''))
    title = f'{args.method} — {args.env} (combined {len(study_dirs)} studies)'
    if len(plot_idx) < n_cfgs:
        title += f' — top {len(plot_idx)} of {n_cfgs}'
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()

    suffix = f'top{len(plot_idx)}of{n_cfgs}' if len(plot_idx) < n_cfgs else f'{n_cfgs}configs'
    out = out_dir / f'{args.method}_{args.env}_hpsweep_combined{len(study_dirs)}_{suffix}.pdf'
    fig.savefig(out)
    print(f'wrote {out}')

    best = int(ranked[0])
    print(f'\nbest config [{best}]: {labels[best]}  '
          f'(mean final return = {final_mean[best]:.3f}, n_seeds = {returns_per_cfg[best].shape[0]})')
    for i in ranked:
        print(f'  [{i}] {labels[i]:>40s}  '
              f'{final_mean[i]:.3f}  (n_seeds={returns_per_cfg[i].shape[0]})')

    if args.save_best:
        study_label = f'{METHOD_PREFIXES[args.method]}_{args.env}_hpsweep* (n={len(study_dirs)})'
        _save_best_yaml(args.method, args.env, best, labels[best],
                        final_mean[best], hparams_per_cfg[best], study_label, varying_keys)


def main(args: SweepPlotArgs):
    out_dir = Path(args.out_dir) if args.out_dir else Path(PROJECT_ROOT_DIR, 'images', 'urnn_plots')
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_sweep(args, out_dir)


if __name__ == '__main__':
    main(SweepPlotArgs().parse_args())
