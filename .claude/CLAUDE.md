# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# POBax - Partial Observability Benchmarks in JAX

**Paper:** "Benchmarking Partial Observability in Reinforcement Learning with a Suite of Memory-Improvable Domains" (RLC 2025)
**Authors:** Ruo Yu (David) Tao, Kaicheng Guo, Cameron Allen, George Konidaris
**OpenReview:** https://openreview.net/forum?id=HUTCbYOW5E

## Environment

ALWAYS use the uv venv at `~/qrl_env` for any Python execution in this repo — running code, installing packages, tests, experiments. Activate with `source ~/qrl_env/bin/activate` (or prefix commands with it in non-interactive shells). Install new deps via `uv pip install ...` inside this env. Do NOT create a new venv, do NOT use the system python, and do NOT use any other env name. The installed package is `pobax`.

## Commands

```bash
# Run tests
source ~/qrl_env/bin/activate && python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_born_rule_actor.py -v

# Run a specific test
python -m pytest tests/test_eunn.py::test_unitarity_f64 -v

# Quick training smoke test (CPU, ~20s)
python -m pobax.algos.ppo --env tmaze_10 --memory_type urnn --hidden_size 32 \
    --num_envs 4 --total_steps 100000 --platform cpu --n_seeds 1

# SLURM batch submission (from login node)
./submit_pobax.sh                          # full matrix
DRY_RUN=1 ./submit_pobax.sh               # print commands only
METHODS="eunn_2" ENVS="tmaze_10" SMOKE=1 ./submit_pobax.sh  # smoke run
```

## What This Repo Is

A JAX-native POMDP benchmark suite designed to test whether memory architectures actually help in RL. All environments are curated to be "memory-improvable" — a memoryless policy provably cannot match a memory-augmented one.

## Stack

- **Flax Linen** (`flax.linen as nn`) for all networks
- **JAX** for computation, `jax.lax.scan` for rollouts
- **Optax** for optimizers (Adam with gradient clipping)
- **Distrax** for probability distributions (Categorical, Normal)
- **TAP** (Typed Argument Parser) for config — CLI args, not Hydra
- **Gymnax** API for JAX-native environments

## Directory Structure

```
pobax/
  config.py              # All hyperparameters (PPOHyperparams class via TAP)
  definitions.py         # Project root path definitions
  algos/
    ppo.py               # Recurrent PPO — main training loop
    transformer_xl.py    # GTrXL variant of PPO
    run_helper.py        # Training utilities (logging, checkpointing)
  models/
    actor_critic.py      # ActorCritic module — composes embedding + memory + heads
    network.py           # ScannedRNN (GRU), ScannedURNN, URNNCell, LegacyURNNCell, EUNNCell, ModReLU
    discrete.py          # DiscreteActor, BornRuleActor, DiscreteActorCriticTransformer
    continuous.py        # ContinuousActor
    value.py             # Critic (value function head)
    embedding.py         # Embedding/preprocessing networks (CNN, SimpleNN)
    transformerXL.py     # Transformer-XL with relative positional attention
    rel_multi_head.py    # Relative multi-head attention
    __init__.py          # Network factory functions
  envs/
    __init__.py          # get_env() dispatcher — routes env name to constructor
    jax/                 # JAX-native environments
      tmaze.py           # T-Maze (classic memory diagnostic)
      rocksample.py      # RockSample 11x11/15x15
      battleship.py      # Battleship 10x10
      pocman.py          # Partially observable Pac-Man
      simple_chain.py    # Simple chain (minimal memory diagnostic)
      compass_world.py   # Compass navigation
      fishing.py         # Fishing domain
      reacher_pomdp.py   # Reacher with missing state
      navix_mazes.py     # DMLab-style Minigrid mazes (3 difficulty levels)
    wrappers/
      gymnax.py          # Gymnax compatibility wrapper
      gymnasium.py       # Gymnasium compatibility wrapper
      nx.py              # Navix compatibility wrapper
      observation.py     # Observation transformations
      pixel.py           # Pixel rendering wrapper (incl. craftax 10px→3px downscale)
    configs/             # Per-environment YAML configs
  utils/
    sweep.py             # Hyperparameter sweep logic (grid/random)
    file_system.py       # Results I/O
    plot.py              # Plotting
    video.py             # Video recording
    grid.py              # Grid utilities
tests/
  test_born_rule_actor.py  # BornRuleActor: shapes, masking, gradients, integration
  test_eunn.py             # EUNNCell unitarity, gradients, shapes
  test_scanned_urnn.py     # ScannedURNN: dtypes, resets, Wirtinger gradients
  test_battleship.py       # Battleship env
  test_rocksample.py       # RockSample env
  ...
run_pobax.sbatch         # SLURM job script — runs one (method, env) combo
submit_pobax.sh          # Submits the full (method × env) matrix via sbatch
scripts/
  hyperparams/           # Tuned hyperparams per environment
  baselines/             # Baseline experiment scripts
  launching/             # SLURM/Onager job submission
  visualizations/        # Plotting scripts
```

## Architecture

### Network Pipeline

```
                                              ┌─ Actor Head ──→ Action Distribution
Observation → Embedding → [Memory module] ──→ │
                                              └─ Critic Head ─→ Value
```

For uRNN/EUNN with `--policy_head born`, the actor/critic paths split:
```
mem_out (complex64)
    ├── BornRuleActor: complex Dense(H→A) → log(|z|²+ε) → Categorical
    └── concat [h.real, h.imag] → Critic (real, 2H→H→1)
```

The `ActorCritic` module in `models/actor_critic.py` composes these:
- `embedding`: `nn.Dense` (vector obs) or `CNN` (pixel obs)
- `memory`: `ScannedRNN` (GRU), `ScannedURNN` (uRNN/EUNN) — omitted if `--memoryless`
- `actor`: `DiscreteActor`, `BornRuleActor` (Born-rule), or `ContinuousActor`
- `critic`: `Critic` (single or double for lambda-discrepancy)

### Memory Modules (models/network.py)

Two scanned wrappers, both using `nn.scan` to unroll over time axis:
- `ScannedRNN`: hardcoded GRU cell (`nn.GRUCell`). Carry is float32.
- `ScannedURNN`: wraps `URNNCell`, `LegacyURNNCell`, or `EUNNCell` (selected by `variant`).
  Carry is complex64. Episode resets via `jnp.where` against `initial_urnn_carry`.

### Hidden State Flow Through Training

1. **Rollout** (`algos/ppo.py`, `env_step`): Hidden state `hstate` is carried through `jax.lax.scan` over rollout steps. Resets on episode boundaries via `jnp.where`.
2. **Loss computation**: Network is re-run on the full trajectory for fresh log probs/values. Initial hidden state is taken from start of trajectory.
3. **No chunked BPTT**: Full rollout length = full BPTT length. No sequence chunking.

### Training Loop (algos/ppo.py)

```
runner_state = (train_state, env_state, obs, done, hstate, rng)
    |
    v
jax.lax.scan(env_step, runner_state, None, num_steps)
    |  - Forward pass: value, action, log_prob, hstate = agent.act(...)
    |  - Environment step
    |  - Store transition (no hstate stored in transition)
    v
GAE computation (reverse scan with done-masking)
    |
    v
PPO update (multiple epochs, minibatch shuffle)
    |  - Re-run network on trajectory
    |  - Clipped policy loss + clipped value loss + entropy bonus
    v
Repeat for num_updates
```

## Implemented Methods

1. **Recurrent PPO** (GRU memory) — `python -m pobax.algos.ppo`
2. **GTrXL PPO** (Gated Transformer-XL) — `python -m pobax.algos.transformer_xl`
3. **Unitary RNN (uRNN)** — drop-in complex-valued memory: `--memory_type urnn` (see section below)
4. **Tunable EUNN** — Jing et al. 2017 unitary cell: `--memory_type eunn` (see section below)
5. **QuRNN (Born-rule actor)** — complex actor head with Born-rule logits: `--policy_head born` (see section below)
6. **Lambda-discrepancy** — dual critic with different GAE lambdas: `--double_critic --ld_weight 0.1`
7. **Memoryless baseline** — `--memoryless`
8. **Perfect memory baseline** — `--perfect_memory` (fully observable)

### uRNN details

Port of `cleanrl_qrl_fork/cleanrl/urnn.py` (Arjovsky et al., W = D₃ R₂ F⁻¹ D₂ Π R₁ F D₁).
Lives in `pobax/models/network.py` as `URNNCell` / `LegacyURNNCell` wrapped by
`ScannedURNN`. Activated with `--memory_type urnn`; ignored otherwise.

- **Carry dtype**: `complex64`, shape `(num_envs, hidden_size)`. The scan output
  is the complex carry itself; the real-concat adapter
  `jnp.concatenate([h.real, h.imag], -1)` lives in `ActorCritic.__call__`
  (models/actor_critic.py), so downstream actor/critic Dense layers auto-adapt
  to `2*hidden_size` input features.
- **Two variants**: `--urnn_variant standard` (default; input-dependent
  D/R via Dense projections) or `legacy` (learnable-fixed D/R).
- **Other uRNN flags**: `--urnn_input_dense` (default True; adds a complex
  input projection each step, standard variant only), `--urnn_norm_scale`
  (default 1.0; scales the initial equal-superposition carry), `--urnn_perm_seed`
  (default 0; seeds the fixed permutation inside the unitary transform).
- **Dual-LR optimizer**: when `memory_type='urnn'`, `pobax/algos/ppo.py`'s
  `_build_optimizer` splits params by dtype — complex leaves use
  `--complex_lr` (default `[8e-5]`, matches torch), real leaves use `--lr`.
  Both schedules anneal linearly when `--anneal_lr` is on. A single
  `optax.clip_by_global_norm` wraps both groups — verified to handle
  complex grads correctly via `optax.global_norm`.
- **Initial carry**: `(√(norm_scale/(2H)) + i·√(norm_scale/(2H)))·1`, reset
  at every episode boundary via the `resets` flag inside the scan.
- **Only supports discrete action spaces** at present — the continuous actor
  path is untouched.

Example:
```bash
python -m pobax.algos.ppo --env tmaze_5 --memory_type urnn --platform gpu --n_seeds 5
python -m pobax.algos.ppo --env rocksample_11_11 --memory_type urnn --urnn_variant legacy
python -m pobax.algos.ppo --env tmaze_5 --memory_type urnn --lr 2.5e-4 1e-4 --complex_lr 8e-5 5e-5  # sweep both LRs
```

### EUNN details

Tunable Efficient Unitary Neural Network (Jing et al. 2017, [arXiv:1612.05231](https://arxiv.org/abs/1612.05231)).
Parameterizes `W = D · F_L · F_{L-1} · ... · F_1`, where each `F_k` is a block-diagonal of `H/2`
independent 2×2 unitary mixing units with alternating cyclic-shift permutations between layers.
`L` is a capacity hyperparameter — at `L=H` the construction provably covers full `U(H)`.
Lives in `pobax/models/network.py` as `EUNNCell`, wrapped by `ScannedURNN(variant='eunn')`.
Activated with `--memory_type eunn`.

- **Params** (all real angles; complex action on complex carry):
  - `angles`: `(H, L)` real — even rows are per-pair phase φ, odd rows are per-pair rotation θ.
  - `diag`: `(H,)` real — final diagonal phase applied as `h * exp(1j * D)`, same shape/role as `d1/d2/d3` in `LegacyURNNCell`.
  - `input_embed`: complex `nn.Dense(H, param_dtype=complex64, use_bias=False)`, glorot-init (reused from uRNN).
- **Unitary composition**: `EUNNCell._apply_unitary(angles, diag, h)` is a `@staticmethod` so
  tests can materialize `U` by feeding identity columns without init/apply. Layer composition uses `jax.lax.scan`
  over the `L` axis — O(1) trace size in `L`, JIT-friendly. Permutation between layers is a `jnp.roll(±1)`
  alternating by layer parity.
- **Carry + reset + real-concat adapter**: shares uRNN's machinery. Carry is complex64; reset uses
  `initial_urnn_carry` via `get_memory_initial_carry` (routes `('urnn','eunn')` to same init);
  `ActorCritic.__call__` concatenates `[mem_out.real, mem_out.imag]` for downstream heads.
- **Dual-LR optimizer**: same dtype-split path as uRNN; gate widened to `memory_type in ('urnn','eunn')`.
  Real `angles`/`diag` → `--lr`, complex `input_embed` kernel → `--complex_lr`.
- **Config**: `--eunn_capacity L` (default `2`, matches the paper's RNN recommendation). Typed as
  `int` (not `list[int]`) because `L` determines tensor shape `(H, L)` and is therefore not
  vmap-sweepable — sweep via separate program runs.
- **Ignored flags under `eunn`**: `--urnn_variant`, `--urnn_input_dense`, `--urnn_perm_seed`
  (EUNN uses its own alternating cyclic-shift permutation, not the uRNN seeded permutation).
  `--urnn_norm_scale` is reused for initial carry scaling.
- **Activation**: same `ModReLU` as uRNN.
- **Unit tests**: `tests/test_eunn.py` — float64 unitarity (tol 1e-10), float32 numerical sanity
  (tol 1e-3) at several `(H, L)` including `L=H`, gradient finiteness at the equal-superposition
  reset carry, shape/dtype end-to-end, and raises on odd `hidden_size`.

Example:
```bash
python -m pobax.algos.ppo --env tmaze_5 --memory_type eunn --platform gpu --n_seeds 5
python -m pobax.algos.ppo --env tmaze_5 --memory_type eunn --eunn_capacity 8 --hidden_size 64
```

### QuRNN (Born-rule actor) details

Quantum-inspired actor head ported from `cleanrl_qrl_fork/cleanrl/ppo_atari_qurnn.py`.
Instead of converting the complex hidden state to real via `[h.real, h.imag]` concatenation
and feeding it to a standard MLP actor, the Born-rule actor operates directly on the complex
hidden state and uses `log(|z|² + ε)` to produce real-valued logits.

Lives in `pobax/models/discrete.py` as `BornRuleActor`. Activated with `--policy_head born`.

- **Architecture**: single complex `nn.Dense(hidden_size → action_dim, param_dtype=complex64)`.
  The complex linear map `z = W·h + b` preserves phase interference patterns; the Born rule
  `log(|z|² + ε)` extracts real logits encoding those patterns as action preferences.
  Optional intermediate complex hidden layer via `complex_hidden_size` constructor arg
  (adds complex Dense → ModReLU before the output projection; default: None = single layer).
- **Initialization**: `_glorot_complex_small(scale=0.01)` — Glorot-uniform scaled by 0.01,
  matching the torch reference's `xavier_uniform_(gain=0.01)`. Produces near-uniform initial
  policy (all `|z|²` values close to zero → similar logits). Verified that Flax Linen's
  `glorot_uniform(dtype=complex64)` correctly initializes both real and imaginary parts
  (unlike NNX which only initializes the real part).
- **Critic**: unchanged — still receives `concat([h.real, h.imag])` (shape `2*hidden_size`).
  Only the actor path changes.
- **Embedding split** in `ActorCritic.__call__` (models/actor_critic.py):
  when `policy_head='born'`, the actor receives raw complex `mem_out` while the critic
  receives the real-concat embedding. When `policy_head='standard'` (default), both
  receive real-concat (existing behavior).
- **Optimizer**: complex Dense params in `BornRuleActor` are automatically routed to
  `complex_lr` via the existing dtype-based `label_fn` in `_build_optimizer`. No changes needed.
- **Constraints**: requires `memory_type in ('urnn','eunn')`, `is_discrete=True`.
  Validated at config parse time (`process_args`) and at module construction (`setup`).
- **Only supports discrete action spaces** — continuous Born-rule theory is not developed.
- **Unit tests**: `tests/test_born_rule_actor.py` — shapes, action masking, near-uniform init,
  complex hidden layer, full ActorCritic integration, gradient sanity (including z=0 edge case),
  config validation, complex dtype routing, and Glorot complex initializer verification.

Example:
```bash
python -m pobax.algos.ppo --env tmaze_10 --memory_type urnn --policy_head born --platform gpu --n_seeds 5
python -m pobax.algos.ppo --env rocksample_11_11 --memory_type eunn --policy_head born --eunn_capacity 4
```

## Environments

All discrete action spaces unless noted:

| Environment | Type | Key Memory Challenge |
|---|---|---|
| T-Maze | Diagnostic | Remember cue at start of hallway |
| Simple Chain | Diagnostic | Minimal memory requirement |
| RockSample 11x11/15x15 | Object uncertainty | Track rock quality beliefs |
| Battleship 10x10 | Object uncertainty | Track hit/miss history |
| PocMan | Partial observability | Pac-Man with limited vision |
| Compass World | Navigation | Track orientation |
| Navix Mazes (3 difficulties) | Spatial uncertainty | DMLab-style grid navigation — closest JAX proxy to Memory Maze |
| Masked MuJoCo (Walker, Ant, Hopper, HalfCheetah) | Feature masking | Continuous control with hidden state dims |
| Visual MuJoCo | Visual occlusion | Pixel-based continuous control |
| Craftax (no-inventory) | Complex multi-object | Requires Craftax dependency |

## Running Experiments

```bash
# Basic recurrent PPO
python -m pobax.algos.ppo --env tmaze_5 --hidden_size 128 --platform gpu --n_seeds 5

# GTrXL
python -m pobax.algos.transformer_xl --env rocksample_11_11 --platform gpu

# With lambda-discrepancy
python -m pobax.algos.ppo --env battleship --double_critic --ld_weight 0.1

# Memoryless baseline
python -m pobax.algos.ppo --env tmaze_5 --memoryless

# Hyperparameter sweep (grid search over list-valued args)
python -m pobax.algos.ppo --env tmaze_5 --lr 0.001 0.0001 --lambda0 0.9 0.95
```

Best hyperparameters for each environment are in `scripts/hyperparams/<env>/best/`.

## Key Config Parameters (config.py)

| Parameter | Default | Purpose |
|---|---|---|
| `num_steps` | 128 | Rollout length (= BPTT length) |
| `num_epochs` | 50 | Total training epochs |
| `update_epochs` | 4 | PPO epochs per update |
| `num_minibatches` | 4 | Minibatches per epoch |
| `hidden_size` | 128 | RNN hidden dimension |
| `lr` | [2.5e-4] | Learning rate (list = sweep) |
| `gamma` | 0.99 | Discount factor |
| `lambda0` | [0.95] | GAE lambda (critic 1) |
| `lambda1` | [0.5] | GAE lambda (critic 2, for LD) |
| `ld_weight` | [0.0] | Lambda-discrepancy loss weight |
| `clip_eps` | 0.2 | PPO clip epsilon |
| `vf_coeff` | [0.5] | Value function loss weight (list = sweep) |
| `entropy_coeff` | [0.01] | Entropy bonus weight |
| `memory_type` | `'gru'` | Memory module: `'gru'`, `'urnn'`, or `'eunn'` |
| `urnn_variant` | `'standard'` | uRNN cell flavor: `'standard'` or `'legacy'` (ignored for EUNN) |
| `urnn_input_dense` | True | Add complex input-embed inside uRNN (standard only) |
| `urnn_norm_scale` | 1.0 | Scales the initial complex carry (shared uRNN + EUNN) |
| `urnn_perm_seed` | 0 | Seed for uRNN fixed permutation (ignored for EUNN) |
| `eunn_capacity` | 2 | EUNN capacity `L`. Scalar `int` (not vmap-sweepable; determines tensor shape) |
| `complex_lr` | [8e-5] | LR for complex params under `memory_type` in `{'urnn','eunn'}` |
| `policy_head` | `'standard'` | Actor head: `'standard'` (MLP) or `'born'` (Born-rule complex actor) |
| `n_seeds` | 5 | Seeds per config |
| `total_steps` | 1.5e6 | Total environment steps |

## SLURM Submission Infrastructure

`submit_pobax.sh` iterates a (method × env) matrix, submitting one `sbatch` job per combo
via `run_pobax.sbatch`. Each job reserves one GPU and runs N_SEEDS vmapped inside a single
JAX process.

- **Methods**: `urnn_standard`, `urnn_legacy`, `gru`, `ppo_ld`, `eunn_<L>` (e.g. `eunn_2`).
  The method string encodes variant + capacity; `run_pobax.sbatch` parses it in a `case` switch.
- **Hyperparams**: `BASE[env]` and `LD[env]` associative arrays in `submit_pobax.sh` hold
  per-env tuned values. PPO-LD overrides lr/lambda from `LD[env]`.
- **Auto-skip**: jobs with existing `results/<study>/` subdirs are skipped (FORCE=1 overrides).
- **Log frequency thinning**: craftax envs use `steps_log_freq=16, update_log_freq=16` to
  avoid ~15GB metric arrays from 100M-step runs.
- **Env vars**: `STUDY_NAME` override, `COMPLEX_LR`, `SEED_BASE`, `N_SEEDS` all configurable.
- **Custom jobs**: set env vars as shell prefix with `--export=ALL` (inline `--export="ALL,KEY=VAL"`
  syntax can fail to set vars).

## Notes for Extending

### Adding a new recurrent cell
1. Write the cell conforming to Flax `nn.RNNCellBase`: `__call__(carry, inputs) -> (new_carry, output)` + `initialize_carry(rng, input_shape)`
2. Modify `ScannedRNN` in `models/network.py` to accept a cell type parameter, or create a new `ScannedXxx` class
3. Add a `--memory_type` flag to `PPOHyperparams` in `config.py`
4. Branch on it in `ActorCritic.setup()` in `models/actor_critic.py`
5. If complex-valued: add carry init to `get_memory_initial_carry()` in `models/network.py`,
   and the dtype-split optimizer in `_build_optimizer()` already handles complex params
6. Training loop, GAE, evaluation — all untouched

### Adding a new environment
1. Implement as a `gymnax.Environment` subclass in `pobax/envs/jax/`
2. Register in `get_env()` in `pobax/envs/__init__.py`
3. Add hyperparameter configs in `scripts/hyperparams/<env>/`
4. For SLURM runs: add `BASE[env]` (and `LD[env]` if using PPO-LD) entries in `submit_pobax.sh`
