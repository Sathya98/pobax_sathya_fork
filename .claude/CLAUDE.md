# POBax - Partial Observability Benchmarks in JAX

**Paper:** "Benchmarking Partial Observability in Reinforcement Learning with a Suite of Memory-Improvable Domains" (RLC 2025)
**Authors:** Ruo Yu (David) Tao, Kaicheng Guo, Cameron Allen, George Konidaris
**OpenReview:** https://openreview.net/forum?id=HUTCbYOW5E

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
posebax/
  config.py              # All hyperparameters (PPOHyperparams class via TAP)
  definitions.py         # Project root path definitions
  algos/
    ppo.py               # Recurrent PPO — main training loop
    transformer_xl.py    # GTrXL variant of PPO
    run_helper.py        # Training utilities (logging, checkpointing)
  models/
    actor_critic.py      # ActorCritic module — composes embedding + memory + heads
    network.py           # ScannedRNN (GRU), CNN, SimpleNN, SmallImageCNN, FullImageCNN
    discrete.py          # DiscreteActor, DiscreteActorCriticTransformer
    continuous.py        # ContinuousActor
    value.py             # Critic (value function head)
    embedding.py         # Embedding/preprocessing networks
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
      pixel.py           # Pixel rendering wrapper
    configs/             # Per-environment YAML configs
  utils/
    sweep.py             # Hyperparameter sweep logic (grid/random)
    file_system.py       # Results I/O
    plot.py              # Plotting
    video.py             # Video recording
    grid.py              # Grid utilities
scripts/
  hyperparams/           # Tuned hyperparams per environment
    rocksample/best/     # Best configs for RockSample
    tmaze/best/
    battleship/best/
    navix/best/
    masked_mujoco/best/
    visual_mujoco/
    craftax/
    pocman/best/
  baselines/             # Baseline experiment scripts
  launching/             # SLURM/Onager job submission
  visualizations/        # Plotting scripts
```

## Architecture

### Network Pipeline

```
Observation -> Embedding -> [ScannedRNN] -> Actor Head -> Action Distribution
                                         -> Critic Head -> Value
```

The `ActorCritic` module in `models/actor_critic.py` composes these:
- `embedding`: `nn.Dense` (vector obs) or `CNN`/`SmallImageCNN`/`FullImageCNN` (pixel obs)
- `memory`: `ScannedRNN` (GRU-based) — omitted if `--memoryless`
- `actor`: `DiscreteActor` or `ContinuousActor`
- `critic`: `Critic` (single or double for lambda-discrepancy)

### ScannedRNN (models/network.py)

The core recurrent module. Uses `nn.scan` to unroll over time axis:
```python
class ScannedRNN(nn.Module):
    hidden_size: int
    # Uses nn.scan with variable_broadcast="params", in_axes=0, out_axes=0
    # Input: (ins, resets) — resets flags trigger hidden state zeroing at episode boundaries
    # Cell: nn.GRUCell(features=self.hidden_size) — hardcoded to GRU
    # Returns: (new_rnn_state, y)
```

**Important:** The cell is hardcoded to GRU. To add a new cell type, modify this class.

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

1. **Recurrent PPO** (GRU memory) — `python -m posebax.algos.ppo`
2. **GTrXL PPO** (Gated Transformer-XL) — `python -m posebax.algos.transformer_xl`
3. **Lambda-discrepancy** — dual critic with different GAE lambdas: `--double_critic --ld_weight 0.1`
4. **Memoryless baseline** — `--memoryless`
5. **Perfect memory baseline** — `--perfect_memory` (fully observable)

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
python -m posebax.algos.ppo --env tmaze_5 --hidden_size 128 --platform gpu --n_seeds 5

# GTrXL
python -m posebax.algos.transformer_xl --env rocksample_11_11 --platform gpu

# With lambda-discrepancy
python -m posebax.algos.ppo --env battleship --double_critic --ld_weight 0.1

# Memoryless baseline
python -m posebax.algos.ppo --env tmaze_5 --memoryless

# Hyperparameter sweep (grid search over list-valued args)
python -m posebax.algos.ppo --env tmaze_5 --lr 0.001 0.0001 --lambda0 0.9 0.95
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
| `entropy_coeff` | [0.01] | Entropy bonus weight |
| `n_seeds` | 5 | Seeds per config |
| `total_steps` | 1.5e6 | Total environment steps |

## Notes for Extending

### Adding a new recurrent cell
1. Write the cell conforming to Flax `nn.RNNCellBase`: `__call__(carry, inputs) -> (new_carry, output)` + `initialize_carry(rng, input_shape)`
2. Modify `ScannedRNN` in `models/network.py` to accept a cell type parameter, or create a new `ScannedXxx` class
3. Add a `--memory_type` flag to `PPOHyperparams` in `config.py`
4. Branch on it in `ActorCritic.setup()` in `models/actor_critic.py`
5. Training loop, GAE, evaluation — all untouched

### Adding a new environment
1. Implement as a `gymnax.Environment` subclass in `posebax/envs/jax/`
2. Register in `get_env()` in `posebax/envs/__init__.py`
3. Add hyperparameter configs in `scripts/hyperparams/<env>/`
