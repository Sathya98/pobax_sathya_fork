# Phase Probing Analysis for Complex-Valued Recurrent Policies

**Date:** 2026-05-07
**Author:** Sathya (with Claude)
**Status:** Design — pending implementation
**Primary env:** Battleship 10×10
**Secondary env:** T-Maze (sanity check)
**Architectures probed:** GRU (control), uRNN-standard, uRNN-legacy, EUNN-2, EUNN-8

## 1. Motivation

The POBax paper claims complex-valued recurrent policies (uRNN, EUNN) and the
Born-rule actor (QuRNN) are useful inductive biases for memory in POMDPs.
Reward curves alone leave the *mechanism* unexplained — we don't know what the
complex hidden state actually carries, and in particular what the **phase** of
each component encodes about the agent's belief over latent state.

The synchrony / oscillator literature (Reichert & Serre 2014; Löwe et al.
2022, 2023; Stanić et al. 2023; Miyato et al. 2024) hypothesizes that in
complex-valued networks **magnitude carries presence and phase carries
identity / grouping / a wrapped accumulator**. This design tests that
hypothesis in an RL setting where the latent variables to be tracked are
unambiguous: which cells of a Battleship grid have been fired at and what
their outcomes were.

A key structural argument makes Battleship + EUNN especially well-posed: the
EUNN cell is exactly norm-preserving, so any monotonic accumulator (number of
hits, number of shots, episode progress) **cannot live in `‖h‖`** — it must
live in phase or nowhere. The standard uRNN with `--urnn_input_dense` breaks
strict norm preservation, so it has more flexibility; comparing the two is
itself an experimental axis.

## 2. Research questions

1. **Decodability.** Are concrete belief-state attributes — number of hits,
   identity of hit cells, fraction of ships sunk — linearly decodable from the
   complex hidden state? With what selectivity over a magnitude-only baseline?
2. **Phase coding.** Does the agent assign distinguishable phases to
   distinguishable latent events (cell identities, hit/miss outcomes), in the
   sense of the Reichert/Löwe binding hypothesis?
3. **Causal use.** Are the phases the agent computes actually used by the
   policy, or are they incidental? Test via phase-shift interventions.
4. **Architecture comparison.** Does the EUNN's strict unitarity force more
   structured phase coding than the uRNN-standard? Do both encode more in
   phase than a GRU encodes in any rotation-like coordinate?

A null result (phase carries nothing decodable beyond magnitude) is also
publishable: it would mean unitary RNNs in RL don't realize the
synchrony-binding hypothesis, and the gains over GRU come from the implicit
regularization of unitarity rather than from learned phase structure.

## 3. Environments and ground-truth attributes

### 3.1 Battleship 10×10 (primary)

State (`BattleShipState` in `pobax/envs/jax/battleship.py`):
- `hits_misses` ∈ {0,1,2}^{10×10}: 0 = unfired, 1 = miss, 2 = hit
- `board` ∈ {0,1}^{10×10}: ground-truth ship positions (hidden from agent)

Probeable attributes (computed deterministically from state + action history):
- **A1 (scalar counts):** `n_hits_t`, `n_misses_t`, `n_total_shots_t`,
  `n_remaining_ship_cells_t`, `t` (step within episode).
- **A2 (vector binary):** 100-dim hit-grid `H_t ∈ {0,1}^{100}`, miss-grid
  `M_t ∈ {0,1}^{100}`, fired-grid `F_t = H_t ∨ M_t`.
- **A3 (action-history):** 100-dim one-hot of last action; categorical of
  most-recent-hit cell index.
- **A4 (latent / hidden from agent):** ship positions `board`, number of
  ships remaining unsunk, expected hit probability over remaining cells under
  the analytic posterior. These let us check whether the agent's hidden state
  encodes the *Bayesian posterior*, not just the *observed* history.

Why this env: belief is binary-per-cell with no transitions, no spatial
movement. The only thing the memory must carry is the bag of past
(action, outcome) pairs. This isolates memory from policy/transition
structure cleanly.

### 3.2 T-Maze (sanity check, secondary)

State (`TMazeState`): `grid_idx`, `goal_dir`.

Probeable attributes:
- **A1:** `goal_dir ∈ {0,1}` (the cue, observable only at `grid_idx == 0`).
- **A2:** `grid_idx` (position along hallway).
- **A3:** `t` (step count).

Use case: 1-bit cue is the textbook memory diagnostic. If we cannot decode
`goal_dir` from a uRNN that achieves >95% return on T-Maze 10, the rest of
the analysis is suspect. T-Maze is the canary, Battleship is the science.

## 4. Hypotheses (numbered, falsifiable)

- **H1 — Cumulative-count phase code.** For trained EUNN-Battleship, ≥1
  hidden unit `u*` exists such that circular-linear regression of `n_hits_t`
  on `(cos φ_{u*}, sin φ_{u*})` yields `R_{cl} ≥ 0.6` averaged across
  episodes. *Falsified if* no unit achieves `R_{cl} ≥ 0.3` above the
  random-init-EUNN control.
- **H2 — Cell-identity binding.** For trained EUNN-Battleship, the
  per-step phase shift `Δφ_u(t) = φ_u(h_t) − φ_u(h_{t-1})` is statistically
  dependent on the cell index of the latest action: a multinomial probe
  `cell_index → Δφ` achieves accuracy ≥ 2× chance (`2/100`). *Falsified if*
  not significantly above random-init-EUNN control after Bonferroni
  correction.
- **H3 — Selective phase decoding.** For the 100-dim hit grid `H_t`,
  per-cell macro-F1 of a phase-only probe `(cos φ, sin φ) → H_t` exceeds
  magnitude-only probe by ≥ 0.10 on held-out episodes. *Falsified if* phase
  selectivity ≤ 0.02.
- **H4 — Causal phase use.** Multiplying the hidden state by `e^{iΔφ}` on a
  small subset (top-K by H1/H3) of units mid-episode produces a measurable
  policy KL of ≥ 0.5 nats and a return drop of ≥ 30% at `Δφ = π`. Random
  units rotated by the same `Δφ` produce ≤ 30% of that effect. *Falsified*
  if either threshold fails.
- **H5 — Architecture differentiation.** EUNN-Battleship phase-only-probe
  selectivity strictly exceeds uRNN-standard's, which strictly exceeds GRU's
  "phase" (defined as `arctan2(h_{2k+1}, h_{2k})` for adjacent real pairs —
  expected to be near-random for GRU).

We expect H1 and H3 to hold, H2 to be the most uncertain (cell identity may
be distributed over many units rather than concentrated in any single unit's
phase shift), H4 to require some search over `K` and `Δφ`, and H5's GRU vs.
uRNN comparison to be cleaner than uRNN vs. EUNN.

## 5. Methodology

### 5.1 Phase 0 — training & checkpointing

Reuse existing best-hparam configs from `scripts/hyperparams/battleship/best/`
and `scripts/hyperparams/tmaze_*/best/`. Train (or reuse if available) one
checkpoint per `(env × architecture × seed)`:

- envs: `battleship_10x10`, `tmaze_10`, `tmaze_5`
- architectures: `gru`, `urnn_standard`, `urnn_legacy`, `eunn_2`, `eunn_8`
- seeds: 5 each (the existing convention)
- save final params **and** params at 25%, 50%, 75% of training to allow
  developmental analysis if interesting (low priority — only if Phase 1
  results suggest mid-training is more informative).

For each architecture we also train one **random-init control** (no
gradient updates, just initial params) per seed. This is the dominant
control in all probes (Section 6).

Storage: `results/phase_probing/checkpoints/<env>/<arch>/seed_<i>/params.pkl`.

### 5.2 Phase 1 — trajectory dataset generation

New script `scripts/phase_probing/collect_rollouts.py`. For each checkpoint:

- run 200 evaluation episodes with deterministic policy
  (argmax on `pi.logits`)
- run 200 evaluation episodes with stochastic policy (sample) — for some
  probes we need entropy in the action distribution
- per timestep, store: `obs`, `action`, `reward`, `done`, ground-truth env
  state (`hits_misses`, `board`, `goal_dir`, `grid_idx`), **and the full
  complex hidden state `h_t ∈ ℂ^{128}`** for uRNN/EUNN architectures, the
  real hidden state for GRU.

Use `jax.lax.scan` over rollouts (already the pattern in the training loop)
and dump to a flat HDF5 (or `.npz`) per checkpoint:
`results/phase_probing/rollouts/<env>/<arch>/seed_<i>/{deterministic,stochastic}.h5`.

Approximate size: 200 episodes × ~70 steps avg × 128-dim complex64 ≈ 14 MB
per checkpoint per policy. Whole sweep ≈ 5 archs × 5 seeds × 2 policies ×
14 MB ≈ 700 MB — fine.

Modify the rollout pass to also expose the **post-step hidden state and
pre-step hidden state** so we can compute `Δh_t` for the cell-binding probe
(H2). Confirmed feasible: the existing PPO scan's carry already holds
`hstate`, just needs to be threaded into the per-step output rather than
discarded.

### 5.3 Phase 2 — probes

Run from `scripts/phase_probing/run_probes.py`. All probes are fit on
80%/20% train/test split *over episodes* (not over timesteps within an
episode — within-episode timesteps are correlated). Report mean ± std over
5 seeds.

#### Probe P1 — count-in-phase (H1)

For each unit `u ∈ {1..128}`:
1. Form `(φ_u, n)` pairs across all (episode, t).
2. Fit circular-linear regression `n ∼ a cos φ_u + b sin φ_u + c`. Report
   the circular-linear correlation coefficient `R_{cl}` (Jammalamadaka &
   Sarma, equivalently `corr(n, cos φ) ⊕ corr(n, sin φ)`).
3. Repeat for each of `{n_hits, n_misses, n_total_shots, t}`.
4. Identify top-K units per attribute. Plot `(n, φ_u)` as a phase-vs-count
   scatter for the top-1 unit; expect a wrapped linear trend.

Output: 4 attributes × 5 archs × 5 seeds → 100 numbers, summarized in a
heatmap (`unit × attribute`, color = `R_{cl}`) per arch; bar chart of
`max_u R_{cl}` per arch with random-init control.

#### Probe P2 — cell-identity binding (H2)

For each timestep `t` with action = fire-at-cell `(r, c)`:
1. Compute `Δφ_t ∈ ℝ^{128}` = wrapped phase difference `h_t − h_{t-1}`.
2. Fit a multinomial logistic probe `Δφ_t → (r, c) ∈ {1..100}`. Use both
   linear and 1-hidden-layer MLP variants.
3. Restrict to non-trivial timesteps (action history excludes "no
   movement" / illegal-action mask outputs).
4. Also fit a probe **conditioned on outcome** — separate models for
   hit-actions vs. miss-actions — to test whether the binding code carries
   `(cell, outcome)` jointly.

Output: probe accuracy + chance baseline + random-init-EUNN baseline + MDL
description length.

#### Probe P3 — hit-grid decoding (H3)

For each timestep, target `H_t ∈ {0,1}^{100}`. Fit four probes:
- (a) **Magnitude-only**: features `|h_t| ∈ ℝ^{128}`.
- (b) **Phase-only**: features `(cos φ_t, sin φ_t) ∈ ℝ^{256}` after
  excluding units with `|h_t| < ε` (phase ill-defined when magnitude is
  near-zero). Default `ε = 1e-3`; report sensitivity at `1e-2` and `1e-4`.
- (c) **Real-concat**: features `(Re h_t, Im h_t) ∈ ℝ^{256}` (this is what
  the policy/value heads actually see).
- (d) **Raw obs**: `obs_t` only — no memory at all. Lower bound.

Each probe: linear ridge classifier (multi-label, one-vs-rest), then 2-layer
MLP. Report **per-cell macro-F1**, **probe accuracy**, and **MDL** (Voita &
Titov 2020 — online code length, simpler to implement than variational).

The selectivity numbers we care about:
- `S_phase = F1(b) − F1(a)` (does phase add over magnitude?)
- `S_total = F1(c) − F1(d)` (does memory add over raw obs?)
- `S_phase_share = S_phase / S_total` (what fraction of memory's
  contribution lives in phase?)

Repeat for the perfect-information attribute `H_t` (observed) and the
**unobserved** attribute `board_t` (ground-truth ship layout). Decoding
`board_t` would mean the agent's hidden state encodes the Bayesian
*posterior over ship positions*, not just the observed event history —
that's a much stronger claim.

#### Probe P4 — causal phase intervention (H4)

For each architecture × seed:
1. Identify top-K units by P3 phase-importance — units whose phase, when
   ablated, drops phase-only probe macro-F1 on the **hit-grid `H_t`**
   target the most (leave-one-out over the 128 units; sweep `K ∈ {1, 4,
   16, 32}`).
2. Replay 200 rollouts. At a randomly chosen step `t* ∈ [10, T-5]`, multiply
   the hidden state on those K units by `e^{iΔφ}` for
   `Δφ ∈ {π/4, π/2, π, 3π/2}`. Continue the rollout from there.
3. Measure (i) policy KL between perturbed and unperturbed action
   distributions at `t*` and the next 5 steps, (ii) episode return relative
   to control.
4. Run the same protocol with K *random* units as the control. The signal
   is `effect(top-K) − effect(random-K)` as a function of `Δφ`.

If H4 holds we expect a periodic (in `Δφ`) response with maximum at `π`,
matching the unitary group's structure.

#### Probe P0 (sanity) — T-Maze cue decoding

For `tmaze_10` × uRNN/EUNN: probe `goal_dir ∈ {0,1}` from `h_t` for `t > 0`
(after the cue has scrolled out of the observation). If `goal_dir` is not
≥95% decodable from `h_t` at the junction, something is broken upstream
and the Battleship analysis isn't trustworthy.

### 5.4 Phase 3 — cross-architecture comparison

Once P1-P4 numbers are in, produce two summary tables:
- Table 1: probe selectivity per architecture (rows: GRU, uRNN-standard,
  uRNN-legacy, EUNN-2, EUNN-8; cols: P1 max-`R_{cl}`, P2 acc, P3
  `S_phase`, P4 KL@π).
- Table 2: same rows, columns: random-init control values for each
  metric. Selectivity = Table 1 − Table 2.

The headline figure is "EUNN concentrates X% of memory's decodable belief
in phase, vs. uRNN-standard at Y%, vs. GRU at Z%."

## 6. Metrics & controls

### 6.1 Metrics

- **Probe accuracy / macro-F1** for classification attributes.
- **Circular-linear correlation `R_{cl}`** for `(scalar, phase)` pairs.
  Reference: Berens 2009 `CircStat`. We'll port the relevant function or
  use `pingouin.circ_corrcl`.
- **Mean resultant length `R = |⟨e^{iφ}⟩|`** per unit per attribute level —
  measures phase concentration (close to 1 → phase tightly clustered).
- **MDL probe (Voita & Titov 2020)** — online code length, reported in
  bits. Distinguishes "linearly trivial" from "extractable with effort."
  Use the online prequential coding variant; reference implementation:
  github.com/lena-voita/description-length-probing.
- **Policy KL** for interventions: `KL(π_perturbed ‖ π_unperturbed)`.

### 6.2 Controls (mandatory, not optional)

- **C1 — Random-init agent** — full `ActorCritic` network (embedding +
  memory + heads) initialized with the same seed protocol but never
  trained. Crucial: a uRNN's fixed unitary input embedding can already
  linearly decode count (rotation = accumulator). The "learning encodes
  belief" claim needs the trained agent to clear this baseline. Roll out
  with the random-init policy (returns will be near zero — that's fine,
  the probe targets are computed from env state, not from rewards).
- **C2 — Magnitude-only baseline** (already P3 (a)).
- **C3 — Raw-obs baseline** (already P3 (d)).
- **C4 — Shuffled labels** within-attribute, across-episode. Estimates
  probe overfitting under no signal.
- **C5 — Memoryless agent** (`--memoryless` flag, already supported). Probe
  accuracy here gives us the ceiling for "what's decodable from the
  current obs alone," matching C3 conceptually but with the agent's actual
  embedding.
- **C6 — GRU agent**. The "phase" definition for GRU is the angle of
  adjacent-pair coordinates `(h_{2k}, h_{2k+1})` — included to demonstrate
  that probe selectivity attributable to *learned* phase structure
  vanishes for a real-valued architecture.

### 6.3 Statistical reporting

- All metrics reported as mean ± std over 5 training seeds.
- Significance tests: paired t-test or Wilcoxon signed-rank on per-seed
  selectivity numbers when comparing two architectures. Bonferroni-correct
  across the 4 architecture pairs we care about.
- Distinguish "decodable" (probe acc above C4) from "selectively phase-coded"
  (selectivity above C2 *and* C1).

## 7. Implementation plan (high-level — detailed plan to follow via writing-plans)

```
scripts/phase_probing/
  collect_rollouts.py         # Phase 1: dump (obs, act, rew, h_t, state)
  run_probes.py               # Phase 2: P0-P4 probes
  intervene.py                # Phase 2/P4: phase-shift causal probe
  metrics/
    circular_stats.py         # R_cl, mean resultant length, von Mises MLE
    mdl_probe.py              # online prequential MDL
  plots/
    phase_count_scatter.py    # H1 figure
    phase_heatmap.py          # unit × attribute R_cl heatmap
    selectivity_bars.py       # cross-architecture summary
  configs/
    battleship_10x10.yaml     # arch list, seeds, n_episodes
    tmaze_10.yaml
```

Key engineering changes outside `scripts/phase_probing/`:
- `pobax/algos/ppo.py`: expose `hstate_pre, hstate_post` per timestep during
  evaluation rollouts (small change — already in scan carry).
- `pobax/models/network.py`: add a one-line method on `ScannedURNN` to
  re-inject a perturbed carry `h * e^{iΔφ}` for the intervention probe.
- Add tests for circular-linear correlation against `pingouin` reference
  values and for MDL probe against a known toy distribution.

No changes needed to `models/actor_critic.py`, `algos/ppo.py` training
path, environment files, or any uRNN/EUNN cell — the probe pipeline reads
checkpoints and rolls out, it doesn't retrain.

## 8. Deliverables

1. Per-arch heatmaps (`unit × attribute`, color = `R_{cl}`) — one per
   `(env, arch)`. Identifies which units carry which counters in their
   phases.
2. Phase-vs-count scatter for the top-`R_{cl}` unit per env and arch —
   the "phase encodes a counter" qualitative figure.
3. Selectivity table (Section 5.4 Table 1) — the headline numerical
   result.
4. Phase-shift response curve (return / KL vs `Δφ` for top-K vs random-K)
   per env per arch — the causal-use figure.
5. T-Maze sanity-check probe accuracy on `goal_dir` (one number per arch).
6. A 2-3 page section draft (markdown) summarizing findings, suitable for
   inclusion as a follow-up to the RLC submission or a workshop paper on
   complex-valued representation interpretation.

## 9. Risks & open questions

- **Risk: phase signal is distributed, not unit-localized.** Per-unit
  probes (P1, P2) might miss real structure. Mitigation: also fit
  *population* probes (full `(cos φ, sin φ) ∈ ℝ^{256}`) — that's already
  P3. If P3 selectivity is high but P1/P2 are flat, the conclusion is
  "phase encodes belief but distributed over units." Still a positive
  result.
- **Risk: training instability for EUNN-8.** Higher capacity may not
  converge as cleanly on Battleship. Mitigation: keep EUNN-2 as the
  primary EUNN comparison; treat EUNN-8 as a stretch goal.
- **Risk: Born-rule actor (`--policy_head born`) confounds the comparison.**
  We should fix actor head across all archs in the main comparison —
  probably standard MLP — and run a *separate* born-vs-standard ablation
  on EUNN only. This isolates the phase question from the actor-head
  question.
- **Risk: rollout determinism.** For interventions (P4), we need
  reproducible rollouts to compare perturbed vs. unperturbed. The
  evaluation rng plus environment rng both need to be seeded and
  duplicated across the two runs. Standard JAX practice but worth
  flagging.
- **Open: should we do a developmental analysis** (probes at 25/50/75/100%
  training)? Defer until Phase 1 results — only worth it if phase
  structure changes substantially over training.
- **Open: should we extend to RockSample as a third env?** The Bayesian
  belief story is more attractive there but the analytic posterior takes
  more work to compute. Stretch goal for a paper, scope-cut for a
  workshop.

## 10. Out of scope

- **Born-rule actor analysis.** The phase-encoded belief question is
  separable from "does the Born actor *use* the phase well." That's a
  follow-up after we know what the phases mean.
- **Continuous-control envs.** uRNN/EUNN in this repo only support
  discrete actions. Continuous envs aren't probed.
- **Craftax.** Belief-state ground truth in Craftax is too high-dimensional
  and noisy to define clean probe targets. Possible future extension once
  Battleship + T-Maze methodology is validated.
- **Theoretical analysis of EUNN dynamical systems.** Sussillo-style
  fixed-point analysis, eigenmode decomposition of the recurrent map —
  interesting but a separate workstream.
- **Re-training with auxiliary phase losses.** This design strictly
  *probes* existing checkpoints, doesn't modify training objectives.

## 11. Effort estimate

- Phase 0 (training/reusing checkpoints): 0.5 day if best-hparams runs
  exist, 2-3 days if we need to retrain.
- Phase 1 (rollout dataset generation): 1 day. Mostly a refactor of the
  existing eval loop to dump hidden states.
- Phase 2 (probes P0-P4): 4-5 days.
- Phase 3 (cross-architecture comparison + figures): 2 days.
- Writing the section: 1-2 days.

**Total: ~1.5-2 weeks of focused work**, parallelizable across probes once
Phase 1 is done.

## 12. Success criteria

This analysis is a success if at least *one* of the following is true at
the end:

(a) **Positive interpretability result.** H1 + H3 + at least partial H4
hold for EUNN-Battleship, with sharp selectivity over magnitude-only and
random-init controls. We have a publishable claim that "EUNN's hidden
state phase linearly encodes the agent's belief over hit cells, and this
phase code is causally used by the policy."

(b) **Calibrated null result.** We rigorously rule out (a) — phase carries
no decodable information beyond magnitude — and the gains EUNN/uRNN show
on the reward curves come from the implicit regularization of unitarity,
not from learned phase structure. This is an honest negative result that
informs how the field should think about complex-valued recurrence in RL.

(c) **Mixed result, well-characterized.** Phase encodes counters cleanly
(H1) but doesn't encode cell identity (H2 fails). This is the most
informative outcome — it tells us which aspects of the synchrony-binding
hypothesis transfer to RL and which don't.

What would make this analysis a *failure* (and we'd discard it before
submission): if probes are uninterpretable noise, controls aren't
properly matched, or no signal survives the random-init baseline. We'll
know this at the end of Phase 2.

## 13. References

- Reichert & Serre 2014 — Neuronal Synchrony in Complex-Valued Deep Networks. arXiv:1312.6115.
- Löwe et al. 2022 — Complex-Valued Autoencoders for Object Discovery. arXiv:2204.02075.
- Löwe et al. 2023 — Rotating Features for Object Discovery. arXiv:2306.00600.
- Stanić et al. 2023 — Contrastive Training of Complex-Valued Autoencoders. arXiv:2305.15001.
- Miyato et al. 2024 — Artificial Kuramoto Oscillatory Neurons. arXiv:2410.13821.
- Mikulik et al. 2020 — Meta-trained agents implement Bayes-optimal agents. arXiv:2010.11223.
- Voita & Titov 2020 — Information-Theoretic Probing with MDL. EMNLP 2020.
- Elazar et al. 2021 — Amnesic Probing. TACL 2021.
- Geiger et al. 2024 — Distributed Alignment Search (DAS). arXiv:2303.02536.
- Sussillo & Barak 2013 — Opening the Black Box: Low-Dimensional Dynamics in Recurrent Neural Networks.
- Berens 2009 — `CircStat`: A MATLAB toolbox for circular statistics.
- Lin et al. 2024 — Learning to Model the World With Language (Dynalang). arXiv:2308.01399.
