# Born-rule Policy Mechanistic Analysis — Design

**Date:** 2026-05-11
**Status:** Phase 1 scoped, Phase 2 deferred.
**Owner:** sathya
**Related:** [phase-probing-analysis-design (2026-05-07)](2026-05-07-phase-probing-analysis-design.md) — the sister doc on probing the *hidden state*. This doc is about probing the *policy head*.

---

## 1. Question

The `BornRuleActor` (pobax/models/discrete.py:40-60) converts a complex hidden state `h ∈ ℂ^H` to a categorical policy by computing `log(|W₂ · ModReLU(W₁ h + b₁) + b₂|² + ε)` and softmaxing. Because `softmax(log(x))` is `x/Σx`, this is exactly the **Born rule**: action probabilities equal squared amplitudes of a learned complex projection. We want to understand mechanistically what this head is doing in a *trained* uRNN/EUNN agent — specifically whether the off-diagonal ("coherent") cross-terms in `|·|²` are load-bearing for the policy, and where in an episode they matter.

The animating physical analogy is quantum measurement: the unitary RNN evolves a "state vector" coherently, the Born head projects + measures, action selection collapses to a single outcome. The empirical observation that motivated this analysis is that Born works but is **not dramatically better** than a standard real-valued MLP head — and we want to know why.

---

## 2. Critical reframing of the framing

Before the analysis, an honest read of what the literature actually says, because it changes the question we should be asking.

### 2.1 What the framing gets right

- `|z_a|² = Σ_j |W_{a,j} h_j|² + 2 Σ_{j<k} Re(W_{a,j} W_{a,k}* h_j h_k*)` does have genuine cross-terms involving relative phases of `h_j, h_k`. These are absent from a real linear logit map.
- Unitary RNNs evolve phase coherently across time: `h_t = U(h_{t-1}, x_t)` with `U` unitary, so phase relations between hidden dims at time `t` reflect a learned encoding of history. A Born head consumes those phase relations directly.
- The Fubini–Study metric on `ℂP^{H-1}`, pulled back along `z ↦ |z|²/Σ|z|²`, *is* the Fisher–Rao metric on the simplex. Born policies are quantum-kernel one-vs-rest classifiers over actions (Schuld & Killoran 2019). This gives a clean information-geometric structure for free.

### 2.2 What the framing overclaims

1. **`|z|²` is a real quadratic form.** Writing `W = A + iB`, `h = u + iv`: `|Wh|² = (Au − Bv)² + (Av + Bu)²`. A real network with that specific tied-weight quadratic head computes *exactly* the same function. Voigtlaender 2020 shows `|·|²` is a valid universal nonlinearity for complex nets, but Glasser et al. (NeurIPS 2019) prove non-negative tensor models and Born machines are *incomparable* — there exist distributions polynomial under softmax-positive parameterizations that need exponential resources as Born machines, and vice versa. So Born is not strictly more expressive than real quadratic.
2. **Real MLPs also produce cross-terms** — between layers via nonlinearities. Framing "interference" as a uniquely complex capability conflates "the cross-terms are explicit in a single layer" with "only complex nets have cross-terms."
3. **The strongest piece of prior empirical evidence cuts against Born.** Jerbi et al. (NeurIPS 2021, "Parametrized Quantum Policies for RL") directly compares `RAW-PQC` (`π(a|s) = ⟨P_a⟩`, literal Born) against `SOFTMAX-PQC` (softmax of expectation values). **SOFTMAX-PQC dominates RAW-PQC** in convergence speed and final return on CartPole et al. Attribution: RAW-PQC has no inverse-temperature parameter, and `∇log π = 2 Re(z̄ ∇z)/|z|²` is unbounded near zero amplitudes. The `+ ε` regulariser in `BornRuleActor` patches the singularity but doesn't add a temperature knob.

### 2.3 The right question

Not *"does Born provide extra expressivity?"* (it doesn't, strictly). Rather:

> **Does a trained Born policy actually use the off-diagonal phase-coherent terms, and where in an episode are they load-bearing?**

This is falsifiable from a single trained checkpoint:
- If trained Born policies are silently dominated by their diagonal (`Σ_j |W_{a,j} h_j|²`) contribution, the head has collapsed to a magnitude-only regime, the quantum framing is decorative, and we have an explanation for why Born ≈ real MLP empirically.
- If the off-diagonal mass concentrates at decision-critical timesteps (T-Maze junction, RockSample sample-action), then interference *is* being used by the trained agent and we can characterize when. Both outcomes are publishable; the design must not depend on which it is.

---

## 3. Architecture refresher (what we are actually analysing)

**Head structure** (pobax/models/discrete.py:50-60):

```
h_mem (complex64, H)
  │
  W₁ + b₁    Dense(H → H, complex64, glorot)
  │   z₁
ModReLU      magnitude-only, phase-preserving
  │   z₂
  W₂ + b₂    Dense(H → A, complex64, glorot · 0.01)
  │   z₃
log(|·|² + ε)
  │   logits
softmax → π(a|s)
```

Two important details that affect the math of the decomposition:

- **ModReLU is phase-preserving.** `ModReLU(z) = z · relu(|z|+β)/(|z|+ε)` — a magnitude-only gate. Phase relations between `z₁`'s components survive through `z₂` (modulo magnitude clipping for components with `|z₁,j|+β_j < 0`). So the phase structure that drives `|z₃|²`'s cross-terms originates in `h_mem` and propagates through `W₁`'s linear remix.
- **Biases are non-zero in general.** `b₁, b₂` are initialised to zero but trained. With non-zero learned bias, a global phase rotation `h_mem ↦ e^{iθ} h_mem` does *not* leave `π` invariant. The empirical magnitude of the learned biases is itself a diagnostic — if `‖b‖ ≪ ‖W h‖`, the policy is approximately Born-invariant in the quantum sense, otherwise the network has drifted toward something more general.

**Parameter count (per action head, real params):**
- `DiscreteActor` (standard, real MLP on `[Re h, Im h] ∈ ℝ^{2H}`): `2H·H + H + H·A + A = 2H² + (A+1)H + A`.
- `BornRuleActor`: `W₁` is `H·H` complex = `2H²` real; `b₁` is `H` complex = `2H` real; `W₂` is `H·A` complex = `2HA` real; `b₂` is `A` complex = `2A` real; ModReLU's `β` is `H` real. Total: `2H² + 2H(A+1) + 2A + H`.

The two heads have similar but not identical capacities. For Phase 2's matched-control ablations we will explicitly match them; for Phase 1 analysis on a single trained Born checkpoint this is informational only.

---

## 4. Mathematical setup for the decomposition

For each action `a ∈ {1, …, A}` and timestep `t` of a rollout, compute the **pre-Born complex projection**

`z_{a,t} = W₂_a · z₂(t) + b₂_a   ∈ ℂ`

where `z₂(t) = ModReLU(W₁ · h_mem(t) + b₁)`. Then decompose the squared amplitude:

```
|z_{a,t}|² = D_{a,t} + O_{a,t}
D_{a,t}   = Σ_j |W₂_{a,j} z₂_j(t) + (1/H) b₂_a|²        # "diagonal" / "incoherent"
O_{a,t}   = |z_{a,t}|² − D_{a,t}                        # off-diagonal / coherent
```

(The bias-attribution split `(1/H) b₂_a` per term is a choice — alternative is to absorb the bias into a 1-hot extra "dim" via `z₂ ← [z₂, 1]`, `W₂ ← [W₂ | b₂]`. We'll use the latter as primary; it makes the decomposition exact: `|z_{a,t}|² = Σ_j |W̃₂_{a,j} z̃₂_j|² + 2 Σ_{j<k} Re(W̃₂_{a,j} W̃₂_{a,k}* z̃₂_j z̃₂_k*)`.)

The **coherent fraction** at `(a, t)` is

`ϕ_{a,t} = O_{a,t} / (D_{a,t} + |O_{a,t}|) ∈ [−1, 1]`.

`ϕ_{a,t} ≈ 0` ↔ Born head silently diagonal at that step (no interference being used).
`ϕ_{a,t} > 0` ↔ coherent terms boost the chosen action's amplitude.
`ϕ_{a,t} < 0` ↔ coherent terms *suppress* the action's amplitude (the policy is using destructive interference).

We will also report the **l₁-coherence** (Baumgratz, Cramer, Plenio 2014):

`C_l1(z_{a,t}) = Σ_{j ≠ k} |W̃₂_{a,j} z̃₂_j| · |W̃₂_{a,k} z̃₂_k*|`

`C_l1` is a basis-dependent but principled measure of quantum coherence; tracking it gives a sign-blind picture of how much interferometric mass the head is generating.

---

## 5. Scope

### 5.1 Environments

Both selected because they isolate complementary aspects of the interference story:

- **T-Maze (75 and 100 hallway lengths).** Single decision-critical step (the junction). Memory requirement is exactly *one bit* (the initial cue). If interference at the junction is load-bearing, we expect `ϕ_{a,T-1}` to spike at the junction step. Hallway steps should be approximately featureless. Existing checkpoints: `results/qurnn_standard_tmaze_{75,100}_hpsweep`, `results/qeunn_2_tmaze_{75,100}_hpsweep`.
- **RockSample 11×11.** Repeated belief updates over up to ~5 rocks; sense and sample actions are decision-critical at different times. Existing checkpoints: `results/qurnn_standard_rocksample_11_11_hpsweep{,extra}`, `results/qeunn_2_rocksample_11_11_hpsweepextra`.

We will *not* tackle Battleship 10×10 or RockSample 15×15 in Phase 1 — the additional combinatorics doesn't add clarity. We will *not* tackle Craftax (too messy for mechanistic analysis at this scale).

### 5.2 Memory variants

- **uRNN-standard + Born** (qurnn_standard). Primary.
- **EUNN(L=2) + Born** (qeunn_2). Secondary — replicates findings under a different unitary parameterisation. If results disagree between qurnn and qeunn this is informative on its own.

### 5.3 Phase split

- **Phase 1 (this design, to be implemented next).** No retraining. Analyse existing best-hyperparam Born checkpoints. Yields: coherent-fraction profile across episodes, phase-intervention causal curves, action templates / FS decision-boundary slices, learned-bias diagnostic.
- **Phase 2 (future, separate design doc).** Retraining ablations: Born-diagonal head, matched real-MLP head on `[Re h, Im h]`, frozen-memory head swap, coherence-over-training trajectory.

---

## 6. Phase 1 — analyses

### 6.1 Pillar A: coherent-fraction decomposition

**What.** For each `(env, memory_variant, seed)` checkpoint, roll out `N_episodes = 256` deterministic-ish (argmax) and stochastic (sample) trajectories. For each rollout, at every timestep:
- Compute `z₂(t)`, then `z̃₂(t) = [z₂(t), 1]`, `W̃₂ = [W₂ | b₂]`.
- Compute `D_{a,t}`, `O_{a,t}`, `ϕ_{a,t}`, `C_l1(z_{a,t})` for *every* action `a` (not only the sampled one).
- Compute the same for the *next* action sampled (for "action-aligned" plots).
- Compute the action-mask-aware version: in RockSample some actions are always available, so this isn't critical; in Battleship it would be (but we're skipping Battleship in Phase 1).

**Plots.**
- `ϕ_{a*,t}` vs episode-step, mean ± SEM over rollouts, one curve per env, panel per memory_variant. T-Maze: expect spike at junction or null.
- `C_l1(z_{a*,t})` vs episode-step, same panels.
- Per-action heatmap: `ϕ_{a,t}` over `(a, t)` for one example episode. Shows whether different actions use coherence at different times.
- Histogram of `ϕ_{a*,t}` over (rollout × step) — what's the population distribution of coherent fractions?

**Falsifiable predictions.** Pre-register at least these:
- *T-Maze:* `mean_t |ϕ_{a*,junction}| > 2 × mean_t |ϕ_{a*,hallway-step}|` — coherence concentrates at decision step.
- *RockSample:* `|ϕ_{a*,t}|` is larger when `a* = sense` or `sample` than when `a* = move`.
- Null result: `|ϕ_{a*,t}| < 0.05` everywhere — Born head is silently diagonal.

### 6.2 Pillar B: causal phase interventions

Take a trained agent, intervene mid-rollout, measure return drop and policy KL.

**Three intervention types**, applied at each step `t`:

1. **Global phase rotation of `h_mem`.** `h_mem ↦ e^{iθ} h_mem`, θ swept over `[0, 2π)`. *Sanity check.* If biases are small, policy should be invariant. Magnitude of deviation = magnitude of bias-driven phase-anchoring. Plot: `KL(π_orig || π_perturbed)` vs `θ`.

2. **Per-dimension random phase on `h_mem`.** `h_mem,j ↦ |h_mem,j| · e^{iφ_j}` with `φ_j ∼ U[0,2π)` iid. Preserves magnitudes, destroys cross-dim phase relations. Average KL and return over 20 random redraws per step. Plot: ΔKL and Δreturn vs episode-step. Headline: at decision-critical steps, the perturbation should hurt more than at non-critical steps.

3. **Per-dimension random phase on `W₂`.** `W₂_{a,j} ↦ |W₂_{a,j}| · e^{iψ_{a,j}}` (one-shot, not per step). Tests whether the actor's *weights* encode phase information that matters. If this kills performance while (2) doesn't, phase information lives mostly in W; vice versa for h. This isolates *where* the interference is being constructed.

**Plots.**
- Return vs intervention strength (interpolate `h_perturbed = (1-α) h + α h_phase-shuffled` for α ∈ [0,1]).
- ΔKL vs episode-step.
- Decomposition: how much of the return drop is from (i) cross-term destruction vs (ii) policy entropy increase? Compute `KL(π_orig || π_perturbed)` and compare to entropy of `π_orig`.

### 6.3 Pillar C: action templates and FS decision-boundary slices

The trained `W̃₂_a` (rows of `W₂` with bias appended) are the **action templates**: the "ideal `z₂` vectors" whose `|⟨W̃₂_a, z̃₂⟩|²` is maximised. Visualise:

1. **Magnitude/phase profile of `W̃₂_a` across `j`**, one panel per `a`. Reveal whether templates are localized in a few hidden dims, or distributed; whether phases are clustered or scattered.

2. **Inter-template overlap matrix** `M_{ab} = |⟨W̃₂_a, W̃₂_b⟩|² / (‖W̃₂_a‖² ‖W̃₂_b‖²)`. Low off-diagonal = well-separated action templates. High off-diagonal pairs reveal action pairs that compete on the same phase pattern (e.g., in RockSample, "sense rock 3" vs "sample rock 3" may share template direction).

3. **2-D slice of the FS decision boundary.** Pick the two hidden dims `(j, k)` with the highest sensitivity (judged by ∂π/∂Re h_j and ∂π/∂Im h_j over the rollout distribution). On the 2-D complex slice `(h_j, h_k)` (4 real coordinates), pick a 2-D real slice through the typical trajectory point and plot the decision regions `argmax_a π(a)`. Born decision boundary `|z_a|² = |z_b|²` is a *quadric*; a real MLP head's would be a piecewise-linear polytope. Side-by-side comparison with the standard `DiscreteActor` (trained on the same memory) makes the qualitative difference visible.

### 6.4 Pillar D: learned-bias diagnostic (small but important)

Report `‖b₁‖ / E_t ‖W₁ h_mem(t)‖` and `‖b₂‖ / E_t ‖W₂ z₂(t)‖` for every trained checkpoint. This tells us how close the trained head is to a pure (bias-free) Born projector. If biases are negligible, Pillar B's global-phase sanity check should confirm invariance; if biases are large, the network has learned to use the bias's symmetry-breaking, and the Born-rule framing is even more decorative.

---

## 7. Phase 2 — retraining ablations (deferred, separate design later)

For completeness, the experiments we *would* do with retraining, in priority order. Phase 2 will get its own design doc when we pick it up.

1. **Born-diagonal head.** Replace `|z₃|²` with `Σ_j |W₂_{a,j} z₂_j|²` (cross-terms surgically removed). Train with PPO. If matched performance ⇒ trained Born head is silently diagonal. Cleanest single ablation.
2. **Matched-capacity real MLP head on `[Re h, Im h]`.** Tests whether the complex head buys anything beyond doubled-channel real heads.
3. **Coherence-over-training trajectory.** Log `mean_t ϕ_{a*,t}` per training checkpoint. Does the agent learn to use interference, or does it converge to diagonal early?
4. **Frozen-memory head swap.** Freeze a trained uRNN, retrain only the head. Isolates head-vs-memory contribution.
5. **Phase-coherence auxiliary loss.** Stanić et al. 2023 add a contrastive loss on phase; does it sharpen the trained policy's interference structure? Curiosity-level priority.

---

## 8. Concrete instrumentation

### 8.1 Code to add

- `pobax/analysis/born_decompose.py` — pure function `decompose_logits(params, h_mem) → {D, O, phi, c_l1}` returning per-(a, t) decompositions for a batch of rollouts. JAX, no side effects. Reuse the trained `BornRuleActor.apply` for `z₂` computation.
- `pobax/analysis/phase_intervene.py` — three intervention closures (global, per-dim-h, per-dim-W) producing perturbed-policy callables. Each returns rollout statistics (return, mean KL).
- `pobax/analysis/action_templates.py` — extract `W̃₂`, compute overlap matrix, plot magnitude/phase profiles, plot FS decision-boundary 2-D slices.
- `scripts/analysis/run_born_analysis.py` — driver: takes `--checkpoint`, `--env`, `--n_episodes`, dispatches the three pillars, writes per-pillar artefacts to `results/<study>/analysis/born_<run_name>/`.

### 8.2 Checkpoint loading

Existing PPO `train_state` dicts under `results/<study>/<run>/` already contain the actor params (real biases + complex Dense kernels). We need to confirm the params include the `BornRuleActor` substree — verify before writing the analysis driver (one-line `print(jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), params))`).

### 8.3 What to log per rollout

Per timestep, per action: `(D_{a,t}, O_{a,t}, ϕ_{a,t}, C_l1_{a,t}, π(a|s_t), a_sampled)`. For T-Maze: also `grid_idx, goal_dir, done`. For RockSample: also `position, sampled-rock-belief` (extract from `RockSampleState`). Save as a single per-checkpoint NPZ for fast re-plotting.

### 8.4 Sample sizes

- T-Maze: 256 episodes per checkpoint × 5 seeds = 1280 episodes. Cheap (each ep ≤ ~120 steps).
- RockSample: 128 episodes × 5 seeds = 640 episodes (longer per ep). Should fit in 1 GPU-minute per checkpoint.
- Phase interventions: 20 random redraws per intervention × 256 rollouts = 5120 perturbed rollouts per intervention type. Still cheap because rollouts share weights.

---

## 9. Expected outcomes and how to read them

For each headline metric, write down what each outcome would mean *before* running:

| Observation | Interpretation |
|---|---|
| `|ϕ_{a*,t}|` peaks at junction (T-Maze) | Interference is used at decision-critical step. Headline figure. |
| `|ϕ_{a*,t}|` < 5% everywhere | Born head is silently diagonal. Explains "Born ≈ real MLP" empirically. Different but still publishable story. |
| Per-dim phase intervention on `h` kills return at junction, not at hallway | Phase information stored in `h` is what matters; uRNN's coherent-phase-evolution claim is supported. |
| Per-dim phase intervention on `W₂` kills return uniformly | Phase information stored in actor weights; uRNN-phase-evolution claim is weaker. |
| Both interventions kill return uniformly | Interference is used everywhere; the analogy holds. Surprising but possible. |
| Action templates have low overlap, distinct phase profiles | Born head learned a "quantum-kernel one-vs-rest" classifier, as predicted. |
| Templates have high overlap, near-zero phases | Born head collapsed to a real-quadratic head with random phases. Negative result. |
| `‖b‖ / ‖Wh‖` ≫ 1 | Head is not Born-like in the strict sense; analysis still informative but quantum framing is even more decorative. |

We should be equally happy with positive and null results — both narrow the design space for future complex-policy architectures.

---

## 10. Risks and caveats

1. **`ε`-regularizer artefacts.** `log(|z|² + ε)` with `ε = 1e-10` is benign in the forward pass but the gradient `2 z̄ / (|z|² + ε)` is large when `|z|² ≪ ε`. We don't backprop here (frozen policy), but the gradient direction matters for any retraining in Phase 2. Note for the future doc.
2. **ModReLU magnitude-clipping.** Components with `|z₁,j| + β_j < 0` get killed entirely; the post-ModReLU `z₂` is effectively sparser than `H`. Compute and report the activation sparsity per timestep — affects the effective `H` for the decomposition.
3. **Basis-dependence of the decomposition.** `D` and `O` depend on the chosen basis (here, hidden-dim coordinates). A unitary basis rotation `Q` applied to `(W₂, z₂) → (W₂ Q^H, Q z₂)` leaves `|z_a|²` invariant but redistributes mass between `D` and `O`. We should also compute basis-invariant quantities (`tr(ρ_a)`, `‖z̃₂‖`) for sanity and pick the *trained-natural* basis (hidden coordinates) for the headline plot, while documenting this limitation.
4. **Small action spaces.** T-Maze has `A=4`, RockSample-11×11 has `A=5+k` where `k` is the rock count. The action templates picture has few enough actions to read. Battleship-10×10 has `A=100`; would need different visualization. Phase 1 sidesteps this.
5. **Memory-init-carry quirks.** `initial_urnn_carry` is `(√(1/2H))(1 + i)·𝟙` — a uniform real-equal-imaginary state. At episode start before any input, all hidden dims have identical magnitudes and phases — `O_{a,0} = (H-1) D_{a,0}` for any constant `W̃₂_a`. Coherent fraction is maximal at `t=0` by construction; we must not over-interpret. Start analyses at `t ≥ 1` or `t ≥ first-non-zero-obs`.

---

## 11. References

- **Jerbi, Gyurik, Marshall, Briegel, Dunjko, 2021.** "Parametrized Quantum Policies for Reinforcement Learning." NeurIPS. <https://arxiv.org/abs/2103.05577> — direct RAW-PQC vs SOFTMAX-PQC comparison; softmax wins.
- **Glasser, Sweke, Pancotti, Eisert, Cirac, 2019.** "Expressive Power of Tensor-Network Factorizations." NeurIPS. <https://arxiv.org/abs/1907.03741> — Born machines vs positive tensor models are incomparable.
- **Voigtlaender, 2020.** "The Universal Approximation Theorem for Complex-Valued Neural Networks." <https://arxiv.org/abs/2012.03351> — `|·|²` is a valid universal nonlinearity.
- **Tan & Yu, 2023.** "Complex-Valued Neurons Can Learn More But Slower Than Real-Valued Neurons via Gradient Descent." NeurIPS. <https://openreview.net/forum?id=qA0uHmaVKk> — single-neuron expressivity vs trainability separation.
- **Baumgratz, Cramer, Plenio, 2014.** "Quantifying Coherence." Phys. Rev. Lett. <https://arxiv.org/abs/1311.0275> — definition of `l₁`-coherence.
- **Schuld & Killoran, 2019.** "Quantum machine learning in feature Hilbert spaces." Phys. Rev. Lett. <https://arxiv.org/abs/1803.07128> — `|⟨W_a, h⟩|²` as quantum kernel.
- **Stoudenmire & Schwab, 2016.** "Supervised Learning with Tensor Networks." <https://arxiv.org/abs/1605.05775> — MPS Born-rule classifier.
- **Mei, Xu, Schuurmans, Dale, 2020.** "On the Global Convergence Rates of Softmax Policy Gradient Methods." <https://arxiv.org/abs/2005.06392> — softmax PG geometry, contrast for Born PG.
- **Elazar, Ravfogel, Jacovi, Goldberg, 2021.** "Amnesic Probing." TACL. <https://aclanthology.org/2021.tacl-1.10/> — methodological template for "is this information *used* by the model".
- **Stanić, Gopalakrishnan, Irie, Schmidhuber, 2023.** "Contrastive Training of Complex-Valued Autoencoders for Object Discovery." <https://arxiv.org/abs/2305.15001> — phase-contrastive aux loss, candidate for Phase 2.

---

## 12. Open questions for sathya before Phase 1 execution

1. Pick which seeds within `qurnn_standard_tmaze_75_hpsweep` (and the others) we treat as "the canonical trained agents" — best return? median? all five? Probably "top-3 by mean eval return" is right.
2. Should we run the same analysis on `qeunn_2_*` checkpoints as a built-in replication, or only as a sanity follow-up? Default: built-in.
3. The `phase-probing-analysis-design.md` sister doc plans to probe the *hidden state* for belief variables. Should that probe and this Born decomposition share rollout datasets (cheaper) or have independent rollouts (cleaner)? Default: share datasets, save NPZs with both sets of metrics.
