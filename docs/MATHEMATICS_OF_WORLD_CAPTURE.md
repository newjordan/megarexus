# Mathematical Structure of Game-World Capture in MegaMind

**Audience:** research mathematicians and mathematical scientists  
**System:** megarexus MegaMind (dual-hemisphere base model)  
**Repository primary:** one brain — observe, structure, and act under environment-only grading  
**Code authority:** `synthesis_lab/orchestration/rdl/brain/`  
**Version note:** pad size \(N_{\mathrm{op}}=231\), feature pad dimension \(D_{\mathrm{extra}}=3696\); 24-game suite for multi-environment training

This document describes **what is actually implemented**: discrete geometry of frames, operator-valued feature maps, dual latent dynamics, LoRA-parameterized policy fusion, and multi-game training selection. It is not a scorecard claim and does not assert optimality of any single estimator.

---

## 1. Discrete game world as a colored grid process

### 1.1 State space

At each discrete time \(t \in \mathbb{N}\), the environment exposes a **board**

\[
s_t \in \mathcal{S} \subseteq \{0,1,\ldots,C-1\}^{H \times W},
\]

with \(C=16\) color classes by default (one-hot channelization in the neural encoder) and typical \((H,W)=(64,64)\). The implementation is **size-agnostic** for geometric operators: any \(H,W \ge 1\) is admissible.

An action is an element of a finite control set

\[
a_t \in \mathcal{A} = \{0,1,\ldots,A-1\}, \qquad A=8
\]

(with environment-available subsets \(\mathcal{A}_t \subseteq \mathcal{A}\) when the env restricts legality).

The env defines a (generally stochastic, partially observed) kernel

\[
s_{t+1} \sim P(\cdot \mid s_t, a_t).
\]

**Truth is external:** only environment transitions and level counters grade success. Internal models never admit “solve truth” flags at construction (fail-closed).

### 1.2 Levels as a filtration of progress

Many suite environments expose an integer progress coordinate

\[
\ell_t = \mathrm{levels\_completed}(s_t) \in \mathbb{N}_0.
\]

Training is organized by **level bands** (closed intervals)

\[
B = [\ell_{\mathrm{lo}}, \ell_{\mathrm{hi}}] \cap \mathbb{N}_0,
\]

e.g. \(B=[0,1]\). A transition is retained for a band cohort iff \(\ell_t \in B\). Progression into higher bands is **honest**: no forced level injection.

### 1.3 Multi-game product

Let \(\mathcal{G} = \{g_1,\ldots,g_{24}\}\) be the public suite. A **level-group cohort** is a multiset of transitions

\[
\mathcal{D}_B = \bigcup_{g \in \mathcal{G}} \mathcal{D}_{B,g},
\]

where each \(\mathcal{D}_{B,g}\) is collected under band \(B\) in game \(g\). Transitions are **round-robin interleaved** so batches mix games, enforcing multi-environment (cross-game) statistics rather than single-game specialization.

---

## 2. Perception as combinatorial geometry on labels

Before neural embedding, frames are reduced by deterministic **operator families** \(\Phi_k\), \(k=1,\ldots,N_{\mathrm{op}}\). Each operator returns a finite set of scalar scores (and optional short vectors), packed into a fixed slot of length \(m=16\):

\[
\phi_k(s) \in \mathbb{R}^{m}, \qquad
\Phi(s) = \big(\phi_1(s),\ldots,\phi_{N_{\mathrm{op}}}(s)\big) \in \mathbb{R}^{D},
\quad D = m\, N_{\mathrm{op}} = 3696.
\]

Missing/degraded operators zero-pad. The map \(\Phi\) is **hypothesis-only**: it never sets admission flags that would treat scores as ground-truth solves.

### 2.1 Connected components (label topology)

For background color \(b\) (modal or map-trust assigned), non-background sites form 4-connected components \(\{C_i\}\). For each component:

- area \(|C_i|\),
- bounding box aspect ratio,
- centroid \(c_i \in \mathbb{R}^2\),
- color label.

This is the discrete analogue of a partition of \(\mathrm{supp}(s \neq b)\) into path-connected pieces in the grid graph \(G_{H,W}\).

### 2.2 Spatial relations

Pairwise geometry uses standard metrics on centroids:

\[
d_1(c_i,c_j)=\|c_i-c_j\|_1, \quad
d_2(c_i,c_j)=\|c_i-c_j\|_2,
\]

plus bearings \(\mathrm{atan2}\), directional gaps, and nearest-neighbor queues (GeoBand-style). These induce a finite **relation graph** on components.

### 2.3 Symmetry scores

Global reflection symmetries on the label field:

\[
\mathrm{Sym}_H(s) = \frac{1}{HW}\sum_{x,y}\mathbf{1}[s_{x,y}=s_{W-1-x,y}],
\quad
\mathrm{Sym}_V(s) = \frac{1}{HW}\sum_{x,y}\mathbf{1}[s_{x,y}=s_{x,H-1-y}],
\]

and rotational analogues (e.g. \(180^\circ\)). Local self/pair symmetries are mined when available.

### 2.4 Signed potential fields (multi-source)

Attractors \(\{(p_j, w_j^+)\}\) and repellers \(\{(q_k, w_k^-)\}\) (component centers with strengths from area / rare-color heuristics) define a scalar potential

\[
U(x)
= \sum_j \frac{w_j^+}{\|x-p_j\|^2 + 1}
- \sum_k \frac{w_k^-}{\|x-q_k\|^2 + 1}
- \frac{w_{\partial}}{\mathrm{dist}(x,\partial\Omega)^2 + \varepsilon}.
\]

Suggested motion follows a unit finite-difference gradient \(\nabla U\). Multi-color variants treat each color class as a separate field channel (multivariate potential).

### 2.5 Dynamic multi-variate pattern growth

A dedicated catalog (G1–G40) treats the board as a **multi-variable discrete field** evolving in time. Representative constructions:

| Concept | Mathematical content |
|---|---|
| Multi-channel variables | Active label support size; entropy of color occupancy \(H = -\sum p_c \log p_c\) |
| Density field | Global fill \(\mu = |\{s\neq b\}|/(HW)\); blockwise local means and variance (heterogeneity) |
| Connectivity | Component count, area spectrum, fragmentation \(\propto n_{\mathrm{comp}}/\sqrt{HW}\) |
| Expansion frontier | Perimeter of solid mask; isoperimetric roughness vs ideal circle \(2\sqrt{\pi \mathrm{Area}}\) |
| Accretion / erosion | Given \(s_{t-1},s_t\): accrete \(=|S_t\setminus S_{t-1}|\), erode \(=|S_{t-1}\setminus S_t|\) |
| Merge / split | \(\Delta n_{\mathrm{comp}} = n_t - n_{t-1}\) |
| Anisotropy | Eigenvalues of covariance of solid pixel coordinates; axis ratio \(\lambda_1/\lambda_2\) |
| Periodicity | Shift-match autocorrelations at small lags |
| Reaction–diffusion proxy | Interface energy: fraction of disagreeing 4-neighbors |
| CA affinity | 3×3 majority rewrite flip rate (local rule pressure) |
| Cavities / genus proxy | Background components not touching \(\partial\Omega\) |
| Adjacency graph | Edges between distinct colors on adjacent pixels |
| Mass conservation | \(\ell_1\) change of per-color histograms between frames |
| Spatial frequency | Discrete Laplacian energy on solid masks |
| Joint state vector | Concatenation of low-order multi-variate statistics for the pad |

These are **descriptive statistics and dynamical signatures**, not claimed complete invariants of the true transition kernel \(P\).

### 2.6 Legal feature mines (392-D pathway)

When available, a legal-search encoder produces a longer feature vector \(f_{\mathrm{legal}}(s) \in \mathbb{R}^{392}\) (full-mine schema), then bin-reduced into operator slots. This aggregates ports, saliency, state-connectivity graphs, affordance proxies, and related discrete structures mined from controller literature—again as **hypothesis features**, not admitted solutions.

### 2.7 Catalog structure of the pad

The \(N_{\mathrm{op}}=231\) families include (among physics/CONTRA layers):

- physics/map: body, ray, field, perception trust  
- legal mines: symmetry, geo, saliency, affordances, statecon, delta, ghost, crucible, explore, legal392  
- full CONTRA fanout layers (vision, control, surfaces, geoband, boundary, …)  
- Loom imagination seeds L1–L30 (world-sim / multi-channel / MPC-style informers as **channels**)  
- Rexus system seeds R1–R60 (admission, structural sense, deltaband, fences, …)  
- Primitive game-design atoms P1–P32  
- Pattern growth G1–G40  
- Weco RSI bi-level / public–private / anti-hack seeds W1–W32  

Each is a named \(\phi_k\). Training is free to drive unused slots toward zero weight.

---

## 3. Dual-hemisphere latent dynamics

### 3.1 Frame embedding

A convolutional encoder \(\mathcal{E}_\theta\) maps one-hot boards to latents

\[
h = \mathcal{E}_\theta(s) \in \mathbb{R}^{d}, \qquad d=128.
\]

Architecture: three \(\mathrm{Conv2d}\) stages with ReLU, global average pool, linear projection. Colors are one-hot on \(C\) channels.

Actions embed as \(e_a \in \mathbb{R}^{32}\) via a learned table.

### 3.2 Internal hemisphere (dream / propose)

\[
\begin{aligned}
h_I &= \mathcal{E}_I(s), \\
\hat{h}'_{\mathrm{dream}} &= D_I([h_I; e_a]), \\
\pi_I(a \mid s) &\propto \exp\bigl((W_{\mathrm{plan}} h_I)_a\bigr).
\end{aligned}
\]

\(D_I\) is a two-layer MLP (dream head). \(\pi_I\) is a **hypothesis policy**, not the committing action distribution.

### 3.3 External hemisphere (world / judge)

\[
\begin{aligned}
h_E &= \mathcal{E}_E(s), \\
\hat{h}'_{\mathrm{pred}} &= F_E([h_E; e_a]), \\
\mathrm{disagreement} &= \|\hat{h}'_{\mathrm{dream}} - \hat{h}'_{\mathrm{pred}}\| \quad\text{(or related residual)}, \\
\mathrm{judge}_I &= J_E(h_E, \hat{h}'_{\mathrm{dream}}, \hat{h}'_{\mathrm{pred}}).
\end{aligned}
\]

Optional world-model plugs may blend additional dynamics predictors (fail-closed if absent).

### 3.4 Oscillation training (three-phase)

On a batch of real transitions \((s,a,s')\), phases update **disjoint parameter sets**:

| Phase | Active params | Loss (schematic) |
|---|---|---|
| **E** (external) | \(\theta_E\) | \(\mathcal{L}_E = \|\hat{h}'_{\mathrm{pred}} - \mathcal{E}_E(s')\|_2^2\) |
| **I** (internal) | \(\theta_I\) | \(\mathcal{L}_I = \|\hat{h}'_{\mathrm{dream}} - \mathrm{sg}(\mathcal{E}_E(s'))\|_2^2 + \tfrac12\|\hat{h}'_{\mathrm{dream}} - \mathrm{sg}(\hat{h}'_{\mathrm{pred}})\|_2^2\) |
| **C** (central) | LoRA on policy/trust/mode only | \(\mathcal{L}_C = \mathrm{CE}(\mathrm{logits}_C, a) + 0.2\,\mathrm{MSE}(\mathrm{trust}, \tau^\*)\) |

where \(\mathrm{sg}\) is stop-gradient (teacher geometry), and \(\tau^\*\) pushes trust toward the external hemisphere when disagreement is high (median split on the batch).

This is a **block-coordinate** learning schedule on a dual latent model with asymmetric roles (propose vs judge), not a single end-to-end RL objective.

---

## 4. Central fusion and low-rank policy adaptation

### 4.1 Feature fusion

The central cortex builds

\[
x = \bigl[h_I;\; h_E;\; h_I-h_E;\; |h_I-h_E|;\; \mathrm{judge}_I;\; \mathrm{disagreement};\; \Phi(s)\bigr]
\in \mathbb{R}^{4d + 2 + D}.
\]

With \(d=128\), \(D=3696\), \(\dim x = 4210\). A trunk MLP maps \(x \mapsto z \in \mathbb{R}^{256}\).

### 4.2 LoRA heads

Policy, trust, and mode heads use **LoRA** on linear maps \(W\):

\[
y = Wx + b + \frac{\alpha}{r}\, B A x,
\quad A \in \mathbb{R}^{r \times n},\; B \in \mathbb{R}^{m \times r},\; r=8,\; \alpha=16.
\]

Base \(W,b\) are frozen in Phase C; only \((A,B)\) train (AdaJEPA-style online adaptation spirit). Outputs:

- action logits \(\in \mathbb{R}^{A}\),
- trust \((\alpha_I,\alpha_E)=\mathrm{softmax}(\cdot)\) (weights on hemispheres),
- mode logits over \(\{\mathrm{gas},\mathrm{brake},\mathrm{probe},\mathrm{dream}\}\).

### 4.3 Online Ada step

After a transition cycle, an Ada-online module may further adapt LoRA parameters on recent residuals (implementation: `ada_online.py`), remaining restricted to low-rank adapters.

---

## 5. Multi-game training mathematics

### 5.1 Level-group cohorts

For each band \(B\), collect interleaved \(\mathcal{D}_B\) across all \(g \in \mathcal{G}\). Policy for collection is **ε-greedy** on internal propose logits:

\[
a =
\begin{cases}
\sim \mathrm{Uniform}(\mathcal{A}_t) & \text{with probability }\varepsilon, \\
\sim \mathrm{Categorical}\bigl(\mathrm{softmax}(\pi_I(\cdot\mid s)|_{\mathcal{A}_t})\bigr) & \text{with probability }1-\varepsilon.
\end{cases}
\]

\(\varepsilon\) decays with outer cycle index but floors (e.g. \(\varepsilon \ge 0.25\)) to avoid pure exploitation freeze.

### 5.2 Private selection score (outer keep-if-better)

Public train loss alone is **not** the selection criterion (reward-hacking risk). A private composite \(S \in [0,1]\) aggregates multi-game and environment axes (version 3), schematically

\[
\begin{aligned}
S
&= 0.12\, r_{\mathrm{robust}}
+ 0.10\, r_{\mathrm{hit}}
+ 0.08\, r_{\mathrm{ok}}
+ 0.08\, r_{\mathrm{arcade}}
+ 0.16\, \eta_{\mathrm{loss}}
+ 0.10\, \delta_{\mathrm{epoch}}
+ 0.06\, v_{\mathrm{vol}}
+ 0.04\, b_{\mathrm{balance}}
+ 0.12\, \ell_{\mathrm{level}}
+ 0.08\, u_{\mathrm{ups}}
+ 0.04\, g_{\mathrm{upfrac}}
+ 0.02\, \Delta_{\mathrm{frame}},
\end{aligned}
\]

with

\[
\eta_{\mathrm{loss}} = \Bigl(1 + \frac{\log(1+L)}{\log(1+800)}\Bigr)^{-1},
\quad
v_{\mathrm{vol}} = \mathrm{clamp}\frac{\log(1+n_{\mathrm{trans}})}{\log(1+1000)},
\quad
\delta_{\mathrm{epoch}} = \mathrm{clamp}\frac{L_0 - L_T}{|L_0|+\varepsilon}.
\]

Level axes use \(\ell\) and level-up counts (environment truth). Checkpoint \(c'\) replaces incumbent \(c\) iff \(S(c') > S(c)\) (or near-tie with substantially better public loss under full multi-game coverage).

This is a **bi-level** design in the sense of outer selection over inner training runs (cf. AIDE² public/private split), **without** claiming recursive self-improvement Level 1 for the system.

### 5.3 Cross-game robustness probe

After training, one frame per game is observed; the fraction of games with healthy pattern-growth channel status contributes \(r_{\mathrm{robust}}\). Shared parameters must support all prefixes simultaneously.

---

## 6. What is *not* claimed

1. **Completeness of \(\Phi\).** Operator catalogs are a spanning set of *named hypotheses*, not a basis for all Markov kernels on grids.  
2. **Identifiability of \(P\).** Latent MSE on embeddings is not equivalent to total variation learning of \(P\).  
3. **Optimality of LoRA rank, ε schedule, or private weights.** These are design parameters under active training.  
4. **Level-band solvability.** Random or partially trained policies may never leave \(B=[0,1]\); higher bands then lack in-band environment data.  
5. **Specialist scorecard equivalence.** Multi-game pad training is not the same object as a single-game competition policy.

---

## 7. Compact formal pipeline

\[
\begin{CD}
s_t @>{\Phi}>> \mathbb{R}^{D} \\
@V{\mathcal{E}_I,\mathcal{E}_E}VV @VV{\mathrm{concat}}V \\
(h_I,h_E,\hat h'_{\mathrm{dream}},\hat h'_{\mathrm{pred}}) @>>> x \in \mathbb{R}^{4d+2+D} \\
@VV{\mathrm{trunk}+LoRA}V \\
(\mathrm{logits}_a, \alpha_I,\alpha_E, \mathrm{mode}) @>>> a_t^{\mathrm{commit}} \\
@VV{P}V \\
s_{t+1},\; \ell_{t+1}
\end{CD}
\]

Training: phase-locked losses on real \((s,a,s')\); outer loop selects parameters by multi-game private \(S\).

---

## 8. References in code (for verification)

| Topic | Location |
|---|---|
| Pad families / \(D\) | `hemisphere_i.py` (`OP_FAMILIES`, `EXTRA_FEAT_DIM`) |
| Fusion / LoRA | `central.py` |
| Phase losses | `oscillate.py` |
| Pattern growth operators | `operators/pattern_growth.py` |
| Potential fields | `operators/field_ops.py` |
| Legal geometry | `legal_search/algorithms/geometry_rel.py` et al. |
| Level cohorts / interleave | `collect_level_group.py`, `suite_games.py` |
| Private score \(S\) | `rsi_outer_gate.py` |
| Outer flywheel | `train_flywheel.py` |
| Doctrine status | `STATUS.md`, `docs/PRIMARY.md` |

---

## 9. Summary for mathematical readers

MegaMind treats each game as a **finite-color grid dynamical system**. It captures structure via a large family of **deterministic geometric and combinatorial functionals** \(\Phi\), embeds boards into **dual latent manifolds** with asymmetric dream/judge dynamics, and commits actions via a **low-rank-adapted central map** on fused hemispheres and \(\Phi\). Learning is **block-coordinate** on real transitions; multi-environment training uses **level-band product measures** over 24 games with a **private multi-criteria selection functional** that privileges generalization and environment progress over raw training loss.

The design priority is **robust multi-game structure capture under fail-closed truth**, not closed-form solution of any single game’s optimal control problem.

---

*Document generated for mathematical review of the implemented megarexus MegaMind world-capture stack. Implementation is the source of truth if this text and code diverge.*
