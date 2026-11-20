# CONVERSATIONAL AI — WINTER SEMESTER 2026/27
## Mission 07: Sagittarius $A^{*}$ — The Gravitational Search Horizon & Process Reward Evaluator

**Subsystem 07:** Gravitational Search Horizon & Process Reward Evaluator ($+7.1\% \rightarrow 49.7\%$ total integrity)  
**Starship Operational Unit:** USSE Noosphere Bridge Crew — Squadron Turing  
**Mission Date:** 20 November 2026  
**Galactic Waypoint:** Sector SEC-07-SGR: Sagittarius $A^*$ Supermassive Black Hole ($26,670\text{ ly}$ from Sol)  
**Course Module:** Week 7 — Test-Time Compute & Reasoning Search Paradigms  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Python 3.11+ (Zero External Framework Bloat)  
**Public Repository:** `https://github.com/shaiksoh01/Sohel-Shaik-conversational-ai-mission-07`  
**Official Course Repository:** `https://github.com/Unisvet/couse-c-ai`  

**Mission Authors:**
* Sumeet Somnath Dighe
* Sohel Shaik
* Amulya Gaddam
* Atharv Santosh Jaiswal

---

## 1. Executive Summary
Mission 07 investigates System 2 Test-Time Compute (TTC) search algorithms, step-level process verification, and adversarial reward-hacking defences in conversational reasoning models. Holding station above Sagittarius $A^*$ (`SEC-07-SGR`, $4.154 \times 10^6 M_\odot$), AURA Subsystem 07 faces severe gravitational time dilation and exposure-bias instability inherent in single-pass left-to-right autoregressive generation.

The work is organised into four engineering components:
1. **Tree-of-Thoughts Solver (`tot_solver.py`):** Breadth-First Beam Search (`ToT-BFS`) and Depth-First Backtracking Search (`ToT-DFS`) under strict expansion budgets.
2. **MCTS & PUCT Engine (`mcts_protocol.py`):** Four-phase Monte Carlo Tree Search guided by policy priors $P_\theta(a \mid s)$ and the PUCT selection criterion.
3. **Process Reward Model & Math-Shepherd (`prm_evaluator.py`):** Step-level trajectory reductions, Outcome Reward Model (ORM) false-positive diagnostics, and automated rollout supervision.
4. **Event Horizon Pruner (`horizon_pruner.py`):** Real-time trajectory severing ($r_{\mathrm{PRM}} < \tau_{\mathrm{horizon}}$) and adversarial reward-hacking detection against sycophantic filler and step-splitting inflation.

The implementation uses native Python 3.11+ with zero external framework dependencies to ensure deterministic reproducibility and bounded test-time compute overhead.

---

## 2. Mission Overview and Deliverables

### 2.1 Mission Context
In Sector `SEC-07-SGR`, single-pass autoregressive decoding generates an output sequence $y = (y_1, \dots, y_T)$ conditioned on prompt $x$ via:

$$P_\theta(y \mid x) = \prod_{t=1}^{T} P_\theta(y_t \mid x, y_{\lt t})$$

Because standard autoregressive decoding commits irrevocably to each emitted token without lookahead or backtracking, any intermediate step error probability $\epsilon > 0$ causes the cumulative probability of committing at least one fatal mistake over $T$ steps to compound exponentially:

$$P(\text{at least one error}) = 1 - (1 - \epsilon)^T$$

Once an erroneous step enters the context window, exposure bias induces snowballing hallucinations that pull the navigation trajectory across the Event Horizon.

### 2.2 Implementation Deliverables

| Deliverable | Purpose |
| :--- | :--- |
| `tot_solver.py` | `ThoughtNode` dataclass, trajectory reconstruction, `bfs_beam_search`, `dfs_backtrack_search`, and `max_expansions` budget enforcement. |
| `mcts_protocol.py` | Exact PUCT formula (`puct_score`) and four-phase MCTS engine (`select_leaf`, `expand`, `simulate_rollout`, `backpropagate`). |
| `prm_evaluator.py` | Step-score reductions (`min`, `prod`, `geometric_mean`), ORM false-positive benchmark, and Math-Shepherd label estimator. |
| `horizon_pruner.py` | Reward-hacking detector (`check_reward_hacking`) and real-time Event Horizon trajectory pruning (`prune_trajectory`). |
| `mission-07-doc.pdf` | Compiled mission report and verification record for Moodle submission. |

---

## 3. Part I: Tree-of-Thoughts Beam & Backtracking Solver (`tot_solver.py`)

### 3.1 ThoughtNode Data Structure
Each node in the search tree is represented by `ThoughtNode`, encapsulating:
* `state: str` — semantic reasoning state at the current step.
* `depth: int` — tree depth relative to the root node ($d = 0$).
* `step_score: float` — local verifier score assigned to the transition.
* `cumulative_score: float` — product of step scores along the path from root.
* `parent: Optional[ThoughtNode]` and `children: List[ThoughtNode]`.

The method `get_trajectory()` walks `parent` pointers back to the root node and reverses the collected sequence to reconstruct the complete derivation chain $z_0 \rightarrow z_1 \rightarrow \dots \rightarrow z_t$.

### 3.2 Search Algorithms & Compute Guardrails
1. **Breadth-First Beam Search (`bfs_beam_search`):** Maintains an active frontier of at most $B = \mathrm{beam\_width}$ nodes at each depth $d \in \{1, \dots, \mathrm{max\_depth}\}$. Candidate children whose step evaluation falls below `prune_threshold` are pruned immediately, and surviving nodes are ranked by `cumulative_score`.
2. **Depth-First Backtracking Search (`dfs_backtrack_search`):** Sorts valid child branches by `step_score` descending and traverses the highest-scoring branch depth-first up to `max_depth`. Whenever a node scores below `prune_threshold` or reaches a dead-end, recursion chronologically backtracks to the next highest-scoring sibling.
3. **Expansion Budget (`max_expansions`):** Both search routines increment the expansion counter before expanding a node and terminate immediately once `expansion_count >= max_expansions`.

---

## 4. Part II: 4-Phase MCTS & PUCT Engine (`mcts_protocol.py`)

### 4.1 Exact PUCT Formulation
During tree traversal, child nodes are selected according to the Predictor + Upper Confidence Bound applied to Trees (PUCT) objective:

$$\mathrm{PUCT}(s, a) = Q(s, a) + c_{\mathrm{puct}} \cdot P_\theta(a \mid s) \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

where $Q(s, a) = \frac{W(s, a)}{N(s, a)}$ if $N(s, a) > 0$ (otherwise $0.0$), $N(s) = \max(1, \mathrm{parent.visit\_count})$, and $c_{\mathrm{puct}}$ is the exploration constant.

### 4.2 Canonical Four-Phase Execution Loop
1. **Phase 1 (`select_leaf`):** Traverses downward from `root` by greedily choosing the best child maximizing `puct_score` until reaching an unexpanded or terminal leaf node.
2. **Phase 2 (`expand`):** Instantiates child `MCTSNode` instances for each `(thought_state, prior_prob)` candidate pair.
3. **Phase 3 (`simulate_rollout`):** Evaluates the newly expanded leaf via a trajectory rollout or PRM value estimate bounded within $[0, 1]$.
4. **Phase 4 (`backpropagate`):** Traverses upward from the leaf to `root`, incrementing `visit_count += 1` and accumulating `value_sum += reward` at every ancestor.

---

## 5. Part III: Process Reward Model & Math-Shepherd (`prm_evaluator.py`)

### 5.1 Trajectory Step Reductions
Given intermediate step scores $[r_1, r_2, \dots, r_T] \in (0, 1]^T$, `ProcessRewardEvaluator` implements three reduction operators:
* **Minimum Reduction (`min`):** $\min_{t=1}^{T} r_t$ — weakest-link detector.
* **Product Reduction (`prod`):** $\prod_{t=1}^{T} r_t$ — joint step-wise validity probability.
* **Geometric Mean (`geometric_mean`):** $\left(\prod_{t=1}^{T} r_t\right)^{1/T}$ — length-normalized trajectory quality.

### 5.2 ORM vs. PRM False-Positive Benchmark
An Outcome Reward Model evaluates only whether the terminal answer matches ground truth $y^*$. In our benchmark test (`run_orm_vs_prm_benchmark`), a flawed two-step derivation commits two canceling algebraic sign errors:
* **Step 1:** $(-6) \times (-7) = -42$ *(Sign error #1; step score $r_1 = 0.05$)*
* **Step 2:** $-(-42) = 42$ *(Sign error #2 canceling the first; step score $r_2 = 0.10$)*

| Evaluator / Metric | Score | Diagnostic Interpretation |
| :--- | :---: | :--- |
| Outcome Reward Model (`ORM`) | `1.0000` | False positive: terminal answer matches $y^* = 42$. |
| PRM Minimum Reduction (`min`) | `0.0500` | Flags invalid intermediate sign error at Step 1. |
| PRM Product Reduction (`prod`) | `0.0050` | Reflects low joint validity across both flawed steps. |
| PRM Geometric Mean (`geometric_mean`) | `0.0707` | Length-normalized score rejects corrupted proof. |

### 5.3 Math-Shepherd Automated Supervision
To estimate step validity without manual step annotations, `math_shepherd_estimation` evaluates $M$ simulated completions $z_{t \rightarrow T}^{(1)}, \dots, z_{t \rightarrow T}^{(M)}$ from intermediate step $z_t$ against ground-truth answer $y^*$:

$$\hat{y}_{z_t}^{\mathrm{soft}} = \frac{1}{M}\sum_{j=1}^{M}\mathbb{I}\!\left(\mathrm{ans}\!\left(z_{t \rightarrow T}^{(j)}\right) = y^*\right), \qquad \hat{y}_{z_t}^{\mathrm{hard}} = \mathbb{I}\!\left(\hat{y}_{z_t}^{\mathrm{soft}} > 0\right)$$

---

## 6. Part IV: Event Horizon Pruner & Reward-Hacking Defense (`horizon_pruner.py`)

### 6.1 Adversarial Reward-Hacking Detector
`EventHorizonPruner.check_reward_hacking` inspects each candidate step against two adversarial manipulation vectors:
* **Sycophantic Filler Injection:** Detects vacuous self-congratulatory phrases (e.g., *"Let us carefully verify this 100% rigorous step..."*, *"Obviously, by trivial inspection..."*) designed to inflate naive PRM logits without advancing mathematical state.
* **Step-Splitting Inflation:** Strips non-symbolic filler to normalize algebraic tokens and detects repeated equations or vacuous micro-steps injected to dilute a low-scoring error under mean aggregation.

### 6.2 Horizon Trajectory Pruning Protocol
`EventHorizonPruner.prune_trajectory` iterates sequentially through `(steps, step_scores)`:
1. Whenever `check_reward_hacking(step)` triggers, the raw step score is penalized multiplicatively by `self.hacking_penalty_factor` (`0.25`).
2. If the adjusted step score falls below `self.tau_horizon` (`0.50`), the trajectory is immediately severed at that index and a structured `TelemetryAlert` is recorded.

---

## 7. Squad Review and Sign-Off

| Mission Author | Contribution Record |
| :--- | :--- |
| **Sumeet Somnath Dighe** | Bridge crew member and mission author. |
| **Sohel Shaik** | Bridge crew member and mission author. |
| **Amulya Gaddam** | Bridge crew member and mission author. |
| **Atharv Santosh Jaiswal** | Bridge crew member and mission author. |

**Technical Responsibility Areas:**
1. **Systems Architect (Lead Engineer):** `tot_solver.py` — ToT-BFS, ToT-DFS, and expansion budget pruning.
2. **Communications and Protocol Officer:** `mcts_protocol.py` — Exact PUCT scoring and four-phase MCTS engine.
3. **Data and Payload Specialist:** `prm_evaluator.py` — Trajectory step reductions, ORM false-positive benchmark, and Math-Shepherd supervision.
4. **Security and Ethics Sentinel:** `horizon_pruner.py` — Reward-hacking audit and Event Horizon trajectory pruning.

---

## 8. Final Submission Checklist

```text
mission-07/
|-- tot_solver.py
|-- mcts_protocol.py
|-- prm_evaluator.py
|-- horizon_pruner.py
|-- mission-07-doc.md
`-- mission-07-doc.pdf
