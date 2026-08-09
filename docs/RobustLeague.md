# Robust multiplayer league training

This mode upgrades return-based opponent rotation without changing the policy network, learner, replay format, or number of self-play games. It is a lightweight league inspired by PSRO/JPSRO, but it is deliberately **not** a full implementation of either algorithm. Set `zero_league_use_deviation=false` to remove step 6 while keeping steps 1-5.

## Recommended Blokus 10 configuration

Start a new run with the following population settings. Behavior conditioning is optional; the league does not require FiLM or any other network module.

```text
zero_use_population=true
zero_use_league=true
zero_population_size=6
zero_population_snapshot_interval=5
zero_population_current_seat_min=1
zero_population_current_seat_max=3
zero_population_balance_seats=true
zero_population_hard_ratio=0.5
zero_population_temperature=0.2

zero_league_refresh_interval=10
zero_league_rank_weight=0.75
zero_league_self_ratio=0.30
zero_league_champion_ratio=0.20
zero_league_frontier_ratio=0.25
zero_league_hard_ratio=0.15
zero_league_coverage_ratio=0.10
zero_league_min_games=32
zero_league_confidence_scale=1.0
zero_league_champion_margin=0.0
zero_league_use_deviation=true
zero_league_deviation_margin=0.05
zero_league_deviation_lineup_ratio=0.5

nn_use_behavior_conditioning=false
```

For the 1-5 ablation, change only:

```text
zero_league_use_deviation=false
```

Keeping every other setting identical makes the two runs directly comparable. Use the same seeds, training budget, MCTS simulations, evaluation opponents, and evaluation seat permutations.

## The six changes

### 1. Rank-aware empirical utility and uncertainty

For player $i$, the league observes terminal value $v_i\in[-1,1]$ and normalized rank $r_i\in[-1,1]$, then computes

\[
u_i=(1-\lambda)v_i+\lambda r_i,
\qquad \lambda=\texttt{zero\_league\_rank\_weight}.
\]

For four players, untied ranks are $1,1/3,-1/3,-1$; tied places receive the average of their rank values. A larger $\lambda$ makes opponent selection follow placement/win rate rather than score magnitude. This affects only server-side league decisions, not MCTS backup or learner targets.

For each historical opponent $h$ and number of current-model seats $k$, the server records separate sample statistics for current and historical seats. If a game contains current-seat set $C$ and historical-seat set $H$, its two observations are

\[
\hat u^{\mathrm{cur}}_{h,k}=\frac{1}{|C|}\sum_{i\in C}u_i,
\qquad
\hat u^{\mathrm{hist}}_{h,k}=\frac{1}{|H|}\sum_{i\in H}u_i.
\]

Mean and sample variance use Welford's online update. Given old count $n$, mean $m$, squared-deviation sum $M_2$, and new observation $x$:

\[
n'=n+1,\quad \delta=x-m,\quad
m'=m+\frac{\delta}{n'},\quad
M_2'=M_2+\delta(x-m').
\]

The sample variance, standard error, and lower confidence bound (LCB) are

\[
s^2=\frac{M_2}{n-1},\qquad
\operatorname{SE}=\sqrt{\frac{s^2}{n}},\qquad
\operatorname{LCB}_c=m-c\operatorname{SE},
\]

where $c=\texttt{zero\_league\_confidence\_scale}$. Decisions requiring confidence are disabled until at least `zero_league_min_games` observations exist.

Why it is needed: raw Blokus score return can improve without improving placement. Rank mixing aligns the scheduler with the requested win/placement objective, while count and uncertainty keep a few lucky games from immediately controlling training.

### 2. Statistics choose opponents

Statistics are pooled over observed lineup sizes $k$ for each opponent. The role selector uses:

\[
h_{\mathrm{frontier}}=\arg\min_h |\bar u_h^{\mathrm{cur}}|,
\]

\[
h_{\mathrm{hard}}=\arg\min_h \operatorname{LCB}_c(u_h^{\mathrm{cur}}),
\]

and

\[
h_{\mathrm{coverage}}=\arg\min_h n_h,
\]

with least-recently-seen as the coverage tie-breaker. Before an opponent reaches the minimum sample count, frontier and hard selection fall back to coverage.

Why it is needed: the previous implementation measured lineup return but still chose historical checkpoints round-robin. These rules close that loop. Frontier supplies informative near-even games, hard targets weaknesses conservatively, and coverage prevents forgotten or newly added opponents from receiving no data.

### 3. Frozen active pool

At refresh time, the server builds at most `zero_population_size` available checkpoints from

\[
t_j=t_{\mathrm{current}}-j\,
\texttt{zero\_population\_snapshot\_interval}\,
\texttt{learner\_training\_step},
\qquad j=1,2,\ldots
\]

and includes the champion when available. The pool and its block statistics remain fixed for `zero_league_refresh_interval` outer iterations. Statistics reset when the pool refreshes.

Why it is needed: a moving pool makes an opponent disappear before enough games have been collected to estimate its difficulty. A short frozen block makes selection meaningful without constructing or evaluating a full payoff tensor.

### 4. Worker-role mixture

Each self-play worker is assigned one of five base roles for an outer iteration:

- `self`: all seats use the current model;
- `champion`: use the gated champion as the historical model;
- `frontier`: use the empirically closest-to-even opponent;
- `hard`: use the opponent with the lowest current-model LCB;
- `coverage`: use the least-observed active opponent.

Let the configured non-negative role weights be $w_j$, normalized as $q_j=w_j/\sum_lw_l$. With $A$ total assignments and $A_j$ assignments already given to role $j$, weighted fair scheduling chooses

\[
j^*=\arg\max_j\left(q_j(A+1)-A_j\right).
\]

The ratios are therefore long-run targets, not exact per-game quotas. This fits the existing architecture, where one worker keeps one opponent for its batch of parallel games. Actual completed games are logged with SGF tag `LR` and `[League Role Games]` entries.

Why it is needed: hard-only training can overfit a narrow counter-cycle, while self-only training forgets old weaknesses. The role mixture explicitly reserves data for current-policy quality, stable anchors, learning-progress games, weaknesses, and diversity.

For a hard-role opponent, lineup-size sampling retains the existing adaptive mixture. First define

\[
b_k\propto
\begin{cases}
1/k,&\texttt{zero\_population\_balance\_seats=true},\\
1,&\text{otherwise},
\end{cases}
\]

then

\[
q_k=\frac{b_k\exp(-\bar u_{h,k}^{\mathrm{cur}}/T)}
{\sum_jb_j\exp(-\bar u_{h,j}^{\mathrm{cur}}/T)},
\qquad
p_k=(1-\rho)b_k+\rho q_k,
\]

where $T=\texttt{zero\_population\_temperature}$ and $\rho=\texttt{zero\_population\_hard\_ratio}$. Other base roles use $b_k$.

### 5. Champion gate

The current checkpoint $t$ replaces champion $c$ only when the current-seat utility against $c$, at the smallest configured current-seat count $k_{\min}$, passes

\[
\operatorname{LCB}_c
\left(u^{\mathrm{cur}}_{c,k_{\min}}\right)
>\texttt{zero\_league\_champion\_margin}
\]

with at least `zero_league_min_games` observations from the current checkpoint. SGFs produced by stale current checkpoints may still enter replay under the existing server setting, but they are excluded from league decisions by checking the `EV` model tag.

Why it is needed: always replacing the anchor allows temporary regressions to redefine “strong.” The gate provides a stable opponent and promotes only confidence-adjusted improvements. The champion and league statistics are in-memory state; restarting the server reconstructs the champion from the newest available active checkpoint.

### 6. Restricted deviation feedback

For each historical policy $h$, the server inspects the lineup with the maximum current-seat count $k_{\max}$. In four-player Blokus with $k_{\max}=3$, this is exactly one historical seat deviating against three current-policy seats. Its restricted deviation score is

\[
g_h=\operatorname{LCB}_c
\left(u^{\mathrm{hist}}_{h,k_{\max}}\right).
\]

If

\[
\max_h g_h>\texttt{zero\_league\_deviation\_margin},
\]

the hard-role budget temporarily becomes a `deviation` role against the maximizing $h$. Its lineup distribution is

\[
p_k^{\mathrm{dev}}=(1-\eta)b_k+\eta\,\mathbf 1[k=k_{\max}],
\qquad
\eta=\texttt{zero\_league\_deviation\_lineup\_ratio}.
\]

Why it is needed: current-seat return alone can hide a policy that one archived strategy still exploits. This feedback explicitly trains the current policy on the detected one-deviator profile. Turning `zero_league_use_deviation` off yields steps 1-5 with no other code or parameter change.

This score is **not an empirical CCE gap**. It searches only archived policies and only the supported shared-current/shared-historical lineup family. If $k_{\max}<N-1$, it measures a multi-seat historical deviation rather than a unilateral one.

## Parameter reference

| Parameter | Default | Meaning and tuning effect |
|---|---:|---|
| `zero_use_population` | `false` | Must be `true`; enables current-versus-history actors and trainable-seat filtering. |
| `zero_use_league` | `false` | Enables all league behavior. `false` preserves legacy round-robin population sampling. |
| `zero_population_size` | `2` | Maximum active historical checkpoints, including the champion when it occupies a slot. Larger values add diversity but spread observations more thinly. |
| `zero_population_snapshot_interval` | `50` | Checkpoint spacing in outer-iteration units before multiplication by `learner_training_step`. Smaller values give more recent opponents. |
| `zero_population_current_seat_min` | `1` | Smallest $k$; also the champion-gate lineup size. For four-player Blokus, keep `1`. |
| `zero_population_current_seat_max` | `3` | Largest $k$; used by the deviation test. Set to $N-1$ for a unilateral historical deviator. |
| `zero_population_balance_seats` | `false` | Uses $b_k\propto1/k$, equalizing expected trainable-position mass across lineup sizes. |
| `zero_population_hard_ratio` | `0.7` | $\rho$, hard-lineup mixture strength. `0` disables hard lineup-size adaptation without disabling hard opponent choice. |
| `zero_population_temperature` | `0.2` | $T$, hard-lineup softmax temperature. Smaller values concentrate more sharply on low-utility lineup sizes. |
| `zero_population_rotation_interval` | `10` | Used only by legacy population rotation; ignored by league scheduling. |
| `zero_league_refresh_interval` | `20` | Number of outer iterations per frozen-pool block. Longer blocks improve estimates but react more slowly to new checkpoints. |
| `zero_league_rank_weight` | `0.75` | $\lambda$, rank contribution to league utility. It is independent of `actor_rank_utility_weight`. |
| `zero_league_self_ratio` | `0.30` | Base worker weight for current-only self-play. |
| `zero_league_champion_ratio` | `0.20` | Base worker weight for the stable champion anchor and gate evidence. |
| `zero_league_frontier_ratio` | `0.25` | Base worker weight for near-even opponents, usually the most learning-efficient games. |
| `zero_league_hard_ratio` | `0.15` | Base worker weight for hard opponents. When step 6 detects a deviation, it uses this same budget, so total game count does not grow. |
| `zero_league_coverage_ratio` | `0.10` | Base worker weight for least-observed opponents. Increase it if a larger pool is sampled unevenly. |
| `zero_league_min_games` | `32` | Minimum $n$ for frontier/hard confidence decisions, champion promotion, and deviation activation. |
| `zero_league_confidence_scale` | `1.0` | $c$, standard-error multiplier. Larger values make gates/deviation more conservative but make hard selection more uncertainty-seeking. |
| `zero_league_champion_margin` | `0.0` | Required champion-gate LCB. Increase to reduce champion churn. |
| `zero_league_use_deviation` | `true` | Enables step 6. Set `false` for the otherwise identical 1-5 method. |
| `zero_league_deviation_margin` | `0.05` | Restricted deviation threshold. Larger values respond only to clearer archived exploiters. |
| `zero_league_deviation_lineup_ratio` | `0.5` | $\eta$, extra probability mass assigned to $k_{\max}$ during deviation response. |

The five base role ratios are normalized, so they do not have to sum to one, but at least one must be positive. Invalid negative ratios, invalid mixing weights, or league-without-population configurations fail at actor startup.

## Runtime and theory boundary

The number of requested self-play games and optimization steps is unchanged. A population worker still holds the current model and at most one historical model; a self-role worker uses only the current model. Server work is $O(|P|K)$ over active opponents and lineup sizes and is negligible compared with MCTS. The method therefore adds no required environment rollouts, payoff-table evaluation phase, or best-response training job. It may change wall time slightly through checkpoint loading and through different game lengths.

Full [PSRO](https://papers.nips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html) constructs and evaluates a restricted meta-game and repeatedly trains response oracles. [JPSRO](https://proceedings.mlr.press/v139/marris21a.html) additionally maintains distributions over joint policy profiles and uses CE/CCE meta-solvers for multiplayer general-sum games. Those components are intentionally absent here. Consequently, this league has no Nash, CE, or CCE convergence guarantee and the restricted deviation score is a diagnostic/control signal, not exploitability.

The implementation is a practical multiplayer training framework for shared-policy $N>2$ games: it should improve opponent coverage, regression resistance, and response to archived exploiters at roughly the current cost. Whether it converges faster or wins more than the existing sampler remains an empirical question; use balanced held-out opponents and seat permutations to compare the two proposed runs.
