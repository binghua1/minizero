# JPSRO-Guided Opponent Pool

This framework keeps MiniZero's single continuing AlphaZero model and uses a
small JPSRO empirical game only to choose useful frozen opponents. It targets
strong final-model performance under a fixed budget while separately retaining
an empirical CCE certificate for the frozen population.

It is not full JPSRO: the training distribution contains hard and historical
coverage in addition to the CCE response objective. The final CURRENT model is
therefore not itself guaranteed to be a CCE. The certificate applies only to
`meta/meta_strategy.json` over the evaluated frozen population.

## Training loop

1. Train CURRENT from random initialization for the bootstrap iterations.
2. Freeze that in-run checkpoint as `p0`, evaluate its empirical game, and
   solve a CCE.
3. Generate a weighted plan containing hard, CCE, and historical joint
   profiles with one through `N-1` CURRENT seats.
4. Continue the same CURRENT model for one meta interval.
5. Freeze the candidate and estimate its unilateral gains against the previous
   CCE support.
6. Admit it only where its lower-confidence gain is positive.
7. If admitted, fill the new restricted payoffs, solve a new CCE, and regenerate
   the plan. Otherwise CURRENT continues, but is not added to the certificate.

There is no separately pretrained AlphaZero warm start. By default one
MiniZero server, learner, and set of self-play worker processes stay alive for
the complete run. At each meta boundary the server pauses after optimization,
the controller updates the empirical game, and those same workers load only
the changed frozen-policy slots before continuing. CUDA is not reinitialized.

## Training distribution

The global distribution is

\[
q_{\mathrm{train}}
=w_0q_{\mathrm{current}}+w_hq_{\mathrm{hard}}
+w_cq_{\mathrm{CCE}}+w_uq_{\mathrm{history}},
\qquad w_0+w_h+w_c+w_u=1.
\]

Defaults are

\[
(w_0,w_h,w_c,w_u)=(0.15,0.60,0.10,0.15).
\]

`current` preserves on-policy learning; `hard` targets low-return roles;
`CCE` prevents total collapse onto one counter; and `history` reduces
forgetting through uniform support and homogeneous-policy coverage.

For frozen profile \(a\), CURRENT-seat mask \(M\), candidate \(\pi\), and
confidence multiplier \(c\), hardness is

\[
H(a,M)=\frac{1}{|M|}\sum_{i\in M}
\left[-\hat u_i(\pi_i,a_{-i})+c\,\widehat{SE}_i(\pi_i,a_{-i})\right].
\]

The hard proposal is

\[
q_{\mathrm{hard}}(a,M)\propto
\sigma_{\mathrm{CCE}}(a)P(M)\exp\left(\frac{H(a,M)}{\tau}\right).
\]

Missing payoff estimates use zero hardness and fall back to the CCE prior.
`GUIDED_HARD_TEMPERATURE` is \(\tau\).

With role balancing, the probability of selecting `k` CURRENT seats is

\[
P(k)=\frac{1/k}{\sum_{j=k_{\min}}^{k_{\max}}1/j},
\]

and masks of the same size are uniform. Thus 1v2/1v3 games occur more often,
while each configuration contributes comparable CURRENT training mass.

## Admission and CCE certificate

A candidate is admitted for player `i` only when

\[
\widehat\Delta_i-c_a\widehat{SE}(\widehat\Delta_i)>\Delta_{\min}.
\]

`GUIDED_ADMISSION_CONFIDENCE` is \(c_a\) and `GUIDED_ADMISSION_GAIN` is
\(\Delta_{\min}\).

For frozen-population distribution \(\sigma\), the reported gap is

\[
\operatorname{Gap}(\sigma)=
\max_{i,\pi'_i}\mathbb E_{a\sim\sigma}
\left[u_i(\pi'_i,a_{-i})-u_i(a)\right].
\]

This is a restricted empirical-game statement subject to payoff sampling
error. It does not guarantee win rate, full-game convergence under a finite
neural oracle, or equilibrium of the final CURRENT checkpoint.

## Commands

TicTacMo, 50 iterations:

```bash
GUIDED_TOTAL_ITERATIONS=50 \
GUIDED_GPU=0 \
GUIDED_SELFPLAY_GPU=0 \
GUIDED_PORT=10031 \
tools/multiplayer-guided-pool.sh \
  tictacmo \
  runs/tictacmo_guided_pool_50_s0 \
  tictacmo_multiplayer_pool_50.cfg
```

Connect3x3, 100 iterations:

```bash
GUIDED_TOTAL_ITERATIONS=100 \
GUIDED_GPU=0 \
GUIDED_SELFPLAY_GPU=0 \
GUIDED_PORT=10032 \
tools/multiplayer-guided-pool.sh \
  connect3x3 \
  runs/connect3x3_guided_pool_100_s0 \
  connect3x3.cfg
```

TicTacMo/Connect3x3 default to three players and Blokus to four. Other games
set `GUIDED_NUM_PLAYERS`. Repeating an identical command resumes the run;
`GUIDED_TOTAL_ITERATIONS` may be increased to extend it. Other changed settings
require a new run directory, and the saved total cannot be decreased. In the
default persistent mode, total iterations must equal bootstrap iterations plus
an integer multiple of the meta interval.

## Parameters

| Variable | Default | Meaning |
|---|---:|---|
| `GUIDED_TOTAL_ITERATIONS` | 30 | Total MiniZero iterations |
| `GUIDED_BOOTSTRAP_ITERATIONS` | 10 | First checkpoint from the same run |
| `GUIDED_META_INTERVAL` | 10 | Iterations between candidate/meta updates |
| `GUIDED_CURRENT_RATIO` | 0.15 | All-CURRENT mass \(w_0\) |
| `GUIDED_HARD_RATIO` | 0.60 | Hard-profile mass \(w_h\) |
| `GUIDED_CCE_RATIO` | 0.10 | CCE-profile mass \(w_c\) |
| `GUIDED_HISTORY_RATIO` | 0.15 | Historical mass \(w_u\) |
| `GUIDED_POPULATION_SIZE` | 8 | Soft cap; CCE-support policies are never pruned |
| `GUIDED_HARD_TEMPERATURE` | 0.20 | Hard softmax temperature \(\tau\) |
| `GUIDED_HARD_CONFIDENCE` | 1.0 | Uncertainty bonus in hard scoring |
| `GUIDED_CURRENT_SEAT_MIN` | 1 | Minimum CURRENT seats |
| `GUIDED_CURRENT_SEAT_MAX` | `N-1` | Maximum CURRENT seats |
| `GUIDED_EVAL_GAMES` | 20 | Payoff samples per requested profile |
| `GUIDED_ADMISSION_GAIN` | 0.02 | Minimum lower-confidence gain |
| `GUIDED_ADMISSION_CONFIDENCE` | 2.0 | Admission confidence multiplier |
| `GUIDED_TOLERANCE` | 0.01 | Empirical CCE-gap target |
| `GUIDED_GAMES_PER_ITERATION` | 2000 | Self-play games per iteration |
| `GUIDED_TRAINING_STEPS` | 500 | Learner steps per iteration |
| `GUIDED_LEARNER_BATCH` | 1024 | Learner batch size |
| `GUIDED_SELFPLAY_WORKERS` | 4 | Parallel self-play workers |
| `GUIDED_SELFPLAY_BATCH` | 64 | Games per self-play worker batch |
| `GUIDED_EVAL_BATCH` | 64 | Batched payoff games |
| `GUIDED_EVAL_THREADS` | 4 | Payoff-evaluation CPU threads |
| `GUIDED_SIMULATIONS` | 50 | Training/evaluation MCTS simulations |
| `GUIDED_SEED` | 0 | Experiment seed |
| `GUIDED_GPU` | 0 | MiniZero GPU-index string; e.g. `0123` uses four GPUs |
| `GUIDED_SELFPLAY_GPU` | `GUIDED_GPU` | One index per self-play worker |
| `GUIDED_EVAL_GPU` | first guided GPU | Single payoff-evaluation GPU |
| `GUIDED_PORT` | 10021 | Zero-server port |
| `GUIDED_PERSISTENT_WORKERS` | true | Keep server, learner, and SP processes alive across meta updates |

The four distribution ratios must be non-negative and sum to one. The
10-iteration/20-game defaults intentionally spend less on payoff evaluation
than the earlier five-iteration/50-game Adaptive JPSRO run.

## Outputs

- `training/model/weight_iter_*.pt`: continuing single model for deployment.
- `frozen/p*.pt`: frozen candidates/population policies.
- `meta/meta_strategy.json`: population CCE and empirical gap.
- `meta/payoffs.json`: payoff means, variances, counts, and sample IDs.
- `guided/guided_after_*.tsv`: weighted plan loaded by self-play workers.
- `guided/guided_after_*.tsv.json`: hard/CCE/history contribution per plan row.
- `guided/admission_i*.json`: gains, standard errors, and admission decision.
- `evaluations/eval_*`: batched payoff games and manifests.
- `controller_state.json`: completed controller stages used for safe resume.

The learner, network architecture, MCTS budget, and rolling replay
implementation are unchanged. The same learner process retains its replay
state across meta boundaries, and every accepted iteration SGF remains on
disk. Workers can finish a small number of games after the iteration quota is
reached; only these uncommitted old-profile queue entries are discarded at a
meta boundary so they cannot leak into the next iteration.

Opponent games may load up to `N-1` frozen networks. Profiles are assigned per
worker and existing network slots are reused, so a boundary reloads only model
paths that changed. Payoff evaluation is isolated from the training server and
never enters training replay. Its actor process is also persistent across
payoff profiles. Set `GUIDED_PERSISTENT_WORKERS=false` only for compatibility
with the older segmented quick-run controller.
