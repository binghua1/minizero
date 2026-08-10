# Adaptive JPSRO-guided AlphaZero

This branch uses JPSRO as a certified multiplayer opponent-pool manager while
keeping ordinary AlphaZero self-play as the main data source. It is intended
for symmetric, general-sum AlphaZero games with two to six players. The first
controller is `tools/tictacmo-jpsro.sh`; the C++ and Python JPSRO layers are
game-independent.

## Why this is not pure JPSRO

Pure JPSRO trains one unilateral best-response seat per game. In an
`n`-player game that retains only about `1/n` of the positions for the learner.
The 30-iteration TicTacMo pilot consequently lost to the compute-matched
Multiplayer AlphaZero baseline, especially in 1-vs-2 play.

The adaptive method separates two goals:

- AlphaZero all-seat self-play preserves general strength and data throughput.
- JPSRO supplies correlated joint opponents and an empirical CCE certificate
  for the frozen population.

The certificate applies to the population meta-strategy, not to the final
single current network's external win rate.

## Training mixture

Let `rho = zero_jpsro_selfplay_ratio`. At each iteration the server assigns a
stratified fraction of self-play workers to:

\[
\rho\,\text{CURRENT-vs-CURRENT all-seat self-play}
 +(1-\rho)\,\text{one-responder JPSRO games}.
\]

All positions from a normal self-play game are trainable. Only the CURRENT
responder's positions are trainable in a JPSRO game. The expected retained
position fraction relative to ordinary AlphaZero is approximately

\[
\rho + \frac{1-\rho}{n}.
\]

For three players and `rho=0.7`, this is `0.8`, rather than pure JPSRO's
`1/3`. The normal rolling replay buffer is retained across candidate checks.

Opponent profiles are sampled jointly from the CCE. Independently sampling
each player's marginal would destroy the CCE correlation.

The server enforces the ratio on accepted games, not merely on launched
workers. Faster ordinary workers cannot fill the iteration before the required
opponent-game quota arrives. With 2000 games and `rho=0.7`, the saved replay is
exactly 1400 ordinary games and 600 JPSRO opponent games.

## Adaptive population admission

At each candidate boundary, the current checkpoint is evaluated as a
unilateral deviation for every player. For player `i`, it is added to
`Pi_i` only when

\[
\widehat g_i-c\,SE(\widehat g_i)>\delta.
\]

`delta` is `JPSRO_ADMISSION_GAIN` and `c` is
`JPSRO_ADMISSION_CONFIDENCE`. A shared checkpoint may therefore be admitted
for player 1 but rejected for players 0 and 2. If no player passes, the
candidate is removed and training continues against the previous certified
pool.

This avoids the pilot's error of placing one shared candidate in every seat
when it improved only one or two seats.

## CCE formula and support compaction

For player `i`, finite policy set `Pi_i`, ordered joint profile `pi`, and
empirical payoff `u_hat_i(pi)`, a distribution `sigma` is an epsilon-CCE when

\[
\max_{i,\pi'_i\in\Pi_i}
\sum_{\boldsymbol\pi}\sigma(\boldsymbol\pi)
\left[
\hat u_i(\pi'_i,\boldsymbol\pi_{-i})-
\hat u_i(\boldsymbol\pi)
\right]\le\epsilon.
\]

The solver first obtains a CCE on the complete current restricted game. It
then tries top-probability supports from smallest to largest, renormalizes each
one, and recomputes every restricted unilateral-deviation constraint. A
compact support is saved only if its measured gap remains below `epsilon`.
Therefore support compaction does not weaken the empirical restricted-game
certificate.

A rejected candidate requires only about `|S| * n` deviation profiles, where
`S` is the compact CCE support. An admitted candidate still triggers the
missing payoff evaluations needed for the new complete restricted game. This
is deliberate: skipping those payoffs would make the reported CCE certificate
invalid. Per-player admission keeps that product much smaller than a shared
`K^n` pool.

Payoffs are finite-sample estimates, so this is an **empirical restricted-game
CCE guarantee**. Full-game convergence additionally requires exact expected
payoffs, exact best responses, unlimited population growth, and no fixed
training budget; neural MCTS training does not satisfy those assumptions.

## Run from scratch

Inside the MiniZero container:

```bash
JPSRO_GPU=0 \
JPSRO_SELFPLAY_GPU=0 \
JPSRO_SELFPLAY_WORKERS=4 \
JPSRO_SELFPLAY_BATCH=64 \
JPSRO_SELFPLAY_RATIO=0.7 \
JPSRO_EVAL_THREADS=4 \
JPSRO_PORT=10021 \
tools/tictacmo-jpsro.sh \
  runs/tictacmo_adaptive_jpsro_s0 \
  tictacmo_jpsro.cfg
```

This is a fresh run: no external Multiplayer AlphaZero checkpoint is loaded.
Do not reuse a directory created by the old fixed `5,15,30` controller.

`JPSRO_SELFPLAY_WORKERS=4` launches four worker processes.
`JPSRO_SELFPLAY_BATCH=64` runs 64 actors concurrently inside each process, for
up to 256 in-flight games on the selected GPU. Reduce the batch or worker count
if a larger game exceeds GPU memory.

## Controller parameters

- `JPSRO_BOOTSTRAP_ITERATIONS` (default `5`): initial all-seat AlphaZero
  iterations before `p0` is frozen.
- `JPSRO_ORACLE_INTERVAL` (default `5`): iterations between candidate tests.
- `JPSRO_TOTAL_ITERATIONS` (default `30`): fixed total training budget.
- `JPSRO_GAMES_PER_ITERATION` (default `2000`): self-play games per iteration.
- `JPSRO_TRAINING_STEPS` (default `500`): learner updates per iteration.
- `JPSRO_LEARNER_BATCH` (default `1024`): learner minibatch size.
- `JPSRO_SELFPLAY_WORKERS` (default `4`): parallel worker processes.
- `JPSRO_SELFPLAY_BATCH` (default `64`): parallel actors per worker.
- `JPSRO_SELFPLAY_RATIO` (default `0.7`): normal AlphaZero worker fraction.
- `JPSRO_CPU_THREADS` (default `4`): CPU threads per self-play worker.
- `JPSRO_EVAL_GAMES` (default `20`): games per evaluated payoff profile.
- `JPSRO_EVAL_THREADS` (default `4`): parallel arena workers.
- `JPSRO_EVAL_NOISE` (default `true`): stochastic evaluation games.
- `JPSRO_SIMULATIONS` (default `50`): MCTS simulations in training and payoff
  evaluation.
- `JPSRO_ADMISSION_GAIN` (default `0.02`): minimum lower-bound gain.
- `JPSRO_ADMISSION_CONFIDENCE` (default `2.0`): standard-error multiplier.
- `JPSRO_TOLERANCE` (default `0.01`): empirical CCE-gap target.
- `JPSRO_GPU`, `JPSRO_SELFPLAY_GPU`, `JPSRO_SEED`, `JPSRO_PORT`: device,
  reproducibility, and server settings.

## Outputs

- `training/model/`: continuously trained single-network checkpoints.
- `frozen/`: only accepted population checkpoints plus `p0`.
- `admission_i*.json`: player-wise gains, errors, and admission decisions.
- `meta/payoffs.json`: idempotently ingested empirical payoff statistics.
- `meta/meta_strategy.json`: latest compact, certified CCE distribution.
- `meta/meta_strategy_g0.json`, `meta/meta_strategy_i*.json`: preserved solver
  generations.
- `oracle_i*.tsv`: correlated opponent plans used by self-play workers.

The latest checkpoint is the primary single-model result. The CCE population
is a certified opponent curriculum and may also be deployed by sampling one
whole joint profile per game.

## Core MiniZero configuration

```ini
zero_use_population=false
zero_use_jpsro=true
zero_jpsro_profile_file=/absolute/path/to/oracle.tsv
zero_jpsro_selfplay_ratio=0.7
zero_jpsro_num_workers=4
```

`zero_jpsro_replay_start_iteration` remains parseable only so old configuration
files do not fail; the adaptive path intentionally ignores it.
