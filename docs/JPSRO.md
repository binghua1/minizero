# MiniZero JPSRO

This branch adds a game-independent JPSRO layer for two to six player
MiniZero AlphaZero environments. It is separate from `zero_use_population`:
the old population sampler is a training heuristic, while JPSRO explicitly
stores an empirical game, solves a joint meta-strategy, trains a unilateral
response, and measures its deviation gain.

## Components

- A frozen-policy registry in `policies.json`.
- An ordered joint payoff table with online means, variances, and counts in
  `payoffs.json`.
- A dependency-free no-regret CCE solver and empirical certificate in
  `meta_strategy.json`.
- Evaluation manifest generation and ingestion through
  `tools/multiplayer-eval.py`.
- A weighted oracle plan consumed by `ZeroServer`.
- Per-seat neural networks in `ZeroActor`; exactly one responder seat is
  trainable and all other seats use frozen policies.
- A shared physical pool for symmetric games such as Blokus, while each joint
  profile may still assign different policies to different seats.

Normal AlphaZero is unchanged when `zero_use_jpsro=false`.

## Automatic budgeted TicTacMo training

`tools/tictacmo-jpsro.sh` runs one from-scratch training directory and performs
the JPSRO generation boundaries automatically. No external AlphaZero model is
loaded. With the default boundaries `5,15,30` it:

1. trains iterations 1--5 from random initialization and freezes `p0`;
2. evaluates `p0`, solves its initial CCE, and writes the first oracle plan;
3. continues the same learner through iteration 15 against that plan, freezes
   `p1`, fills the expanded payoff table, and updates the CCE;
4. continues through iteration 30 against the new CCE, freezes `p2`, and
   writes the final empirical CCE and deviation reports.

Run it inside the MiniZero container:

```bash
tools/tictacmo-jpsro.sh runs/tictacmo_jpsro_s0 path/to/tictacmo.cfg
```

The command is restartable: running the identical command again continues the
current training phase and resumes arena games. The defaults match the existing
TicTacMo budget: 2000 games and 500 learner steps per iteration, learner batch
1024, 50 MCTS simulations, self-play batch 32, four CPU threads, and seed 0.

Set generation boundaries explicitly for a different total budget:

```bash
JPSRO_BOUNDARIES=50,100,150,200,250,300 \
  tools/tictacmo-jpsro.sh runs/tictacmo_jpsro_300_s0 path/to/tictacmo.cfg
```

Other overrides are `JPSRO_GAMES_PER_ITERATION`, `JPSRO_TRAINING_STEPS`,
`JPSRO_LEARNER_BATCH`, `JPSRO_SELFPLAY_BATCH`, `JPSRO_CPU_THREADS`,
`JPSRO_EVAL_GAMES`, `JPSRO_EVAL_THREADS`, `JPSRO_EVAL_NOISE`,
`JPSRO_SIMULATIONS`, `JPSRO_GPU`, `JPSRO_SEED`, `JPSRO_PORT`, and
`JPSRO_TOLERANCE`.

`training/model/` contains the continuously warm-started learner checkpoints;
`frozen/p*.pt` are immutable population members. The latest checkpoint is the
single-model result. `meta/meta_strategy.json` is the population result: sample
one whole joint profile once per game to preserve CCE correlation. Each solved
generation is also preserved as `meta/meta_strategy_g0.json`,
`meta/meta_strategy_g1.json`, and so on; `meta_strategy.json` always points to
the latest solved generation.

## Formulas and guarantees

For player (i), let (\Pi_i) be its finite policy population and let
(u_i(\boldsymbol\pi)) be its payoff under ordered joint profile
(\boldsymbol\pi=(\pi_1,\ldots,\pi_n)). The stored empirical payoff is

\[
\hat u_i(\boldsymbol\pi)=\frac{1}{m_{\boldsymbol\pi}}
\sum_{g=1}^{m_{\boldsymbol\pi}}r_{i,g}.
\]

A distribution (\sigma) over joint profiles is an (\epsilon)-CCE when

\[
\max_{i,\pi'_i\in\Pi_i}
\sum_{\boldsymbol\pi}\sigma(\boldsymbol\pi)
\left[\hat u_i(\pi'_i,\boldsymbol\pi_{-i})-
      \hat u_i(\boldsymbol\pi)\right]\le\epsilon.
\]

`tools/jpsro.py solve` reports the left side as `gap`. This is an exact
certificate for the **complete empirical restricted game** represented by the
stored payoff means. It is not automatically a certificate for the original
game. That stronger statement additionally requires exact expected payoffs and
exact best-response oracles. With finite matches and AlphaZero as an
approximate oracle, this is scalable approximate JPSRO. The payoff standard
errors and frozen-candidate gains make the approximation visible.

The meta-solver uses simultaneous full-information Hedge. At step (t),

\[
q_i^t(a)=\frac{\exp(w_{i,a}^t)}{\sum_b\exp(w_{i,b}^t)},\qquad
w_{i,a}^{t+1}=w_{i,a}^t+\eta\,\tilde u_i(a,q_{-i}^t),
\]

where (\tilde u\in[0,1]) is the configured normalized utility. The output is
(\sigma_T=T^{-1}\sum_t\prod_iq_i^t). For payoff range
(R=u_{max}-u_{min}), the reported conservative Hedge bound is

\[
R\left(\frac{\log |\Pi|}{\eta T}+\frac{\eta}{8}\right).
\]

The measured empirical gap is normally the useful stopping signal.

## Why only one trainable seat

A JPSRO oracle estimates a unilateral deviation:

\[
BR_i(\sigma)=\arg\max_{\pi'_i}
\mathbb E_{\boldsymbol\pi\sim\sigma}
[u_i(\pi'_i,\boldsymbol\pi_{-i})].
\]

Each oracle game therefore marks exactly one responder seat `TR=1`. Moves from
all other seats are `TR=0`, and the existing replay loader excludes them. For a
shared-policy Blokus run, the responder position rotates, so one network learns
all four positions without turning a game into four simultaneous deviations.
The 1v3, 2v2, and 3v1 compositions still exist as payoff/evaluation profiles;
they are not simultaneous trainable deviations.

## Recorded fields and parameters

`ZeroActor` records:

- `JI`: oracle profile identifier.
- `JP`: policy ID assigned to every seat.
- `TM`: profile-level trainable mask.
- `PI`: policy ID that produced a move.
- `TR`: whether that move may enter replay.

New MiniZero configuration:

- `zero_use_jpsro` (default `false`): enable oracle-profile training.
- `zero_jpsro_profile_file` (default empty): absolute path to the TSV generated
  by `oracle-plan`.
- `zero_jpsro_replay_start_iteration` (default `1`): first replay iteration
  belonging to the current response oracle. The automatic controller advances
  it at each generation boundary.

Set `zero_use_population=false`; the server rejects enabling both modes.

Workflow and solver parameters:

- `--shared-pool`: allow every logical player to use the same frozen artifacts.
- `--utility-min`, `--utility-max`: known payoff range used by Hedge.
- `--games-per-profile`: arena repeats for each ordered profile. Enable noise or
  random rotations if deterministic repeats would be identical.
- `--min-games`: samples required before a payoff is accepted.
- `--iterations`: maximum CPU-only Hedge updates.
- `--tolerance`: target empirical CCE gap.
- `--eta`: optional Hedge learning rate; normally leave unset.
- `--responders`: zero-based seats trained by an oracle plan. Use `all` for
  shared-policy Blokus.
- `--max-profiles`: split a large payoff evaluation into batches.

## One generation

Start with a frozen four-player Blokus10 policy:

```bash
python3 tools/jpsro.py init runs/blokus10_jpsro --players 4 --shared-pool
python3 tools/jpsro.py add-policy runs/blokus10_jpsro p0 /abs/path/weight_iter_50000.pt

python3 tools/jpsro.py make-eval runs/blokus10_jpsro runs/blokus10_jpsro/eval_g0/arena.json \
  --game blokus10 --conf-file /abs/path/blokus10.cfg \
  --executable build/blokus10/minizero_blokus10 --games-per-profile 8 \
  --num-simulations 50 --noise
python3 tools/multiplayer-eval.py runs/blokus10_jpsro/eval_g0/arena.json \
  runs/blokus10_jpsro/eval_g0 -g 0 --num_threads 2
python3 tools/jpsro.py ingest runs/blokus10_jpsro \
  runs/blokus10_jpsro/eval_g0/arena.json runs/blokus10_jpsro/eval_g0/games.jsonl
python3 tools/jpsro.py solve runs/blokus10_jpsro --min-games 8 --tolerance 0.01

python3 tools/jpsro.py oracle-plan runs/blokus10_jpsro \
  runs/blokus10_jpsro/oracle_g1.tsv --responders all
```

Train a fresh AlphaZero oracle from the desired initialization with:

```ini
zero_use_population=false
zero_use_jpsro=true
zero_jpsro_profile_file=/abs/path/runs/blokus10_jpsro/oracle_g1.tsv
```

Freeze and register its final checkpoint, then first evaluate only unilateral
deviations:

```bash
python3 tools/jpsro.py add-policy runs/blokus10_jpsro p1 /abs/path/oracle_g1.pt --generation 1
python3 tools/jpsro.py make-eval runs/blokus10_jpsro runs/blokus10_jpsro/deviation_g1/arena.json \
  --candidate p1 --game blokus10 --conf-file /abs/path/blokus10.cfg \
  --executable build/blokus10/minizero_blokus10 --games-per-profile 8 --noise
# Run multiplayer-eval.py and ingest its games.jsonl as above.
python3 tools/jpsro.py deviation-gap runs/blokus10_jpsro p1
```

If the frozen candidate has a meaningful positive gain, run `make-eval`
without `--candidate` to fill the expanded payoff table, then solve again. If
the gain is below the tolerance and uncertainty is small, stop. This two-stage
evaluation avoids paying for the full cross product when an oracle did not
improve.

## Runtime, scaling, and deployment

Oracle self-play still performs one MCTS search per move. Extra training time
comes from loading at most (n-1) frozen models once per self-play worker and
MiniZero iteration. Identical frozen artifacts used at several seats are
deduplicated on each GPU. Frozen models keep stable GPU slots across lineup and
seat changes, so an already resident model is not loaded again. Memory is
approximately one current model plus the distinct frozen models in the
selected profile, not the full population.

`tools/quick-run.sh -b` controls self-play parallel games and therefore the
neural-network inference batch. It is different from `learner_batch_size`,
which only changes optimizer batches. The automatic experiment retains
`-b 32 -c 4` so that the primary compute-matched comparison changes as few
variables as possible. Increasing `-b` is an optional throughput experiment,
not part of the primary result.

Only one of (n) seats is trainable in each oracle game. This reduces usable
replay positions per game to about (1/n), but it does not make each MCTS game
(n) times slower. Matching the baseline's number of distinct trainable
positions would require more games. For a clean from-scratch method comparison,
keep the baseline and oracle compute budgets equal and explicitly report that
the oracle receives fewer trainable positions; a separate data-matched run
would require about (n) times as many games and answers a different question.

The complete payoff table grows as (\prod_i|\Pi_i|), or (K^n) for a shared
pool of size (K). Four-player generations with (K=2) and (K=3) need 16
and 81 ordered profiles. Keep early populations small, use the unilateral test,
split evaluation with `--max-profiles`, and stop based on gain and uncertainty.
Pruning is faster, but then the certificate applies only to the retained game.

`meta_strategy.json` contains the joint CCE distribution and each seat's
marginal. Joint sampling preserves CCE correlation. Independently sampling the
marginals is easier but generally loses the CCE guarantee. A single checkpoint
can still be selected for ordinary competition, but its win rate is empirical,
not guaranteed by CCE theory. Use the existing checkpoint self-eval and
1v3/2v2/3v1 arenas to choose that deployment checkpoint.
