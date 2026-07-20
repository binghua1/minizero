# Research roadmap: MuZero in multiplayer competitive games

## Recommended thesis

The strongest project is not merely "MuZero plays four-player Blokus." A publishable claim is:

> A population-conditioned, vector-valued MuZero with equilibrium-aware search remains robust under opponent and coalition shift while scaling to large combinatorial action spaces.

Blokus is a useful hard benchmark: it is deterministic and perfect-information, yet it has four selfish players, score margins, seat effects, temporary blocking incentives, a large structured action space, and policies that can overfit to a particular opponent population. Add small OpenSpiel general-sum games for exact equilibrium diagnostics and at least one second scalable domain to establish generality.

## Core algorithm

1. Predict an absolute utility vector `v in R^N`, not a sign-flipped scalar. Predict immediate reward vectors as well if the environment has intermediate rewards.
2. Condition representation, dynamics, policy, and value on the acting player and a population/opponent code. Use a permutation-equivariant player encoder so changing seat labels does not create a new game.
3. Replace scalar minimax backup with an explicit solution concept:
   - MaxN as the selfish baseline;
   - Paranoid as coalition-robust baseline;
   - response-conditioned or CCE-aware backup as the proposed method.
4. Train a population, not a single self-play checkpoint. A JPSRO/NeuPL-style meta-solver supplies opponent mixtures and measures unilateral deviation incentives.
5. Use sampled or progressive-widening search over Blokus's structured `(piece, orientation, anchor)` action space. Correct the sampled priors as in Sampled MuZero.
6. Add counterfactual opponent-policy consistency: recurrent latent states reached by actions from held-out opponents should predict utility vectors and policies correctly, not only trajectories near the current self-play policy.

## Falsifiable hypotheses

- A scalar/root-relative MuZero overfits to the training lineup; vector prediction improves cross-play but does not by itself solve strategic non-stationarity.
- Population-conditioned latent dynamics reduce value error and regret against unseen opponent mixtures.
- CCE/response-aware search lowers empirical deviation incentive and coalition exploitability compared with MaxN, without sacrificing average score.
- Sampled spatial search matches exhaustive-search strength at a fixed wall-clock budget and scales to action spaces where enumeration is impractical.
- Counterfactual consistency improves deeper search specifically under opponent shift; if it only improves in-distribution value error, the proposed mechanism is not supported.

## Required baselines and ablations

- random, greedy largest-piece, flat rollout, and uninformed/vanilla MCTS;
- multiplayer AlphaZero with MaxN and Paranoid;
- vector MuZero with MaxN and Paranoid;
- proposed population-conditioned/equilibrium-aware MuZero;
- exhaustive versus sampled/progressive-widening action search;
- single latest-opponent self-play, league self-play, PSRO/JPSRO mixture;
- scalar, winner-only vector, raw-score vector, and centered score-margin targets;
- with and without player permutation equivariance and counterfactual consistency.

Report environment steps, model inferences, simulations, wall-clock time, GPU hours, and parameter count. Match inference budgets rather than comparing only equal simulation counts.

## Pure algorithmic Blokus baselines in this repo

`tools/blokus-baseline-agent.py` is a model-free Blokus console engine.  It uses the
same action id catalogue as `minizero/environment/blokus`: 91 piece orientations,
400 board anchors, and `PASS=36400`.  It can therefore be seated directly against
MiniZero neural console engines by `tools/multiplayer-eval.py`.

Implemented policies:

- `random`: uniformly sample a legal action.
- `greedy`: play the largest legal remaining piece.
- `greedy_mobility`: prefer large pieces, then more future diagonal contact points.
- `rollout`: a flat rollout baseline.  It evaluates top legal candidate moves by
  random/greedy playouts using only game rules and final centered Blokus utilities.
  This is not the same as the `UninformedMCTSPlayer` baseline in Petosa and Balch's
  Multiplayer AlphaZero code, which runs tree-search simulations with a uniform-prior,
  zero-value `DumbNet`.
- `uct_maxn`: pure UCT-style tree search with no neural network.  At each node the
  acting player selects by its own backed-up centered return plus UCB exploration.
  The default leaf evaluator is `zero`, matching a DumbNet-style uninformed MCTS.
- `uct_paranoid`: pure UCT-style tree search with no neural network.  The root player
  maximizes its own backed-up return; non-root players are treated as a coalition
  minimizing the root player's return.  The default leaf evaluator is also `zero`.

Run model-versus-algorithm baselines:

```bash
MODEL=55500 GAMES_PER_SEATING=1 GPU=0123 \
  scripts/run-blokus-uct-sweep.sh blokus_maxn_smoke_01
```

The script calls `tools/multiplayer-eval.py blokus-baseline` and sweeps:

- model search: `SEARCH_TYPE=maxn` by default, override with `SEARCH_TYPE=paranoid`;
- model simulations: `MODEL_SIMS=50` by default;
- baselines: `BASELINES="random greedy greedy_mobility rollout uct_maxn uct_paranoid"` by default;
- flat rollout playouts: `ROLLOUTS=64` by default;
- pure-MCTS simulations: `MCTS_SIMS="50 100 200 400"` by default;
- pure-MCTS leaf evaluation: `MCTS_LEAF_EVAL=zero` by default, matching a
  DumbNet-style uninformed MCTS rather than rollout-to-terminal evaluation;
- pure-MCTS branching cap: `MCTS_CANDIDATE_LIMIT=64` by default.

The generated arena is pairwise seat-balanced: for each baseline it runs model
versus that baseline in 1-vs-3, 2-vs-2, and 3-vs-1 lineups with all unique seat
permutations.  Each `uct_*` simulation count becomes a distinct agent name such
as `alg_uct_maxn_s50` and `alg_uct_paranoid_s400`.  Results are written under
`TRAINING_DIR/evaluation/WEIGHT_SEARCH_vs_BASELINES.../` unless `OUTPUT=...` is set:

- `games.jsonl`: one record per game, including seating, moves, returns, and errors.
- `agent_summary.csv`: win rate and average centered return by agent.
- `seat_summary.csv`: the same metrics split by Blokus seat.
- `seating_summary.csv`: per-lineup/per-seating diagnostics.
- `agent_summary.png`: win rate and average return bar charts.
- `seat_summary.png`: per-seat average return chart.
- `mcts_sweep_summary.csv`: `uct_maxn`/`uct_paranoid` metrics by simulation count.
- `mcts_sweep.png`: paper-style sweep plot with simulations on a log-scale x-axis
  and win rate / average centered return on y-axes.

With the default script settings there are 12 expanded baseline agents:
4 non-MCTS baselines plus `2 * 4 = 8` UCT baselines.  Each pairwise
model-vs-baseline comparison has 14 unique seatings, so one game per seating
requires `12 * 14 = 168` games.  Use multiples of 168 for cleaner balance, for
example `GAMES_PER_SEATING=1` -> 168 games, `GAMES_PER_SEATING=5` -> 840 games.

## Evaluation

- all unique four-seat assignments and multiple random seeds;
- full checkpoint cross-play matrix, not only latest-versus-latest Elo;
- average official score, win/tie rate, centered utility, and per-seat confidence intervals;
- approximate best-response gain / empirical NashConv on Blokus;
- CCE deviation incentive on tractable OpenSpiel games;
- 2-versus-1-versus-1 and 3-versus-1 coalition stress tests;
- held-out scripted strategies and held-out population mixtures;
- latent multi-step value error stratified by search depth and opponent distance.

## Paper-shaping advice

For NeurIPS/ICML/ICLR, the work needs a general algorithmic idea, more than one domain, strong equilibrium diagnostics, compute-matched baselines, and evidence that the learned model—not merely population training—causes the gain. A Blokus-only performance paper is more naturally scoped for AAAI/IJCAI/AAMAS/AIIDE. The most credible top-tier route is to make opponent-shift generalization of value-equivalent models the central problem, derive the response-conditioned objective, and use Blokus as the large structured test bed.

## Starting references

- Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero), 2020.
- Petosa and Balch, *Multiplayer AlphaZero*, 2019.
- Hubert et al., *Learning and Planning in Complex Action Spaces* (Sampled MuZero), ICML 2021.
- Danihelka et al., *Policy Improvement by Planning with Gumbel*, ICLR 2022.
- Liu et al., *Efficient Multi-Agent Reinforcement Learning by Planning* (MAZero), ICLR 2024. This is cooperative CTDE, not competitive general-sum MuZero.
- Liu et al., *Neural Population Learning beyond Symmetric Zero-sum Games* (NeuPL-JPSRO), AAMAS 2024.
- He et al., *What Model Does MuZero Learn?*, ECAI 2024.
- Yang et al., *Incentivize without Bonus: Provably Efficient Model-based Online Multi-agent RL for Markov Games*, ICML 2025.
- Sasaki, *Monte-Carlo Tree Search in the Game of Blokus-Duo*, GPW 2009.
