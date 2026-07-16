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

- random, greedy largest-piece, and rollout MCTS;
- multiplayer AlphaZero with MaxN and Paranoid;
- vector MuZero with MaxN and Paranoid;
- proposed population-conditioned/equilibrium-aware MuZero;
- exhaustive versus sampled/progressive-widening action search;
- single latest-opponent self-play, league self-play, PSRO/JPSRO mixture;
- scalar, winner-only vector, raw-score vector, and centered score-margin targets;
- with and without player permutation equivariance and counterfactual consistency.

Report environment steps, model inferences, simulations, wall-clock time, GPU hours, and parameter count. Match inference budgets rather than comparing only equal simulation counts.

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
