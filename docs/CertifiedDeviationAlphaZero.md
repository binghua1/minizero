# Certified-Deviation AlphaZero (CD-AZ)

CD-AZ is an opt-in multiplayer AlphaZero policy-improvement interface. It is
designed to reduce the destructive policy jumps caused by fitting finite,
high-variance MCTS visit counts. It does not change a game's action encoding
and does not assume a spatial or factorized action space.

The implementation is backward compatible. `actor_policy_target_type=visit`
is the default and reproduces the existing replay target. All existing games,
console modes, arena evaluation, and self-evaluation continue to use the same
interfaces.

## Core update

At a root state, let `pi_ref` be the frozen network policy restricted and
renormalized over legal actions, and let `pi_search` be the normalized MCTS
visit distribution. For each root edge, MCTS now retains its sample mean,
second moment, visit count, and a prior-regularized uncertainty radius.

CD-AZ computes

```text
V_ref       = sum_a pi_ref(a) Q(a)
Delta       = sum_a pi_search(a) [Q(a) - V_ref]
uncertainty = sqrt(sum_a [pi_search(a) - pi_ref(a)]^2 radius(a)^2)
gate        = sigmoid((Delta - uncertainty) / temperature)
```

It then selects the largest `alpha <= gate` satisfying

```text
KL((1-alpha) pi_ref + alpha pi_search || pi_ref) <= kl_budget
```

and writes that projected policy as the replay target. The update therefore
has one central mechanism: search proposes a policy, deviation evidence
controls how much of that proposal is accepted, and a KL ball caps the accepted
step. An inconclusive finite search still supplies a soft learning signal;
stronger evidence permits a larger step.

The learner also receives the frozen root policy and can apply

```text
lambda_ref * KL(pi_ref || pi_theta)
```

as an anchor. This is deliberately a separate ablation switch. The default is
zero for conventional AlphaZero and the smoke configuration uses `0.05` for
CD-AZ.

## Why this can be more stable

Standard AlphaZero treats every MCTS visit target as equally trustworthy. In a
multiplayer tree, the target can move sharply because every player's policy
changes the continuation distribution, shallow MaxN estimates are noisy, and
the replay buffer mixes targets produced by several historical joint policies.
CD-AZ limits that feedback loop at the search-to-learner boundary. The reference
policy is fixed during each self-play iteration, uncertainty decreases with
visits, and every target has an explicit measured KL from its reference.

This is an empirical finite-search safeguard, not a formal neural-network
convergence theorem. The stored confidence radius does not account for all
function-approximation, non-stationarity, or tree-correlation errors. The name
"certified" refers to the checked empirical update condition; papers and
reports must not describe it as a formal high-probability exploitability
certificate without additional calibration theory.

## Generality

The method requires a turn-based environment with discrete legal actions, a
simulator, a policy prior, and scalar utility for the acting player on each
edge. It supports any player count handled by `PlayerValues`, both MaxN and
Paranoid backup, and arbitrary policy sizes. It does not use Blokus pieces,
board geometry, action factorization, or a game-specific neural head.

The current MiniZero multiplayer restrictions still apply: multiplayer MuZero,
Gumbel search, resignation, value rescaling, and intermediate sequences are not
enabled. These are implementation boundaries, not assumptions of the update.

## Replay diagnostics

When CD-AZ is enabled, each move stores:

- `P`: projected CD-AZ training target;
- `A`: frozen legal reference policy;
- `N`: raw MCTS visit distribution;
- `D`: `action_id:empirical_deviation:visit_count` diagnostics;
- `K`: achieved `KL(P || A)`.

Old records without `A` remain loadable; the data loader falls back to `P`.

## Running the paired experiment

Build and test first:

```bash
scripts/build.sh tictacmo release
cmake -S . -B build/tictacmo-test -DGAME_TYPE=TICTACMO \
  -DCMAKE_BUILD_TYPE=Debug -DMINIZERO_BUILD_TESTS=ON
cmake --build build/tictacmo-test -j
ctest --test-dir build/tictacmo-test --output-on-failure
```

Run a same-seed baseline/CD-AZ smoke ablation:

```bash
scripts/run-cdaz-ablation.sh tictacmo 8 \
  config/cdaz/tictacmo_smoke.cfg experiments/cdaz_smoke 60
```

The script trains both modes and runs the existing multiplayer checkpoint
self-evaluation. For a paper experiment, use at least five training seeds,
hundreds of balanced arena games per checkpoint, a common-opponent arena, and
report bootstrap confidence intervals for area under the strength curve,
regressions between adjacent checkpoints, final strength, target KL, and wall
clock/sample efficiency.

## Initial smoke result (2026-07-21)

A three-iteration TicTacMo run (128 self-play games and 50 learner steps per
iteration, 24 simulations, seed 17) established that the path compiles, trains,
serializes the new replay fields, and evaluates without engine errors.

The first per-action exponential version was too conservative and lost badly to
the conventional final checkpoint. It was replaced by the current
evidence-gated projection of the search visit policy. In a 24-game balanced
mixed-lineup check, that revision narrowed the gap but did not beat baseline:
baseline recorded 12 wins and 8 draws across its 36 seat appearances, while
CD-AZ recorded 8 wins and 4 draws. In a cleaner arena containing one baseline
checkpoint, one CD-AZ checkpoint, and one shared initial checkpoint in every
game, baseline won 8/24, CD-AZ won 4/24, and the initial checkpoint won 12/24.

These samples are small and highly deterministic, so they are not statistical
evidence of convergence. More importantly, they do not support a claim that
CD-AZ is already better. The branch should be treated as a complete experimental
implementation with a falsifiable protocol, not as a confirmed top-conference
result. The next decision gate is whether longer, multi-seed runs improve area
under the strength curve and reduce checkpoint regressions without lowering
final strength.
