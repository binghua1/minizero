# Multiplayer AlphaZero

The classic four-player Blokus environment and its large spatial action representation are documented in [Blokus.md](Blokus.md). A proposed path from the current AlphaZero-only multiplayer implementation to competitive general-sum MuZero is in [MultiplayerMuZeroResearch.md](MultiplayerMuZeroResearch.md).

MiniZero's multiplayer AlphaZero path uses an absolute utility vector at network leaves and terminal states. MCTS edges remain scalar, with selectable MaxN or Paranoid backup semantics.

- `actor_multiplayer_search_type=maxn` stores the utility component of the player who selected each edge, so every player maximizes their own value.
- `actor_multiplayer_search_type=paranoid` stores the root player's utility on root-player edges and its negation on every opponent edge, so all opponents act as a coalition minimizing the root player's value.

## Behavior-conditioned population training

The optional population path trains one current policy against historical checkpoints without changing MaxN Q-values or adding Rank/lambda heads. Each self-play worker loads the current model and one historical model. A sampled subset of seats uses the current model for the whole game; only moves from those seats enter replay. Recent actions from every player are encoded by a shared attention encoder and fused into the current model's residual trunk with FiLM, so the policy and absolute vector-value losses can learn state-dependent responses to observed play styles.

Enable it with the following additions to a multiplayer AlphaZero config:

```text
zero_use_population=true
zero_population_size=2
zero_population_snapshot_interval=10
zero_population_rotation_interval=5
zero_population_hard_ratio=0.7
zero_population_temperature=0.2
zero_population_current_seat_min=1
zero_population_current_seat_max=3
nn_use_behavior_conditioning=true
nn_behavior_history_length=16
nn_behavior_embedding_dim=32
```

For a four-player game, the seat range above samples one, two, or three current-model seats. The server increases the probability of lineups on which the current model has lower mean return. Old SGFs remain fully trainable because the per-move `TR` tag defaults to true when absent.

`quick-run.sh` starts one self-play worker per GPU. With one GPU, use `-g 0`; the worker loads only the current model and one historical opponent on GPU 0:

```bash
tools/quick-run.sh train blokus10 blokus10_population.cfg 500 \
  -n blokus10_behavior_population_01 -g 0 -c 8 -p 10001
```

If more GPUs become available, `-g 0123` starts four independent self-play workers and shards the historical opponents across them.

The first historical opponent becomes available after `zero_population_snapshot_interval` iterations. A baseline checkpoint can warm-start the shared trunk; the new behavior modules and optimizer state are initialized from scratch.

The neural network and terminal targets remain the same vector in both modes. Models trained with different search types should use separate training directories because their MCTS policy targets differ.

## Robust league training

The optional robust league adds rank-aware utility, confidence statistics, metric-driven opponent selection, frozen active pools, worker roles, a champion gate, and an independently switchable restricted-deviation response. It does not require behavior conditioning or modify the learner/network architecture. See [RobustLeague.md](RobustLeague.md) for the complete formulas, every parameter, the recommended Blokus 10 settings, runtime implications, and the boundary relative to PSRO/JPSRO.

## Paper implementation audit

The four multiplayer extensions in [Petosa and Balch, *Multiplayer AlphaZero*](https://arxiv.org/pdf/1910.13012) are implemented in this branch:

| Paper extension | MiniZero implementation |
|---|---|
| Rotate through every player instead of alternating two players | `getNextPlayer` and each multiplayer action rotate through all three players. |
| Return a terminal score vector | Multiplayer environments and SGF/self-play records use `[P1, P2, P3]`, such as `[1, -1, -1]`. |
| Back up the component belonging to the player selecting an edge | `MCTS::backup(PlayerValues)` implements player-relative MaxN. Paranoid is an additional baseline not used in the paper. |
| Predict a value vector and train with vector MSE | The AlphaZero value head emits one value per player; the loader preserves the complete vector and the loss averages squared error across players. |

This is an implementation of the paper's multiplayer algorithm, not an exact reproduction of its experimental system. The paper used an eight-block squeeze-and-excitation network, Adam with its reported hyperparameters, first-move-only Dirichlet noise, an unbounded replay buffer, and uninformed-MCTS controls. MiniZero retains its ResNet, training pipeline, replay policy, and root-noise behavior unless a separate replication experiment explicitly changes them.

## Tic-Tac-Mo

Tic-Tac-Mo is the initial three-player correctness environment. Players take turns placing stones on an empty 3x5 board. The first player to place three stones consecutively in a horizontal, vertical, or diagonal line wins. A full board without a line is a draw. A win for player 1 has utility `[1, -1, -1]`; draws have utility `[0, 0, 0]`.

The observation contains six 3x5 planes: one stone plane and one to-play plane for each player. Rectangular-board rotations currently map to the identity transformation.

## Connect 3x3

Connect 3x3 is the paper's second three-player game. It uses the standard 6x7 Connect Four board, but three equal-colored stones in a horizontal, vertical, or diagonal line win instead of four. A move selects one of seven columns and gravity places the current player's stone in the lowest empty cell. Full columns are illegal. Players rotate P1, P2, P3; a winner receives `[1, -1, -1]` in the corresponding player order, and a full board without a line returns `[0, 0, 0]`. Games last at most 42 moves.

The policy has seven actions, one per column. The observation matches the paper: six 6x7 planes, consisting of one absolute stone plane and one to-play plane for each player. Rectangular-board rotations map to the identity transformation.

Build it inside the supported MiniZero Linux environment:

```bash
scripts/build.sh connect3x3 release
```

Generate a complete paper-inspired MaxN config using MiniZero's own ResNet architecture:

```bash
build/connect3x3/minizero_connect3x3 \
  -gen connect3x3_maxn.cfg \
  -conf_str "nn_type_name=alphazero:actor_multiplayer_search_type=maxn:actor_num_simulation=50:actor_mcts_puct_init=3:actor_use_gumbel=false:actor_mcts_value_rescale=false:actor_dirichlet_noise_alpha=1:actor_dirichlet_noise_epsilon=0.25:zero_disable_resign_ratio=1:zero_actor_intermediate_sequence_length=0:learner_use_per=false:learner_optimizer=Adam:learner_learning_rate=0.001:learner_batch_size=64:learner_weight_decay=0.0001"
```

The generated config intentionally does not claim to reproduce the paper's SENet. Train it with:

```bash
tools/quick-run.sh train connect3x3 connect3x3_maxn.cfg 100 -n connect3x3_maxn_01
```

## Build and test

Inside the supported MiniZero Linux environment:

```bash
scripts/build.sh tictacmo release
```

To include the focused multiplayer test, configure the build with `MINIZERO_BUILD_TESTS=ON`, build, and run CTest:

```bash
cmake -S . -B build/tictacmo-test \
  -DGAME_TYPE=TICTACMO \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINIZERO_BUILD_TESTS=ON
cmake --build build/tictacmo-test -j
ctest --test-dir build/tictacmo-test --output-on-failure
```

The test covers player rotation, a complete Tic-Tac-Mo win, vector-result record round-tripping, player-relative MaxN backup, Paranoid opponent selection, and two-player zero-sum backup equivalence.

Connect 3x3 has a separate environment test covering full columns, horizontal/vertical/diagonal wins, a complete 42-move draw, the six-plane observation, console actions, and vector-result record round-tripping:

```bash
cmake -S . -B build/connect3x3-test \
  -DGAME_TYPE=CONNECT3X3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINIZERO_BUILD_TESTS=ON
cmake --build build/connect3x3-test -j
ctest --test-dir build/connect3x3-test --output-on-failure
```

## Multi-model evaluation

`tools/multiplayer-eval.py` runs one MiniZero console process per seat, forwards every move to the other engines, and balances seat advantage by evaluating fixed, cyclic, or all unique permutations of each lineup. Each agent can use a different model, configuration, simulation budget, or search algorithm.

### Checkpoint self-evaluation

Multiplayer checkpoint self-evaluation follows the original `quick-run.sh self-eval` pairing rule. Checkpoints are sorted by iteration, and an interval of 10 evaluates checkpoint indices 0 vs. 10, 10 vs. 20, and so on. Both checkpoint agents use the same search algorithm. For every pair, the evaluator creates the two mixed three-player lineups, evaluates all seat assignments, and distributes the requested total game count as evenly as possible across them.

The existing quick-run interface automatically selects this evaluator for Tic-Tac-Mo and Connect 3x3:

```bash
tools/quick-run.sh self-eval connect3x3 connect3x3_maxn_smoke_01 \
  connect3x3_maxn_smoke_01/connect3x3_maxn_smoke_01.cfg 10 100 \
  -g 0123 --num_threads 2
```

Here, `10` is the checkpoint-index interval and `100` is the total number of games for each checkpoint pair, not the number of games per seating. Results use the original layout `TRAINING_DIR/self_eval/NEWER_vs_OLDER/`, plus `self_eval/elo.csv` and `self_eval/elo.png`.

The same evaluator can be launched directly:

```bash
python3 tools/multiplayer-eval.py self-eval connect3x3 connect3x3_maxn_smoke_01 \
  --conf-file connect3x3_maxn_smoke_01/connect3x3_maxn_smoke_01.cfg \
  --interval 10 --games 100 --search-type maxn --noise \
  -g 0123 --num_threads 2
```

Use `--search-type paranoid` to evaluate Paranoid-trained checkpoints. Like the original quick-run self-evaluation, action selection and Dirichlet noise are inherited from the selected config by default. Use `--noise` or `--no-noise` to override only the noise setting. Every engine receives a deterministic per-seating seed, so noise produces different games while the complete experiment remains reproducible. A checkpoint wins a multiplayer game when any seat using that checkpoint wins; the sequential Elo score uses the balanced pair win rate, with a draw worth one half.

Rerunning an identical command automatically skips completed game IDs. If the current config or generated settings differ, add `--resume` to continue each existing pair with its saved `arena.json` settings; this prevents old and new settings from being mixed within a pair. Use `--overwrite` only when the existing pair results should be replaced.

### Search-algorithm arenas

For the common case of comparing MaxN and Paranoid with the same trained model, use auto mode:

```bash
python3 tools/multiplayer-eval.py auto tictacmo tictacmo_maxn_smoke_02 \
  --num-simulations 200 --noise --games-per-seating 20 -g 0 --num_threads 1
```

Auto mode selects the numerically latest `model/weight_iter_*.pt` and the newest `*.cfg`, locates `build/GAME/minizero_GAME`, creates the two balanced lineups `[maxn, maxn, paranoid]` and `[maxn, paranoid, paranoid]`, and evaluates all unique seat permutations. It saves the resolved manifest under `TRAINING_DIR/evaluation/WEIGHT_maxn_vs_paranoid_nSIM_noise/arena.json`. `--noise` enables identical Dirichlet-noise settings for both search algorithms, so repeated games vary while the fixed seeds keep the experiment reproducible. Use `--model 15000` to select an iteration, `--conf-file PATH` to select another config, and `--output PATH` to override the result directory. `--dry-run` only generates the manifest.

To compare independently trained models without writing JSON, override either agent model:

```bash
python3 tools/multiplayer-eval.py auto tictacmo tictacmo_maxn_run \
  --maxn-model tictacmo_maxn_run/model/weight_iter_15000.pt \
  --paranoid-model tictacmo_paranoid_run/model/weight_iter_15000.pt \
  --num-simulations 200 --noise --games-per-seating 20 -g 0123 --num_threads 1
```

When an agent-specific model is under a `TRAINING_DIR/model/` directory, auto mode also selects the newest `*.cfg` from that model's training directory. `--maxn-conf-file` and `--paranoid-conf-file` override those choices explicitly. MiniZero loads each `-conf_file` first, then applies the generated `-conf_str`; consequently the arena's model path, search type, simulation override, seed, noise, action selection, and multiplayer safety settings take precedence over values saved in the training config.

The explicit JSON form below remains available for experiments involving different models or per-agent settings.

Create a JSON manifest such as:

```json
{
  "game": "tictacmo",
  "players": ["b", "w", "r"],
  "agents": [
    {
      "name": "iter_1000",
      "cwd": "/workspace/minizero",
      "env": {"OMP_NUM_THREADS": "2"},
      "command": [
        "build/tictacmo/minizero_tictacmo", "-mode", "console",
        "-conf_file", "/workspace/tictacmo/run.cfg",
        "-conf_str", "nn_file_name=/workspace/tictacmo/model/weight_iter_1000.pt:actor_multiplayer_search_type=maxn:program_seed={seed}:program_auto_seed=false:actor_use_gumbel=false:actor_use_dirichlet_noise=false:actor_use_random_rotation_features=false:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_mcts_value_rescale=false:zero_disable_resign_ratio=1:zero_actor_intermediate_sequence_length=0"
      ]
    },
    {
      "name": "iter_5000",
      "cwd": "/workspace/minizero",
      "env": {"OMP_NUM_THREADS": "2"},
      "command": [
        "build/tictacmo/minizero_tictacmo", "-mode", "console",
        "-conf_file", "/workspace/tictacmo/run.cfg",
        "-conf_str", "nn_file_name=/workspace/tictacmo/model/weight_iter_5000.pt:actor_multiplayer_search_type=maxn:program_seed={seed}:program_auto_seed=false:actor_use_gumbel=false:actor_use_dirichlet_noise=false:actor_use_random_rotation_features=false:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_mcts_value_rescale=false:zero_disable_resign_ratio=1:zero_actor_intermediate_sequence_length=0"
      ]
    },
    {
      "name": "iter_15000",
      "cwd": "/workspace/minizero",
      "env": {"OMP_NUM_THREADS": "2"},
      "command": [
        "build/tictacmo/minizero_tictacmo", "-mode", "console",
        "-conf_file", "/workspace/tictacmo/run.cfg",
        "-conf_str", "nn_file_name=/workspace/tictacmo/model/weight_iter_15000.pt:actor_multiplayer_search_type=maxn:program_seed={seed}:program_auto_seed=false:actor_use_gumbel=false:actor_use_dirichlet_noise=false:actor_use_random_rotation_features=false:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_mcts_value_rescale=false:zero_disable_resign_ratio=1:zero_actor_intermediate_sequence_length=0"
      ]
    }
  ],
  "lineups": [["iter_1000", "iter_5000", "iter_15000"]],
  "seat_mode": "all_permutations",
  "games_per_seating": 10,
  "max_moves": 15,
  "command_timeout": 300,
  "seed": 0
}
```

To evaluate or train a Paranoid agent, change only its configuration override:

```text
actor_multiplayer_search_type=paranoid
```

For a mixed arena, give the MaxN and Paranoid entries distinct agent names and commands, then place them in explicit lineups such as `["maxn", "maxn", "paranoid"]`. Unique seat permutations are generated automatically.

`{seed}`, `{seat}`, `{seat_index}`, `{agent}`, `{gpu}`, `{worker_id}`, and `{task_id}` in commands or environment variables are replaced when each engine starts. Run the arena inside the normal MiniZero/Podman environment:

```bash
python3 tools/multiplayer-eval.py arena.json evaluation/tictacmo_crossplay -g 0123 --num_threads 1
```

The output directory contains:

- `games.jsonl`: moves, seating, utility vector, winner, duration, and error for every game.
- `sgf/`: one neutral game record per successful game.
- `agent_summary.csv`: aggregate win rate, draw rate, and average return for every model.
- `seat_summary.csv`: the same metrics split by player seat.
- `seating_summary.csv`: matchup results for every exact model-to-seat assignment.
- `errors.csv` and `engine_logs/`: failed games and engine diagnostics.

Use `--resume` to skip recorded game IDs after an interrupted evaluation. `--overwrite` starts the scheduled evaluation again. `-g 0123 --num_threads 2` uses GPUs 0, 1, 2, and 3 with two parallel seating workers per GPU. All engines in one seating share that worker's physical GPU and see it as logical CUDA device 0. One worker launches one engine process per seat, so increase `--num_threads` only when each GPU has enough memory for the additional model copies. Agent `env` entries can set CPU controls such as `OMP_NUM_THREADS` and can explicitly override `CUDA_VISIBLE_DEVICES` when a custom placement is required.

## Training constraints

The first multiplayer baseline supports AlphaZero with standard PUCT only. Use the following configuration values:

```text
nn_type_name=alphazero
actor_multiplayer_search_type=maxn
actor_use_gumbel=false
actor_mcts_value_rescale=false
learner_use_per=false
zero_disable_resign_ratio=1
zero_actor_intermediate_sequence_length=0
```

Unsupported combinations fail at startup instead of silently using two-player semantics. Multiplayer MuZero, Gumbel search, resignation, categorical values, prioritized replay, and intermediate self-play sequences are intentionally deferred.
