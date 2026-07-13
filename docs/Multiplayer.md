# Multiplayer AlphaZero

MiniZero's multiplayer AlphaZero path uses an absolute utility vector at network leaves and terminal states. MCTS edges remain scalar, with selectable MaxN or Paranoid backup semantics.

- `actor_multiplayer_search_type=maxn` stores the utility component of the player who selected each edge, so every player maximizes their own value.
- `actor_multiplayer_search_type=paranoid` stores the root player's utility on root-player edges and its negation on every opponent edge, so all opponents act as a coalition minimizing the root player's value.

The neural network and terminal targets remain the same vector in both modes. Models trained with different search types should use separate training directories because their MCTS policy targets differ.

## Tic-Tac-Mo

Tic-Tac-Mo is the initial three-player correctness environment. Players take turns placing stones on an empty 3x5 board. The first player to place three stones consecutively in a horizontal, vertical, or diagonal line wins. A full board without a line is a draw. A win for player 1 has utility `[1, -1, -1]`; draws have utility `[0, 0, 0]`.

The observation contains six 3x5 planes: one stone plane and one to-play plane for each player. Rectangular-board rotations currently map to the identity transformation.

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

## Multi-model evaluation

`tools/multiplayer-eval.py` runs one MiniZero console process per seat, forwards every move to the other engines, and balances seat advantage by evaluating fixed, cyclic, or all unique permutations of each lineup. Each agent can use a different model, configuration, simulation budget, or search algorithm.

For the common case of comparing MaxN and Paranoid with the same trained model, use auto mode:

```bash
python3 tools/multiplayer-eval.py auto tictacmo tictacmo_maxn_smoke_02 \
  --num-simulations 200 --noise --games-per-seating 20 -g 0 --num_threads 1
```

Auto mode selects the numerically latest `model/weight_iter_*.pt` and the newest `*.cfg`, locates `build/GAME/minizero_GAME`, creates the two balanced lineups `[maxn, maxn, paranoid]` and `[maxn, paranoid, paranoid]`, and evaluates all unique seat permutations. It saves the resolved manifest under `TRAINING_DIR/evaluation/WEIGHT_maxn_vs_paranoid_nSIM_noise/arena.json`. `--noise` enables identical Dirichlet-noise settings for both search algorithms, so repeated games vary while the fixed seeds keep the experiment reproducible. Use `--model 15000` to select an iteration, `--conf-file PATH` to select another config, and `--output PATH` to override the result directory. `--dry-run` only generates the manifest.

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
