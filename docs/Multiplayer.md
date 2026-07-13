# Multiplayer AlphaZero

MiniZero's multiplayer AlphaZero path uses an absolute utility vector at network leaves and terminal states. MCTS edges remain scalar: during backup, each edge receives the utility component of the player who selected that edge. This is the MaxN backup used by the `nplayer` branch of the reference Multiplayer AlphaZero implementation.

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

The test covers player rotation, a complete Tic-Tac-Mo win, vector-result record round-tripping, and player-relative MaxN backup.

## Multi-model evaluation

`tools/multiplayer-eval.py` runs one MiniZero console process per seat, forwards every move to the other engines, and balances seat advantage by evaluating fixed, cyclic, or all unique permutations of each lineup. Each agent can use a different model, configuration, simulation budget, or search algorithm.

Create a JSON manifest such as:

```json
{
  "game": "tictacmo",
  "players": ["b", "w", "r"],
  "agents": [
    {
      "name": "iter_1000",
      "cwd": "/workspace/minizero",
      "command": [
        "build/tictacmo/minizero_tictacmo", "-mode", "console",
        "-conf_file", "/workspace/tictacmo/run.cfg",
        "-conf_str", "nn_file_name=/workspace/tictacmo/model/weight_iter_1000.pt:program_seed={seed}:program_auto_seed=false:actor_use_gumbel=false:actor_use_dirichlet_noise=false:actor_use_random_rotation_features=false:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_mcts_value_rescale=false:zero_disable_resign_ratio=1:zero_actor_intermediate_sequence_length=0"
      ]
    },
    {
      "name": "iter_5000",
      "cwd": "/workspace/minizero",
      "command": [
        "build/tictacmo/minizero_tictacmo", "-mode", "console",
        "-conf_file", "/workspace/tictacmo/run.cfg",
        "-conf_str", "nn_file_name=/workspace/tictacmo/model/weight_iter_5000.pt:program_seed={seed}:program_auto_seed=false:actor_use_gumbel=false:actor_use_dirichlet_noise=false:actor_use_random_rotation_features=false:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_mcts_value_rescale=false:zero_disable_resign_ratio=1:zero_actor_intermediate_sequence_length=0"
      ]
    },
    {
      "name": "iter_15000",
      "cwd": "/workspace/minizero",
      "command": [
        "build/tictacmo/minizero_tictacmo", "-mode", "console",
        "-conf_file", "/workspace/tictacmo/run.cfg",
        "-conf_str", "nn_file_name=/workspace/tictacmo/model/weight_iter_15000.pt:program_seed={seed}:program_auto_seed=false:actor_use_gumbel=false:actor_use_dirichlet_noise=false:actor_use_random_rotation_features=false:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_mcts_value_rescale=false:zero_disable_resign_ratio=1:zero_actor_intermediate_sequence_length=0"
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

`{seed}`, `{seat}`, `{seat_index}`, and `{agent}` in commands or environment variables are replaced when each engine starts. Run the arena inside the normal MiniZero/Podman environment:

```bash
python3 tools/multiplayer-eval.py arena.json evaluation/tictacmo_crossplay --threads 1
```

The output directory contains:

- `games.jsonl`: moves, seating, utility vector, winner, duration, and error for every game.
- `sgf/`: one neutral game record per successful game.
- `agent_summary.csv`: aggregate win rate, draw rate, and average return for every model.
- `seat_summary.csv`: the same metrics split by player seat.
- `seating_summary.csv`: matchup results for every exact model-to-seat assignment.
- `errors.csv` and `engine_logs/`: failed games and engine diagnostics.

Use `--resume` to skip recorded game IDs after an interrupted evaluation. `--overwrite` starts the scheduled evaluation again. One evaluation thread launches one engine process per seat, so increase `--threads` only when GPU memory can hold the additional models.

## Training constraints

The first multiplayer baseline supports AlphaZero with standard PUCT only. Use the following configuration values:

```text
nn_type_name=alphazero
actor_use_gumbel=false
actor_mcts_value_rescale=false
learner_use_per=false
zero_disable_resign_ratio=1
zero_actor_intermediate_sequence_length=0
```

Unsupported combinations fail at startup instead of silently using two-player semantics. Multiplayer MuZero, Gumbel search, resignation, categorical values, prioritized replay, and intermediate self-play sequences are intentionally deferred.
