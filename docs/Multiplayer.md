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
