# Four-player Blokus

MiniZero implements the classic four-player Blokus rules on a 20x20 board. Each player owns the complete set of 21 free polyominoes (89 unit squares). The first piece covers that player's starting corner; later pieces must touch the player's existing pieces diagonally and must not touch them orthogonally. Opponents' pieces may touch in either way. A player with no legal placement passes once and is skipped for the rest of the game. The game ends after all four players have been eliminated.

The implementation follows Mattel's classic scoring: every unplayed unit square is -1; playing all pieces earns +15; playing the monomino last raises that score to +20. Each player's official score is independently normalized from `[-89, +20]` to the value head's `[-1, 1]` range:

```text
v_i = 2 * (score_i + 89) / 109 - 1
```

This preserves absolute score margins without forcing the four-player returns to be zero-sum. Game records contain all four normalized scores in the `RE` property.

## State and action representation

The observation has 100 absolute-seat planes:

- 4 occupancy planes;
- 84 piece-availability planes (21 per player);
- 4 side-to-move planes;
- 4 eliminated-player planes;
- 4 monomino-last bonus planes.

There are 91 distinct oriented polyominoes. Each owns one 20x20 policy plane whose cell is the orientation's normalized bounding-box anchor. A 92nd plane represents PASS at its first cell, giving 36,800 policy logits. Blokus uses a direct 1x1-convolutional policy head; the previous fully connected head would be prohibitively large. Console placement actions are their numeric action IDs; PASS is printed as `PASS`.

Seat-preserving board rotations are deliberately disabled. Rotating a board also permutes the four player-specific starting corners and value-vector components, which MiniZero's current augmentation interface cannot express. Keep `actor_use_random_rotation_features=false`.

## Build and test

Use the project's supported Linux container, which supplies LibTorch, Boost, OpenCV, and ALE:

```bash
cmake -S . -B build/blokus-test \
  -DGAME_TYPE=BLOKUS \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINIZERO_BUILD_TESTS=ON
cmake --build build/blokus-test -j
ctest --test-dir build/blokus-test --output-on-failure
```

The focused test checks the 21-piece/91-orientation catalogue, all 89 squares, the 58 legal opening placements, all four starting corners, diagonal versus edge contact, action features, official initial score, normalized score returns, and record round-tripping.

## AlphaZero baseline

The current multiplayer learner supports AlphaZero, MaxN or Paranoid search, vector values, and seat-balanced evaluation. A practical smoke-test configuration can be generated with:

```bash
build/blokus-test/minizero_blokus \
  -gen blokus_maxn.cfg \
  -conf_str "nn_type_name=alphazero:actor_multiplayer_search_type=maxn:actor_use_gumbel=false:actor_use_random_rotation_features=false:actor_mcts_value_rescale=false:zero_disable_resign_ratio=1:zero_actor_intermediate_sequence_length=0:learner_use_per=false"
```

Blokus has a much larger branching factor and policy than the existing multiplayer toy games. Start with a small network and batch while validating throughput, then increase simulations and model size. Compare MaxN and Paranoid under identical compute and balance all four seats:

```bash
python3 tools/multiplayer-eval.py auto blokus TRAINING_DIR \
  --num-simulations 200 --games-per-seating 10 -g 0 --num_threads 1
```

Auto and self-evaluation modes infer four players, a 400-move safety cap, and four elimination passes for Blokus.

## Current MuZero boundary

This repository still rejects MuZero when `getNumPlayer() > 2`. The Blokus environment already provides two-channel recurrent action features (placed cells plus PASS) and the spatial MuZero policy head, but multiplayer MuZero additionally needs vector-valued prediction, player-aware latent-tree backup, vector training targets at every unroll step, and a deliberate equilibrium/self-play objective. Treating that missing algorithm as a research contribution is preferable to silently applying the existing scalar two-player backup.
