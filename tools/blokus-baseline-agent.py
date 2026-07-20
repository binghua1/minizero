#!/usr/bin/env python3

"""Pure algorithmic Blokus console agent for MiniZero multiplayer evaluation.

This process speaks the same minimal console protocol as MiniZero's
``-mode console`` engine, so it can be seated together with neural agents by
``tools/multiplayer-eval.py``.
"""

import argparse
import random
import sys
from dataclasses import dataclass


BOARD_SIZE = 20
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
NUM_PLAYERS = 4
NUM_PIECES = 21
NUM_ORIENTATIONS = 91
PASS_ACTION = NUM_ORIENTATIONS * BOARD_AREA
PLAYER_CODES = ("b", "w", "r", "g")
EMPTY = -1

BASE_PIECES = (
    ((0, 0),),                                                   # I1
    ((0, 0), (0, 1)),                                           # I2
    ((0, 0), (0, 1), (0, 2)),                                   # I3
    ((0, 0), (1, 0), (1, 1)),                                   # V3
    ((0, 0), (0, 1), (0, 2), (0, 3)),                           # I4
    ((0, 0), (0, 1), (1, 0), (1, 1)),                           # O4
    ((0, 0), (0, 1), (0, 2), (1, 1)),                           # T4
    ((0, 0), (1, 0), (2, 0), (2, 1)),                           # L4
    ((0, 0), (0, 1), (1, 1), (1, 2)),                           # Z4
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),                   # F5
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),                   # I5
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),                   # L5
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),                   # P5
    ((0, 0), (1, 0), (1, 1), (2, 1), (3, 1)),                   # N5
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),                   # T5
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),                   # U5
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),                   # V5
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),                   # W5
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),                   # X5
    ((0, 0), (1, 0), (2, 0), (3, 0), (2, 1)),                   # Y5
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),                   # Z5
)


@dataclass(frozen=True)
class Orientation:
    piece: int
    height: int
    width: int
    cells: tuple


def transform_shape(shape, transform):
    cells = []
    for row, col in shape:
        if transform >= 4:
            col = -col
        for _ in range(transform % 4):
            row, col = col, -row
        cells.append((row, col))
    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    return tuple(sorted((row - min_row, col - min_col) for row, col in cells))


def make_orientations():
    orientations = []
    for piece, shape in enumerate(BASE_PIECES):
        # C++ uses std::set<Shape>, so the stable action id order is
        # lexicographic shape order, not transform generation order.
        for cells in sorted({transform_shape(shape, transform) for transform in range(8)}):
            height = max(row for row, _ in cells) + 1
            width = max(col for _, col in cells) + 1
            orientations.append(Orientation(piece, height, width, cells))
    if len(orientations) != NUM_ORIENTATIONS:
        raise RuntimeError(f"expected {NUM_ORIENTATIONS} orientations, got {len(orientations)}")
    return tuple(orientations)


ORIENTATIONS = make_orientations()
PIECE_SIZES = tuple(len(piece) for piece in BASE_PIECES)


def player_index(player):
    if player not in PLAYER_CODES:
        raise ValueError(f"unknown player: {player}")
    return PLAYER_CODES.index(player)


def starting_corner(index):
    return ((0, 0), (0, BOARD_SIZE - 1), (BOARD_SIZE - 1, BOARD_SIZE - 1), (BOARD_SIZE - 1, 0))[index]


def position(row, col):
    return row * BOARD_SIZE + col


class BlokusState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turn = 0
        self.board = [EMPTY] * BOARD_AREA
        self.available = [[True] * NUM_PIECES for _ in range(NUM_PLAYERS)]
        self.eliminated = [False] * NUM_PLAYERS
        self.monomino_last = [False] * NUM_PLAYERS
        self.moves = []

    def clone(self):
        state = BlokusState.__new__(BlokusState)
        state.turn = self.turn
        state.board = self.board[:]
        state.available = [pieces[:] for pieces in self.available]
        state.eliminated = self.eliminated[:]
        state.monomino_last = self.monomino_last[:]
        state.moves = self.moves[:]
        return state

    def is_terminal(self):
        return all(self.eliminated)

    def is_first_move(self, player):
        return all(self.available[player])

    def required_contact_points(self, player):
        if self.is_first_move(player):
            return [starting_corner(player)]
        contacts = set()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[position(row, col)] != player:
                    continue
                for drow, dcol in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    next_row = row + drow
                    next_col = col + dcol
                    if (0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE and
                            self.board[position(next_row, next_col)] == EMPTY):
                        contacts.add((next_row, next_col))
        return sorted(contacts)

    def candidate_anchors(self, orientation, contact_points):
        anchors = set()
        for contact_row, contact_col in contact_points:
            for cell_row, cell_col in orientation.cells:
                anchor_row = contact_row - cell_row
                anchor_col = contact_col - cell_col
                if (0 <= anchor_row and anchor_row + orientation.height <= BOARD_SIZE and
                        0 <= anchor_col and anchor_col + orientation.width <= BOARD_SIZE):
                    anchors.add(position(anchor_row, anchor_col))
        return sorted(anchors)

    def placement_legal(self, action_id, player):
        if action_id < 0 or action_id >= PASS_ACTION or self.eliminated[player]:
            return False
        orientation_index, anchor = divmod(action_id, BOARD_AREA)
        orientation = ORIENTATIONS[orientation_index]
        if not self.available[player][orientation.piece]:
            return False
        anchor_row, anchor_col = divmod(anchor, BOARD_SIZE)
        if anchor_row + orientation.height > BOARD_SIZE or anchor_col + orientation.width > BOARD_SIZE:
            return False

        diagonal_contact = False
        for cell_row, cell_col in orientation.cells:
            row = anchor_row + cell_row
            col = anchor_col + cell_col
            if self.board[position(row, col)] != EMPTY:
                return False
            for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_row = row + drow
                next_col = col + dcol
                if (0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE and
                        self.board[position(next_row, next_col)] == player):
                    return False
            for drow, dcol in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                next_row = row + drow
                next_col = col + dcol
                if (0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE and
                        self.board[position(next_row, next_col)] == player):
                    diagonal_contact = True

        if self.is_first_move(player):
            corner = starting_corner(player)
            return any((anchor_row + row, anchor_col + col) == corner for row, col in orientation.cells)
        return diagonal_contact

    def has_placement(self, player):
        if self.eliminated[player]:
            return False
        contacts = self.required_contact_points(player)
        for orientation_index, orientation in enumerate(ORIENTATIONS):
            if not self.available[player][orientation.piece]:
                continue
            for anchor in self.candidate_anchors(orientation, contacts):
                if self.placement_legal(orientation_index * BOARD_AREA + anchor, player):
                    return True
        return False

    def legal_actions(self, player=None):
        if player is None:
            player = self.turn
        if self.is_terminal() or self.eliminated[player]:
            return [PASS_ACTION]
        actions = []
        contacts = self.required_contact_points(player)
        for orientation_index, orientation in enumerate(ORIENTATIONS):
            if not self.available[player][orientation.piece]:
                continue
            for anchor in self.candidate_anchors(orientation, contacts):
                action_id = orientation_index * BOARD_AREA + anchor
                if self.placement_legal(action_id, player):
                    actions.append(action_id)
        return actions or [PASS_ACTION]

    def is_legal_action(self, action_id, player):
        if self.is_terminal() or player != self.turn:
            return False
        if action_id == PASS_ACTION:
            return self.eliminated[player] or not self.has_placement(player)
        return self.placement_legal(action_id, player)

    def play(self, action_id, player):
        if not self.is_legal_action(action_id, player):
            raise ValueError(f"illegal action for {PLAYER_CODES[player]}: {action_to_string(action_id)}")
        self.moves.append((player, action_id))
        if action_id == PASS_ACTION:
            self.eliminated[player] = True
        else:
            orientation_index, anchor = divmod(action_id, BOARD_AREA)
            orientation = ORIENTATIONS[orientation_index]
            anchor_row, anchor_col = divmod(anchor, BOARD_SIZE)
            for cell_row, cell_col in orientation.cells:
                self.board[position(anchor_row + cell_row, anchor_col + cell_col)] = player
            self.available[player][orientation.piece] = False
            if not any(self.available[player]):
                self.monomino_last[player] = orientation.piece == 0
        self.turn = (player + 1) % NUM_PLAYERS
        if not self.is_terminal():
            while self.eliminated[self.turn]:
                self.turn = (self.turn + 1) % NUM_PLAYERS

    def player_score(self, player):
        remaining = sum(PIECE_SIZES[piece] for piece, available in enumerate(self.available[player]) if available)
        if remaining:
            return -remaining
        return 15 + (5 if self.monomino_last[player] else 0)

    def returns(self):
        scores = [float(self.player_score(player)) for player in range(NUM_PLAYERS)]
        total = sum(scores)
        return [(score - (total - score) / 3.0) / 109.0 for score in scores]

    def board_string(self):
        symbols = [".", "b", "w", "r", "g"]
        lines = ["    " + " ".join(chr(ord("A") + col) for col in range(BOARD_SIZE))]
        for row in range(BOARD_SIZE - 1, -1, -1):
            cells = []
            for col in range(BOARD_SIZE):
                value = self.board[position(row, col)]
                cells.append(symbols[value + 1] if value != EMPTY else ".")
            lines.append(f"{row + 1:2d}  " + " ".join(cells))
        return "\n".join(lines)


def action_to_string(action_id):
    return "PASS" if action_id == PASS_ACTION else str(action_id)


def parse_action(value):
    value = value.strip()
    if value.upper() == "PASS":
        return PASS_ACTION
    return int(value)


def action_piece_size(action_id):
    if action_id == PASS_ACTION:
        return 0
    orientation = ORIENTATIONS[action_id // BOARD_AREA]
    return PIECE_SIZES[orientation.piece]


def choose_random(state, player, rng):
    return rng.choice(state.legal_actions(player))


def choose_greedy(state, player, rng):
    actions = state.legal_actions(player)
    best_size = max(action_piece_size(action) for action in actions)
    best = [action for action in actions if action_piece_size(action) == best_size]
    return rng.choice(best)


def choose_greedy_mobility(state, player, rng):
    actions = state.legal_actions(player)
    if actions == [PASS_ACTION]:
        return PASS_ACTION
    sample_limit = min(len(actions), 256)
    candidates = actions if len(actions) <= sample_limit else rng.sample(actions, sample_limit)
    best_score = None
    best = []
    for action in candidates:
        child = state.clone()
        child.play(action, player)
        next_mobility = len(child.legal_actions(child.turn)) if not child.is_terminal() else 0
        score = (
            action_piece_size(action),
            len(child.required_contact_points(player)),
            -next_mobility,
        )
        if best_score is None or score > best_score:
            best_score = score
            best = [action]
        elif score == best_score:
            best.append(action)
    return rng.choice(best)


def playout_policy_action(state, player, rng, policy):
    if policy == "greedy":
        return choose_greedy(state, player, rng)
    if policy == "greedy_mobility":
        return choose_greedy_mobility(state, player, rng)
    return choose_random(state, player, rng)


def choose_rollout(state, player, rng, rollouts, candidate_limit, playout_policy, max_plies):
    actions = state.legal_actions(player)
    if actions == [PASS_ACTION]:
        return PASS_ACTION
    ordered = sorted(actions, key=lambda action: (action_piece_size(action), -action), reverse=True)
    candidates = ordered[:min(candidate_limit, len(ordered))]
    if len(actions) > len(candidates):
        # Keep a little exploration among legal but not top-sized moves.
        candidate_set = set(candidates)
        tail = [action for action in actions if action not in candidate_set]
        candidates.extend(rng.sample(tail, min(max(0, candidate_limit // 4), len(tail))))

    visits = {action: 0 for action in candidates}
    value_sums = {action: 0.0 for action in candidates}
    for index in range(max(rollouts, len(candidates))):
        action = candidates[index % len(candidates)]
        child = state.clone()
        child.play(action, player)
        plies = 0
        while not child.is_terminal() and plies < max_plies:
            current = child.turn
            child.play(playout_policy_action(child, current, rng, playout_policy), current)
            plies += 1
        value_sums[action] += child.returns()[player]
        visits[action] += 1
    best_mean = max(value_sums[action] / visits[action] for action in candidates)
    best = [action for action in candidates if value_sums[action] / visits[action] == best_mean]
    return rng.choice(best)


class UCTNode:
    def __init__(self, state, root_player, mode, rng, candidate_limit):
        self.player = state.turn
        self.terminal = state.is_terminal()
        self.visits = 0
        self.value_sums = [0.0] * NUM_PLAYERS
        self.children = {}
        self.unexpanded = []
        if not self.terminal:
            actions = state.legal_actions(self.player)
            self.unexpanded = mcts_candidate_actions(actions, root_player, mode, rng, candidate_limit)
            rng.shuffle(self.unexpanded)

    def mean_value(self, player):
        return self.value_sums[player] / self.visits if self.visits else 0.0


def mcts_candidate_actions(actions, root_player, mode, rng, candidate_limit):
    if actions == [PASS_ACTION] or len(actions) <= candidate_limit:
        return list(actions)
    ordered = sorted(actions, key=lambda action: (action_piece_size(action), -action), reverse=True)
    candidates = ordered[:candidate_limit]
    # Add a small amount of non-top-piece diversity; this keeps the MCTS
    # baseline from being only "largest-piece UCT" while staying tractable.
    candidate_set = set(candidates)
    tail = [action for action in actions if action not in candidate_set]
    extra = min(max(0, candidate_limit // 4), len(tail))
    if extra:
        candidates.extend(rng.sample(tail, extra))
    return candidates


def select_uct_child(node, root_player, mode, cpuct, rng):
    parent_visits = max(1, node.visits)
    best_score = None
    best_actions = []
    for action, child in node.children.items():
        child_visits = max(1, child.visits)
        if mode == "maxn":
            q = child.mean_value(node.player)
        else:
            root_q = child.mean_value(root_player)
            q = root_q if node.player == root_player else -root_q
        exploration = cpuct * (parent_visits ** 0.5) / (1 + child.visits)
        score = q + exploration
        if best_score is None or score > best_score:
            best_score = score
            best_actions = [action]
        elif score == best_score:
            best_actions.append(action)
    return rng.choice(best_actions)


def rollout_value(state, root_player, rng, playout_policy, max_plies):
    child = state.clone()
    plies = 0
    while not child.is_terminal() and plies < max_plies:
        current = child.turn
        child.play(playout_policy_action(child, current, rng, playout_policy), current)
        plies += 1
    return child.returns()


def choose_uct(state, root_player, rng, mode, simulations, cpuct, candidate_limit, leaf_eval, playout_policy, max_plies):
    root = UCTNode(state, root_player, mode, rng, candidate_limit)
    if root.terminal:
        return PASS_ACTION
    if root.unexpanded == [PASS_ACTION] and not root.children:
        return PASS_ACTION

    for _ in range(simulations):
        search_state = state.clone()
        node = root
        path = [node]

        while not node.terminal:
            if node.unexpanded:
                action = node.unexpanded.pop()
                search_state.play(action, search_state.turn)
                child = UCTNode(search_state, root_player, mode, rng, candidate_limit)
                node.children[action] = child
                node = child
                path.append(node)
                break
            if not node.children:
                break
            action = select_uct_child(node, root_player, mode, cpuct, rng)
            search_state.play(action, search_state.turn)
            node = node.children[action]
            path.append(node)

        if search_state.is_terminal():
            values = search_state.returns()
        elif leaf_eval == "rollout":
            values = rollout_value(search_state, root_player, rng, playout_policy, max_plies)
        else:
            values = [0.0] * NUM_PLAYERS
        for visited in path:
            visited.visits += 1
            for player, value in enumerate(values):
                visited.value_sums[player] += value

    if not root.children:
        return choose_greedy(state, root_player, rng)
    best_visits = max(child.visits for child in root.children.values())
    best = [action for action, child in root.children.items() if child.visits == best_visits]
    return rng.choice(best)


def choose_action(state, player, args, rng):
    if args.policy == "random":
        return choose_random(state, player, rng)
    if args.policy == "greedy":
        return choose_greedy(state, player, rng)
    if args.policy == "greedy_mobility":
        return choose_greedy_mobility(state, player, rng)
    if args.policy == "rollout":
        return choose_rollout(
            state, player, rng,
            rollouts=args.rollouts,
            candidate_limit=args.candidate_limit,
            playout_policy=args.playout_policy,
            max_plies=args.max_plies,
        )
    if args.policy == "uct_maxn":
        return choose_uct(
            state, player, rng,
            mode="maxn",
            simulations=args.mcts_simulations,
            cpuct=args.mcts_cpuct,
            candidate_limit=args.mcts_candidate_limit,
            leaf_eval=args.mcts_leaf_eval,
            playout_policy=args.mcts_playout_policy,
            max_plies=args.mcts_max_plies,
        )
    if args.policy == "uct_paranoid":
        return choose_uct(
            state, player, rng,
            mode="paranoid",
            simulations=args.mcts_simulations,
            cpuct=args.mcts_cpuct,
            candidate_limit=args.mcts_candidate_limit,
            leaf_eval=args.mcts_leaf_eval,
            playout_policy=args.mcts_playout_policy,
            max_plies=args.mcts_max_plies,
        )
    raise AssertionError(args.policy)


def success(payload=""):
    print("=" + (f" {payload}" if payload else ""))
    print()
    sys.stdout.flush()


def failure(payload):
    print(f"? {payload}")
    print()
    sys.stdout.flush()


def game_string(state):
    returns = ",".join(f"{value:.9g}" for value in state.returns())
    moves = "".join(f";{PLAYER_CODES[player].upper()}[{action_to_string(action)}]" for player, action in state.moves)
    return f"(;GM[blokus]RE[{returns}]{moves})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("random", "greedy", "greedy_mobility", "rollout", "uct_maxn", "uct_paranoid"), default="greedy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--playout-policy", choices=("random", "greedy", "greedy_mobility"), default="random")
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--mcts-simulations", type=int, default=50)
    parser.add_argument("--mcts-cpuct", type=float, default=1.0)
    parser.add_argument("--mcts-candidate-limit", type=int, default=64)
    parser.add_argument("--mcts-leaf-eval", choices=("zero", "rollout"), default="zero")
    parser.add_argument("--mcts-playout-policy", choices=("random", "greedy", "greedy_mobility"), default="random")
    parser.add_argument("--mcts-max-plies", type=int, default=400)
    args = parser.parse_args()
    if (args.rollouts <= 0 or args.candidate_limit <= 0 or args.max_plies <= 0 or
            args.mcts_simulations <= 0 or args.mcts_candidate_limit <= 0 or
            args.mcts_cpuct < 0 or args.mcts_max_plies <= 0):
        parser.error("rollout and MCTS counts/limits must be positive; mcts-cpuct cannot be negative")

    rng = random.Random(args.seed)
    state = BlokusState()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        command = parts[0].lower()
        try:
            if command == "quit":
                success()
                return
            if command == "protocol_version":
                success("2")
            elif command == "name":
                success(f"blokus_{args.policy}_baseline")
            elif command == "version":
                success("1")
            elif command == "list_commands":
                success("protocol_version\nname\nversion\nlist_commands\nclear_board\nplay\ngenmove\ngame_string\nshowboard\nquit")
            elif command == "clear_board":
                state.reset()
                success()
            elif command == "play":
                if len(parts) != 3:
                    raise ValueError("usage: play PLAYER ACTION")
                player = player_index(parts[1].lower())
                action = parse_action(parts[2])
                state.play(action, player)
                success()
            elif command == "genmove":
                if len(parts) != 2:
                    raise ValueError("usage: genmove PLAYER")
                player = player_index(parts[1].lower())
                if player != state.turn:
                    raise ValueError(f"expected turn {PLAYER_CODES[state.turn]}, got {parts[1]}")
                action = choose_action(state, player, args, rng)
                state.play(action, player)
                success(action_to_string(action))
            elif command == "game_string":
                success(game_string(state))
            elif command == "showboard":
                success(state.board_string())
            else:
                failure(f"unknown command: {parts[0]}")
        except Exception as exc:
            failure(str(exc))


if __name__ == "__main__":
    main()
