#include "blokus15.h"
#include "random.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace minizero::env::blokus15 {

namespace {

    using Shape = std::vector<Blokus15Cell>;

    const std::array<Shape, kBlokus15NumPieces>& getBasePieces()
    {
        static const std::array<Shape, kBlokus15NumPieces> pieces = {{
            {{0, 0}},                         // I1
            {{0, 0}, {0, 1}},                 // I2
            {{0, 0}, {0, 1}, {0, 2}},         // I3
            {{0, 0}, {1, 0}, {1, 1}},         // V3
            {{0, 0}, {0, 1}, {0, 2}, {0, 3}}, // I4
            {{0, 0}, {0, 1}, {1, 0}, {1, 1}}, // O4
            {{0, 0}, {0, 1}, {0, 2}, {1, 1}}, // T4
            {{0, 0}, {1, 0}, {2, 0}, {2, 1}}, // L4
            {{0, 0}, {0, 1}, {1, 1}, {1, 2}}, // Z4
            {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}}, // I5
            {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {3, 1}}, // L5
            {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}}, // P5
            {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {2, 1}}, // T5
            {{0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}}, // V5
        }};
        return pieces;
    }

    Shape transformShape(const Shape& shape, int transform)
    {
        Shape result;
        result.reserve(shape.size());
        for (const Blokus15Cell& cell : shape) {
            int row = cell.row;
            int col = cell.col;
            if (transform >= 4) { col = -col; }
            for (int rotation = 0; rotation < transform % 4; ++rotation) {
                const int next_row = col;
                const int next_col = -row;
                row = next_row;
                col = next_col;
            }
            result.push_back({row, col});
        }
        const int min_row = std::min_element(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) { return lhs.row < rhs.row; })->row;
        const int min_col = std::min_element(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) { return lhs.col < rhs.col; })->col;
        for (Blokus15Cell& cell : result) {
            cell.row -= min_row;
            cell.col -= min_col;
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    std::pair<int, int> startingCorner(Player player)
    {
        switch (player) {
            case Player::kPlayer1: return {0, 0};
            case Player::kPlayer2: return {0, kBlokus15BoardSize - 1};
            case Player::kPlayer3: return {kBlokus15BoardSize - 1, kBlokus15BoardSize - 1};
            case Player::kPlayer4: return {kBlokus15BoardSize - 1, 0};
            default: return {-1, -1};
        }
    }

} // namespace

const std::vector<Blokus15Orientation>& getBlokus15Orientations()
{
    static const std::vector<Blokus15Orientation> orientations = [] {
        std::vector<Blokus15Orientation> result;
        for (int piece = 0; piece < kBlokus15NumPieces; ++piece) {
            std::set<Shape> unique;
            for (int transform = 0; transform < 8; ++transform) { unique.insert(transformShape(getBasePieces()[piece], transform)); }
            for (const Shape& shape : unique) {
                int height = 0;
                int width = 0;
                for (const Blokus15Cell& cell : shape) {
                    height = std::max(height, cell.row + 1);
                    width = std::max(width, cell.col + 1);
                }
                result.push_back({piece, height, width, shape});
            }
        }
        if (result.size() != kBlokus15NumOrientations) { throw std::runtime_error("Blokus15 orientation catalogue must contain 54 entries"); }
        return result;
    }();
    return orientations;
}

Blokus15Action::Blokus15Action(const std::vector<std::string>& action_string_args)
{
    assert(action_string_args.size() == 2);
    assert(action_string_args[0].size() == 1);
    player_ = charToPlayer(action_string_args[0][0]);
    std::string action = action_string_args[1];
    std::transform(action.begin(), action.end(), action.begin(), [](unsigned char c) { return std::toupper(c); });
    if (action == "PASS") {
        action_id_ = kBlokus15PassAction;
    } else {
        try {
            size_t parsed = 0;
            action_id_ = std::stoi(action, &parsed);
            if (parsed != action.size()) { action_id_ = -1; }
        } catch (...) {
            action_id_ = -1;
        }
    }
}

std::string Blokus15Action::toConsoleString() const
{
    return getActionID() == kBlokus15PassAction ? "PASS" : std::to_string(getActionID());
}

void Blokus15Env::reset()
{
    turn_ = Player::kPlayer1;
    actions_.clear();
    observations_.clear();
    board_.fill(Player::kPlayerNone);
    for (auto& pieces : available_) { pieces.fill(true); }
    eliminated_.fill(false);
    monomino_last_.fill(false);
}

bool Blokus15Env::act(const Blokus15Action& action)
{
    if (!isLegalAction(action)) { return false; }
    actions_.push_back(action);
    const int player_index = playerToIndex(action.getPlayer());
    if (action.getActionID() == kBlokus15PassAction) {
        eliminated_[player_index] = true;
    } else {
        const int orientation_index = action.getActionID() / kBlokus15BoardArea;
        const int anchor = action.getActionID() % kBlokus15BoardArea;
        const int anchor_row = anchor / kBlokus15BoardSize;
        const int anchor_col = anchor % kBlokus15BoardSize;
        const Blokus15Orientation& orientation = getBlokus15Orientations()[orientation_index];
        for (const Blokus15Cell& cell : orientation.cells) {
            board_[getPosition(anchor_row + cell.row, anchor_col + cell.col)] = action.getPlayer();
        }
        available_[player_index][orientation.piece] = false;
        const bool no_pieces_left = std::none_of(available_[player_index].begin(), available_[player_index].end(), [](bool available) { return available; });
        if (no_pieces_left) { monomino_last_[player_index] = orientation.piece == 0; }
    }
    turn_ = action.nextPlayer();
    if (!isTerminal()) {
        while (eliminated_[playerToIndex(turn_)]) { turn_ = getNextPlayer(turn_, kBlokus15NumPlayer); }
    }
    return true;
}

bool Blokus15Env::act(const std::vector<std::string>& action_string_args)
{
    return act(Blokus15Action(action_string_args));
}

std::vector<Blokus15Action> Blokus15Env::getLegalActions() const
{
    std::vector<Blokus15Action> actions;
    if (isTerminal()) { return actions; }
    const int player_index = playerToIndex(turn_);
    if (!eliminated_[player_index]) {
        const auto& orientations = getBlokus15Orientations();
        const std::vector<Blokus15Cell> contact_points = getRequiredContactPoints(turn_);
        for (int orientation_index = 0; orientation_index < static_cast<int>(orientations.size()); ++orientation_index) {
            const Blokus15Orientation& orientation = orientations[orientation_index];
            if (!available_[player_index][orientation.piece]) { continue; }
            const std::array<bool, kBlokus15BoardArea> candidate_anchors = getCandidateAnchors(orientation, contact_points);
            for (int anchor = 0; anchor < kBlokus15BoardArea; ++anchor) {
                if (!candidate_anchors[anchor]) { continue; }
                const int action_id = orientation_index * kBlokus15BoardArea + anchor;
                if (isPlacementLegal(action_id, turn_)) { actions.emplace_back(action_id, turn_); }
            }
        }
    }
    if (actions.empty()) { actions.emplace_back(kBlokus15PassAction, turn_); }
    return actions;
}

std::vector<Blokus15Cell> Blokus15Env::getRequiredContactPoints(Player player) const
{
    if (isFirstMove(player)) {
        const auto [row, col] = startingCorner(player);
        return {{row, col}};
    }

    std::array<bool, kBlokus15BoardArea> is_contact{};
    constexpr int diagonals[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (int row = 0; row < kBlokus15BoardSize; ++row) {
        for (int col = 0; col < kBlokus15BoardSize; ++col) {
            if (board_[getPosition(row, col)] != player) { continue; }
            for (const auto& diagonal : diagonals) {
                const int contact_row = row + diagonal[0];
                const int contact_col = col + diagonal[1];
                if (contact_row < 0 || contact_row >= kBlokus15BoardSize ||
                    contact_col < 0 || contact_col >= kBlokus15BoardSize ||
                    board_[getPosition(contact_row, contact_col)] != Player::kPlayerNone) {
                    continue;
                }
                is_contact[getPosition(contact_row, contact_col)] = true;
            }
        }
    }

    std::vector<Blokus15Cell> contact_points;
    for (int position = 0; position < kBlokus15BoardArea; ++position) {
        if (is_contact[position]) { contact_points.push_back({position / kBlokus15BoardSize, position % kBlokus15BoardSize}); }
    }
    return contact_points;
}

std::array<bool, kBlokus15BoardArea> Blokus15Env::getCandidateAnchors(const Blokus15Orientation& orientation, const std::vector<Blokus15Cell>& contact_points) const
{
    std::array<bool, kBlokus15BoardArea> is_candidate{};
    for (const Blokus15Cell& contact : contact_points) {
        for (const Blokus15Cell& cell : orientation.cells) {
            const int anchor_row = contact.row - cell.row;
            const int anchor_col = contact.col - cell.col;
            if (anchor_row < 0 || anchor_row + orientation.height > kBlokus15BoardSize ||
                anchor_col < 0 || anchor_col + orientation.width > kBlokus15BoardSize) {
                continue;
            }
            is_candidate[getPosition(anchor_row, anchor_col)] = true;
        }
    }
    return is_candidate;
}

bool Blokus15Env::isLegalAction(const Blokus15Action& action) const
{
    if (isTerminal() || action.getPlayer() != turn_) { return false; }
    const int player_id = static_cast<int>(action.getPlayer());
    if (player_id < 1 || player_id > kBlokus15NumPlayer) { return false; }
    if (action.getActionID() == kBlokus15PassAction) {
        return eliminated_[playerToIndex(action.getPlayer())] || !hasPlacement(action.getPlayer());
    }
    return isPlacementLegal(action.getActionID(), action.getPlayer());
}

bool Blokus15Env::isPlacementLegal(int action_id, Player player) const
{
    if (action_id < 0 || action_id >= kBlokus15PassAction) { return false; }
    const int player_index = playerToIndex(player);
    if (eliminated_[player_index]) { return false; }
    const int orientation_index = action_id / kBlokus15BoardArea;
    const int anchor = action_id % kBlokus15BoardArea;
    const Blokus15Orientation& orientation = getBlokus15Orientations()[orientation_index];
    if (!available_[player_index][orientation.piece]) { return false; }
    const int anchor_row = anchor / kBlokus15BoardSize;
    const int anchor_col = anchor % kBlokus15BoardSize;
    if (anchor_row + orientation.height > kBlokus15BoardSize || anchor_col + orientation.width > kBlokus15BoardSize) { return false; }

    bool diagonal_contact = false;
    constexpr int sides[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    constexpr int diagonals[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (const Blokus15Cell& cell : orientation.cells) {
        const int row = anchor_row + cell.row;
        const int col = anchor_col + cell.col;
        if (board_[getPosition(row, col)] != Player::kPlayerNone) { return false; }
        for (const auto& side : sides) {
            const int next_row = row + side[0];
            const int next_col = col + side[1];
            if (next_row >= 0 && next_row < kBlokus15BoardSize && next_col >= 0 && next_col < kBlokus15BoardSize &&
                board_[getPosition(next_row, next_col)] == player) {
                return false;
            }
        }
        for (const auto& diagonal : diagonals) {
            const int next_row = row + diagonal[0];
            const int next_col = col + diagonal[1];
            if (next_row >= 0 && next_row < kBlokus15BoardSize && next_col >= 0 && next_col < kBlokus15BoardSize &&
                board_[getPosition(next_row, next_col)] == player) {
                diagonal_contact = true;
            }
        }
    }
    return isFirstMove(player) ? coversStartingCorner(orientation, anchor_row, anchor_col, player) : diagonal_contact;
}

bool Blokus15Env::hasPlacement(Player player) const
{
    const int player_index = playerToIndex(player);
    if (eliminated_[player_index]) { return false; }
    const auto& orientations = getBlokus15Orientations();
    const std::vector<Blokus15Cell> contact_points = getRequiredContactPoints(player);
    for (int orientation_index = 0; orientation_index < static_cast<int>(orientations.size()); ++orientation_index) {
        const Blokus15Orientation& orientation = orientations[orientation_index];
        if (!available_[player_index][orientation.piece]) { continue; }
        const std::array<bool, kBlokus15BoardArea> candidate_anchors = getCandidateAnchors(orientation, contact_points);
        for (int anchor = 0; anchor < kBlokus15BoardArea; ++anchor) {
            if (!candidate_anchors[anchor]) { continue; }
            if (isPlacementLegal(orientation_index * kBlokus15BoardArea + anchor, player)) { return true; }
        }
    }
    return false;
}

bool Blokus15Env::isFirstMove(Player player) const
{
    return std::all_of(available_[playerToIndex(player)].begin(), available_[playerToIndex(player)].end(), [](bool available) { return available; });
}

bool Blokus15Env::coversStartingCorner(const Blokus15Orientation& orientation, int anchor_row, int anchor_col, Player player) const
{
    const std::pair<int, int> corner = startingCorner(player);
    return std::any_of(orientation.cells.begin(), orientation.cells.end(), [&](const Blokus15Cell& cell) {
        return anchor_row + cell.row == corner.first && anchor_col + cell.col == corner.second;
    });
}

bool Blokus15Env::isTerminal() const
{
    return std::all_of(eliminated_.begin(), eliminated_.end(), [](bool eliminated) { return eliminated; });
}

int Blokus15Env::getPlayerScore(Player player) const
{
    const int player_index = playerToIndex(player);
    int remaining_squares = 0;
    const auto& pieces = getBasePieces();
    for (int piece = 0; piece < kBlokus15NumPieces; ++piece) {
        if (available_[player_index][piece]) { remaining_squares += static_cast<int>(pieces[piece].size()); }
    }
    if (remaining_squares != 0) { return -remaining_squares; }
    return 15 + (monomino_last_[player_index] ? 5 : 0);
}

float Blokus15Env::getEvalScore(bool is_resign /* = false */) const
{
    return getEvalScores(is_resign)[0];
}

PlayerValues Blokus15Env::getEvalScores(bool is_resign /* = false */) const
{
    PlayerValues values{};
    if (is_resign) {
        const int resigned = playerToIndex(turn_);
        for (int player = 0; player < kBlokus15NumPlayer; ++player) { values[player] = player == resigned ? -1.0f : 1.0f; }
        return values;
    }
    for (int player = 0; player < kBlokus15NumPlayer; ++player) {
        const float score = static_cast<float>(getPlayerScore(indexToPlayer(player)));
        values[player] = 2.0f * (score + static_cast<float>(kBlokus15TotalSquares)) /
                             static_cast<float>(kBlokus15TotalSquares + 20) -
                         1.0f;
    }
    return values;
}

std::vector<float> Blokus15Env::getFeatures(utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    (void)rotation;
    std::vector<float> features;
    features.reserve(kBlokus15NumInputChannels * kBlokus15BoardArea);
    for (int player = 0; player < kBlokus15NumPlayer; ++player) {
        for (Player cell : board_) { features.push_back(cell == indexToPlayer(player) ? 1.0f : 0.0f); }
    }
    for (int player = 0; player < kBlokus15NumPlayer; ++player) {
        for (int piece = 0; piece < kBlokus15NumPieces; ++piece) {
            features.insert(features.end(), kBlokus15BoardArea, available_[player][piece] ? 1.0f : 0.0f);
        }
    }
    for (int player = 0; player < kBlokus15NumPlayer; ++player) {
        features.insert(features.end(), kBlokus15BoardArea, turn_ == indexToPlayer(player) ? 1.0f : 0.0f);
    }
    for (int player = 0; player < kBlokus15NumPlayer; ++player) {
        features.insert(features.end(), kBlokus15BoardArea, eliminated_[player] ? 1.0f : 0.0f);
    }
    for (int player = 0; player < kBlokus15NumPlayer; ++player) {
        features.insert(features.end(), kBlokus15BoardArea, monomino_last_[player] ? 1.0f : 0.0f);
    }
    return features;
}

std::vector<float> Blokus15Env::getActionFeatures(const Blokus15Action& action, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    (void)rotation;
    std::vector<float> features(kBlokus15NumActionFeatureChannels * kBlokus15BoardArea, 0.0f);
    if (action.getActionID() == kBlokus15PassAction) {
        std::fill(features.begin() + kBlokus15BoardArea, features.end(), 1.0f);
        return features;
    }
    if (action.getActionID() < 0 || action.getActionID() >= kBlokus15PassAction) { return features; }
    const int orientation_index = action.getActionID() / kBlokus15BoardArea;
    const int anchor = action.getActionID() % kBlokus15BoardArea;
    const int anchor_row = anchor / kBlokus15BoardSize;
    const int anchor_col = anchor % kBlokus15BoardSize;
    for (const Blokus15Cell& cell : getBlokus15Orientations()[orientation_index].cells) {
        const int row = anchor_row + cell.row;
        const int col = anchor_col + cell.col;
        if (row < kBlokus15BoardSize && col < kBlokus15BoardSize) { features[getPosition(row, col)] = 1.0f; }
    }
    return features;
}

std::string Blokus15Env::toString() const
{
    std::ostringstream oss;
    oss << "    ";
    for (int col = 0; col < kBlokus15BoardSize; ++col) { oss << static_cast<char>('A' + col) << ' '; }
    oss << '\n';
    for (int row = kBlokus15BoardSize - 1; row >= 0; --row) {
        oss << std::setw(2) << row + 1 << "  ";
        for (int col = 0; col < kBlokus15BoardSize; ++col) {
            const Player player = board_[getPosition(row, col)];
            char stone = '.';
            if (player == Player::kPlayer1) { stone = 'X'; }
            if (player == Player::kPlayer2) { stone = 'O'; }
            if (player == Player::kPlayer3) { stone = 'R'; }
            if (player == Player::kPlayer4) { stone = 'G'; }
            oss << stone << ' ';
        }
        oss << '\n';
    }
    return oss.str();
}

bool Blokus15Env::isPieceAvailable(Player player, int piece) const
{
    return piece >= 0 && piece < kBlokus15NumPieces && available_[playerToIndex(player)][piece];
}

Player Blokus15Env::getPlayerAt(int row, int col) const
{
    if (row < 0 || row >= kBlokus15BoardSize || col < 0 || col >= kBlokus15BoardSize) { return Player::kPlayerNone; }
    return board_[getPosition(row, col)];
}

void Blokus15EnvLoader::loadFromEnvironment(const Blokus15Env& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history /* = {} */)
{
    BaseEnvLoader<Blokus15Action, Blokus15Env>::loadFromEnvironment(env, action_info_history);
    addTag("RE", playerValuesToString(env.getEvalScores(), kBlokus15NumPlayer));
}

std::vector<float> Blokus15EnvLoader::getValue(const int pos) const
{
    return playerValuesToVector(stringToPlayerValues(getTag("RE"), kBlokus15NumPlayer), kBlokus15NumPlayer);
}

std::vector<float> Blokus15EnvLoader::getActionFeatures(const int pos, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    (void)rotation;
    std::vector<float> action_features(kBlokus15NumActionFeatureChannels * kBlokus15BoardArea, 0.0f);
    const int action_id = pos < static_cast<int>(action_pairs_.size())
                              ? action_pairs_[pos].first.getActionID()
                              : utils::Random::randInt() % kBlokus15PolicySize;
    Blokus15Env env;
    (void)env;
    if (action_id == kBlokus15PassAction) {
        std::fill(action_features.begin() + kBlokus15BoardArea, action_features.end(), 1.0f);
    } else if (action_id >= 0 && action_id < kBlokus15PassAction) {
        const int orientation_index = action_id / kBlokus15BoardArea;
        const int anchor = action_id % kBlokus15BoardArea;
        const int anchor_row = anchor / kBlokus15BoardSize;
        const int anchor_col = anchor % kBlokus15BoardSize;
        for (const Blokus15Cell& cell : getBlokus15Orientations()[orientation_index].cells) {
            const int row = anchor_row + cell.row;
            const int col = anchor_col + cell.col;
            if (row < kBlokus15BoardSize && col < kBlokus15BoardSize) { action_features[row * kBlokus15BoardSize + col] = 1.0f; }
        }
    }
    return action_features;
}

} // namespace minizero::env::blokus15
