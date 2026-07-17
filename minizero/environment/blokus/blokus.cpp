#include "blokus.h"
#include "random.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace minizero::env::blokus {

namespace {

using Shape = std::vector<BlokusCell>;

const std::array<Shape, kBlokusNumPieces>& getBasePieces()
{
    static const std::array<Shape, kBlokusNumPieces> pieces = {{
        {{0, 0}},                                                   // I1
        {{0, 0}, {0, 1}},                                           // I2
        {{0, 0}, {0, 1}, {0, 2}},                                   // I3
        {{0, 0}, {1, 0}, {1, 1}},                                   // V3
        {{0, 0}, {0, 1}, {0, 2}, {0, 3}},                           // I4
        {{0, 0}, {0, 1}, {1, 0}, {1, 1}},                           // O4
        {{0, 0}, {0, 1}, {0, 2}, {1, 1}},                           // T4
        {{0, 0}, {1, 0}, {2, 0}, {2, 1}},                           // L4
        {{0, 0}, {0, 1}, {1, 1}, {1, 2}},                           // Z4
        {{0, 1}, {0, 2}, {1, 0}, {1, 1}, {2, 1}},                   // F5
        {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}},                   // I5
        {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {3, 1}},                   // L5
        {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}},                   // P5
        {{0, 0}, {1, 0}, {1, 1}, {2, 1}, {3, 1}},                   // N5
        {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {2, 1}},                   // T5
        {{0, 0}, {0, 2}, {1, 0}, {1, 1}, {1, 2}},                   // U5
        {{0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}},                   // V5
        {{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 2}},                   // W5
        {{0, 1}, {1, 0}, {1, 1}, {1, 2}, {2, 1}},                   // X5
        {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {2, 1}},                   // Y5
        {{0, 0}, {0, 1}, {1, 1}, {2, 1}, {2, 2}},                   // Z5
    }};
    return pieces;
}

Shape transformShape(const Shape& shape, int transform)
{
    Shape result;
    result.reserve(shape.size());
    for (const BlokusCell& cell : shape) {
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
    for (BlokusCell& cell : result) {
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
        case Player::kPlayer2: return {0, kBlokusBoardSize - 1};
        case Player::kPlayer3: return {kBlokusBoardSize - 1, kBlokusBoardSize - 1};
        case Player::kPlayer4: return {kBlokusBoardSize - 1, 0};
        default: return {-1, -1};
    }
}

} // namespace

const std::vector<BlokusOrientation>& getBlokusOrientations()
{
    static const std::vector<BlokusOrientation> orientations = [] {
        std::vector<BlokusOrientation> result;
        for (int piece = 0; piece < kBlokusNumPieces; ++piece) {
            std::set<Shape> unique;
            for (int transform = 0; transform < 8; ++transform) { unique.insert(transformShape(getBasePieces()[piece], transform)); }
            for (const Shape& shape : unique) {
                int height = 0;
                int width = 0;
                for (const BlokusCell& cell : shape) {
                    height = std::max(height, cell.row + 1);
                    width = std::max(width, cell.col + 1);
                }
                result.push_back({piece, height, width, shape});
            }
        }
        if (result.size() != kBlokusNumOrientations) { throw std::runtime_error("Blokus orientation catalogue must contain 91 entries"); }
        return result;
    }();
    return orientations;
}

BlokusAction::BlokusAction(const std::vector<std::string>& action_string_args)
{
    assert(action_string_args.size() == 2);
    assert(action_string_args[0].size() == 1);
    player_ = charToPlayer(action_string_args[0][0]);
    std::string action = action_string_args[1];
    std::transform(action.begin(), action.end(), action.begin(), [](unsigned char c) { return std::toupper(c); });
    if (action == "PASS") {
        action_id_ = kBlokusPassAction;
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

std::string BlokusAction::toConsoleString() const
{
    return getActionID() == kBlokusPassAction ? "PASS" : std::to_string(getActionID());
}

void BlokusEnv::reset()
{
    turn_ = Player::kPlayer1;
    actions_.clear();
    observations_.clear();
    board_.fill(Player::kPlayerNone);
    for (auto& pieces : available_) { pieces.fill(true); }
    eliminated_.fill(false);
    monomino_last_.fill(false);
}

bool BlokusEnv::act(const BlokusAction& action)
{
    if (!isLegalAction(action)) { return false; }
    actions_.push_back(action);
    const int player_index = playerToIndex(action.getPlayer());
    if (action.getActionID() == kBlokusPassAction) {
        eliminated_[player_index] = true;
    } else {
        const int orientation_index = action.getActionID() / kBlokusBoardArea;
        const int anchor = action.getActionID() % kBlokusBoardArea;
        const int anchor_row = anchor / kBlokusBoardSize;
        const int anchor_col = anchor % kBlokusBoardSize;
        const BlokusOrientation& orientation = getBlokusOrientations()[orientation_index];
        for (const BlokusCell& cell : orientation.cells) {
            board_[getPosition(anchor_row + cell.row, anchor_col + cell.col)] = action.getPlayer();
        }
        available_[player_index][orientation.piece] = false;
        const bool no_pieces_left = std::none_of(available_[player_index].begin(), available_[player_index].end(), [](bool available) { return available; });
        if (no_pieces_left) { monomino_last_[player_index] = orientation.piece == 0; }
    }
    turn_ = action.nextPlayer();
    if (!isTerminal()) {
        while (eliminated_[playerToIndex(turn_)]) { turn_ = getNextPlayer(turn_, kBlokusNumPlayer); }
    }
    return true;
}

bool BlokusEnv::act(const std::vector<std::string>& action_string_args)
{
    return act(BlokusAction(action_string_args));
}

std::vector<BlokusAction> BlokusEnv::getLegalActions() const
{
    std::vector<BlokusAction> actions;
    if (isTerminal()) { return actions; }
    const int player_index = playerToIndex(turn_);
    if (!eliminated_[player_index]) {
        const auto& orientations = getBlokusOrientations();
        const std::vector<BlokusCell> contact_points = getRequiredContactPoints(turn_);
        for (int orientation_index = 0; orientation_index < static_cast<int>(orientations.size()); ++orientation_index) {
            const BlokusOrientation& orientation = orientations[orientation_index];
            if (!available_[player_index][orientation.piece]) { continue; }
            const std::array<bool, kBlokusBoardArea> candidate_anchors = getCandidateAnchors(orientation, contact_points);
            for (int anchor = 0; anchor < kBlokusBoardArea; ++anchor) {
                if (!candidate_anchors[anchor]) { continue; }
                const int action_id = orientation_index * kBlokusBoardArea + anchor;
                if (isPlacementLegal(action_id, turn_)) { actions.emplace_back(action_id, turn_); }
            }
        }
    }
    if (actions.empty()) { actions.emplace_back(kBlokusPassAction, turn_); }
    return actions;
}

std::vector<BlokusCell> BlokusEnv::getRequiredContactPoints(Player player) const
{
    if (isFirstMove(player)) {
        const auto [row, col] = startingCorner(player);
        return {{row, col}};
    }

    std::array<bool, kBlokusBoardArea> is_contact{};
    constexpr int diagonals[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (int row = 0; row < kBlokusBoardSize; ++row) {
        for (int col = 0; col < kBlokusBoardSize; ++col) {
            if (board_[getPosition(row, col)] != player) { continue; }
            for (const auto& diagonal : diagonals) {
                const int contact_row = row + diagonal[0];
                const int contact_col = col + diagonal[1];
                if (contact_row < 0 || contact_row >= kBlokusBoardSize ||
                    contact_col < 0 || contact_col >= kBlokusBoardSize ||
                    board_[getPosition(contact_row, contact_col)] != Player::kPlayerNone) {
                    continue;
                }
                is_contact[getPosition(contact_row, contact_col)] = true;
            }
        }
    }

    std::vector<BlokusCell> contact_points;
    contact_points.reserve(kBlokusBoardArea);
    for (int position = 0; position < kBlokusBoardArea; ++position) {
        if (is_contact[position]) { contact_points.push_back({position / kBlokusBoardSize, position % kBlokusBoardSize}); }
    }
    return contact_points;
}

std::array<bool, kBlokusBoardArea> BlokusEnv::getCandidateAnchors(const BlokusOrientation& orientation, const std::vector<BlokusCell>& contact_points) const
{
    std::array<bool, kBlokusBoardArea> is_candidate{};
    for (const BlokusCell& contact : contact_points) {
        for (const BlokusCell& cell : orientation.cells) {
            const int anchor_row = contact.row - cell.row;
            const int anchor_col = contact.col - cell.col;
            if (anchor_row < 0 || anchor_row + orientation.height > kBlokusBoardSize ||
                anchor_col < 0 || anchor_col + orientation.width > kBlokusBoardSize) {
                continue;
            }
            is_candidate[getPosition(anchor_row, anchor_col)] = true;
        }
    }
    return is_candidate;
}

bool BlokusEnv::isLegalAction(const BlokusAction& action) const
{
    if (isTerminal() || action.getPlayer() != turn_) { return false; }
    const int player_id = static_cast<int>(action.getPlayer());
    if (player_id < 1 || player_id > kBlokusNumPlayer) { return false; }
    if (action.getActionID() == kBlokusPassAction) {
        return eliminated_[playerToIndex(action.getPlayer())] || !hasPlacement(action.getPlayer());
    }
    return isPlacementLegal(action.getActionID(), action.getPlayer());
}

bool BlokusEnv::isPlacementLegal(int action_id, Player player) const
{
    if (action_id < 0 || action_id >= kBlokusPassAction) { return false; }
    const int player_index = playerToIndex(player);
    if (eliminated_[player_index]) { return false; }
    const int orientation_index = action_id / kBlokusBoardArea;
    const int anchor = action_id % kBlokusBoardArea;
    const BlokusOrientation& orientation = getBlokusOrientations()[orientation_index];
    if (!available_[player_index][orientation.piece]) { return false; }
    const int anchor_row = anchor / kBlokusBoardSize;
    const int anchor_col = anchor % kBlokusBoardSize;
    if (anchor_row + orientation.height > kBlokusBoardSize || anchor_col + orientation.width > kBlokusBoardSize) { return false; }

    bool diagonal_contact = false;
    constexpr int sides[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    constexpr int diagonals[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (const BlokusCell& cell : orientation.cells) {
        const int row = anchor_row + cell.row;
        const int col = anchor_col + cell.col;
        if (board_[getPosition(row, col)] != Player::kPlayerNone) { return false; }
        for (const auto& side : sides) {
            const int next_row = row + side[0];
            const int next_col = col + side[1];
            if (next_row >= 0 && next_row < kBlokusBoardSize && next_col >= 0 && next_col < kBlokusBoardSize &&
                board_[getPosition(next_row, next_col)] == player) {
                return false;
            }
        }
        for (const auto& diagonal : diagonals) {
            const int next_row = row + diagonal[0];
            const int next_col = col + diagonal[1];
            if (next_row >= 0 && next_row < kBlokusBoardSize && next_col >= 0 && next_col < kBlokusBoardSize &&
                board_[getPosition(next_row, next_col)] == player) {
                diagonal_contact = true;
            }
        }
    }
    return isFirstMove(player) ? coversStartingCorner(orientation, anchor_row, anchor_col, player) : diagonal_contact;
}

bool BlokusEnv::hasPlacement(Player player) const
{
    const int player_index = playerToIndex(player);
    if (eliminated_[player_index]) { return false; }
    const auto& orientations = getBlokusOrientations();
    const std::vector<BlokusCell> contact_points = getRequiredContactPoints(player);
    for (int orientation_index = 0; orientation_index < static_cast<int>(orientations.size()); ++orientation_index) {
        const BlokusOrientation& orientation = orientations[orientation_index];
        if (!available_[player_index][orientation.piece]) { continue; }
        const std::array<bool, kBlokusBoardArea> candidate_anchors = getCandidateAnchors(orientation, contact_points);
        for (int anchor = 0; anchor < kBlokusBoardArea; ++anchor) {
            if (!candidate_anchors[anchor]) { continue; }
            if (isPlacementLegal(orientation_index * kBlokusBoardArea + anchor, player)) { return true; }
        }
    }
    return false;
}

bool BlokusEnv::isFirstMove(Player player) const
{
    return std::all_of(available_[playerToIndex(player)].begin(), available_[playerToIndex(player)].end(), [](bool available) { return available; });
}

bool BlokusEnv::coversStartingCorner(const BlokusOrientation& orientation, int anchor_row, int anchor_col, Player player) const
{
    const std::pair<int, int> corner = startingCorner(player);
    return std::any_of(orientation.cells.begin(), orientation.cells.end(), [&](const BlokusCell& cell) {
        return anchor_row + cell.row == corner.first && anchor_col + cell.col == corner.second;
    });
}

bool BlokusEnv::isTerminal() const
{
    return std::all_of(eliminated_.begin(), eliminated_.end(), [](bool eliminated) { return eliminated; });
}

int BlokusEnv::getPlayerScore(Player player) const
{
    const int player_index = playerToIndex(player);
    int remaining_squares = 0;
    const auto& pieces = getBasePieces();
    for (int piece = 0; piece < kBlokusNumPieces; ++piece) {
        if (available_[player_index][piece]) { remaining_squares += static_cast<int>(pieces[piece].size()); }
    }
    if (remaining_squares != 0) { return -remaining_squares; }
    return 15 + (monomino_last_[player_index] ? 5 : 0);
}

float BlokusEnv::getEvalScore(bool is_resign /* = false */) const
{
    return getEvalScores(is_resign)[0];
}

PlayerValues BlokusEnv::getEvalScores(bool is_resign /* = false */) const
{
    PlayerValues values{};
    if (is_resign) {
        const int resigned = playerToIndex(turn_);
        for (int player = 0; player < kBlokusNumPlayer; ++player) { values[player] = player == resigned ? -1.0f : 1.0f / 3.0f; }
        return values;
    }
    std::array<float, kBlokusNumPlayer> scores;
    float total = 0.0f;
    for (int player = 0; player < kBlokusNumPlayer; ++player) {
        scores[player] = static_cast<float>(getPlayerScore(indexToPlayer(player)));
        total += scores[player];
    }
    // Official Blokus scores range from -89 to +20.  Centering each score
    // against the other three players yields a bounded, zero-sum utility.
    for (int player = 0; player < kBlokusNumPlayer; ++player) {
        const float opponent_mean = (total - scores[player]) / 3.0f;
        values[player] = (scores[player] - opponent_mean) / 109.0f;
    }
    return values;
}

std::vector<float> BlokusEnv::getFeatures(utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    (void)rotation; // Player-specific starting corners make seat-preserving rotations invalid.
    std::vector<float> features;
    features.reserve(kBlokusNumInputChannels * kBlokusBoardArea);
    for (int player = 0; player < kBlokusNumPlayer; ++player) {
        for (Player cell : board_) { features.push_back(cell == indexToPlayer(player) ? 1.0f : 0.0f); }
    }
    for (int player = 0; player < kBlokusNumPlayer; ++player) {
        for (int piece = 0; piece < kBlokusNumPieces; ++piece) {
            features.insert(features.end(), kBlokusBoardArea, available_[player][piece] ? 1.0f : 0.0f);
        }
    }
    for (int player = 0; player < kBlokusNumPlayer; ++player) {
        features.insert(features.end(), kBlokusBoardArea, turn_ == indexToPlayer(player) ? 1.0f : 0.0f);
    }
    for (int player = 0; player < kBlokusNumPlayer; ++player) {
        features.insert(features.end(), kBlokusBoardArea, eliminated_[player] ? 1.0f : 0.0f);
    }
    for (int player = 0; player < kBlokusNumPlayer; ++player) {
        features.insert(features.end(), kBlokusBoardArea, monomino_last_[player] ? 1.0f : 0.0f);
    }
    return features;
}

std::vector<float> BlokusEnv::getActionFeatures(const BlokusAction& action, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    (void)rotation;
    std::vector<float> features(kBlokusNumActionFeatureChannels * kBlokusBoardArea, 0.0f);
    if (action.getActionID() == kBlokusPassAction) {
        std::fill(features.begin() + kBlokusBoardArea, features.end(), 1.0f);
        return features;
    }
    if (action.getActionID() < 0 || action.getActionID() >= kBlokusPassAction) { return features; }
    const int orientation_index = action.getActionID() / kBlokusBoardArea;
    const int anchor = action.getActionID() % kBlokusBoardArea;
    const int anchor_row = anchor / kBlokusBoardSize;
    const int anchor_col = anchor % kBlokusBoardSize;
    for (const BlokusCell& cell : getBlokusOrientations()[orientation_index].cells) {
        const int row = anchor_row + cell.row;
        const int col = anchor_col + cell.col;
        if (row < kBlokusBoardSize && col < kBlokusBoardSize) { features[getPosition(row, col)] = 1.0f; }
    }
    return features;
}

std::string BlokusEnv::toString() const
{
    std::ostringstream oss;
    oss << "    ";
    for (int col = 0; col < kBlokusBoardSize; ++col) { oss << static_cast<char>('A' + col) << ' '; }
    oss << '\n';
    for (int row = kBlokusBoardSize - 1; row >= 0; --row) {
        oss << std::setw(2) << row + 1 << "  ";
        for (int col = 0; col < kBlokusBoardSize; ++col) {
            const Player player = board_[getPosition(row, col)];
            oss << (player == Player::kPlayerNone ? '.' : playerToChar(player)) << ' ';
        }
        oss << '\n';
    }
    oss << "score";
    for (int player = 0; player < kBlokusNumPlayer; ++player) {
        oss << ' ' << playerToChar(indexToPlayer(player)) << '=' << getPlayerScore(indexToPlayer(player));
    }
    oss << '\n';
    return oss.str();
}

bool BlokusEnv::isPieceAvailable(Player player, int piece) const
{
    return piece >= 0 && piece < kBlokusNumPieces && available_[playerToIndex(player)][piece];
}

Player BlokusEnv::getPlayerAt(int row, int col) const
{
    if (row < 0 || row >= kBlokusBoardSize || col < 0 || col >= kBlokusBoardSize) { return Player::kPlayerNone; }
    return board_[getPosition(row, col)];
}

void BlokusEnvLoader::loadFromEnvironment(const BlokusEnv& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history /* = {} */)
{
    BaseEnvLoader<BlokusAction, BlokusEnv>::loadFromEnvironment(env, action_info_history);
    addTag("RE", playerValuesToString(env.getEvalScores(), kBlokusNumPlayer));
}

std::vector<float> BlokusEnvLoader::getValue(const int pos) const
{
    (void)pos;
    return playerValuesToVector(stringToPlayerValues(getTag("RE"), kBlokusNumPlayer), kBlokusNumPlayer);
}

std::vector<float> BlokusEnvLoader::getActionFeatures(const int pos, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    const int action_id = pos < static_cast<int>(action_pairs_.size())
                              ? action_pairs_[pos].first.getActionID()
                              : utils::Random::randInt() % kBlokusPassAction;
    BlokusEnv env;
    return env.getActionFeatures(BlokusAction(action_id, Player::kPlayer1), rotation);
}

} // namespace minizero::env::blokus
