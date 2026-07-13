#include "connect3x3.h"
#include "random.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <sstream>
#include <utility>

namespace minizero::env::connect3x3 {

Connect3x3Action::Connect3x3Action(const std::vector<std::string>& action_string_args)
{
    assert(action_string_args.size() == 2);
    assert(action_string_args[0].size() == 1);
    player_ = charToPlayer(action_string_args[0][0]);

    std::string column = action_string_args[1];
    std::transform(column.begin(), column.end(), column.begin(), ::toupper);
    if (column.size() == 1 && column[0] >= 'A' && column[0] < 'A' + kConnect3x3BoardWidth) {
        action_id_ = column[0] - 'A';
    } else if (column.size() == 1 && column[0] >= '1' && column[0] <= '0' + kConnect3x3BoardWidth) {
        action_id_ = column[0] - '1';
    } else {
        action_id_ = -1;
    }
}

std::string Connect3x3Action::toConsoleString() const
{
    assert(getActionID() >= 0 && getActionID() < kConnect3x3PolicySize);
    return std::string(1, static_cast<char>('A' + getActionID()));
}

void Connect3x3Env::reset()
{
    turn_ = Player::kPlayer1;
    actions_.clear();
    observations_.clear();
    board_.assign(kConnect3x3BoardHeight * kConnect3x3BoardWidth, Player::kPlayerNone);
    column_heights_.assign(kConnect3x3BoardWidth, 0);
}

bool Connect3x3Env::act(const Connect3x3Action& action)
{
    if (!isLegalAction(action)) { return false; }
    const int column = action.getActionID();
    const int row = column_heights_[column]++;
    actions_.push_back(action);
    board_[getPosition(row, column)] = action.getPlayer();
    turn_ = action.nextPlayer();
    return true;
}

bool Connect3x3Env::act(const std::vector<std::string>& action_string_args)
{
    return act(Connect3x3Action(action_string_args));
}

std::vector<Connect3x3Action> Connect3x3Env::getLegalActions() const
{
    std::vector<Connect3x3Action> actions;
    for (int column = 0; column < kConnect3x3PolicySize; ++column) {
        Connect3x3Action action(column, turn_);
        if (isLegalAction(action)) { actions.push_back(action); }
    }
    return actions;
}

bool Connect3x3Env::isLegalAction(const Connect3x3Action& action) const
{
    const int action_id = action.getActionID();
    const int player_id = static_cast<int>(action.getPlayer());
    return !isTerminal() && action_id >= 0 && action_id < kConnect3x3PolicySize &&
           player_id >= 1 && player_id <= kConnect3x3NumPlayer && action.getPlayer() == turn_ &&
           column_heights_[action_id] < kConnect3x3BoardHeight;
}

bool Connect3x3Env::isTerminal() const
{
    return getWinner() != Player::kPlayerNone ||
           std::all_of(column_heights_.begin(), column_heights_.end(), [](int height) { return height == kConnect3x3BoardHeight; });
}

float Connect3x3Env::getEvalScore(bool is_resign /* = false */) const
{
    return getEvalScores(is_resign)[0];
}

PlayerValues Connect3x3Env::getEvalScores(bool is_resign /* = false */) const
{
    PlayerValues values{};
    if (is_resign) {
        values[playerToIndex(turn_)] = -1.0f;
        return values;
    }

    const Player winner = getWinner();
    if (winner == Player::kPlayerNone) { return values; }
    for (int player_index = 0; player_index < kConnect3x3NumPlayer; ++player_index) { values[player_index] = -1.0f; }
    values[playerToIndex(winner)] = 1.0f;
    return values;
}

std::vector<float> Connect3x3Env::getFeatures(utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    std::vector<float> features;
    const int board_area = kConnect3x3BoardHeight * kConnect3x3BoardWidth;
    features.reserve(getNumInputChannels() * board_area);
    for (int channel = 0; channel < getNumInputChannels(); ++channel) {
        for (int position = 0; position < board_area; ++position) {
            if (channel < kConnect3x3NumPlayer) {
                features.push_back(board_[position] == indexToPlayer(channel) ? 1.0f : 0.0f);
            } else {
                features.push_back(turn_ == indexToPlayer(channel - kConnect3x3NumPlayer) ? 1.0f : 0.0f);
            }
        }
    }
    return features;
}

std::vector<float> Connect3x3Env::getActionFeatures(const Connect3x3Action& action, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    std::vector<float> action_features(kConnect3x3BoardHeight * kConnect3x3BoardWidth, 0.0f);
    for (int row = 0; row < kConnect3x3BoardHeight; ++row) { action_features[getPosition(row, action.getActionID())] = 1.0f; }
    return action_features;
}

std::string Connect3x3Env::toString() const
{
    std::ostringstream oss;
    oss << "   A  B  C  D  E  F  G" << std::endl;
    for (int row = kConnect3x3BoardHeight - 1; row >= 0; --row) {
        oss << row + 1 << " ";
        for (int col = 0; col < kConnect3x3BoardWidth; ++col) {
            const Player player = board_[getPosition(row, col)];
            const char stone = player == Player::kPlayer1 ? 'X' : player == Player::kPlayer2 ? 'O'
                                                              : player == Player::kPlayer3   ? 'Y'
                                                                                             : '.';
            oss << ' ' << stone << ' ';
        }
        oss << std::endl;
    }
    return oss.str();
}

Player Connect3x3Env::getWinner() const
{
    constexpr int directions[][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (int row = 0; row < kConnect3x3BoardHeight; ++row) {
        for (int col = 0; col < kConnect3x3BoardWidth; ++col) {
            const Player player = board_[getPosition(row, col)];
            if (player == Player::kPlayerNone) { continue; }
            for (const auto& direction : directions) {
                const int end_row = row + 2 * direction[0];
                const int end_col = col + 2 * direction[1];
                if (end_row < 0 || end_row >= kConnect3x3BoardHeight || end_col < 0 || end_col >= kConnect3x3BoardWidth) { continue; }
                if (board_[getPosition(row + direction[0], col + direction[1])] == player &&
                    board_[getPosition(end_row, end_col)] == player) {
                    return player;
                }
            }
        }
    }
    return Player::kPlayerNone;
}

void Connect3x3EnvLoader::loadFromEnvironment(const Connect3x3Env& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history /* = {} */)
{
    BaseEnvLoader<Connect3x3Action, Connect3x3Env>::loadFromEnvironment(env, action_info_history);
    addTag("RE", playerValuesToString(env.getEvalScores(), kConnect3x3NumPlayer));
}

std::vector<float> Connect3x3EnvLoader::getValue(const int pos) const
{
    return playerValuesToVector(stringToPlayerValues(getTag("RE"), kConnect3x3NumPlayer), kConnect3x3NumPlayer);
}

std::vector<float> Connect3x3EnvLoader::getActionFeatures(const int pos, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    const int action_id = pos < static_cast<int>(action_pairs_.size())
                              ? action_pairs_[pos].first.getActionID()
                              : utils::Random::randInt() % kConnect3x3PolicySize;
    std::vector<float> action_features(kConnect3x3BoardHeight * kConnect3x3BoardWidth, 0.0f);
    for (int row = 0; row < kConnect3x3BoardHeight; ++row) { action_features[row * kConnect3x3BoardWidth + action_id] = 1.0f; }
    return action_features;
}

} // namespace minizero::env::connect3x3
