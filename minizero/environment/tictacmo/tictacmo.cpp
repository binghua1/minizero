#include "tictacmo.h"
#include "random.h"
#include "sgf_loader.h"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <utility>

namespace minizero::env::tictacmo {

TicTacMoAction::TicTacMoAction(const std::vector<std::string>& action_string_args)
{
    assert(action_string_args.size() == 2);
    assert(action_string_args[0].size() == 1);
    player_ = charToPlayer(action_string_args[0][0]);
    action_id_ = utils::SGFLoader::boardCoordinateStringToActionID(action_string_args[1], kTicTacMoBoardWidth);
}

std::string TicTacMoAction::toConsoleString() const
{
    return utils::SGFLoader::actionIDToBoardCoordinateString(getActionID(), kTicTacMoBoardWidth);
}

void TicTacMoEnv::reset()
{
    turn_ = Player::kPlayer1;
    actions_.clear();
    observations_.clear();
    board_.assign(kTicTacMoPolicySize, Player::kPlayerNone);
}

bool TicTacMoEnv::act(const TicTacMoAction& action)
{
    if (!isLegalAction(action)) { return false; }
    actions_.push_back(action);
    board_[action.getActionID()] = action.getPlayer();
    turn_ = action.nextPlayer();
    return true;
}

bool TicTacMoEnv::act(const std::vector<std::string>& action_string_args)
{
    return act(TicTacMoAction(action_string_args));
}

std::vector<TicTacMoAction> TicTacMoEnv::getLegalActions() const
{
    std::vector<TicTacMoAction> actions;
    for (int action_id = 0; action_id < kTicTacMoPolicySize; ++action_id) {
        TicTacMoAction action(action_id, turn_);
        if (isLegalAction(action)) { actions.push_back(action); }
    }
    return actions;
}

bool TicTacMoEnv::isLegalAction(const TicTacMoAction& action) const
{
    const int player_id = static_cast<int>(action.getPlayer());
    return !isTerminal() && action.getActionID() >= 0 && action.getActionID() < kTicTacMoPolicySize &&
           player_id >= 1 && player_id <= kTicTacMoNumPlayer && action.getPlayer() == turn_ &&
           board_[action.getActionID()] == Player::kPlayerNone;
}

bool TicTacMoEnv::isTerminal() const
{
    return getWinner() != Player::kPlayerNone ||
           std::find(board_.begin(), board_.end(), Player::kPlayerNone) == board_.end();
}

float TicTacMoEnv::getEvalScore(bool is_resign /* = false */) const
{
    return getEvalScores(is_resign)[0];
}

PlayerValues TicTacMoEnv::getEvalScores(bool is_resign /* = false */) const
{
    PlayerValues values{};
    if (is_resign) {
        values[playerToIndex(turn_)] = -1.0f;
        return values;
    }

    const Player winner = getWinner();
    if (winner == Player::kPlayerNone) { return values; }
    for (int player_index = 0; player_index < kTicTacMoNumPlayer; ++player_index) { values[player_index] = -1.0f; }
    values[playerToIndex(winner)] = 1.0f;
    return values;
}

std::vector<float> TicTacMoEnv::getFeatures(utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    std::vector<float> features;
    features.reserve(getNumInputChannels() * kTicTacMoPolicySize);
    for (int channel = 0; channel < getNumInputChannels(); ++channel) {
        for (int position = 0; position < kTicTacMoPolicySize; ++position) {
            if (channel < kTicTacMoNumPlayer) {
                features.push_back(board_[position] == indexToPlayer(channel) ? 1.0f : 0.0f);
            } else {
                features.push_back(turn_ == indexToPlayer(channel - kTicTacMoNumPlayer) ? 1.0f : 0.0f);
            }
        }
    }
    return features;
}

std::vector<float> TicTacMoEnv::getActionFeatures(const TicTacMoAction& action, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    std::vector<float> action_features(kTicTacMoPolicySize, 0.0f);
    action_features[action.getActionID()] = 1.0f;
    return action_features;
}

std::string TicTacMoEnv::toString() const
{
    std::ostringstream oss;
    oss << "   A  B  C  D  E" << std::endl;
    for (int row = kTicTacMoBoardHeight - 1; row >= 0; --row) {
        oss << row + 1 << " ";
        for (int col = 0; col < kTicTacMoBoardWidth; ++col) {
            const Player player = board_[getPosition(row, col)];
            const char stone = player == Player::kPlayer1 ? 'X' : player == Player::kPlayer2 ? 'O'
                                                                                           : player == Player::kPlayer3 ? 'Y'
                                                                                                                        : '.';
            oss << ' ' << stone << ' ';
        }
        oss << std::endl;
    }
    return oss.str();
}

Player TicTacMoEnv::getWinner() const
{
    constexpr int directions[][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (int row = 0; row < kTicTacMoBoardHeight; ++row) {
        for (int col = 0; col < kTicTacMoBoardWidth; ++col) {
            const Player player = board_[getPosition(row, col)];
            if (player == Player::kPlayerNone) { continue; }
            for (const auto& direction : directions) {
                const int end_row = row + 2 * direction[0];
                const int end_col = col + 2 * direction[1];
                if (end_row < 0 || end_row >= kTicTacMoBoardHeight || end_col < 0 || end_col >= kTicTacMoBoardWidth) { continue; }
                if (board_[getPosition(row + direction[0], col + direction[1])] == player &&
                    board_[getPosition(end_row, end_col)] == player) {
                    return player;
                }
            }
        }
    }
    return Player::kPlayerNone;
}

void TicTacMoEnvLoader::loadFromEnvironment(const TicTacMoEnv& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history /* = {} */)
{
    BaseEnvLoader<TicTacMoAction, TicTacMoEnv>::loadFromEnvironment(env, action_info_history);
    addTag("RE", playerValuesToString(env.getEvalScores(), kTicTacMoNumPlayer));
}

std::vector<float> TicTacMoEnvLoader::getValue(const int pos) const
{
    return playerValuesToVector(stringToPlayerValues(getTag("RE"), kTicTacMoNumPlayer), kTicTacMoNumPlayer);
}

std::vector<float> TicTacMoEnvLoader::getActionFeatures(const int pos, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    std::vector<float> action_features(kTicTacMoPolicySize, 0.0f);
    const int action_id = pos < static_cast<int>(action_pairs_.size())
                              ? action_pairs_[pos].first.getActionID()
                              : utils::Random::randInt() % kTicTacMoPolicySize;
    action_features[action_id] = 1.0f;
    return action_features;
}

} // namespace minizero::env::tictacmo
