#include "go3.h"
#include "configuration.h"
#include "random.h"
#include "sgf_loader.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

namespace minizero::env::go3 {

namespace {

    char playerStone(Player player)
    {
        switch (player) {
            case Player::kPlayer1: return 'X';
            case Player::kPlayer2: return 'O';
            case Player::kPlayer3: return 'R';
            default: return '.';
        }
    }

} // namespace

void Go3Env::setUpEnv()
{
    if (config::env_board_size <= 0) { config::env_board_size = 5; }
}

Go3Action::Go3Action(const std::vector<std::string>& action_string_args)
{
    assert(action_string_args.size() == 2);
    assert(action_string_args[0].size() == 1);
    player_ = charToPlayer(action_string_args[0][0]);
    std::string action = action_string_args[1];
    std::transform(action.begin(), action.end(), action.begin(), [](unsigned char c) { return std::toupper(c); });
    if (action == "PASS") {
        action_id_ = config::env_board_size * config::env_board_size;
    } else {
        action_id_ = utils::SGFLoader::boardCoordinateStringToActionID(action_string_args[1], config::env_board_size);
    }
}

std::string Go3Action::toConsoleString() const
{
    return getActionID() == config::env_board_size * config::env_board_size
               ? "PASS"
               : utils::SGFLoader::actionIDToBoardCoordinateString(getActionID(), config::env_board_size);
}

void Go3Env::reset()
{
    board_size_ = config::env_board_size <= 0 ? 5 : config::env_board_size;
    if (board_size_ < 2 || board_size_ > kGo3MaxBoardSize) { throw std::runtime_error("go3 env_board_size must be between 2 and 19"); }
    turn_ = Player::kPlayer1;
    actions_.clear();
    observations_.clear();
    board_.assign(board_size_ * board_size_, Player::kPlayerNone);
    board_history_.clear();
    board_history_.push_back(boardKey(board_));
    consecutive_passes_ = 0;
}

bool Go3Env::act(const Go3Action& action)
{
    if (!isLegalAction(action)) { return false; }
    actions_.push_back(action);
    if (isPassAction(action)) {
        ++consecutive_passes_;
    } else {
        std::vector<Player> next_board;
        if (!playOnBoard(action.getActionID(), action.getPlayer(), next_board)) { return false; }
        board_ = next_board;
        board_history_.push_back(boardKey(board_));
        consecutive_passes_ = 0;
    }
    turn_ = action.nextPlayer();
    return true;
}

bool Go3Env::act(const std::vector<std::string>& action_string_args)
{
    assert(action_string_args.size() == 2);
    assert(action_string_args[0].size() == 1);
    Player player = charToPlayer(action_string_args[0][0]);
    std::string action = action_string_args[1];
    std::transform(action.begin(), action.end(), action.begin(), [](unsigned char c) { return std::toupper(c); });
    int action_id = action == "PASS"
                        ? getPassActionID()
                        : utils::SGFLoader::boardCoordinateStringToActionID(action_string_args[1], board_size_);
    return act(Go3Action(action_id, player));
}

std::vector<Go3Action> Go3Env::getLegalActions() const
{
    std::vector<Go3Action> actions;
    for (int action_id = 0; action_id < getPassActionID(); ++action_id) {
        Go3Action action(action_id, turn_);
        if (isLegalAction(action)) { actions.push_back(action); }
    }
    actions.emplace_back(getPassActionID(), turn_);
    return actions;
}

bool Go3Env::isLegalAction(const Go3Action& action) const
{
    const int player_id = static_cast<int>(action.getPlayer());
    if (isTerminal() || player_id < 1 || player_id > kGo3NumPlayer || action.getPlayer() != turn_) { return false; }
    if (isPassAction(action)) { return true; }
    if (action.getActionID() < 0 || action.getActionID() >= getPassActionID() || board_[action.getActionID()] != Player::kPlayerNone) { return false; }
    std::vector<Player> next_board;
    if (!playOnBoard(action.getActionID(), action.getPlayer(), next_board)) { return false; }
    const std::string key = boardKey(next_board);
    return std::find(board_history_.begin(), board_history_.end(), key) == board_history_.end();
}

bool Go3Env::isTerminal() const
{
    return consecutive_passes_ >= kGo3NumPlayer;
}

float Go3Env::getEvalScore(bool is_resign /* = false */) const
{
    return getEvalScores(is_resign)[0];
}

PlayerValues Go3Env::getEvalScores(bool is_resign /* = false */) const
{
    PlayerValues values{};
    if (is_resign) {
        values[playerToIndex(turn_)] = -1.0f;
        return values;
    }

    const std::array<int, kGo3NumPlayer> scores = getChineseScores();
    const float board_area = static_cast<float>(board_size_ * board_size_);
    for (int player = 0; player < kGo3NumPlayer; ++player) {
        values[player] = 2.0f * static_cast<float>(scores[player]) / board_area - 1.0f;
    }
    return values;
}

std::vector<float> Go3Env::getFeatures(utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    (void)rotation;
    std::vector<float> features;
    features.reserve(getNumInputChannels() * board_size_ * board_size_);
    for (int player = 0; player < kGo3NumPlayer; ++player) {
        for (Player cell : board_) { features.push_back(cell == indexToPlayer(player) ? 1.0f : 0.0f); }
    }
    for (int player = 0; player < kGo3NumPlayer; ++player) {
        features.insert(features.end(), board_.size(), turn_ == indexToPlayer(player) ? 1.0f : 0.0f);
    }
    return features;
}

std::vector<float> Go3Env::getActionFeatures(const Go3Action& action, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    (void)rotation;
    std::vector<float> action_features(board_size_ * board_size_, 0.0f);
    if (action.getActionID() >= 0 && action.getActionID() < getPassActionID()) { action_features[action.getActionID()] = 1.0f; }
    return action_features;
}

std::string Go3Env::toString() const
{
    std::ostringstream oss;
    oss << "   ";
    for (int col = 0; col < board_size_; ++col) { oss << ' ' << static_cast<char>('A' + col) << ' '; }
    oss << '\n';
    for (int row = board_size_ - 1; row >= 0; --row) {
        oss << row + 1 << (row + 1 < 10 ? "  " : " ");
        for (int col = 0; col < board_size_; ++col) { oss << ' ' << playerStone(board_[getPosition(row, col)]) << ' '; }
        oss << '\n';
    }
    return oss.str();
}

std::vector<int> Go3Env::getNeighbors(int position) const
{
    const int row = position / board_size_;
    const int col = position % board_size_;
    std::vector<int> neighbors;
    constexpr int directions[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (const auto& direction : directions) {
        const int next_row = row + direction[0];
        const int next_col = col + direction[1];
        if (isOnBoard(next_row, next_col)) { neighbors.push_back(getPosition(next_row, next_col)); }
    }
    return neighbors;
}

std::vector<int> Go3Env::collectString(int start, const std::vector<Player>& board) const
{
    const Player color = board[start];
    std::vector<int> stones;
    if (color == Player::kPlayerNone) { return stones; }
    std::vector<bool> visited(board.size(), false);
    std::queue<int> q;
    q.push(start);
    visited[start] = true;
    while (!q.empty()) {
        const int pos = q.front();
        q.pop();
        stones.push_back(pos);
        for (int neighbor : getNeighbors(pos)) {
            if (!visited[neighbor] && board[neighbor] == color) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
    return stones;
}

bool Go3Env::stringHasLiberty(const std::vector<int>& stones, const std::vector<Player>& board) const
{
    for (int stone : stones) {
        for (int neighbor : getNeighbors(stone)) {
            if (board[neighbor] == Player::kPlayerNone) { return true; }
        }
    }
    return false;
}

bool Go3Env::playOnBoard(int action_id, Player player, std::vector<Player>& next_board) const
{
    if (action_id < 0 || action_id >= getPassActionID() || board_[action_id] != Player::kPlayerNone) { return false; }
    next_board = board_;
    next_board[action_id] = player;

    for (int neighbor : getNeighbors(action_id)) {
        if (next_board[neighbor] == Player::kPlayerNone || next_board[neighbor] == player) { continue; }
        const std::vector<int> block = collectString(neighbor, next_board);
        if (!stringHasLiberty(block, next_board)) {
            for (int stone : block) { next_board[stone] = Player::kPlayerNone; }
        }
    }

    const std::vector<int> own_block = collectString(action_id, next_board);
    return stringHasLiberty(own_block, next_board);
}

std::string Go3Env::boardKey(const std::vector<Player>& board) const
{
    std::string key;
    key.reserve(board.size());
    for (Player player : board) { key.push_back(static_cast<char>('0' + static_cast<int>(player))); }
    return key;
}

std::array<int, kGo3NumPlayer> Go3Env::getChineseScores() const
{
    std::array<int, kGo3NumPlayer> scores{};
    std::vector<bool> visited(board_.size(), false);
    for (int pos = 0; pos < static_cast<int>(board_.size()); ++pos) {
        const Player player = board_[pos];
        if (player != Player::kPlayerNone) {
            ++scores[playerToIndex(player)];
            continue;
        }
        if (visited[pos]) { continue; }
        std::vector<int> region;
        std::set<Player> bordering_players;
        std::queue<int> q;
        q.push(pos);
        visited[pos] = true;
        while (!q.empty()) {
            const int current = q.front();
            q.pop();
            region.push_back(current);
            for (int neighbor : getNeighbors(current)) {
                if (board_[neighbor] == Player::kPlayerNone) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                } else {
                    bordering_players.insert(board_[neighbor]);
                }
            }
        }
        if (bordering_players.size() == 1) { scores[playerToIndex(*bordering_players.begin())] += static_cast<int>(region.size()); }
    }
    return scores;
}

void Go3EnvLoader::loadFromEnvironment(const Go3Env& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history /* = {} */)
{
    BaseEnvLoader<Go3Action, Go3Env>::loadFromEnvironment(env, action_info_history);
    addTag("SZ", std::to_string(env.getInputChannelWidth()));
    addTag("RE", playerValuesToString(env.getEvalScores(), kGo3NumPlayer));
}

std::vector<float> Go3EnvLoader::getValue(const int pos) const
{
    return playerValuesToVector(stringToPlayerValues(getTag("RE"), kGo3NumPlayer), kGo3NumPlayer);
}

std::vector<float> Go3EnvLoader::getActionFeatures(const int pos, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    (void)rotation;
    const int board_size = getTag("SZ").empty() ? config::env_board_size : std::stoi(getTag("SZ"));
    std::vector<float> action_features(board_size * board_size, 0.0f);
    const int action_id = pos < static_cast<int>(action_pairs_.size())
                              ? action_pairs_[pos].first.getActionID()
                              : utils::Random::randInt() % (board_size * board_size + 1);
    if (action_id >= 0 && action_id < board_size * board_size) { action_features[action_id] = 1.0f; }
    return action_features;
}

} // namespace minizero::env::go3
