#pragma once

#include "base_env.h"
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace minizero::env::go3 {

constexpr int kGo3NumPlayer = 3;
constexpr int kGo3MaxBoardSize = 19;
constexpr int kGo3NumInputChannels = 2 * kGo3NumPlayer;
constexpr int kGo3NumActionFeatureChannels = 1;
const std::string kGo3Name = "go3";

class Go3Action : public BaseAction {
public:
    Go3Action() : BaseAction() {}
    Go3Action(int action_id, Player player) : BaseAction(action_id, player) {}
    Go3Action(const std::vector<std::string>& action_string_args);

    inline Player nextPlayer() const override { return getNextPlayer(getPlayer(), kGo3NumPlayer); }
    std::string toConsoleString() const override;
};

class Go3Env : public BaseEnv<Go3Action> {
public:
    Go3Env() { reset(); }

    void reset() override;
    bool act(const Go3Action& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<Go3Action> getLegalActions() const override;
    bool isLegalAction(const Go3Action& action) const override;
    bool isTerminal() const override;
    inline float getReward() const override { return 0.0f; }
    float getEvalScore(bool is_resign = false) const override;
    PlayerValues getEvalScores(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const Go3Action& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline int getNumInputChannels() const override { return kGo3NumInputChannels; }
    inline int getNumActionFeatureChannels() const override { return kGo3NumActionFeatureChannels; }
    inline int getInputChannelHeight() const override { return board_size_; }
    inline int getInputChannelWidth() const override { return board_size_; }
    inline int getHiddenChannelHeight() const override { return board_size_; }
    inline int getHiddenChannelWidth() const override { return board_size_; }
    inline int getPolicySize() const override { return board_size_ * board_size_ + 1; }
    inline int getDiscreteValueSize() const override { return 1; }
    inline int getRotatePosition(int position, utils::Rotation /* rotation */) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation /* rotation */) const override { return action_id; }
    std::string toString() const override;
    inline std::string name() const override { return kGo3Name + "_" + std::to_string(board_size_) + "x" + std::to_string(board_size_); }
    inline int getNumPlayer() const override { return kGo3NumPlayer; }
    static void setUpEnv();

    std::array<int, kGo3NumPlayer> getChineseScores() const;

private:
    int getPassActionID() const { return board_size_ * board_size_; }
    int getPosition(int row, int col) const { return row * board_size_ + col; }
    bool isOnBoard(int row, int col) const { return row >= 0 && row < board_size_ && col >= 0 && col < board_size_; }
    bool isPassAction(const Go3Action& action) const { return action.getActionID() == getPassActionID(); }
    std::vector<int> getNeighbors(int position) const;
    std::vector<int> collectString(int start, const std::vector<Player>& board) const;
    bool stringHasLiberty(const std::vector<int>& stones, const std::vector<Player>& board) const;
    bool playOnBoard(int action_id, Player player, std::vector<Player>& next_board) const;
    std::string boardKey(const std::vector<Player>& board) const;

    int board_size_;
    std::vector<Player> board_;
    std::vector<std::string> board_history_;
    int consecutive_passes_;
};

class Go3EnvLoader : public BaseEnvLoader<Go3Action, Go3Env> {
public:
    void loadFromEnvironment(const Go3Env& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override;
    std::vector<float> getValue(const int pos) const override;
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline std::string name() const override { return kGo3Name; }
    inline int getPolicySize() const override { return getTag("SZ").empty() ? 26 : std::stoi(getTag("SZ")) * std::stoi(getTag("SZ")) + 1; }
    inline int getRotatePosition(int position, utils::Rotation /* rotation */) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation /* rotation */) const override { return action_id; }
};

} // namespace minizero::env::go3
