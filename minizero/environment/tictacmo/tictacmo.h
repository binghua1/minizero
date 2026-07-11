#pragma once

#include "base_env.h"
#include <string>
#include <utility>
#include <vector>

namespace minizero::env::tictacmo {

const std::string kTicTacMoName = "tictacmo";
const int kTicTacMoNumPlayer = 3;
const int kTicTacMoBoardHeight = 3;
const int kTicTacMoBoardWidth = 5;
const int kTicTacMoPolicySize = kTicTacMoBoardHeight * kTicTacMoBoardWidth;

class TicTacMoAction : public BaseAction {
public:
    TicTacMoAction() : BaseAction() {}
    TicTacMoAction(int action_id, Player player) : BaseAction(action_id, player) {}
    TicTacMoAction(const std::vector<std::string>& action_string_args);

    inline Player nextPlayer() const override { return getNextPlayer(getPlayer(), kTicTacMoNumPlayer); }
    std::string toConsoleString() const override;
};

class TicTacMoEnv : public BaseEnv<TicTacMoAction> {
public:
    TicTacMoEnv() { reset(); }

    void reset() override;
    bool act(const TicTacMoAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<TicTacMoAction> getLegalActions() const override;
    bool isLegalAction(const TicTacMoAction& action) const override;
    bool isTerminal() const override;
    inline float getReward() const override { return 0.0f; }
    float getEvalScore(bool is_resign = false) const override;
    PlayerValues getEvalScores(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const TicTacMoAction& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline int getNumInputChannels() const override { return 2 * kTicTacMoNumPlayer; }
    inline int getNumActionFeatureChannels() const override { return 1; }
    inline int getInputChannelHeight() const override { return kTicTacMoBoardHeight; }
    inline int getInputChannelWidth() const override { return kTicTacMoBoardWidth; }
    inline int getHiddenChannelHeight() const override { return kTicTacMoBoardHeight; }
    inline int getHiddenChannelWidth() const override { return kTicTacMoBoardWidth; }
    inline int getPolicySize() const override { return kTicTacMoPolicySize; }
    inline int getDiscreteValueSize() const override { return 1; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
    std::string toString() const override;
    inline std::string name() const override { return kTicTacMoName; }
    inline int getNumPlayer() const override { return kTicTacMoNumPlayer; }
    static void setUpEnv() { config::env_board_size = kTicTacMoBoardWidth; }

private:
    Player getWinner() const;
    inline int getPosition(int row, int col) const { return row * kTicTacMoBoardWidth + col; }

    std::vector<Player> board_;
};

class TicTacMoEnvLoader : public BaseEnvLoader<TicTacMoAction, TicTacMoEnv> {
public:
    void loadFromEnvironment(const TicTacMoEnv& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override;
    std::vector<float> getValue(const int pos) const override;
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline std::string name() const override { return kTicTacMoName; }
    inline int getPolicySize() const override { return kTicTacMoPolicySize; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
};

} // namespace minizero::env::tictacmo
