#pragma once

#include "base_env.h"
#include <string>
#include <utility>
#include <vector>

namespace minizero::env::connect3x3 {

const std::string kConnect3x3Name = "connect3x3";
const int kConnect3x3NumPlayer = 3;
const int kConnect3x3BoardHeight = 6;
const int kConnect3x3BoardWidth = 7;
const int kConnect3x3PolicySize = kConnect3x3BoardWidth;

class Connect3x3Action : public BaseAction {
public:
    Connect3x3Action() : BaseAction() {}
    Connect3x3Action(int action_id, Player player) : BaseAction(action_id, player) {}
    Connect3x3Action(const std::vector<std::string>& action_string_args);

    inline Player nextPlayer() const override { return getNextPlayer(getPlayer(), kConnect3x3NumPlayer); }
    std::string toConsoleString() const override;
};

class Connect3x3Env : public BaseEnv<Connect3x3Action> {
public:
    Connect3x3Env() { reset(); }

    void reset() override;
    bool act(const Connect3x3Action& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<Connect3x3Action> getLegalActions() const override;
    bool isLegalAction(const Connect3x3Action& action) const override;
    bool isTerminal() const override;
    inline float getReward() const override { return 0.0f; }
    float getEvalScore(bool is_resign = false) const override;
    PlayerValues getEvalScores(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const Connect3x3Action& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline int getNumInputChannels() const override { return 2 * kConnect3x3NumPlayer; }
    inline int getNumActionFeatureChannels() const override { return 1; }
    inline int getInputChannelHeight() const override { return kConnect3x3BoardHeight; }
    inline int getInputChannelWidth() const override { return kConnect3x3BoardWidth; }
    inline int getHiddenChannelHeight() const override { return kConnect3x3BoardHeight; }
    inline int getHiddenChannelWidth() const override { return kConnect3x3BoardWidth; }
    inline int getPolicySize() const override { return kConnect3x3PolicySize; }
    inline int getDiscreteValueSize() const override { return 1; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
    std::string toString() const override;
    inline std::string name() const override { return kConnect3x3Name; }
    inline int getNumPlayer() const override { return kConnect3x3NumPlayer; }
    static void setUpEnv() { config::env_board_size = kConnect3x3BoardWidth; }

private:
    Player getWinner() const;
    inline int getPosition(int row, int col) const { return row * kConnect3x3BoardWidth + col; }

    std::vector<Player> board_;
    std::vector<int> column_heights_;
};

class Connect3x3EnvLoader : public BaseEnvLoader<Connect3x3Action, Connect3x3Env> {
public:
    void loadFromEnvironment(const Connect3x3Env& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override;
    std::vector<float> getValue(const int pos) const override;
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline std::string name() const override { return kConnect3x3Name; }
    inline int getPolicySize() const override { return kConnect3x3PolicySize; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
};

} // namespace minizero::env::connect3x3
