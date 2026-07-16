#pragma once

#include "base_env.h"
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace minizero::env::blokus {

constexpr int kBlokusNumPlayer = 4;
constexpr int kBlokusBoardSize = 20;
constexpr int kBlokusBoardArea = kBlokusBoardSize * kBlokusBoardSize;
constexpr int kBlokusNumPieces = 21;
constexpr int kBlokusNumOrientations = 91;
// Each orientation owns one 20x20 policy plane.  The final plane represents
// PASS at its first position; its other positions are intentionally illegal.
constexpr int kBlokusPassAction = kBlokusNumOrientations * kBlokusBoardArea;
constexpr int kBlokusPolicySize = (kBlokusNumOrientations + 1) * kBlokusBoardArea;
constexpr int kBlokusNumInputChannels = kBlokusNumPlayer +
                                         kBlokusNumPlayer * kBlokusNumPieces +
                                         3 * kBlokusNumPlayer;
constexpr int kBlokusNumActionFeatureChannels = 2;
const std::string kBlokusName = "blokus";

struct BlokusCell {
    int row;
    int col;

    bool operator==(const BlokusCell& rhs) const { return row == rhs.row && col == rhs.col; }
    bool operator<(const BlokusCell& rhs) const { return row < rhs.row || (row == rhs.row && col < rhs.col); }
};

struct BlokusOrientation {
    int piece;
    int height;
    int width;
    std::vector<BlokusCell> cells;
};

const std::vector<BlokusOrientation>& getBlokusOrientations();

class BlokusAction : public BaseAction {
public:
    BlokusAction() : BaseAction() {}
    BlokusAction(int action_id, Player player) : BaseAction(action_id, player) {}
    BlokusAction(const std::vector<std::string>& action_string_args);

    inline Player nextPlayer() const override { return getNextPlayer(getPlayer(), kBlokusNumPlayer); }
    std::string toConsoleString() const override;
};

class BlokusEnv : public BaseEnv<BlokusAction> {
public:
    BlokusEnv() { reset(); }

    void reset() override;
    bool act(const BlokusAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<BlokusAction> getLegalActions() const override;
    bool isLegalAction(const BlokusAction& action) const override;
    bool isTerminal() const override;
    inline float getReward() const override { return 0.0f; }
    float getEvalScore(bool is_resign = false) const override;
    PlayerValues getEvalScores(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const BlokusAction& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline int getNumInputChannels() const override { return kBlokusNumInputChannels; }
    inline int getNumActionFeatureChannels() const override { return kBlokusNumActionFeatureChannels; }
    inline int getInputChannelHeight() const override { return kBlokusBoardSize; }
    inline int getInputChannelWidth() const override { return kBlokusBoardSize; }
    inline int getHiddenChannelHeight() const override { return kBlokusBoardSize; }
    inline int getHiddenChannelWidth() const override { return kBlokusBoardSize; }
    inline int getPolicySize() const override { return kBlokusPolicySize; }
    inline int getDiscreteValueSize() const override { return 1; }
    inline int getRotatePosition(int position, utils::Rotation /* rotation */) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation /* rotation */) const override { return action_id; }
    std::string toString() const override;
    inline std::string name() const override { return kBlokusName; }
    inline int getNumPlayer() const override { return kBlokusNumPlayer; }
    static void setUpEnv() { config::env_board_size = kBlokusBoardSize; }

    int getPlayerScore(Player player) const;
    bool isPieceAvailable(Player player, int piece) const;
    Player getPlayerAt(int row, int col) const;

private:
    bool isPlacementLegal(int action_id, Player player) const;
    bool hasPlacement(Player player) const;
    bool isFirstMove(Player player) const;
    bool coversStartingCorner(const BlokusOrientation& orientation, int anchor_row, int anchor_col, Player player) const;
    inline int getPosition(int row, int col) const { return row * kBlokusBoardSize + col; }

    std::array<Player, kBlokusBoardArea> board_;
    std::array<std::array<bool, kBlokusNumPieces>, kBlokusNumPlayer> available_;
    std::array<bool, kBlokusNumPlayer> eliminated_;
    std::array<bool, kBlokusNumPlayer> monomino_last_;
};

class BlokusEnvLoader : public BaseEnvLoader<BlokusAction, BlokusEnv> {
public:
    void loadFromEnvironment(const BlokusEnv& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override;
    std::vector<float> getValue(const int pos) const override;
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline std::string name() const override { return kBlokusName; }
    inline int getPolicySize() const override { return kBlokusPolicySize; }
    inline int getRotatePosition(int position, utils::Rotation /* rotation */) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation /* rotation */) const override { return action_id; }
};

} // namespace minizero::env::blokus
