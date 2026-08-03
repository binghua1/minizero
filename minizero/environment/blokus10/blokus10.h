#pragma once

#include "base_env.h"
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace minizero::env::blokus10 {

constexpr int kBlokus10NumPlayer = 4;
constexpr int kBlokus10BoardSize = 10;
constexpr int kBlokus10BoardArea = kBlokus10BoardSize * kBlokus10BoardSize;
constexpr int kBlokus10NumPieces = 9;
constexpr int kBlokus10NumOrientations = 28;
constexpr int kBlokus10TotalSquares = 29;
constexpr int kBlokus10PassAction = kBlokus10NumOrientations * kBlokus10BoardArea;
constexpr int kBlokus10PolicySize = (kBlokus10NumOrientations + 1) * kBlokus10BoardArea;
constexpr int kBlokus10NumInputChannels = kBlokus10NumPlayer +
                                          kBlokus10NumPlayer * kBlokus10NumPieces +
                                          3 * kBlokus10NumPlayer;
constexpr int kBlokus10NumActionFeatureChannels = 2;
const std::string kBlokus10Name = "blokus10";

struct Blokus10Cell {
    int row;
    int col;

    bool operator==(const Blokus10Cell& rhs) const { return row == rhs.row && col == rhs.col; }
    bool operator<(const Blokus10Cell& rhs) const { return row < rhs.row || (row == rhs.row && col < rhs.col); }
};

struct Blokus10Orientation {
    int piece;
    int height;
    int width;
    std::vector<Blokus10Cell> cells;
};

const std::vector<Blokus10Orientation>& getBlokus10Orientations();

class Blokus10Action : public BaseAction {
public:
    Blokus10Action() : BaseAction() {}
    Blokus10Action(int action_id, Player player) : BaseAction(action_id, player) {}
    Blokus10Action(const std::vector<std::string>& action_string_args);

    inline Player nextPlayer() const override { return getNextPlayer(getPlayer(), kBlokus10NumPlayer); }
    std::string toConsoleString() const override;
};

class Blokus10Env : public BaseEnv<Blokus10Action> {
public:
    Blokus10Env() { reset(); }

    void reset() override;
    bool act(const Blokus10Action& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<Blokus10Action> getLegalActions() const override;
    bool isLegalAction(const Blokus10Action& action) const override;
    bool isTerminal() const override;
    inline float getReward() const override { return 0.0f; }
    float getEvalScore(bool is_resign = false) const override;
    PlayerValues getEvalScores(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const Blokus10Action& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline int getNumInputChannels() const override { return kBlokus10NumInputChannels; }
    inline int getNumActionFeatureChannels() const override { return kBlokus10NumActionFeatureChannels; }
    inline int getInputChannelHeight() const override { return kBlokus10BoardSize; }
    inline int getInputChannelWidth() const override { return kBlokus10BoardSize; }
    inline int getHiddenChannelHeight() const override { return kBlokus10BoardSize; }
    inline int getHiddenChannelWidth() const override { return kBlokus10BoardSize; }
    inline int getPolicySize() const override { return kBlokus10PolicySize; }
    inline int getDiscreteValueSize() const override { return 1; }
    inline int getRotatePosition(int position, utils::Rotation /* rotation */) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation /* rotation */) const override { return action_id; }
    std::string toString() const override;
    inline std::string name() const override { return kBlokus10Name; }
    inline int getNumPlayer() const override { return kBlokus10NumPlayer; }
    static void setUpEnv() { config::env_board_size = kBlokus10BoardSize; }

    int getPlayerScore(Player player) const;
    bool isPieceAvailable(Player player, int piece) const;
    Player getPlayerAt(int row, int col) const;

private:
    std::vector<Blokus10Cell> getRequiredContactPoints(Player player) const;
    std::array<bool, kBlokus10BoardArea> getCandidateAnchors(const Blokus10Orientation& orientation, const std::vector<Blokus10Cell>& contact_points) const;
    bool isPlacementLegal(int action_id, Player player) const;
    bool hasPlacement(Player player) const;
    bool isFirstMove(Player player) const;
    bool coversStartingCorner(const Blokus10Orientation& orientation, int anchor_row, int anchor_col, Player player) const;
    inline int getPosition(int row, int col) const { return row * kBlokus10BoardSize + col; }

    std::array<Player, kBlokus10BoardArea> board_;
    std::array<std::array<bool, kBlokus10NumPieces>, kBlokus10NumPlayer> available_;
    std::array<bool, kBlokus10NumPlayer> eliminated_;
    std::array<bool, kBlokus10NumPlayer> monomino_last_;
};

class Blokus10EnvLoader : public BaseEnvLoader<Blokus10Action, Blokus10Env> {
public:
    void loadFromEnvironment(const Blokus10Env& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override;
    std::vector<float> getValue(const int pos) const override;
    std::vector<float> getRank(const int pos) const override;
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline std::string name() const override { return kBlokus10Name; }
    inline int getPolicySize() const override { return kBlokus10PolicySize; }
    inline int getRotatePosition(int position, utils::Rotation /* rotation */) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation /* rotation */) const override { return action_id; }
};

} // namespace minizero::env::blokus10
