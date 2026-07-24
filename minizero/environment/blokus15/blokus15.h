#pragma once

#include "base_env.h"
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace minizero::env::blokus15 {

constexpr int kBlokus15NumPlayer = 4;
constexpr int kBlokus15BoardSize = 15;
constexpr int kBlokus15BoardArea = kBlokus15BoardSize * kBlokus15BoardSize;
constexpr int kBlokus15NumPieces = 14;
constexpr int kBlokus15NumOrientations = 54;
constexpr int kBlokus15TotalSquares = 54;
constexpr int kBlokus15PassAction = kBlokus15NumOrientations * kBlokus15BoardArea;
constexpr int kBlokus15PolicySize = (kBlokus15NumOrientations + 1) * kBlokus15BoardArea;
constexpr int kBlokus15NumInputChannels = kBlokus15NumPlayer +
                                          kBlokus15NumPlayer * kBlokus15NumPieces +
                                          3 * kBlokus15NumPlayer;
constexpr int kBlokus15NumActionFeatureChannels = 2;
const std::string kBlokus15Name = "blokus15";

struct Blokus15Cell {
    int row;
    int col;

    bool operator==(const Blokus15Cell& rhs) const { return row == rhs.row && col == rhs.col; }
    bool operator<(const Blokus15Cell& rhs) const { return row < rhs.row || (row == rhs.row && col < rhs.col); }
};

struct Blokus15Orientation {
    int piece;
    int height;
    int width;
    std::vector<Blokus15Cell> cells;
};

const std::vector<Blokus15Orientation>& getBlokus15Orientations();

class Blokus15Action : public BaseAction {
public:
    Blokus15Action() : BaseAction() {}
    Blokus15Action(int action_id, Player player) : BaseAction(action_id, player) {}
    Blokus15Action(const std::vector<std::string>& action_string_args);

    inline Player nextPlayer() const override { return getNextPlayer(getPlayer(), kBlokus15NumPlayer); }
    std::string toConsoleString() const override;
};

class Blokus15Env : public BaseEnv<Blokus15Action> {
public:
    Blokus15Env() { reset(); }

    void reset() override;
    bool act(const Blokus15Action& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<Blokus15Action> getLegalActions() const override;
    bool isLegalAction(const Blokus15Action& action) const override;
    bool isTerminal() const override;
    inline float getReward() const override { return 0.0f; }
    float getEvalScore(bool is_resign = false) const override;
    PlayerValues getEvalScores(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const Blokus15Action& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline int getNumInputChannels() const override { return kBlokus15NumInputChannels; }
    inline int getNumActionFeatureChannels() const override { return kBlokus15NumActionFeatureChannels; }
    inline int getInputChannelHeight() const override { return kBlokus15BoardSize; }
    inline int getInputChannelWidth() const override { return kBlokus15BoardSize; }
    inline int getHiddenChannelHeight() const override { return kBlokus15BoardSize; }
    inline int getHiddenChannelWidth() const override { return kBlokus15BoardSize; }
    inline int getPolicySize() const override { return kBlokus15PolicySize; }
    inline int getDiscreteValueSize() const override { return 1; }
    inline int getRotatePosition(int position, utils::Rotation /* rotation */) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation /* rotation */) const override { return action_id; }
    std::string toString() const override;
    inline std::string name() const override { return kBlokus15Name; }
    inline int getNumPlayer() const override { return kBlokus15NumPlayer; }
    static void setUpEnv() { config::env_board_size = kBlokus15BoardSize; }

    int getPlayerScore(Player player) const;
    bool isPieceAvailable(Player player, int piece) const;
    Player getPlayerAt(int row, int col) const;

private:
    std::vector<Blokus15Cell> getRequiredContactPoints(Player player) const;
    std::array<bool, kBlokus15BoardArea> getCandidateAnchors(const Blokus15Orientation& orientation, const std::vector<Blokus15Cell>& contact_points) const;
    bool isPlacementLegal(int action_id, Player player) const;
    bool hasPlacement(Player player) const;
    bool isFirstMove(Player player) const;
    bool coversStartingCorner(const Blokus15Orientation& orientation, int anchor_row, int anchor_col, Player player) const;
    inline int getPosition(int row, int col) const { return row * kBlokus15BoardSize + col; }

    std::array<Player, kBlokus15BoardArea> board_;
    std::array<std::array<bool, kBlokus15NumPieces>, kBlokus15NumPlayer> available_;
    std::array<bool, kBlokus15NumPlayer> eliminated_;
    std::array<bool, kBlokus15NumPlayer> monomino_last_;
};

class Blokus15EnvLoader : public BaseEnvLoader<Blokus15Action, Blokus15Env> {
public:
    void loadFromEnvironment(const Blokus15Env& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override;
    std::vector<float> getValue(const int pos) const override;
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline std::string name() const override { return kBlokus15Name; }
    inline int getPolicySize() const override { return kBlokus15PolicySize; }
    inline int getRotatePosition(int position, utils::Rotation /* rotation */) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation /* rotation */) const override { return action_id; }
};

} // namespace minizero::env::blokus15
