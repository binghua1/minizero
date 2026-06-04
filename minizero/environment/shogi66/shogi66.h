#pragma once

#include "base_env.h"
#include <string>
#include <vector>

namespace minizero::env::shogi66{

const std::string kshogi66Name = "shogi66";
const int kshogi66NumPlayer = 2;
const int kshogi66BoardSize = 6; // 66shogi

enum class Phase {
    kSetup,
    kPlay,
};

enum class PieceType {
    kEmpty, // 空
    kKing, // 王
    kGold, // 金將
    kSilver, kPromSilver, // 銀將 -> 成銀
    kKnight, kPromKnight, // 桂馬 -> 成桂
    kLance, kPromLance, // 香車 -> 成香
    kRook, kDragon, // 飛車 -> 龍王
    kBishop, kHorse, // 角行 -> 龍馬
    kPawn, kTokin, // 步兵 -> と金
};

struct Piece {
    Player owner;
    PieceType type;
};

class shogi66Env : public BaseBoardEnv<shogi66Action> {
public:
    shogi66Env() : BaseBoardEnv<shogi66Action>(kshogi66BoardSize) { reset(); }

    void reset() override;
    bool act(const shogi66Action& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;

    std::vector<shogi66Action> getLegalActions() const override;
    bool isLegalAction(const shogi66Action& action) const override;
    bool isTerminal() const override;
    float getReward() const override { return 0.0f; }
    float getEvalScore(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const shogi66Action& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline int getNumInputChannels() const override { return 4; }
    inline int getPolicySize() const override { return getBoardSize() * getBoardSize(); }

    std::string toString() const override;
    inline std::string name() const override { return kshogi66Name; }
    inline int getNumPlayer() const override { return kshogi66NumPlayer; }
    
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return utils::getPositionByRotating(rotation, position, getBoardSize()); };
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return getRotatePosition(action_id, rotation); };
    static void setUpEnv() { config::env_board_size = kshogi66BoardSize; }

private:
    Player eval() const;
};

} // namespace minizero::env::shogi66
