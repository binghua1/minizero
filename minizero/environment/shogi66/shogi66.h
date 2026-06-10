#pragma once

#include "base_env.h"
#include <string>
#include <vector>
#include <array>
#include <unordered_map>


namespace minizero::env::shogi66{

const std::string kshogi66Name = "shogi66";
constexpr int kshogi66NumPlayer = 2;
constexpr int kshogi66BoardSize = 6;
constexpr int kshogi66BoardArea = kshogi66BoardSize * kshogi66BoardSize; // 36
constexpr int kshogi66NumActionPieceTypes = 8;

constexpr int kshogi66SetupActionSize = kshogi66NumActionPieceTypes * kshogi66BoardArea;
constexpr int kshogi66MoveActionSize = kshogi66BoardArea * kshogi66BoardArea * 2; // 升
constexpr int kshogi66DropActionSize = kshogi66NumActionPieceTypes * kshogi66BoardArea;
constexpr int kshogi66PolicySize = kshogi66SetupActionSize + kshogi66MoveActionSize + kshogi66DropActionSize;
using shogi66Hand = std::array<int, kshogi66NumActionPieceTypes>;

enum class Phase {
    kSetup,
    kPlay,
};

enum class ActionType {
    kSetup,
    kMove,
    kDrop,
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

    Piece(Player p = Player::kPlayerNone, PieceType t = PieceType::kEmpty) : owner(p), type(t) {}
};

class shogi66Action : public BaseBoardAction<kshogi66NumPlayer> {
public:
    shogi66Action();
    shogi66Action(int action_id, Player player);
    shogi66Action(ActionType type, PieceType piece_type, int from, int to, bool promote, Player player);
    shogi66Action(const std::vector<std::string>& action_string_args);

    std::string toConsoleString() const override;

    inline ActionType getType() const {return type_;}
    inline PieceType getPieceType() const {return piece_type_;}
    inline int getFrom() const {return from_;}
    inline int getTo() const {return to_;}
    inline bool isPromote() const {return promote_;}
    static int pieceTypeToActionId(PieceType piece_type);
    static PieceType actionIdToPieceType(int index);
    static int encodeActionId(ActionType type, PieceType piece_type, int from, int to, bool promote);

private:
    void decodeActionId();
    ActionType type_;
    PieceType piece_type_;
    int from_;
    int to_;
    bool promote_;
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
    inline int getPolicySize() const override { return kshogi66PolicySize; }

    std::string toString() const override;
    inline std::string name() const override { return kshogi66Name; }
    inline int getNumPlayer() const override { return kshogi66NumPlayer; }
    
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return utils::getPositionByRotating(rotation, position, getBoardSize()); };
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return getRotatePosition(action_id, rotation); };
    static void setUpEnv() { config::env_board_size = kshogi66BoardSize; }

private:
    Player eval() const;
    Phase phase_;
    Player winner_;
    std::vector<Piece> board_;

    bool repetition_draw;
    GamePair<shogi66Hand> hand_;
    std::unordered_map<std::string, int> state_count_;
};

class shogi66EnvLoader : public BaseBoardEnvLoader<shogi66Action, shogi66Env> {
public:
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override {
        return {};
    }
    inline std::vector<float> getValue(const int pos) const { return {getReturn()}; }
    inline std::string name() const override { return kshogi66Name; }
    inline int getPolicySize() const override { return kshogi66PolicySize; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; };
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; };
};

} // namespace minizero::env::shogi66
