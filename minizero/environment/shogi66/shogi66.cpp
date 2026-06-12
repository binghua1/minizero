#include "shogi66.h"
#include <algorithm>
#include <sstream>
#include <string>

namespace minizero::env::shogi66 {

using namespace minizero::utils;
namespace {

    int row(int pos) { return pos / kshogi66BoardSize; }
    int col(int pos) { return pos % kshogi66BoardSize; }
    bool inBoard(int pos) { return 0 <= pos && pos < kshogi66BoardArea; }
    int sign(int x) { return x == 0 ? 0 : (x > 0 ? 1 : -1); }
    int dir(Player p) { return p == Player::kPlayer1 ? 1 : -1; }
    bool isPlayer(Player p) { return p == Player::kPlayer1 || p == Player::kPlayer2; }
    bool isPromoted(PieceType type)
    {
        return type == PieceType::kPromSilver || type == PieceType::kPromKnight || type == PieceType::kPromLance || type == PieceType::kDragon || type == PieceType::kHorse || type == PieceType::kTokin;
    }
    bool isGoldLike(PieceType type)
    {
        return type == PieceType::kGold || type == PieceType::kPromSilver || type == PieceType::kPromKnight || type == PieceType::kPromLance || type == PieceType::kTokin;
    }

    PieceType promoted(PieceType type)
    {
        switch (type) {
            case PieceType::kSilver: return PieceType::kPromSilver;
            case PieceType::kKnight: return PieceType::kPromKnight;
            case PieceType::kLance: return PieceType::kPromLance;
            case PieceType::kRook: return PieceType::kDragon;
            case PieceType::kBishop: return PieceType::kHorse;
            case PieceType::kPawn: return PieceType::kTokin;
            default: return type;
        }
    }

    PieceType unpromote(PieceType type)
    {
        switch (type) {
            case PieceType::kPromSilver: return PieceType::kSilver;
            case PieceType::kPromKnight: return PieceType::kKnight;
            case PieceType::kPromLance: return PieceType::kLance;
            case PieceType::kDragon: return PieceType::kRook;
            case PieceType::kHorse: return PieceType::kBishop;
            case PieceType::kTokin: return PieceType::kPawn;
            default: return type;
        }
    }

    bool isPromotable(PieceType type)
    {
        return promoted(type) != type;
    }

    std::string squareToString(int pos)
    {
        // 0 -> A1, 1 -> B1, ..., 5 -> F1, 6 -> A2, ..., 35 -> F6
        return std::string(1, static_cast<char>('A' + col(pos))) + std::to_string(row(pos) + 1);
    }

    int stringToSquare(std::string s)
    {
        if (size(s) < 2) return -1;
        int x = toupper(s[0]) - 'A', y = std::stoi(s.substr(1)) - 1;
        if (x < 0 || x >= kshogi66BoardSize || y < 0 || y >= kshogi66BoardSize) return -1;
        return y * kshogi66BoardSize + x;
    }

    std::string pieceToString(PieceType piece)
    {
        switch (piece) {
            case PieceType::kKing: return "K";
            case PieceType::kGold: return "G";
            case PieceType::kSilver: return "S";
            case PieceType::kPromSilver: return "+S";
            case PieceType::kKnight: return "N";
            case PieceType::kPromKnight: return "+N";
            case PieceType::kLance: return "L";
            case PieceType::kPromLance: return "+L";
            case PieceType::kRook: return "R";
            case PieceType::kDragon: return "+R";
            case PieceType::kBishop: return "B";
            case PieceType::kHorse: return "+B";
            case PieceType::kPawn: return "P";
            case PieceType::kTokin: return "+P";
            default: return ".";
        }
    }

    PieceType stringToPiece(const std::string& s)
    {
        if (s == "K") return PieceType::kKing;
        if (s == "G") return PieceType::kGold;
        if (s == "S") return PieceType::kSilver;
        if (s == "+S") return PieceType::kPromSilver;
        if (s == "N") return PieceType::kKnight;
        if (s == "+N") return PieceType::kPromKnight;
        if (s == "L") return PieceType::kLance;
        if (s == "+L") return PieceType::kPromLance;
        if (s == "R") return PieceType::kRook;
        if (s == "+R") return PieceType::kDragon;
        if (s == "B") return PieceType::kBishop;
        if (s == "+B") return PieceType::kHorse;
        if (s == "P") return PieceType::kPawn;
        if (s == "+P") return PieceType::kTokin;
        return PieceType::kEmpty;
    }
} // namespace

shogi66Action::shogi66Action() : BaseBoardAction<kshogi66NumPlayer>(-1, Player::kPlayerNone), type_(ActionType::kMove), piece_type_(PieceType::kEmpty), from_(-1), to_(-1), promote_(false) {}

shogi66Action::shogi66Action(int action_id, Player player) : BaseBoardAction<kshogi66NumPlayer>(action_id, player), type_(ActionType::kMove), piece_type_(PieceType::kEmpty), from_(-1), to_(-1), promote_(false)
{
    decodeActionId();
}

shogi66Action::shogi66Action(ActionType type, PieceType piece_type, int from, int to, bool promote, Player player)
    : BaseBoardAction<kshogi66NumPlayer>(encodeActionId(type, piece_type, from, to, promote), player), // id, player
      type_(type), piece_type_(piece_type), from_(from), to_(to), promote_(promote)
{
}

shogi66Action::shogi66Action(const std::vector<std::string>& action_string_args) : shogi66Action()
{
    player_ = charToPlayer(action_string_args[0][0]);
    std::string s = action_string_args[1];
    if (s.rfind("S:", 0) == 0) {
        type_ = ActionType::kSetup;
        int money_pos = s.find('$');
        piece_type_ = stringToPiece(s.substr(2, money_pos - 2));
        to_ = stringToSquare(s.substr(money_pos + 1));
    } else if (find(begin(s), end(s), '*') != s.end()) {
        type_ = ActionType::kDrop;
        int star_pos = s.find('*');
        piece_type_ = stringToPiece(s.substr(0, star_pos));
        to_ = stringToSquare(s.substr(star_pos + 1));
    } else {
        type_ = ActionType::kMove;
        promote_ = !s.empty() && s.back() == '+';
        if (promote_) s.pop_back();
        from_ = stringToSquare(s.substr(0, 2));
        to_ = stringToSquare(s.substr(2, 2));
    }
}

int shogi66Action::setupPieceToActionId(PieceType piece_type)
{
    switch (piece_type) {
        case PieceType::kKing: return 0;
        case PieceType::kGold: return 1;
        case PieceType::kSilver: return 2;
        case PieceType::kKnight: return 3;
        case PieceType::kLance: return 4;
        case PieceType::kRook: return 5;
        case PieceType::kBishop: return 6;
        default: return -1;
    }
}

int shogi66Action::handPieceToActionId(PieceType piece_type)
{
    switch (unpromote(piece_type)) {
        case PieceType::kGold: return 0;
        case PieceType::kSilver: return 1;
        case PieceType::kKnight: return 2;
        case PieceType::kLance: return 3;
        case PieceType::kRook: return 4;
        case PieceType::kBishop: return 5;
        case PieceType::kPawn: return 6;
        default: return -1;
    }
}

PieceType shogi66Action::setupActionIdToPieceType(int index)
{
    static const std::array<PieceType, kshogi66SetupPieceTypes> pieces = {
        PieceType::kKing, PieceType::kGold, PieceType::kSilver, PieceType::kKnight,
        PieceType::kLance, PieceType::kRook, PieceType::kBishop};
    return 0 <= index && index < kshogi66SetupPieceTypes ? pieces[index] : PieceType::kEmpty;
}

PieceType shogi66Action::handActionIdToPieceType(int index)
{
    static const std::array<PieceType, kshogi66HandPieceTypes> pieces = {
        PieceType::kGold, PieceType::kSilver, PieceType::kKnight, PieceType::kLance,
        PieceType::kRook, PieceType::kBishop, PieceType::kPawn};
    return 0 <= index && index < kshogi66HandPieceTypes ? pieces[index] : PieceType::kEmpty;
}

int shogi66Action::encodeActionId(ActionType type, PieceType piece_type, int from, int to, bool promote)
{
    if (type == ActionType::kSetup) { // 1
        int id = setupPieceToActionId(piece_type);
        return id < 0 || !inBoard(to) ? -1 : id * kshogi66BoardArea + to;
    } else if (type == ActionType::kMove) {
        return !inBoard(from) || !inBoard(to) ? -1 : kshogi66SetupActionSize + ((from * kshogi66BoardArea + to) * 2 + static_cast<int>(promote));
    } else if (type == ActionType::kDrop) {
        int id = handPieceToActionId(piece_type);
        return id < 0 || !inBoard(to) ? -1 : kshogi66SetupActionSize + kshogi66MoveActionSize + id * kshogi66BoardArea + to;
    }
    return -1;
}

void shogi66Action::decodeActionId()
{
    int id = action_id_;
    if (0 <= id && id < kshogi66SetupActionSize) {
        type_ = ActionType::kSetup;
        piece_type_ = setupActionIdToPieceType(id / kshogi66BoardArea);
        from_ = -1;
        to_ = id % kshogi66BoardArea;
        promote_ = false;
    } else if (kshogi66SetupActionSize <= id && id < kshogi66SetupActionSize + kshogi66MoveActionSize) {
        id -= kshogi66SetupActionSize;
        type_ = ActionType::kMove;
        promote_ = (id % 2) != 0;
        id >>= 1;
        from_ = id / kshogi66BoardArea;
        to_ = id % kshogi66BoardArea;
        piece_type_ = PieceType::kEmpty;
    } else if (kshogi66SetupActionSize + kshogi66MoveActionSize <= id && id < kshogi66PolicySize) {
        id -= kshogi66SetupActionSize + kshogi66MoveActionSize;
        type_ = ActionType::kDrop;
        piece_type_ = handActionIdToPieceType(id / kshogi66BoardArea);
        from_ = -1;
        to_ = id % kshogi66BoardArea;
        promote_ = false;
    }
}

static PieceType actionIdToPieceType(int index)
{
    switch (index) {
        case 0: return PieceType::kKing;
        case 1: return PieceType::kGold;
        case 2: return PieceType::kSilver;
        case 3: return PieceType::kKnight;
        case 4: return PieceType::kLance;
        case 5: return PieceType::kRook;
        case 6: return PieceType::kBishop;
        case 7: return PieceType::kPawn;
        default: return PieceType::kEmpty;
    }
}

std::string shogi66Action::toConsoleString() const
{
    /*
    Setup: "S " + piece + "$" + to
    Move: from + to + (promote ? "+": "")
    Drop: piece + "*" + to
    */
    if (type_ == ActionType::kSetup) {
        return "S " + pieceToString(piece_type_) + "$" + squareToString(to_);
    } else if (type_ == ActionType::kDrop) {
        return pieceToString(piece_type_) + "*" + squareToString(to_);
    } else { // Move
        return squareToString(from_) + squareToString(to_) + (promote_ ? "+" : "");
    }
}

void shogi66Env::reset()
{
    turn_ = Player::kPlayer1; // random?
    phase_ = Phase::kSetup;
    winner_ = Player::kPlayerNone;
    board_.assign(kshogi66BoardArea, Piece());

    actions_.clear();
    observations_.clear();

    for (int i = 0; i < kshogi66BoardSize; i++) {
        board_[1 * kshogi66BoardSize + i] = Piece(Player::kPlayer1, PieceType::kPawn);
        board_[4 * kshogi66BoardSize + i] = Piece(Player::kPlayer2, PieceType::kPawn);
    }
}

bool shogi66Env::act(const shogi66Action& action)
{
    if (!isLegalAction(action)) return false;

    actions_.push_back(action);
    return true;
}

bool shogi66Env::act(const std::vector<std::string>& args)
{
    return act(shogi66Action(args));
}

std::vector<shogi66Action> shogi66Env::getLegalActions() const
{
    return {};
}

bool shogi66Env::isLegalAction(const shogi66Action& action) const
{
    return false;
}

bool shogi66Env::isTerminal() const
{
    return false;
}

float shogi66Env::getEvalScore(bool is_resign) const
{
    return 0.0f;
}

std::vector<float> shogi66Env::getFeatures(utils::Rotation rotation) const
{
    return {};
}

std::vector<float> shogi66Env::getActionFeatures(const shogi66Action& action, utils::Rotation rotation) const
{
    return {};
}

std::string shogi66Env::toString() const
{
    return "";
}

} // namespace minizero::env::shogi66
