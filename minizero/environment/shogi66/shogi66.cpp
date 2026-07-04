#include "shogi66.h"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <string>
#include <utility>

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

struct EncodedMove {
    int from;
    int to;
    bool promote;
};

bool isPotentialMove(int from, int to)
{
    if (!inBoard(from) || !inBoard(to) || from == to) return false;
    int dr = row(to) - row(from);
    int dc = col(to) - col(from);
    int adr = std::abs(dr);
    int adc = std::abs(dc);
    return dr == 0 || dc == 0 || adr == adc || (adr == 2 && adc == 1);
}

bool isPotentialPromotion(int from, int to)
{
    auto in_promotion_zone = [](int pos) {
        int r = row(pos);
        return r <= 1 || r >= kshogi66BoardSize - 2;
    };
    return in_promotion_zone(from) || in_promotion_zone(to);
}

int moveKey(int from, int to, bool promote)
{
    return ((from * kshogi66BoardArea + to) << 1) + static_cast<int>(promote);
}

int toRelativePosition(int pos, Player player)
{
    if (!inBoard(pos)) return -1;
    if (player == Player::kPlayer2) { return (kshogi66BoardSize - 1 - row(pos)) * kshogi66BoardSize + col(pos); }
    return pos;
}

int fromRelativePosition(int pos, Player player)
{
    return toRelativePosition(pos, player);
}

int dropKey(PieceType piece_type, int relative_to)
{
    return shogi66Action::handPieceToActionId(piece_type) * kshogi66BoardArea + relative_to;
}

struct EncodedDrop {
    PieceType piece_type;
    int relative_to;
};

bool isPotentialDrop(PieceType piece_type, int relative_to)
{
    if (!inBoard(relative_to)) return false;
    int r = row(relative_to);
    if ((piece_type == PieceType::kPawn || piece_type == PieceType::kLance) && r == kshogi66BoardSize - 1) return false;
    if (piece_type == PieceType::kKnight && r >= kshogi66BoardSize - 2) return false;
    return true;
}

const std::vector<EncodedMove>& getEncodedMoves()
{
    static const std::vector<EncodedMove> moves = [] {
        std::vector<EncodedMove> ret;
        ret.reserve(kshogi66MoveActionSize);
        for (int from = 0; from < kshogi66BoardArea; ++from) {
            for (int to = 0; to < kshogi66BoardArea; ++to) {
                if (!isPotentialMove(from, to)) continue;
                ret.push_back({from, to, false});
                if (isPotentialPromotion(from, to)) { ret.push_back({from, to, true}); }
            }
        }
        assert(static_cast<int>(ret.size()) == kshogi66MoveActionSize);
        return ret;
    }();
    return moves;
}

const std::unordered_map<int, int>& getEncodedMoveIndex()
{
    static const std::unordered_map<int, int> index = [] {
        std::unordered_map<int, int> ret;
        const auto& moves = getEncodedMoves();
        ret.reserve(moves.size());
        for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
            ret[moveKey(moves[i].from, moves[i].to, moves[i].promote)] = i;
        }
        return ret;
    }();
    return index;
}

const std::vector<EncodedDrop>& getEncodedDrops()
{
    static const std::vector<EncodedDrop> drops = [] {
        std::vector<EncodedDrop> ret;
        ret.reserve(kshogi66DropActionSize);
        for (int piece = 0; piece < kshogi66HandPieceTypes; ++piece) {
            PieceType type = shogi66Action::handActionIdToPieceType(piece);
            for (int relative_to = 0; relative_to < kshogi66BoardArea; ++relative_to) {
                if (isPotentialDrop(type, relative_to)) { ret.push_back({type, relative_to}); }
            }
        }
        assert(static_cast<int>(ret.size()) == kshogi66DropActionSize);
        return ret;
    }();
    return drops;
}

const std::unordered_map<int, int>& getEncodedDropIndex()
{
    static const std::unordered_map<int, int> index = [] {
        std::unordered_map<int, int> ret;
        const auto& drops = getEncodedDrops();
        ret.reserve(drops.size());
        for (int i = 0; i < static_cast<int>(drops.size()); ++i) {
            ret[dropKey(drops[i].piece_type, drops[i].relative_to)] = i;
        }
        return ret;
    }();
    return index;
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
    : BaseBoardAction<kshogi66NumPlayer>(encodeActionId(type, piece_type, from, to, promote, player), player), // id, player
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
    action_id_ = encodeActionId(type_, piece_type_, from_, to_, promote_, player_);
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

int shogi66Action::encodeActionId(ActionType type, PieceType piece_type, int from, int to, bool promote, Player player)
{
    if (type == ActionType::kSetup) { // 1
        int id = setupPieceToActionId(piece_type);
        int relative_to = toRelativePosition(to, player);
        return id < 0 || !inBoard(relative_to) || row(relative_to) != 0 ? -1 : id * kshogi66BoardSize + col(relative_to);
    } else if (type == ActionType::kMove) {
        auto it = getEncodedMoveIndex().find(moveKey(from, to, promote));
        return it == getEncodedMoveIndex().end() ? -1 : kshogi66SetupActionSize + it->second;
    } else if (type == ActionType::kDrop) {
        int relative_to = toRelativePosition(to, player);
        auto it = getEncodedDropIndex().find(dropKey(piece_type, relative_to));
        return it == getEncodedDropIndex().end() ? -1 : kshogi66SetupActionSize + kshogi66MoveActionSize + it->second;
    }
    return -1;
}

void shogi66Action::decodeActionId()
{
    int id = action_id_;
    if (0 <= id && id < kshogi66SetupActionSize) {
        type_ = ActionType::kSetup;
        piece_type_ = setupActionIdToPieceType(id / kshogi66BoardSize);
        from_ = -1;
        to_ = fromRelativePosition(id % kshogi66BoardSize, player_);
        promote_ = false;
    } else if (kshogi66SetupActionSize <= id && id < kshogi66SetupActionSize + kshogi66MoveActionSize) {
        id -= kshogi66SetupActionSize;
        const EncodedMove& move = getEncodedMoves()[id];
        type_ = ActionType::kMove;
        promote_ = move.promote;
        from_ = move.from;
        to_ = move.to;
        piece_type_ = PieceType::kEmpty;
    } else if (kshogi66SetupActionSize + kshogi66MoveActionSize <= id && id < kshogi66PolicySize) {
        id -= kshogi66SetupActionSize + kshogi66MoveActionSize;
        const EncodedDrop& drop = getEncodedDrops()[id];
        type_ = ActionType::kDrop;
        piece_type_ = drop.piece_type;
        from_ = -1;
        to_ = fromRelativePosition(drop.relative_to, player_);
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
        return "S:" + pieceToString(piece_type_) + "$" + squareToString(to_);
    } else if (type_ == ActionType::kDrop) {
        return pieceToString(piece_type_) + "*" + squareToString(to_);
    } else { // Move
        return squareToString(from_) + squareToString(to_) + (promote_ ? "+" : "");
    }
}

void shogi66Env::reset()
{
    turn_ = Player::kPlayer1;
    phase_ = Phase::kSetup;
    winner_ = Player::kPlayerNone;
    board_.assign(kshogi66BoardArea, Piece());
    repetition_draw_ = false;
    state_count_.clear();
    hand_.reset();
    setup_pool_.reset();

    actions_.clear();
    observations_.clear();

    for (auto p : {Player::kPlayer1, Player::kPlayer2}) {
        hand_.get(p).fill(0);
        setup_pool_.get(p).fill(1);
    }
    for (int i = 0; i < kshogi66BoardSize; i++) {
        board_[1 * kshogi66BoardSize + i] = Piece(Player::kPlayer1, PieceType::kPawn);
        board_[4 * kshogi66BoardSize + i] = Piece(Player::kPlayer2, PieceType::kPawn);
    }
}

bool shogi66Env::act(const shogi66Action& action)
{
    if (!isLegalAction(action)) return false;

    actions_.push_back(action);
    applyActionNoCheck(action);
    if (winner_ == Player::kPlayerNone) winner_ = winnerByMissingKing();
    if (phase_ == Phase::kSetup && isSetupFinished()) phase_ = Phase::kPlay;
    turn_ = action.nextPlayer();
    if (phase_ == Phase::kPlay) {
        int count = ++state_count_[stateKey()];
        if (count >= 4) repetition_draw_ = true;
    }
    return true;
}

bool shogi66Env::act(const std::vector<std::string>& args)
{
    return act(shogi66Action(args));
}

std::vector<shogi66Action> shogi66Env::getLegalActions() const
{
    return getLegalActions(true);
}

std::vector<shogi66Action> shogi66Env::getLegalActions(bool check_pawn_drop_mate) const
{
    static thread_local std::unordered_map<std::string, std::vector<shogi66Action>> legal_action_cache;
    std::string cache_key = legalActionCacheKey(check_pawn_drop_mate);
    auto cache_it = legal_action_cache.find(cache_key);
    if (cache_it != legal_action_cache.end()) { return cache_it->second; }

    std::vector<shogi66Action> actions;
    if (winner_ != Player::kPlayerNone) {
        legal_action_cache.emplace(std::move(cache_key), actions);
        return actions;
    }
    if (phase_ == Phase::kSetup) {
        for (int action_id = 0; action_id < kshogi66SetupActionSize; ++action_id) {
            shogi66Action action(action_id, turn_);
            if (isLegalAction(action, check_pawn_drop_mate)) actions.push_back(action);
        }
        auto ret = legal_action_cache.emplace(std::move(cache_key), std::move(actions));
        return ret.first->second;
    }
    for (int move_id = 0; move_id < kshogi66MoveActionSize; ++move_id) {
        shogi66Action action(kshogi66SetupActionSize + move_id, turn_);
        if (board_[action.getFrom()].owner != turn_) continue;
        if (isLegalAction(action, check_pawn_drop_mate)) actions.push_back(action);
    }
    for (int drop_id = 0; drop_id < kshogi66DropActionSize; ++drop_id) {
        shogi66Action action(kshogi66SetupActionSize + kshogi66MoveActionSize + drop_id, turn_);
        if (isLegalAction(action, check_pawn_drop_mate)) actions.push_back(action);
    }
    auto ret = legal_action_cache.emplace(std::move(cache_key), std::move(actions));
    return ret.first->second;
}

bool shogi66Env::isLegalAction(const shogi66Action& action) const
{
    return isLegalAction(action, true);
}

bool shogi66Env::isLegalAction(const shogi66Action& action, bool check_pawn_drop_mate) const
{
    if (action.getPlayer() != turn_ || action.getActionID() < 0 || !inBoard(action.getTo())) return false;
    if (phase_ == Phase::kSetup) return action.getType() == ActionType::kSetup && isLegalSetupAction(action);
    if (action.getType() == ActionType::kSetup) return false;
    if (action.getType() == ActionType::kMove && !isLegalMove(action)) return false;
    if (action.getType() == ActionType::kDrop && !isLegalDrop(action)) return false;

    bool legal = !isKingInCheckAfterAction(action, turn_);
    bool gives_check = legal && isKingInCheckAfterAction(action, action.nextPlayer());
    if (legal && gives_check) {
        shogi66Env nxt = *this;
        nxt.applyActionNoCheck(action);
        nxt.turn_ = action.nextPlayer();
        auto it = state_count_.find(nxt.stateKey());
        if (it != state_count_.end() && it->second >= 3) legal = false;
    }
    if (legal && check_pawn_drop_mate && gives_check &&
        action.getType() == ActionType::kDrop && action.getPieceType() == PieceType::kPawn && isPawnDropMate(action)) {
        legal = false;
    }

#ifdef SHOGI66_VERIFY_FAST_LEGAL
    assert(legal == isLegalActionSlow(action, check_pawn_drop_mate));
#endif
    return legal;
}

bool shogi66Env::isLegalActionSlow(const shogi66Action& action, bool check_pawn_drop_mate) const
{
    if (action.getPlayer() != turn_ || action.getActionID() < 0 || !inBoard(action.getTo())) return false;
    if (phase_ == Phase::kSetup) return action.getType() == ActionType::kSetup && isLegalSetupAction(action);
    if (action.getType() == ActionType::kSetup) return false;
    if (action.getType() == ActionType::kMove && !isLegalMove(action)) return false;
    if (action.getType() == ActionType::kDrop && !isLegalDrop(action)) return false;
    shogi66Env nxt = *this;
    nxt.applyActionNoCheck(action);
    if (nxt.isKingInCheck(turn_)) return false;
    Player next_turn = action.nextPlayer();
    nxt.turn_ = next_turn;
    if (nxt.isKingInCheck(next_turn)) {
        auto it = state_count_.find(nxt.stateKey());
        if (it != state_count_.end() && it->second >= 3) return false;
    }
    if (check_pawn_drop_mate && action.getType() == ActionType::kDrop && action.getPieceType() == PieceType::kPawn && isPawnDropMate(action)) return false;
    return true;
}

bool shogi66Env::isLegalSetupAction(const shogi66Action& action) const
{
    int id = shogi66Action::setupPieceToActionId(action.getPieceType());
    if (id < 0 || setup_pool_.get(turn_)[id] <= 0) return false;
    if (board_[action.getTo()].owner != Player::kPlayerNone) return false;
    int r = row(action.getTo());
    if (turn_ == Player::kPlayer1 && !(r == 0)) return false;
    if (turn_ == Player::kPlayer2 && !(r == kshogi66BoardSize - 1)) return false;
    // Rook, Bishop choose one
    if (action.getPieceType() == PieceType::kRook && setup_pool_.get(turn_)[shogi66Action::setupPieceToActionId(PieceType::kBishop)] == 0) return false;
    if (action.getPieceType() == PieceType::kBishop && setup_pool_.get(turn_)[shogi66Action::setupPieceToActionId(PieceType::kRook)] == 0) return false;
    return true;
}

bool shogi66Env::isLegalMove(const shogi66Action& action) const
{
    int from = action.getFrom(), to = action.getTo();
    if (!inBoard(from) || !inBoard(to) || from == to) return false;
    if (board_[from].owner != turn_ || board_[to].owner == turn_) return false;
    if (board_[to].type == PieceType::kKing) return false;
    PieceType type = board_[from].type;
    if (action.isPromote()) {
        if (!isPromotable(type) || isPromoted(type) || !canPromote(turn_, from, to, type)) {
            return false;
        }
    } else if (mustPromote(turn_, to, type)) {
        return false;
    }
    return attacksSquare(from, to);
}

bool shogi66Env::isLegalDrop(const shogi66Action& action) const
{
    if (isPromoted(action.getPieceType())) return false;
    int id = shogi66Action::handPieceToActionId(action.getPieceType());
    int to = action.getTo();
    if (id < 0 || hand_.get(turn_)[id] <= 0) return false;
    if (board_[to].owner != Player::kPlayerNone) return false;
    int r = row(to);
    if ((action.getPieceType() == PieceType::kPawn || action.getPieceType() == PieceType::kLance) &&
        r == (turn_ == Player::kPlayer1 ? kshogi66BoardSize - 1 : 0)) return false;
    if (action.getPieceType() == PieceType::kKnight) {
        if (turn_ == Player::kPlayer1 && r >= kshogi66BoardSize - 2) return false;
        if (turn_ == Player::kPlayer2 && r <= 1) return false;
    }
    if (action.getPieceType() == PieceType::kPawn && hasUnpromotedPawnOnFile(turn_, col(to))) return false;
    return true;
}

bool shogi66Env::attacksSquare(int from, int to) const
{
    const Piece& piece = board_[from];
    int dr = row(to) - row(from);
    int dc = col(to) - col(from);
    int forward = dir(piece.owner);

    auto check_line = [&]() {
        int sr = dr == 0 ? 0 : (dr > 0 ? 1 : -1);
        int sc = dc == 0 ? 0 : (dc > 0 ? 1 : -1);
        int r = row(from) + sr;
        int c = col(from) + sc;
        while (r != row(to) || c != col(to)) {
            if (board_[r * kshogi66BoardSize + c].owner != Player::kPlayerNone) return false;
            r += sr, c += sc;
        }
        return true;
    };

    switch (piece.type) {
        case PieceType::kKing:
            return std::max(std::abs(dr), std::abs(dc)) == 1;
        case PieceType::kGold:
        case PieceType::kPromSilver:
        case PieceType::kPromKnight:
        case PieceType::kPromLance:
        case PieceType::kTokin:
            return (dr == forward && std::abs(dc) <= 1) ||
                   (dr == 0 && std::abs(dc) == 1) ||
                   (dr == -forward && dc == 0);
        case PieceType::kSilver:
            return (dr == forward && std::abs(dc) <= 1) ||
                   (dr == -forward && std::abs(dc) == 1);
        case PieceType::kKnight:
            return dr == 2 * forward && std::abs(dc) == 1;
        case PieceType::kLance:
            return dc == 0 && dr * forward > 0 && check_line();
        case PieceType::kRook:
            return (dr == 0 || dc == 0) && check_line();
        case PieceType::kBishop:
            return std::abs(dr) == std::abs(dc) && check_line();
        case PieceType::kDragon:
            return (dr == 0 || dc == 0) && check_line() || (std::abs(dr) == std::abs(dc) && std::abs(dr) == 1);
        case PieceType::kHorse:
            return (std::abs(dr) == std::abs(dc) && check_line()) ||
                   ((std::abs(dr) == 1 && dc == 0) || (dr == 0 && std::abs(dc) == 1));
        case PieceType::kPawn:
            return dr == forward && dc == 0;
        default:
            return false;
    }
}

Piece shogi66Env::getPieceAfterAction(const shogi66Action& action, int pos) const
{
    if (!inBoard(pos)) return Piece();

    if (action.getType() == ActionType::kSetup) {
        return pos == action.getTo() ? Piece(action.getPlayer(), action.getPieceType()) : board_[pos];
    }
    if (action.getType() == ActionType::kDrop) {
        return pos == action.getTo() ? Piece(action.getPlayer(), action.getPieceType()) : board_[pos];
    }
    if (action.getType() != ActionType::kMove) return board_[pos];
    if (pos == action.getFrom()) return Piece();
    if (pos != action.getTo()) return board_[pos];

    Piece piece = board_[action.getFrom()];
    if (action.isPromote()) piece.type = promoted(piece.type);
    return piece;
}

bool shogi66Env::attacksSquareAfterAction(const shogi66Action& action, int from, int to) const
{
    Piece piece = getPieceAfterAction(action, from);
    if (!isPlayer(piece.owner) || !inBoard(to) || from == to) return false;

    int dr = row(to) - row(from);
    int dc = col(to) - col(from);
    int forward = dir(piece.owner);

    auto check_line = [&]() {
        int sr = dr == 0 ? 0 : (dr > 0 ? 1 : -1);
        int sc = dc == 0 ? 0 : (dc > 0 ? 1 : -1);
        int r = row(from) + sr;
        int c = col(from) + sc;
        while (r != row(to) || c != col(to)) {
            if (getPieceAfterAction(action, r * kshogi66BoardSize + c).owner != Player::kPlayerNone) return false;
            r += sr, c += sc;
        }
        return true;
    };

    switch (piece.type) {
        case PieceType::kKing:
            return std::max(std::abs(dr), std::abs(dc)) == 1;
        case PieceType::kGold:
        case PieceType::kPromSilver:
        case PieceType::kPromKnight:
        case PieceType::kPromLance:
        case PieceType::kTokin:
            return (dr == forward && std::abs(dc) <= 1) ||
                   (dr == 0 && std::abs(dc) == 1) ||
                   (dr == -forward && dc == 0);
        case PieceType::kSilver:
            return (dr == forward && std::abs(dc) <= 1) ||
                   (dr == -forward && std::abs(dc) == 1);
        case PieceType::kKnight:
            return dr == 2 * forward && std::abs(dc) == 1;
        case PieceType::kLance:
            return dc == 0 && dr * forward > 0 && check_line();
        case PieceType::kRook:
            return (dr == 0 || dc == 0) && check_line();
        case PieceType::kBishop:
            return std::abs(dr) == std::abs(dc) && check_line();
        case PieceType::kDragon:
            return ((dr == 0 || dc == 0) && check_line()) ||
                   (std::abs(dr) == std::abs(dc) && std::abs(dr) == 1);
        case PieceType::kHorse:
            return (std::abs(dr) == std::abs(dc) && check_line()) ||
                   ((std::abs(dr) == 1 && dc == 0) || (dr == 0 && std::abs(dc) == 1));
        case PieceType::kPawn:
            return dr == forward && dc == 0;
        default:
            return false;
    }
}

bool shogi66Env::isKingInCheck(Player player) const
{
    int king = -1;
    for (int pos = 0; pos < kshogi66BoardArea; pos++) {
        if (board_[pos].owner == player && board_[pos].type == PieceType::kKing) {
            king = pos;
            break;
        }
    }
    if (king < 0) return true;
    Player opponent = getNextPlayer(player, kshogi66NumPlayer);
    for (int pos = 0; pos < kshogi66BoardArea; pos++) {
        if (board_[pos].owner == opponent && attacksSquare(pos, king)) return true;
    }
    return false;
}

bool shogi66Env::isKingInCheckAfterAction(const shogi66Action& action, Player player) const
{
    int king = -1;
    for (int pos = 0; pos < kshogi66BoardArea; pos++) {
        Piece piece = getPieceAfterAction(action, pos);
        if (piece.owner == player && piece.type == PieceType::kKing) {
            king = pos;
            break;
        }
    }
    if (king < 0) return true;

    Player opponent = getNextPlayer(player, kshogi66NumPlayer);
    for (int pos = 0; pos < kshogi66BoardArea; pos++) {
        if (getPieceAfterAction(action, pos).owner == opponent && attacksSquareAfterAction(action, pos, king)) return true;
    }
    return false;
}

bool shogi66Env::isPawnDropMate(const shogi66Action& action) const
{
    shogi66Env nxt = *this;
    nxt.applyActionNoCheck(action);
    nxt.turn_ = getNextPlayer(nxt.turn_, kshogi66NumPlayer);
    if (!nxt.isKingInCheck(nxt.turn_)) return false;
    return nxt.getLegalActions(false).empty();
}

bool shogi66Env::canPromote(Player player, int from, int to, PieceType type) const
{
    if (!isPromotable(type) || isPromoted(type)) return false;
    auto in_pos = [&](int pos) {
        int r = row(pos);
        return player == Player::kPlayer1 ? r >= kshogi66BoardSize - 2 : r <= 1;
    };
    return in_pos(from) || in_pos(to);
}

bool shogi66Env::mustPromote(Player player, int to, PieceType type) const
{
    int r = row(to);
    if ((type == PieceType::kPawn || type == PieceType::kLance) && r == (player == Player::kPlayer1 ? kshogi66BoardSize - 1 : 0))
        return true;
    if (type == PieceType::kKnight)
        return player == Player::kPlayer1 ? r >= kshogi66BoardSize - 2 : r <= 1;
    return false;
}

bool shogi66Env::hasUnpromotedPawnOnFile(Player player, int file) const
{
    for (int i = 0; i < kshogi66BoardSize; i++) {
        const Piece& piece = board_[i * kshogi66BoardSize + file];
        if (piece.owner == player && piece.type == PieceType::kPawn) return true;
    }
    return false;
}

void shogi66Env::applyActionNoCheck(const shogi66Action& action)
{
    Player player = action.getPlayer();
    ActionType ac_type = action.getType();
    PieceType pc_type = action.getPieceType();
    if (ac_type == ActionType::kSetup) {
        board_[action.getTo()] = Piece(player, pc_type);
        setup_pool_.get(player)[shogi66Action::setupPieceToActionId(pc_type)]--;
        if (pc_type == PieceType::kRook) setup_pool_.get(player)[shogi66Action::setupPieceToActionId(PieceType::kBishop)] = 0;
        if (pc_type == PieceType::kBishop) setup_pool_.get(player)[shogi66Action::setupPieceToActionId(PieceType::kRook)] = 0;
    } else if (ac_type == ActionType::kMove) {
        Piece move = board_[action.getFrom()];
        Piece capture = board_[action.getTo()];
        if (capture.owner != Player::kPlayerNone) {
            int id = shogi66Action::handPieceToActionId(unpromote(capture.type));
            hand_.get(player)[id]++;
        }
        if (action.isPromote()) move.type = promoted(move.type);
        board_[action.getTo()] = move;
        board_[action.getFrom()] = Piece();
    } else { // Drop
        board_[action.getTo()] = Piece(player, pc_type);
        hand_.get(player)[shogi66Action::handPieceToActionId(pc_type)]--;
    }
}

bool shogi66Env::isTerminal() const
{
    if (winner_ != Player::kPlayerNone) return true;
    if (repetition_draw_) return true;
    if (phase_ == Phase::kPlay && getLegalActions().empty()) return true;
    return false;
}

float shogi66Env::getEvalScore(bool is_resign) const
{
    Player result = Player::kPlayerNone;
    if (is_resign) {
        result = getNextPlayer(turn_, kshogi66NumPlayer);
    } else if (winner_ != Player::kPlayerNone) {
        result = winner_;
    } else if (phase_ == Phase::kPlay && getLegalActions().empty()) {
        result = getNextPlayer(turn_, kshogi66NumPlayer);
    }
    return result == Player::kPlayer1 ? 1.0f : (result == Player::kPlayer2 ? -1.0f : 0.0f);
}

std::vector<float> shogi66Env::getFeatures(utils::Rotation rotation) const
{
    const int spatial = kshogi66BoardArea;
    std::vector<float> features(getNumInputChannels() * spatial, 0.0f);
    auto piece = [&](PieceType type) {
        switch (type) {
            case PieceType::kKing: return 0;
            case PieceType::kGold: return 1;
            case PieceType::kSilver: return 2;
            case PieceType::kPromSilver: return 3;
            case PieceType::kKnight: return 4;
            case PieceType::kPromKnight: return 5;
            case PieceType::kLance: return 6;
            case PieceType::kPromLance: return 7;
            case PieceType::kRook: return 8;
            case PieceType::kDragon: return 9;
            case PieceType::kBishop: return 10;
            case PieceType::kHorse: return 11;
            case PieceType::kPawn: return 12;
            case PieceType::kTokin: return 13;
            default: return -1;
        }
    };

    Player nxt_player = getNextPlayer(turn_, kshogi66NumPlayer);
    for (int pos = 0; pos < spatial; pos++) {
        const Piece& p = board_[pos];
        int idx = piece(p.type);
        if (idx < 0) continue;
        if (p.owner == turn_)
            features[idx * spatial + pos] = 1.0f;
        else if (p.owner == nxt_player)
            features[(14 + idx) * spatial + pos] = 1.0f;
    }
    for (int i = 0; i < kshogi66HandPieceTypes; i++) {
        for (int pos = 0; pos < spatial; pos++) {
            features[(28 + i) * spatial + pos] = static_cast<float>(hand_.get(turn_)[i]);
            features[(35 + i) * spatial + pos] = static_cast<float>(hand_.get(nxt_player)[i]);
        }
    }
    for (int pos = 0; pos < spatial; pos++) {
        features[42 * spatial + pos] = turn_ == Player::kPlayer1 ? 1.0f : 0.0f;
        features[43 * spatial + pos] = phase_ == Phase::kSetup ? 1.0f : 0.0f;
    }
    return features;
}

std::vector<float> shogi66Env::getActionFeatures(const shogi66Action& action, utils::Rotation rotation) const
{
    std::vector<float> action_features(kshogi66PolicySize, 0.0f);
    int id = getRotateAction(action.getActionID(), rotation);
    if (0 <= id && id < kshogi66PolicySize) action_features[id] = 1.0f;
    return action_features;
}

std::string shogi66Env::toString() const
{
    std::ostringstream oss;
    oss << (phase_ == Phase::kSetup ? "setup" : "play") << playerToChar(turn_) << "\n";
    oss << "    A   B   C   D   E   F\n";
    for (int r = kshogi66BoardSize - 1; r >= 0; r--) {
        oss << r + 1 << " ";
        for (int c = 0; c < kshogi66BoardSize; c++) {
            const Piece& piece = board_[r * kshogi66BoardSize + c];
            auto s = pieceToString(piece.type);
            if (piece.owner == Player::kPlayer2) {
                std::transform(begin(s), end(s), begin(s), [&](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
            }
            if (s.size() == 1) s = " " + s;
            oss << " " << s << " ";
        }
        oss << " " << r + 1 << "\n";
    }
    oss << "    A   B   C   D   E   F\n";
    for (Player p : {Player::kPlayer1, Player::kPlayer2}) {
        oss << (p == Player::kPlayer1 ? "P1" : "P2") << " hand: ";
        for (int piece = 0; piece < kshogi66HandPieceTypes; piece++) {
            PieceType type = shogi66Action::handActionIdToPieceType(piece);
            int count = hand_.get(p)[piece];
            if (count > 0) oss << pieceToString(type) << "x" << count << " ";
        }
        oss << "\n";
    }
    return oss.str();
}

std::vector<float> shogi66EnvLoader::getActionFeatures(const int pos, utils::Rotation rotation) const
{
    std::vector<float> action_features(kshogi66PolicySize, 0.0f);
    if (pos < static_cast<int>(action_pairs_.size())) {
        int id = getRotateAction(action_pairs_[pos].first.getActionID(), rotation);
        if (0 <= id && id < kshogi66PolicySize) action_features[id] = 1.0f;
    }
    return action_features;
}

Player shogi66Env::winnerByMissingKing() const
{
    if (phase_ == Phase::kSetup) return Player::kPlayerNone;

    bool p1_king = false, p2_king = false;
    for (const Piece& piece : board_) {
        if (piece.type == PieceType::kKing && piece.owner == Player::kPlayer1) p1_king = true;
        if (piece.type == PieceType::kKing && piece.owner == Player::kPlayer2) p2_king = true;
    }
    if (!p1_king) return Player::kPlayer2;
    if (!p2_king) return Player::kPlayer1;
    return Player::kPlayerNone;
}

bool shogi66Env::isSetupFinished() const
{
    for (Player p : {Player::kPlayer1, Player::kPlayer2}) {
        for (int count : setup_pool_.get(p)) {
            if (count > 0) return false;
        }
    }
    return true;
}

std::string shogi66Env::stateKey() const
{
    std::ostringstream oss;
    oss << static_cast<int>(turn_) << "|";
    for (const Piece& piece : board_)
        oss << static_cast<int>(piece.owner) << ":" << static_cast<int>(piece.type) << ",";
    oss << "|";
    for (Player p : {Player::kPlayer1, Player::kPlayer2})
        for (int count : hand_.get(p))
            oss << count << "|";
    return oss.str();
}

std::string shogi66Env::legalActionCacheKey(bool check_pawn_drop_mate) const
{
    std::ostringstream oss;
    oss << static_cast<int>(check_pawn_drop_mate) << "|"
        << static_cast<int>(phase_) << "|"
        << static_cast<int>(turn_) << "|"
        << static_cast<int>(winner_) << "|"
        << static_cast<int>(repetition_draw_) << "|";

    for (const Piece& piece : board_) {
        oss << static_cast<int>(piece.owner) << ":" << static_cast<int>(piece.type) << ",";
    }
    oss << "|";

    for (Player p : {Player::kPlayer1, Player::kPlayer2}) {
        for (int count : hand_.get(p)) { oss << count << ","; }
        oss << "|";
    }

    for (Player p : {Player::kPlayer1, Player::kPlayer2}) {
        for (int count : setup_pool_.get(p)) { oss << count << ","; }
        oss << "|";
    }

    std::vector<std::string> repeated_states;
    repeated_states.reserve(state_count_.size());
    for (const auto& item : state_count_) {
        if (item.second >= 3) { repeated_states.push_back(item.first); }
    }
    std::sort(repeated_states.begin(), repeated_states.end());
    for (const std::string& key : repeated_states) { oss << key << ";"; }

    return oss.str();
}

} // namespace minizero::env::shogi66
