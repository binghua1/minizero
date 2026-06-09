#include "shogi66.h"
#include <algorithm>
#include <string>

namespace minizero::env::shogi66 {

using namespace minizero::utils;

shogi66Action::shogi66Action() : from_(-1), to_(-1), piece_type_(PieceType::kEmpty), promote_(false) {}

shogi66Action::shogi66Action(int action_id, Player player) : BaseBoardAction(action_id, player) {}

shogi66Action::shogi66Action(ActionType type, PieceType piece_type, int from, int to, bool promote, Player player) 
    : BaseBoardAction(-1, player), type_(type), piece_type_(piece_type), from_(from), to_(to), promote_(promote) {}
    
shogi66Action::shogi66Action(const std::vector<std::string>& action_string_args) : BaseBoardAction(-1, Player::kPlayerNone) {
    ;
}
    
void shogi66Env::reset(){
    turn_ = Player::kPlayer1; // random?
    phase_ = Phase::kSetup;
    winner_ = Player::kPlayerNone;
    board_.assign(kshogi66BoardArea, Piece());

    actions_.clear();
    observations_.clear();
}

bool shogi66Env::act(const shogi66Action &action) {
    if(!isLegalAction(action)) return false;
}

bool shogi66Env::act(const std::vector<std::string> &args) {
    return act(shogi66Action(args));
}

std::vector<shogi66Action> shogi66Env::getLegalActions() const {
    ;
}

bool shogi66Env::isLegalAction(const shogi66Action &action) const {
    ;
}

bool shogi66Env::isTerminal() const {
    ;
}

Player shogi66Env::eval() const {
    for(int i = 0; i < kshogi66BoardSize; ++i) {
        for(int j = 0; j < kshogi66BoardSize; ++j) {
            if(board_[i * kshogi66BoardSize + j].type == PieceType::kKing) {

            }
        }
    }
} 

float shogi66Env::getEvalScore(bool is_resign) const {
    ;
}

std::vector<float> shogi66Env::getFeatures(utils::Rotation rotation) const {
    ;
}

std::vector<float> shogi66Env::getActionFeatures(const shogi66Action &action, utils::Rotation rotation) const {
    ;
}

std::string shogi66Env::toString() const {
    ;
}


} // namespace minizero::env::shogi66
