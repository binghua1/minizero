#include "blokus.h"
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

using namespace minizero::env;
using namespace minizero::env::blokus;

namespace {

int findAction(const BlokusEnv& env, int piece, int anchor_row, int anchor_col)
{
    for (const BlokusAction& action : env.getLegalActions()) {
        if (action.getActionID() == kBlokusPassAction) { continue; }
        const int orientation = action.getActionID() / kBlokusBoardArea;
        const int anchor = action.getActionID() % kBlokusBoardArea;
        if (getBlokusOrientations()[orientation].piece == piece &&
            anchor / kBlokusBoardSize == anchor_row && anchor % kBlokusBoardSize == anchor_col) {
            return action.getActionID();
        }
    }
    return -1;
}

} // namespace

int main()
{
    assert(getBlokusOrientations().size() == kBlokusNumOrientations);
    int total_cells = 0;
    for (int piece = 0; piece < kBlokusNumPieces; ++piece) {
        for (const auto& orientation : getBlokusOrientations()) {
            if (orientation.piece == piece) {
                total_cells += static_cast<int>(orientation.cells.size());
                break;
            }
        }
    }
    assert(total_cells == 89);

    BlokusEnv env;
    assert(env.getFeatures().size() == kBlokusNumInputChannels * kBlokusBoardArea);
    assert(env.getLegalActions().size() == 58); // every distinct placement covering P1's corner
    const int p1_monomino = findAction(env, 0, 0, 0);
    assert(p1_monomino >= 0);
    assert(env.act(BlokusAction(p1_monomino, Player::kPlayer1)));
    assert(env.getPlayerAt(0, 0) == Player::kPlayer1);
    assert(!env.isPieceAvailable(Player::kPlayer1, 0));

    const int p2_monomino = findAction(env, 0, 0, 19);
    assert(p2_monomino >= 0);
    assert(env.act(BlokusAction(p2_monomino, Player::kPlayer2)));
    const int p3_monomino = findAction(env, 0, 19, 19);
    assert(p3_monomino >= 0);
    assert(env.act(BlokusAction(p3_monomino, Player::kPlayer3)));
    const int p4_monomino = findAction(env, 0, 19, 0);
    assert(p4_monomino >= 0);
    assert(env.act(BlokusAction(p4_monomino, Player::kPlayer4)));

    // P1 must touch its own piece diagonally, never edge-to-edge.
    const int legal_domino = findAction(env, 1, 1, 1);
    assert(legal_domino >= 0);
    assert(env.act(BlokusAction(legal_domino, Player::kPlayer1)));

    BlokusAction pass({"b", "PASS"});
    assert(pass.getActionID() == kBlokusPassAction);
    assert(pass.toConsoleString() == "PASS");
    const auto pass_features = env.getActionFeatures(pass);
    assert(std::accumulate(pass_features.begin(), pass_features.begin() + kBlokusBoardArea, 0.0f) == 0.0f);
    assert(std::accumulate(pass_features.begin() + kBlokusBoardArea, pass_features.end(), 0.0f) == kBlokusBoardArea);

    BlokusEnv initial_score;
    const PlayerValues values = initial_score.getEvalScores();
    assert(std::fabs(values[0] + values[1] + values[2] + values[3]) < 1e-6f);
    assert(initial_score.getPlayerScore(Player::kPlayer1) == -89);

    BlokusEnv complete_game;
    int plies = 0;
    while (!complete_game.isTerminal()) {
        const std::vector<BlokusAction> legal_actions = complete_game.getLegalActions();
        assert(!legal_actions.empty());
        assert(complete_game.act(legal_actions.front()));
        assert(++plies <= kBlokusNumPlayer * (kBlokusNumPieces + 1));
    }
    assert(complete_game.getLegalActions().empty());
    const PlayerValues final_values = complete_game.getEvalScores();
    assert(std::fabs(final_values[0] + final_values[1] + final_values[2] + final_values[3]) < 1e-5f);

    BlokusEnvLoader loader;
    loader.loadFromEnvironment(env);
    BlokusEnvLoader loaded;
    assert(loaded.loadFromString(loader.toString()));
    assert(loaded.getActionPairs().size() == env.getActionHistory().size());
    assert(loaded.getValue(0).size() == kBlokusNumPlayer);
    return 0;
}
