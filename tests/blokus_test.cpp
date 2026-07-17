#include "blokus.h"
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace minizero::env;
using namespace minizero::env::blokus;

namespace {

void require(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}

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

std::vector<int> getActionIDs(const std::vector<BlokusAction>& actions)
{
    std::vector<int> action_ids;
    action_ids.reserve(actions.size());
    for (const BlokusAction& action : actions) { action_ids.push_back(action.getActionID()); }
    return action_ids;
}

std::vector<int> getExhaustiveLegalActionIDs(const BlokusEnv& env)
{
    std::vector<int> action_ids;
    if (env.isTerminal()) { return action_ids; }
    for (int action_id = 0; action_id < kBlokusPassAction; ++action_id) {
        if (env.isLegalAction(BlokusAction(action_id, env.getTurn()))) { action_ids.push_back(action_id); }
    }
    if (action_ids.empty()) { action_ids.push_back(kBlokusPassAction); }
    return action_ids;
}

void testOptimizedGeneratorMatchesExhaustiveGenerator()
{
    std::mt19937 random(0xB10C05u);
    for (int game = 0; game < 8; ++game) {
        BlokusEnv env;
        while (!env.isTerminal()) {
            const std::vector<BlokusAction> legal_actions = env.getLegalActions();
            const std::vector<int> exhaustive_actions = getExhaustiveLegalActionIDs(env);
            require(getActionIDs(legal_actions) == exhaustive_actions,
                    "optimized Blokus legal actions differ from exhaustive generation");
            const BlokusAction& action = legal_actions[random() % legal_actions.size()];
            require(env.act(action), "generated Blokus action is not legal");
        }
        require(env.getLegalActions().empty() && getExhaustiveLegalActionIDs(env).empty(),
                "terminal Blokus environment has legal actions");
    }
}

void requireNormalizedScoreValues(const BlokusEnv& env)
{
    const PlayerValues values = env.getEvalScores();
    for (int player = 0; player < kBlokusNumPlayer; ++player) {
        const float score = static_cast<float>(env.getPlayerScore(indexToPlayer(player)));
        const float expected = 2.0f * (score + 89.0f) / 109.0f - 1.0f;
        require(std::fabs(values[player] - expected) < 1e-6f,
                "Blokus value is not the normalized official score");
    }
}

} // namespace

int main()
{
    require(getBlokusOrientations().size() == kBlokusNumOrientations,
            "Blokus orientation catalogue size is incorrect");
    int total_cells = 0;
    for (int piece = 0; piece < kBlokusNumPieces; ++piece) {
        for (const auto& orientation : getBlokusOrientations()) {
            if (orientation.piece == piece) {
                total_cells += static_cast<int>(orientation.cells.size());
                break;
            }
        }
    }
    require(total_cells == 89, "Blokus piece catalogue must contain 89 cells");

    BlokusEnv env;
    require(env.getFeatures().size() == kBlokusNumInputChannels * kBlokusBoardArea,
            "Blokus feature size is incorrect");
    require(env.getLegalActions().size() == 58,
            "Blokus opening must have 58 distinct placements covering P1's corner");
    require(!env.isLegalAction(BlokusAction(kBlokusPassAction, env.getTurn())),
            "PASS must be illegal while the current player has a placement");
    const int p1_monomino = findAction(env, 0, 0, 0);
    require(p1_monomino >= 0, "P1 monomino opening is missing");
    require(env.act(BlokusAction(p1_monomino, Player::kPlayer1)), "P1 monomino opening is illegal");
    require(env.getPlayerAt(0, 0) == Player::kPlayer1, "P1 opening does not cover its corner");
    require(!env.isPieceAvailable(Player::kPlayer1, 0), "placed piece remains available");

    const int p2_monomino = findAction(env, 0, 0, 19);
    require(p2_monomino >= 0, "P2 monomino opening is missing");
    require(env.act(BlokusAction(p2_monomino, Player::kPlayer2)), "P2 monomino opening is illegal");
    const int p3_monomino = findAction(env, 0, 19, 19);
    require(p3_monomino >= 0, "P3 monomino opening is missing");
    require(env.act(BlokusAction(p3_monomino, Player::kPlayer3)), "P3 monomino opening is illegal");
    const int p4_monomino = findAction(env, 0, 19, 0);
    require(p4_monomino >= 0, "P4 monomino opening is missing");
    require(env.act(BlokusAction(p4_monomino, Player::kPlayer4)), "P4 monomino opening is illegal");

    // P1 must touch its own piece diagonally, never edge-to-edge.
    int horizontal_domino_orientation = -1;
    for (int orientation = 0; orientation < static_cast<int>(getBlokusOrientations().size()); ++orientation) {
        const BlokusOrientation& shape = getBlokusOrientations()[orientation];
        if (shape.piece == 1 && shape.height == 1 && shape.width == 2) {
            horizontal_domino_orientation = orientation;
            break;
        }
    }
    require(horizontal_domino_orientation >= 0, "horizontal domino orientation is missing");
    const int edge_touching_domino = horizontal_domino_orientation * kBlokusBoardArea + 1;
    require(!env.isLegalAction(BlokusAction(edge_touching_domino, Player::kPlayer1)),
            "same-color edge contact must be illegal");
    const int legal_domino = findAction(env, 1, 1, 1);
    require(legal_domino >= 0, "same-color diagonal domino placement is missing");
    require(env.act(BlokusAction(legal_domino, Player::kPlayer1)),
            "same-color diagonal contact must be legal");

    BlokusAction pass({"b", "PASS"});
    require(pass.getActionID() == kBlokusPassAction, "PASS action parsing failed");
    require(pass.toConsoleString() == "PASS", "PASS console formatting failed");
    const auto pass_features = env.getActionFeatures(pass);
    require(std::accumulate(pass_features.begin(), pass_features.begin() + kBlokusBoardArea, 0.0f) == 0.0f,
            "PASS must not mark placement cells");
    require(std::accumulate(pass_features.begin() + kBlokusBoardArea, pass_features.end(), 0.0f) == kBlokusBoardArea,
            "PASS action feature plane is incorrect");

    BlokusEnv initial_score;
    require(initial_score.getPlayerScore(Player::kPlayer1) == -89,
            "initial Blokus score must be -89");
    requireNormalizedScoreValues(initial_score);

    BlokusEnv complete_game;
    int plies = 0;
    while (!complete_game.isTerminal()) {
        const std::vector<BlokusAction> legal_actions = complete_game.getLegalActions();
        require(!legal_actions.empty(), "non-terminal Blokus environment has no legal action");
        require(complete_game.act(legal_actions.front()), "generated complete-game action is illegal");
        ++plies;
        require(plies <= kBlokusNumPlayer * (kBlokusNumPieces + 1),
                "Blokus game exceeded the placement and PASS limit");
    }
    require(complete_game.getLegalActions().empty(), "terminal Blokus environment has legal actions");
    requireNormalizedScoreValues(complete_game);

    BlokusEnvLoader loader;
    loader.loadFromEnvironment(env);
    BlokusEnvLoader loaded;
    require(loaded.loadFromString(loader.toString()), "Blokus record round-trip failed");
    require(loaded.getActionPairs().size() == env.getActionHistory().size(),
            "Blokus record action count changed after round-trip");
    require(loaded.getValue(0).size() == kBlokusNumPlayer,
            "Blokus record does not preserve the four-player value vector");

    // Candidate generation is only an optimization.  On complete random games,
    // it must preserve the exact exhaustive action set and ordering at every ply.
    testOptimizedGeneratorMatchesExhaustiveGenerator();
    return 0;
}
