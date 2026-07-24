#include "blokus15.h"
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace minizero::env;
using namespace minizero::env::blokus15;

namespace {

void require(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}

int findAction(const Blokus15Env& env, int piece, int anchor_row, int anchor_col)
{
    for (const Blokus15Action& action : env.getLegalActions()) {
        if (action.getActionID() == kBlokus15PassAction) { continue; }
        const int orientation = action.getActionID() / kBlokus15BoardArea;
        const int anchor = action.getActionID() % kBlokus15BoardArea;
        if (getBlokus15Orientations()[orientation].piece == piece &&
            anchor / kBlokus15BoardSize == anchor_row && anchor % kBlokus15BoardSize == anchor_col) {
            return action.getActionID();
        }
    }
    return -1;
}

std::vector<int> getActionIDs(const std::vector<Blokus15Action>& actions)
{
    std::vector<int> action_ids;
    action_ids.reserve(actions.size());
    for (const Blokus15Action& action : actions) { action_ids.push_back(action.getActionID()); }
    return action_ids;
}

std::vector<int> getExhaustiveLegalActionIDs(const Blokus15Env& env)
{
    std::vector<int> action_ids;
    if (env.isTerminal()) { return action_ids; }
    for (int action_id = 0; action_id < kBlokus15PassAction; ++action_id) {
        if (env.isLegalAction(Blokus15Action(action_id, env.getTurn()))) { action_ids.push_back(action_id); }
    }
    if (action_ids.empty()) { action_ids.push_back(kBlokus15PassAction); }
    return action_ids;
}

void testOptimizedGeneratorMatchesExhaustiveGenerator()
{
    std::mt19937 random(0xB10C05u);
    for (int game = 0; game < 8; ++game) {
        Blokus15Env env;
        while (!env.isTerminal()) {
            const std::vector<Blokus15Action> legal_actions = env.getLegalActions();
            const std::vector<int> exhaustive_actions = getExhaustiveLegalActionIDs(env);
            require(getActionIDs(legal_actions) == exhaustive_actions,
                    "optimized Blokus15 legal actions differ from exhaustive generation");
            const Blokus15Action& action = legal_actions[random() % legal_actions.size()];
            require(env.act(action), "generated Blokus15 action is not legal");
        }
        require(env.getLegalActions().empty() && getExhaustiveLegalActionIDs(env).empty(),
                "terminal Blokus15 environment has legal actions");
    }
}

} // namespace

int main()
{
    require(getBlokus15Orientations().size() == kBlokus15NumOrientations,
            "Blokus15 orientation catalogue size is incorrect");
    require(kBlokus15PassAction == 12150, "Blokus15 PASS action ID is incorrect");
    require(kBlokus15PolicySize == 12375, "Blokus15 policy size is incorrect");
    require(kBlokus15NumInputChannels == 72, "Blokus15 input channel count is incorrect");
    int total_cells = 0;
    for (int piece = 0; piece < kBlokus15NumPieces; ++piece) {
        for (const auto& orientation : getBlokus15Orientations()) {
            if (orientation.piece == piece) {
                total_cells += static_cast<int>(orientation.cells.size());
                break;
            }
        }
    }
    require(total_cells == 54, "Blokus15 piece catalogue must contain 54 cells");

    Blokus15Env env;
    require(env.getFeatures().size() == kBlokus15NumInputChannels * kBlokus15BoardArea,
            "Blokus15 feature size is incorrect");
    require(!env.getLegalActions().empty(),
            "Blokus15 opening must have at least one placement covering P1's corner");
    require(!env.isLegalAction(Blokus15Action(kBlokus15PassAction, env.getTurn())),
            "PASS must be illegal while the current player has a placement");
    const int p1_monomino = findAction(env, 0, 0, 0);
    require(p1_monomino >= 0, "P1 monomino opening is missing");
    require(env.act(Blokus15Action(p1_monomino, Player::kPlayer1)), "P1 monomino opening is illegal");
    require(env.getPlayerAt(0, 0) == Player::kPlayer1, "P1 opening does not cover its corner");
    require(!env.isPieceAvailable(Player::kPlayer1, 0), "placed piece remains available");

    const int p2_monomino = findAction(env, 0, 0, 14);
    require(p2_monomino >= 0, "P2 monomino opening is missing");
    require(env.act(Blokus15Action(p2_monomino, Player::kPlayer2)), "P2 monomino opening is illegal");
    const int p3_monomino = findAction(env, 0, 14, 14);
    require(p3_monomino >= 0, "P3 monomino opening is missing");
    require(env.act(Blokus15Action(p3_monomino, Player::kPlayer3)), "P3 monomino opening is illegal");
    const int p4_monomino = findAction(env, 0, 14, 0);
    require(p4_monomino >= 0, "P4 monomino opening is missing");
    require(env.act(Blokus15Action(p4_monomino, Player::kPlayer4)), "P4 monomino opening is illegal");

    // P1 must touch its own piece diagonally, never edge-to-edge.
    int horizontal_domino_orientation = -1;
    for (int orientation = 0; orientation < static_cast<int>(getBlokus15Orientations().size()); ++orientation) {
        const Blokus15Orientation& shape = getBlokus15Orientations()[orientation];
        if (shape.piece == 1 && shape.height == 1 && shape.width == 2) {
            horizontal_domino_orientation = orientation;
            break;
        }
    }
    require(horizontal_domino_orientation >= 0, "horizontal domino orientation is missing");
    const int edge_touching_domino = horizontal_domino_orientation * kBlokus15BoardArea + 1;
    require(!env.isLegalAction(Blokus15Action(edge_touching_domino, Player::kPlayer1)),
            "same-color edge contact must be illegal");
    const int legal_domino = findAction(env, 1, 1, 1);
    require(legal_domino >= 0, "same-color diagonal domino placement is missing");
    require(env.act(Blokus15Action(legal_domino, Player::kPlayer1)),
            "same-color diagonal contact must be legal");

    Blokus15Action pass({"b", "PASS"});
    require(pass.getActionID() == kBlokus15PassAction, "PASS action parsing failed");
    require(pass.toConsoleString() == "PASS", "PASS console formatting failed");
    const auto pass_features = env.getActionFeatures(pass);
    require(std::accumulate(pass_features.begin(), pass_features.begin() + kBlokus15BoardArea, 0.0f) == 0.0f,
            "PASS must not mark placement cells");
    require(std::accumulate(pass_features.begin() + kBlokus15BoardArea, pass_features.end(), 0.0f) == kBlokus15BoardArea,
            "PASS action feature plane is incorrect");

    Blokus15Env initial_score;
    const PlayerValues values = initial_score.getEvalScores();
    require(std::fabs(values[0] + values[1] + values[2] + values[3]) < 1e-6f,
            "initial Blokus15 utilities are not zero-sum");
    require(initial_score.getPlayerScore(Player::kPlayer1) == -54,
            "initial Blokus15 score must be -54");

    Blokus15Env complete_game;
    int plies = 0;
    while (!complete_game.isTerminal()) {
        const std::vector<Blokus15Action> legal_actions = complete_game.getLegalActions();
        require(!legal_actions.empty(), "non-terminal Blokus15 environment has no legal action");
        require(complete_game.act(legal_actions.front()), "generated complete-game action is illegal");
        ++plies;
        require(plies <= kBlokus15NumPlayer * (kBlokus15NumPieces + 1),
                "Blokus15 game exceeded the placement and PASS limit");
    }
    require(complete_game.getLegalActions().empty(), "terminal Blokus15 environment has legal actions");
    const PlayerValues final_values = complete_game.getEvalScores();
    require(std::fabs(final_values[0] + final_values[1] + final_values[2] + final_values[3]) < 1e-5f,
            "final Blokus15 utilities are not zero-sum");

    Blokus15EnvLoader loader;
    loader.loadFromEnvironment(env);
    Blokus15EnvLoader loaded;
    require(loaded.loadFromString(loader.toString()), "Blokus15 record round-trip failed");
    require(loaded.getActionPairs().size() == env.getActionHistory().size(),
            "Blokus15 record action count changed after round-trip");
    require(loaded.getValue(0).size() == kBlokus15NumPlayer,
            "Blokus15 record does not preserve the four-player value vector");

    // Candidate generation is only an optimization.  On complete random games,
    // it must preserve the exact exhaustive action set and ordering at every ply.
    testOptimizedGeneratorMatchesExhaustiveGenerator();
    return 0;
}
