#include "mcts.h"
#include "tictacmo.h"
#include "zero_server.h"
#include <cassert>
#include <vector>

int main()
{
    using namespace minizero;
    using namespace minizero::actor;
    using namespace minizero::env;
    using namespace minizero::env::tictacmo;

    assert(getNextPlayer(Player::kPlayer1, 3) == Player::kPlayer2);
    assert(getNextPlayer(Player::kPlayer2, 3) == Player::kPlayer3);
    assert(getNextPlayer(Player::kPlayer3, 3) == Player::kPlayer1);
    assert(getPreviousPlayer(Player::kPlayer1, 3) == Player::kPlayer3);

    TicTacMoEnv env;
    const int moves[] = {0, 5, 10, 1, 6, 11, 2};
    for (int move : moves) { assert(env.act(TicTacMoAction(move, env.getTurn()))); }
    assert(env.isTerminal());
    assert(env.getLegalActions().empty());
    const PlayerValues terminal_values = env.getEvalScores();
    assert(terminal_values[0] == 1.0f && terminal_values[1] == -1.0f && terminal_values[2] == -1.0f);
    assert(env.getFeatures().size() == 90);

    TicTacMoEnvLoader loader;
    loader.loadFromEnvironment(env);
    TicTacMoEnvLoader loaded;
    assert(loaded.loadFromString(loader.toString()));
    assert(loaded.getActionPairs().size() == 7);
    assert(loaded.getValue(0) == std::vector<float>({1.0f, -1.0f, -1.0f}));

    MCTS mcts(8);
    mcts.reset();
    mcts.setRootPlayer(Player::kPlayer1);
    MCTSNode* root = mcts.getRootNode();
    mcts.expand(root, {{Action(0, Player::kPlayer1), 1.0f, 0.0f}});
    MCTSNode* player1_edge = root->getChild(0);
    mcts.expand(player1_edge, {{Action(1, Player::kPlayer2), 1.0f, 0.0f}});
    MCTSNode* player2_edge = player1_edge->getChild(0);
    mcts.expand(player2_edge, {{Action(2, Player::kPlayer3), 1.0f, 0.0f}});
    MCTSNode* player3_edge = player2_edge->getChild(0);
    const PlayerValues search_values{1.0f, -0.25f, -0.75f};
    mcts.backup({root, player1_edge, player2_edge, player3_edge}, search_values);
    assert(root->getMean() == 1.0f);
    assert(player1_edge->getMean() == 1.0f);
    assert(player2_edge->getMean() == -0.25f);
    assert(player3_edge->getMean() == -0.75f);

    MCTS scalar_two_player_mcts(4);
    scalar_two_player_mcts.reset();
    MCTSNode* scalar_root = scalar_two_player_mcts.getRootNode();
    scalar_two_player_mcts.expand(scalar_root, {{Action(0, Player::kPlayer1), 1.0f, 0.0f}});
    MCTSNode* scalar_player1_edge = scalar_root->getChild(0);
    scalar_two_player_mcts.expand(scalar_player1_edge, {{Action(1, Player::kPlayer2), 1.0f, 0.0f}});
    MCTSNode* scalar_player2_edge = scalar_player1_edge->getChild(0);
    scalar_two_player_mcts.backup({scalar_root, scalar_player1_edge, scalar_player2_edge}, 0.6f);

    MCTS vector_two_player_mcts(4);
    vector_two_player_mcts.reset();
    vector_two_player_mcts.setRootPlayer(Player::kPlayer1);
    MCTSNode* vector_root = vector_two_player_mcts.getRootNode();
    vector_two_player_mcts.expand(vector_root, {{Action(0, Player::kPlayer1), 1.0f, 0.0f}});
    MCTSNode* vector_player1_edge = vector_root->getChild(0);
    vector_two_player_mcts.expand(vector_player1_edge, {{Action(1, Player::kPlayer2), 1.0f, 0.0f}});
    MCTSNode* vector_player2_edge = vector_player1_edge->getChild(0);
    const PlayerValues zero_sum_values{0.6f, -0.6f};
    vector_two_player_mcts.backup({vector_root, vector_player1_edge, vector_player2_edge}, zero_sum_values);

    assert(scalar_root->getNormalizedMean(scalar_two_player_mcts.getTreeValueBound()) ==
           vector_root->getNormalizedMean(vector_two_player_mcts.getTreeValueBound(), true));
    assert(scalar_player1_edge->getNormalizedMean(scalar_two_player_mcts.getTreeValueBound()) ==
           vector_player1_edge->getNormalizedMean(vector_two_player_mcts.getTreeValueBound(), true));
    assert(scalar_player2_edge->getNormalizedMean(scalar_two_player_mcts.getTreeValueBound()) ==
           vector_player2_edge->getNormalizedMean(vector_two_player_mcts.getTreeValueBound(), true));

    zero::ZeroSelfPlayData multiplayer_data("SelfPlay true 10 10 1,-1,-1 (;GM[tictacmo]RE[1,-1,-1]) #");
    assert(multiplayer_data.return_ == 1.0f);
    assert(multiplayer_data.returns_ == std::vector<float>({1.0f, -1.0f, -1.0f}));

    zero::ZeroSelfPlayData legacy_data("SelfPlay true 9 9 -1 (;GM[tictactoe]RE[-1]) #");
    assert(legacy_data.return_ == -1.0f);
    assert(legacy_data.returns_ == std::vector<float>({-1.0f}));
    return 0;
}
