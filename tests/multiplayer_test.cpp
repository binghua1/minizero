#include "mcts.h"
#include "tictacmo.h"
#include "zero_server.h"
#include <cassert>
#include <cmath>
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

    TicTacMoEnv replay_env;
    assert(replay_env.act(TicTacMoAction(0, replay_env.getTurn())));
    const std::vector<std::vector<std::pair<std::string, std::string>>> replay_info{{
        {"P", "0:1,1:1"},
        {"A", "0:3,1:1"},
        {"N", "0:2,1:1"},
        {"D", "0:0.2:2,1:-0.1:1"},
        {"K", "0.01"},
    }};
    TicTacMoEnvLoader replay_loader;
    replay_loader.loadFromEnvironment(replay_env, replay_info);
    TicTacMoEnvLoader reloaded_replay;
    assert(reloaded_replay.loadFromString(replay_loader.toString()));
    const std::vector<float> replay_policy = reloaded_replay.getPolicy(0);
    const std::vector<float> replay_reference = reloaded_replay.getReferencePolicy(0);
    assert(replay_policy[0] == 0.5f && replay_policy[1] == 0.5f);
    assert(replay_reference[0] == 0.75f && replay_reference[1] == 0.25f);

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

    MCTS paranoid_mcts(8);
    paranoid_mcts.reset();
    paranoid_mcts.setRootPlayer(Player::kPlayer1);
    paranoid_mcts.setMultiplayerSearchType(MCTS::MultiplayerSearchType::kParanoid);
    MCTSNode* paranoid_root = paranoid_mcts.getRootNode();
    paranoid_mcts.expand(paranoid_root, {{Action(0, Player::kPlayer1), 1.0f, 0.0f}});
    MCTSNode* paranoid_player1_edge = paranoid_root->getChild(0);
    const std::vector<MCTS::ActionCandidate> player2_candidates{
        {Action(1, Player::kPlayer2), 0.5f, 0.0f},
        {Action(2, Player::kPlayer2), 0.5f, 0.0f},
    };
    paranoid_mcts.expand(paranoid_player1_edge, player2_candidates);
    MCTSNode* paranoid_high_root_value = paranoid_player1_edge->getChild(0);
    MCTSNode* paranoid_low_root_value = paranoid_player1_edge->getChild(1);
    paranoid_mcts.backup(
        {paranoid_root, paranoid_player1_edge, paranoid_high_root_value},
        PlayerValues{0.8f, 0.9f, -0.5f});
    paranoid_mcts.backup(
        {paranoid_root, paranoid_player1_edge, paranoid_low_root_value},
        PlayerValues{0.2f, 0.1f, -0.1f});
    assert(paranoid_high_root_value->getMean() == -0.8f);
    assert(paranoid_low_root_value->getMean() == -0.2f);
    assert(paranoid_mcts.selectFromNode(paranoid_player1_edge).back() == paranoid_low_root_value);

    MCTS maxn_selection_mcts(8);
    maxn_selection_mcts.reset();
    maxn_selection_mcts.setRootPlayer(Player::kPlayer1);
    MCTSNode* maxn_selection_root = maxn_selection_mcts.getRootNode();
    maxn_selection_mcts.expand(maxn_selection_root, {{Action(0, Player::kPlayer1), 1.0f, 0.0f}});
    MCTSNode* maxn_selection_player1_edge = maxn_selection_root->getChild(0);
    maxn_selection_mcts.expand(maxn_selection_player1_edge, player2_candidates);
    MCTSNode* maxn_high_player2_value = maxn_selection_player1_edge->getChild(0);
    MCTSNode* maxn_low_player2_value = maxn_selection_player1_edge->getChild(1);
    maxn_selection_mcts.backup(
        {maxn_selection_root, maxn_selection_player1_edge, maxn_high_player2_value},
        PlayerValues{0.8f, 0.9f, -0.5f});
    maxn_selection_mcts.backup(
        {maxn_selection_root, maxn_selection_player1_edge, maxn_low_player2_value},
        PlayerValues{0.2f, 0.1f, -0.1f});
    assert(maxn_high_player2_value->getMean() == 0.9f);
    assert(maxn_low_player2_value->getMean() == 0.1f);
    assert(maxn_selection_mcts.selectFromNode(maxn_selection_player1_edge).back() == maxn_high_player2_value);

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

    // Certified-deviation targets must improve a supported profitable action
    // while respecting the configured trust-region budget.
    MCTS deviation_mcts(16);
    deviation_mcts.reset();
    deviation_mcts.setRootPlayer(Player::kPlayer1);
    MCTSNode* deviation_root = deviation_mcts.getRootNode();
    deviation_mcts.expand(deviation_root, {
                                             {Action(3, Player::kPlayer1), 0.5f, 0.0f},
                                             {Action(4, Player::kPlayer1), 0.3f, 0.0f},
                                             {Action(5, Player::kPlayer1), 0.2f, 0.0f},
                                         });
    for (int sample = 0; sample < 6; ++sample) {
        deviation_mcts.backup({deviation_root, deviation_root->getChild(0)}, PlayerValues{0.8f, -0.2f, -0.4f});
    }
    for (int sample = 0; sample < 3; ++sample) {
        deviation_mcts.backup({deviation_root, deviation_root->getChild(1)}, PlayerValues{0.1f, 0.0f, -0.1f});
    }
    deviation_mcts.backup({deviation_root, deviation_root->getChild(2)}, PlayerValues{-0.5f, 0.3f, 0.2f});
    const float old_confidence_scale = config::actor_deviation_confidence_scale;
    const float old_kl_budget = config::actor_deviation_kl_budget;
    config::actor_deviation_confidence_scale = 0.0f;
    config::actor_deviation_kl_budget = 0.02f;
    const std::vector<float> deviation_policy = deviation_mcts.calculateCertifiedDeviationPolicy();
    assert(deviation_policy.size() == 3);
    assert(deviation_policy[0] > 0.5f);
    assert(deviation_policy[2] < 0.2f);
    assert(deviation_mcts.getCertifiedDeviationPolicyKL() <= config::actor_deviation_kl_budget + 1e-5f);
    assert(std::abs(deviation_policy[0] + deviation_policy[1] + deviation_policy[2] - 1.0f) < 1e-5f);
    config::actor_deviation_confidence_scale = old_confidence_scale;
    config::actor_deviation_kl_budget = old_kl_budget;

    zero::ZeroSelfPlayData multiplayer_data("SelfPlay true 10 10 1,-1,-1 (;GM[tictacmo]RE[1,-1,-1]) #");
    assert(multiplayer_data.return_ == 1.0f);
    assert(multiplayer_data.returns_ == std::vector<float>({1.0f, -1.0f, -1.0f}));

    zero::ZeroSelfPlayData legacy_data("SelfPlay true 9 9 -1 (;GM[tictactoe]RE[-1]) #");
    assert(legacy_data.return_ == -1.0f);
    assert(legacy_data.returns_ == std::vector<float>({-1.0f}));
    return 0;
}
