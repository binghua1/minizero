#pragma once

#include "alphazero_network.h"
#include "base_actor.h"
#include "gumbel_zero.h"
#include "mcts.h"
#include "muzero_network.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minizero::actor {

class MCTSSearchData {
public:
    std::string search_info_;
    MCTSNode* selected_node_;
    std::vector<MCTSNode*> node_path_;
    void clear();
};

class ZeroActor : public BaseActor {
public:
    ZeroActor(uint64_t tree_node_size)
        : tree_node_size_(tree_node_size)
    {
        alphazero_network_ = nullptr;
        muzero_network_ = nullptr;
    }

    void reset() override;
    void resetSearch() override;
    Action think(bool with_play = false, bool display_board = false) override;
    void beforeNNEvaluation() override;
    void afterNNEvaluation(const std::shared_ptr<network::NetworkOutput>& network_output) override;
    bool isSearchDone() const override { return getMCTS()->reachMaximumSimulation(); }
    Action getSearchAction() const override { return mcts_search_data_.selected_node_->getAction(); }
    bool isResign() const override { return enable_resign_ && getMCTS()->isResign(mcts_search_data_.selected_node_); }
    std::string getSearchInfo() const override { return mcts_search_data_.search_info_; }
    void setNetwork(const std::shared_ptr<network::Network>& network) override;
    int getNNEvaluationNetworkID() const override { return nn_evaluation_network_id_; }
    void setPopulationNetworks(int current_network_id,
                               const std::shared_ptr<network::Network>& current_network,
                               int historical_network_id,
                               const std::shared_ptr<network::Network>& historical_network,
                               int historical_iteration,
                               const std::vector<float>& current_seat_weights);
    std::string getRecord(const std::unordered_map<std::string, std::string>& tags = {}) const override;
    std::shared_ptr<Search> createSearch() override { return std::make_shared<MCTS>(tree_node_size_); }
    std::shared_ptr<MCTS> getMCTS() { return std::static_pointer_cast<MCTS>(search_); }
    const std::shared_ptr<MCTS> getMCTS() const { return std::static_pointer_cast<MCTS>(search_); }

protected:
    std::vector<std::pair<std::string, std::string>> getActionInfo() const override;
    std::string getMCTSPolicy() const override { return (config::actor_use_gumbel ? gumbel_zero_.getMCTSPolicy(getMCTS()) : getMCTS()->getSearchDistributionString()); }
    std::string getMCTSValue() const override { return std::to_string(getMCTS()->getRootNode()->getMean()); }
    std::string getEnvReward() const override;

    virtual void step();
    virtual void handleSearchDone();
    virtual MCTSNode* decideActionNode();
    virtual void addNoiseToNodeChildren(MCTSNode* node);
    virtual std::vector<MCTSNode*> selection() { return (config::actor_use_gumbel ? gumbel_zero_.selection(getMCTS()) : getMCTS()->select()); }

    std::vector<MCTS::ActionCandidate> calculateAlphaZeroActionPolicy(const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation);
    std::vector<MCTS::ActionCandidate> calculateMuZeroActionPolicy(MCTSNode* leaf_node, const std::shared_ptr<network::MuZeroNetworkOutput>& muzero_output, const Environment* legal_action_env);
    virtual Environment getEnvironmentTransition(const std::vector<MCTSNode*>& node_path);
    void activateNetworkForTurn();
    void samplePopulationLineup();

    bool enable_resign_;
    GumbelZero gumbel_zero_;
    uint64_t tree_node_size_;
    MCTSSearchData mcts_search_data_;
    utils::Rotation feature_rotation_;
    std::shared_ptr<network::AlphaZeroNetwork> alphazero_network_;
    std::shared_ptr<network::MuZeroNetwork> muzero_network_;
    int active_network_id_ = 0;
    int nn_evaluation_network_id_ = 0;
    int active_model_id_ = 0;
    int current_network_id_ = 0;
    int historical_network_id_ = -1;
    int historical_iteration_ = -1;
    std::shared_ptr<network::Network> current_network_;
    std::shared_ptr<network::Network> historical_network_;
    std::vector<int> seat_model_ids_;
    std::vector<float> current_seat_weights_;
};

} // namespace minizero::actor
