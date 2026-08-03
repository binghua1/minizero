#include "zero_actor.h"
#include "random.h"
#include "time_system.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minizero::actor {

using namespace minizero;
using namespace network;

void MCTSSearchData::clear()
{
    search_info_ = "";
    selected_node_ = nullptr;
    node_path_.clear();
}

void ZeroActor::reset()
{
    BaseActor::reset();
    samplePopulationLineup();
    enable_resign_ = (utils::Random::randReal() < config::zero_disable_resign_ratio ? false : true);
}

void ZeroActor::resetSearch()
{
    BaseActor::resetSearch();
    mcts_search_data_.node_path_.clear();
    getMCTS()->setRootPlayer(env_.getTurn());
    getMCTS()->setMultiplayerSearchType(
        config::actor_multiplayer_search_type == "paranoid"
            ? MCTS::MultiplayerSearchType::kParanoid
            : MCTS::MultiplayerSearchType::kMaxN);
    getMCTS()->getRootNode()->setAction(Action(-1, env::getPreviousPlayer(env_.getTurn(), env_.getNumPlayer())));
}

Action ZeroActor::think(bool with_play /*= false*/, bool display_board /*= false*/)
{
    resetSearch();
    boost::posix_time::ptime start_ptime = utils::TimeSystem::getLocalTime();
    while (!isSearchDone()) {
        step();
        int spent_million_second = (utils::TimeSystem::getLocalTime() - start_ptime).total_milliseconds();
        if (config::actor_mcts_think_time_limit > 0 && spent_million_second >= config::actor_mcts_think_time_limit * 1000) { break; }
    }
    if (!isSearchDone()) { handleSearchDone(); }
    if (with_play) { act(getSearchAction()); }
    if (display_board) { std::cerr << env_.toString() << mcts_search_data_.search_info_ << std::endl; }
    return getSearchAction();
}

void ZeroActor::beforeNNEvaluation()
{
    activateNetworkForTurn();
    nn_evaluation_network_id_ = active_network_id_;
    mcts_search_data_.node_path_ = selection();
    if (alphazero_network_) {
        Environment env_transition = getEnvironmentTransition(mcts_search_data_.node_path_);
        feature_rotation_ = config::actor_use_random_rotation_features ? static_cast<utils::Rotation>(utils::Random::randInt() % static_cast<int>(utils::Rotation::kRotateSize)) : utils::Rotation::kRotationNone;
        nn_evaluation_batch_id_ = alphazero_network_->pushBack(
            env_transition.getFeatures(feature_rotation_),
            env_transition.getBehaviorHistory(alphazero_network_->getBehaviorHistoryLength(), feature_rotation_),
            env::playerToIndex(env_transition.getTurn()));
    } else if (muzero_network_) {
        if (getMCTS()->getNumSimulation() == 0) { // initial inference for root node
            nn_evaluation_batch_id_ = muzero_network_->pushBackInitialData(env_.getFeatures());
        } else { // for non-root nodes
            const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
            MCTSNode* leaf_node = node_path.back();
            MCTSNode* parent_node = node_path[node_path.size() - 2];
            assert(parent_node && parent_node->getHiddenStateDataIndex() != -1);
            const std::vector<float>& hidden_state = getMCTS()->getTreeHiddenStateData().getData(parent_node->getHiddenStateDataIndex()).hidden_state_;
            nn_evaluation_batch_id_ = muzero_network_->pushBackRecurrentData(hidden_state, env_.getActionFeatures(leaf_node->getAction()));
        }
    } else {
        assert(false);
    }
}

void ZeroActor::afterNNEvaluation(const std::shared_ptr<NetworkOutput>& network_output)
{
    const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
    MCTSNode* leaf_node = node_path.back();
    if (alphazero_network_) {
        Environment env_transition = getEnvironmentTransition(node_path);
        if (!env_transition.isTerminal()) {
            std::shared_ptr<AlphaZeroNetworkOutput> alphazero_output = std::static_pointer_cast<AlphaZeroNetworkOutput>(network_output);
            getMCTS()->expand(leaf_node, calculateAlphaZeroActionPolicy(env_transition, alphazero_output, feature_rotation_));
            env::PlayerValues values{};
            assert(static_cast<int>(alphazero_output->values_.size()) >= env_transition.getNumPlayer());
            std::copy(alphazero_output->values_.begin(),
                      alphazero_output->values_.begin() + env_transition.getNumPlayer(),
                      values.begin());
            getMCTS()->backup(node_path, values);
        } else {
            getMCTS()->backup(node_path, env_transition.getEvalScores());
        }
    } else if (muzero_network_) {
        std::shared_ptr<MuZeroNetworkOutput> muzero_output = std::static_pointer_cast<MuZeroNetworkOutput>(network_output);
        // The root corresponds to the real environment, so its exact legal actions
        // are available. Non-root MuZero nodes only have learned latent states and
        // must keep the full action space unless a real environment is supplied.
        const Environment* legal_action_env = (leaf_node == getMCTS()->getRootNode() ? &env_ : nullptr);
        getMCTS()->expand(leaf_node, calculateMuZeroActionPolicy(leaf_node, muzero_output, legal_action_env));
        getMCTS()->backup(node_path, muzero_output->value_, muzero_output->reward_);
        leaf_node->setHiddenStateDataIndex(getMCTS()->getTreeHiddenStateData().store(HiddenStateData(muzero_output->hidden_state_)));
    } else {
        assert(false);
    }
    if (leaf_node == getMCTS()->getRootNode()) { addNoiseToNodeChildren(leaf_node); }
    if (isSearchDone()) { handleSearchDone(); }
    if (config::actor_use_gumbel) { gumbel_zero_.sequentialHalving(getMCTS()); }
}

void ZeroActor::setNetwork(const std::shared_ptr<network::Network>& network)
{
    assert(network);
    alphazero_network_ = nullptr;
    muzero_network_ = nullptr;
    if (network->getNetworkTypeName() == "alphazero") {
        alphazero_network_ = std::static_pointer_cast<AlphaZeroNetwork>(network);
    } else if (network->getNetworkTypeName() == "muzero" || network->getNetworkTypeName() == "muzero_atari") {
        muzero_network_ = std::static_pointer_cast<MuZeroNetwork>(network);
    } else {
        assert(false);
    }
    assert((alphazero_network_ && !muzero_network_) || (!alphazero_network_ && muzero_network_));

    if (env_.getNumPlayer() > 2) {
        if (config::actor_multiplayer_search_type != "maxn" && config::actor_multiplayer_search_type != "paranoid") {
            throw std::runtime_error("multiplayer search type must be maxn or paranoid");
        }
        if (!alphazero_network_) { throw std::runtime_error("multiplayer environments currently support AlphaZero only"); }
        if (network->getNumPlayers() != env_.getNumPlayer()) { throw std::runtime_error("network and environment player counts do not match"); }
        if (config::actor_use_gumbel) { throw std::runtime_error("multiplayer Gumbel search is not validated yet"); }
        if (config::actor_mcts_value_rescale) { throw std::runtime_error("multiplayer value rescaling is not supported yet"); }
        if (config::zero_disable_resign_ratio < 1.0f) { throw std::runtime_error("multiplayer resignation must be disabled"); }
        if (config::zero_actor_intermediate_sequence_length != 0) { throw std::runtime_error("multiplayer intermediate self-play sequences are not supported yet"); }
    }
}

void ZeroActor::setPopulationNetworks(int current_network_id,
                                      const std::shared_ptr<network::Network>& current_network,
                                      int historical_network_id,
                                      const std::shared_ptr<network::Network>& historical_network,
                                      int historical_iteration,
                                      const std::vector<float>& current_seat_weights)
{
    if (!current_network) { throw std::runtime_error("population current network is missing"); }
    current_network_id_ = current_network_id;
    current_network_ = current_network;
    historical_network_id_ = historical_network_id;
    historical_network_ = historical_network;
    historical_iteration_ = historical_iteration;
    current_seat_weights_ = current_seat_weights;
    samplePopulationLineup();
    activateNetworkForTurn();
}

void ZeroActor::samplePopulationLineup()
{
    const int num_players = env_.getNumPlayer();
    seat_model_ids_.assign(num_players, 0);
    if (!historical_network_ || historical_iteration_ < 0 || num_players <= 2) { return; }

    const int min_current = std::clamp(config::zero_population_current_seat_min, 1, num_players - 1);
    const int max_current = std::clamp(config::zero_population_current_seat_max, min_current, num_players - 1);
    const int num_choices = max_current - min_current + 1;
    std::vector<float> weights = current_seat_weights_;
    if (static_cast<int>(weights.size()) != num_choices ||
        std::accumulate(weights.begin(), weights.end(), 0.0f) <= 0.0f) {
        weights.assign(num_choices, 1.0f);
    }
    float draw = utils::Random::randReal(std::accumulate(weights.begin(), weights.end(), 0.0f));
    int num_current = min_current;
    for (int i = 0; i < num_choices; ++i) {
        draw -= std::max(0.0f, weights[i]);
        if (draw <= 0.0f) {
            num_current = min_current + i;
            break;
        }
    }

    seat_model_ids_.assign(num_players, 1);
    std::vector<int> seats(num_players);
    std::iota(seats.begin(), seats.end(), 0);
    for (int i = num_players - 1; i > 0; --i) {
        const int j = static_cast<unsigned int>(utils::Random::randInt()) % (i + 1);
        std::swap(seats[i], seats[j]);
    }
    for (int i = 0; i < num_current; ++i) { seat_model_ids_[seats[i]] = 0; }
}

void ZeroActor::activateNetworkForTurn()
{
    if (!current_network_) { return; }
    const int player_index = env::playerToIndex(env_.getTurn());
    const bool use_history = historical_network_ && historical_iteration_ >= 0 &&
                             player_index >= 0 && player_index < static_cast<int>(seat_model_ids_.size()) &&
                             seat_model_ids_[player_index] == 1;
    active_model_id_ = use_history ? 1 : 0;
    active_network_id_ = use_history ? historical_network_id_ : current_network_id_;
    setNetwork(use_history ? historical_network_ : current_network_);
}

std::string ZeroActor::getRecord(const std::unordered_map<std::string, std::string>& tags) const
{
    auto population_tags = tags;
    if (historical_iteration_ >= 0) {
        std::ostringstream lineup;
        for (size_t i = 0; i < seat_model_ids_.size(); ++i) {
            if (i > 0) { lineup << ","; }
            lineup << seat_model_ids_[i];
        }
        population_tags["LM"] = lineup.str();
        population_tags["HI"] = std::to_string(historical_iteration_);
    }
    return BaseActor::getRecord(population_tags);
}

std::vector<std::pair<std::string, std::string>> ZeroActor::getActionInfo() const
{
    // ignore recording mcts action info if there is no search
    if (getMCTS()->getRootNode()->getCount() > 0) {
        auto action_info = BaseActor::getActionInfo();
        if (historical_iteration_ >= 0) {
            action_info.push_back({"MI", std::to_string(active_model_id_)});
            action_info.push_back({"TR", active_model_id_ == 0 ? "1" : "0"});
        }
        return action_info;
    }
    return {};
}

std::string ZeroActor::getEnvReward() const
{
    std::ostringstream oss;
    oss << env_.getReward();
    return oss.str();
}

void ZeroActor::step()
{
    assert(alphazero_network_ || muzero_network_);
    int num_simulation = getMCTS()->getNumSimulation();
    int num_simulation_left = config::actor_num_simulation + 1 - num_simulation;
    int batch_size = std::min(config::actor_mcts_think_batch_size,
                              (alphazero_network_ || num_simulation > 0) ? num_simulation_left : 1 /* initial inference for root node */);
    assert(batch_size > 0);

    std::vector<std::tuple<int, utils::Rotation, decltype(mcts_search_data_.node_path_)>> batch_queries; // batch id, rotation, search path
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        beforeNNEvaluation();
        assert(nn_evaluation_batch_id_ == batch_id);
        if (mcts_search_data_.node_path_.back()->getVirtualLoss() == 0) {
            batch_queries.emplace_back(nn_evaluation_batch_id_, feature_rotation_, mcts_search_data_.node_path_);
        }
        for (auto node : mcts_search_data_.node_path_) { node->addVirtualLoss(); }
    }
    auto network_output = alphazero_network_ ? alphazero_network_->forward()
                                             : (num_simulation == 0 ? muzero_network_->initialInference() : muzero_network_->recurrentInference());
    for (auto& query : batch_queries) {
        nn_evaluation_batch_id_ = std::get<0>(query);
        feature_rotation_ = std::get<1>(query);
        mcts_search_data_.node_path_ = std::get<2>(query);
        afterNNEvaluation(network_output[nn_evaluation_batch_id_]);
        auto virtual_loss = mcts_search_data_.node_path_.back()->getVirtualLoss();
        for (auto node : mcts_search_data_.node_path_) { node->removeVirtualLoss(virtual_loss); }
    }
}

void ZeroActor::handleSearchDone()
{
    mcts_search_data_.selected_node_ = decideActionNode();
    const Action action = getSearchAction();
    std::ostringstream oss;
    oss << "model file name: " << config::nn_file_name << std::endl
        << utils::TimeSystem::getTimeString("[Y/m/d H:i:s.f] ")
        << "move number: " << env_.getActionHistory().size()
        << ", action: " << action.toConsoleString()
        << " (" << action.getActionID() << ")"
        << ", reward: " << env_.getReward()
        << ", player: " << env::playerToChar(action.getPlayer());
    if (config::actor_mcts_value_rescale) { oss << ", value bound: (" << getMCTS()->getTreeValueBound().begin()->first << ", " << getMCTS()->getTreeValueBound().rbegin()->first << ")"; }
    oss << std::endl
        << "  root node info: " << getMCTS()->getRootNode()->toString() << std::endl
        << "action node info: " << mcts_search_data_.selected_node_->toString() << std::endl;
    mcts_search_data_.search_info_ = oss.str();
}

MCTSNode* ZeroActor::decideActionNode()
{
    if (config::actor_use_gumbel) {
        return gumbel_zero_.decideActionNode(getMCTS());
    } else {
        if (config::actor_select_action_by_count) {
            return getMCTS()->selectChildByMaxCount(getMCTS()->getRootNode());
        } else if (config::actor_select_action_by_softmax_count) {
            return getMCTS()->selectChildBySoftmaxCount(getMCTS()->getRootNode(), config::actor_select_action_softmax_temperature);
        }

        assert(false);
        return nullptr;
    }
}

void ZeroActor::addNoiseToNodeChildren(MCTSNode* node)
{
    assert(node && node->getNumChildren() > 0);
    if (config::actor_use_dirichlet_noise) {
        const float epsilon = config::actor_dirichlet_noise_epsilon;
        std::vector<float> dirichlet_noise = utils::Random::randDirichlet(config::actor_dirichlet_noise_alpha, node->getNumChildren());
        for (int i = 0; i < node->getNumChildren(); ++i) {
            MCTSNode* child = node->getChild(i);
            child->setPolicyNoise(dirichlet_noise[i]);
            child->setPolicy((1 - epsilon) * child->getPolicy() + epsilon * dirichlet_noise[i]);
        }
    } else if (config::actor_use_gumbel_noise) {
        std::vector<float> gumbel_noise = utils::Random::randGumbel(node->getNumChildren());
        for (int i = 0; i < node->getNumChildren(); ++i) {
            MCTSNode* child = node->getChild(i);
            child->setPolicyNoise(gumbel_noise[i]);
            child->setPolicyLogit(child->getPolicyLogit() + gumbel_noise[i]);
        }
    }
}

std::vector<MCTS::ActionCandidate> ZeroActor::calculateAlphaZeroActionPolicy(const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation)
{
    assert(alphazero_network_);
    if (alphazero_output->policy_.size() != alphazero_output->policy_logits_.size()) {
        throw std::runtime_error("AlphaZero policy and policy-logit sizes do not match");
    }

    const std::vector<Action> legal_actions = env_transition.getLegalActions();
    std::vector<MCTS::ActionCandidate> action_candidates;
    action_candidates.reserve(legal_actions.size());
    for (const Action& action : legal_actions) {
        const int action_id = action.getActionID();
        if (action_id < 0 || action_id >= static_cast<int>(alphazero_output->policy_.size())) {
            throw std::runtime_error("legal action ID is outside the AlphaZero policy range");
        }
#ifndef NDEBUG
        assert(env_transition.isLegalAction(action));
#endif
        int rotated_id = env_transition.getRotateAction(action_id, rotation);
        if (rotated_id < 0 || rotated_id >= static_cast<int>(alphazero_output->policy_.size())) {
            throw std::runtime_error("rotated action ID is outside the AlphaZero policy range");
        }
        action_candidates.push_back(MCTS::ActionCandidate(action, alphazero_output->policy_[rotated_id], alphazero_output->policy_logits_[rotated_id]));
    }
    sort(action_candidates.begin(), action_candidates.end(), [](const MCTS::ActionCandidate& lhs, const MCTS::ActionCandidate& rhs) {
        return lhs.policy_ > rhs.policy_;
    });
    return action_candidates;
}

std::vector<MCTS::ActionCandidate> ZeroActor::calculateMuZeroActionPolicy(MCTSNode* leaf_node, const std::shared_ptr<network::MuZeroNetworkOutput>& muzero_output, const Environment* legal_action_env)
{
    assert(muzero_network_);
    if (muzero_output->policy_.size() != muzero_output->policy_logits_.size()) {
        throw std::runtime_error("MuZero policy and policy-logit sizes do not match");
    }

    std::vector<MCTS::ActionCandidate> action_candidates;
    if (legal_action_env) {
        const std::vector<Action> legal_actions = legal_action_env->getLegalActions();
        action_candidates.reserve(legal_actions.size());
        for (const Action& action : legal_actions) {
            const int action_id = action.getActionID();
            if (action_id < 0 || action_id >= static_cast<int>(muzero_output->policy_.size())) {
                throw std::runtime_error("legal action ID is outside the MuZero policy range");
            }
#ifndef NDEBUG
            assert(legal_action_env->isLegalAction(action));
#endif
            action_candidates.push_back(MCTS::ActionCandidate(action, muzero_output->policy_[action_id], muzero_output->policy_logits_[action_id]));
        }
    } else {
        const env::Player turn = leaf_node->getAction().nextPlayer();
        action_candidates.reserve(muzero_output->policy_.size());
        for (size_t action_id = 0; action_id < muzero_output->policy_.size(); ++action_id) {
            const Action action(action_id, turn);
            action_candidates.push_back(MCTS::ActionCandidate(action, muzero_output->policy_[action_id], muzero_output->policy_logits_[action_id]));
        }
    }
    sort(action_candidates.begin(), action_candidates.end(), [](const MCTS::ActionCandidate& lhs, const MCTS::ActionCandidate& rhs) {
        return lhs.policy_ > rhs.policy_;
    });
    return action_candidates;
}

Environment ZeroActor::getEnvironmentTransition(const std::vector<MCTSNode*>& node_path)
{
    Environment env = env_;
    for (size_t i = 1; i < node_path.size(); ++i) { env.act(node_path[i]->getAction()); }
    return env;
}

} // namespace minizero::actor
