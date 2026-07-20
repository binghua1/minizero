#include "mcts.h"

namespace minizero::actor {

void MCTSNode::reset()
{
    num_children_ = 0;
    hidden_state_data_index_ = -1;
    mean_ = 0.0f;
    count_ = 0.0f;
    virtual_loss_ = 0.0f;
    policy_ = 0.0f;
    reference_policy_ = 0.0f;
    policy_logit_ = 0.0f;
    policy_noise_ = 0.0f;
    value_ = 0.0f;
    reward_ = 0.0f;
    value_square_sum_ = 0.0f;
    first_child_ = nullptr;
}

void MCTSNode::add(float value, float weight /* = 1.0f */)
{
    if (count_ + weight <= 0) {
        reset();
    } else {
        count_ += weight;
        value_square_sum_ += weight * value * value;
        mean_ += weight * (value - mean_) / count_;
    }
}

void MCTSNode::remove(float value, float weight /* = 1.0f */)
{
    if (count_ - weight <= 0) {
        reset();
    } else {
        count_ -= weight;
        value_square_sum_ -= weight * value * value;
        value_square_sum_ = std::max(0.0f, value_square_sum_);
        mean_ -= weight * (value - mean_) / count_;
    }
}

float MCTSNode::getNormalizedMean(const std::map<float, int>& tree_value_bound, bool value_is_actor_relative /* = false */) const
{
    float value = reward_ + config::actor_mcts_reward_discount * mean_;
    if (config::actor_mcts_value_rescale) {
        if (tree_value_bound.size() < 2) { return 1.0f; }
        const float value_lower_bound = tree_value_bound.begin()->first;
        const float value_upper_bound = tree_value_bound.rbegin()->first;
        value = (value - value_lower_bound) / (value_upper_bound - value_lower_bound);
        value = fmin(1, fmax(-1, 2 * value - 1)); // normalize to [-1, 1]
    }
    if (!value_is_actor_relative) {
        value = (action_.getPlayer() == env::charToPlayer(config::actor_mcts_value_flipping_player) ? -value : value); // flip value according to player
    }
    value = (value * count_ - virtual_loss_) / getCountWithVirtualLoss();    // value with virtual loss
    return value;
}

float MCTSNode::getNormalizedPUCTScore(int total_simulation, const std::map<float, int>& tree_value_bound, float init_q_value /* = -1.0f */, bool value_is_actor_relative /* = false */) const
{
    float puct_bias = config::actor_mcts_puct_init + log((1 + total_simulation + config::actor_mcts_puct_base) / config::actor_mcts_puct_base);
    float value_u = (puct_bias * getPolicy() * sqrt(total_simulation)) / (1 + getCountWithVirtualLoss());
    float value_q = (getCountWithVirtualLoss() == 0 ? init_q_value : getNormalizedMean(tree_value_bound, value_is_actor_relative));
    return value_u + value_q;
}

std::string MCTSNode::toString() const
{
    std::ostringstream oss;
    oss.precision(4);
    oss << std::fixed << "p = " << policy_
        << ", p_logit = " << policy_logit_
        << ", p_noise = " << policy_noise_
        << ", p_ref = " << reference_policy_
        << ", v = " << value_
        << ", r = " << reward_
        << ", mean = " << mean_
        << ", count = " << count_;
    return oss.str();
}

void MCTS::reset()
{
    Tree::reset();
    tree_hidden_state_data_.reset();
    tree_value_bound_.clear();
    root_player_ = env::Player::kPlayerNone;
    multiplayer_search_type_ = MultiplayerSearchType::kMaxN;
    use_player_value_backup_ = false;
}

bool MCTS::isResign(const MCTSNode* selected_node) const
{
    float root_win_rate = getRootNode()->getNormalizedMean(tree_value_bound_, use_player_value_backup_);
    float action_win_rate = selected_node->getNormalizedMean(tree_value_bound_, use_player_value_backup_);
    return (-root_win_rate < config::actor_resign_threshold && action_win_rate < config::actor_resign_threshold);
}

MCTSNode* MCTS::selectChildByMaxCount(const MCTSNode* node) const
{
    assert(node && !node->isLeaf());
    float max_count = 0.0f;
    MCTSNode* selected = nullptr;
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        if (child->getCount() <= max_count) { continue; }
        max_count = child->getCount();
        selected = child;
    }
    assert(selected != nullptr);
    return selected;
}

MCTSNode* MCTS::selectChildBySoftmaxCount(const MCTSNode* node, float temperature /* = 1.0f */, float value_threshold /* = 0.1f */) const
{
    assert(node && !node->isLeaf());
    MCTSNode* selected = nullptr;
    MCTSNode* best_child = selectChildByMaxCount(node);
    float best_mean = best_child->getNormalizedMean(tree_value_bound_, use_player_value_backup_);
    float sum = 0.0f;
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        float count = std::pow(child->getCount(), 1 / temperature);
        float mean = child->getNormalizedMean(tree_value_bound_, use_player_value_backup_);
        if (count == 0 || (mean < best_mean - value_threshold)) { continue; }
        sum += count;
        float rand = utils::Random::randReal(sum);
        if (selected == nullptr || rand < count) { selected = child; }
    }
    assert(selected != nullptr);
    return selected;
}

std::string MCTS::getSearchDistributionString() const
{
    const MCTSNode* root = getRootNode();
    std::ostringstream oss;
    for (int i = 0; i < root->getNumChildren(); ++i) {
        MCTSNode* child = root->getChild(i);
        if (child->getCount() == 0) { continue; }
        oss << (oss.str().empty() ? "" : ",")
            << child->getAction().getActionID() << ":" << child->getCount();
    }
    return oss.str();
}

std::string MCTS::getReferencePolicyString() const
{
    const MCTSNode* root = getRootNode();
    float sum = 0.0f;
    for (int i = 0; i < root->getNumChildren(); ++i) { sum += std::max(1e-8f, root->getChild(i)->getReferencePolicy()); }
    std::ostringstream oss;
    oss.precision(8);
    for (int i = 0; i < root->getNumChildren(); ++i) {
        if (i > 0) { oss << ","; }
        oss << root->getChild(i)->getAction().getActionID() << ":"
            << std::max(1e-8f, root->getChild(i)->getReferencePolicy()) / sum;
    }
    return oss.str();
}

std::vector<float> MCTS::calculateCertifiedDeviationPolicy() const
{
    const MCTSNode* root = getRootNode();
    const int num_actions = root->getNumChildren();
    if (num_actions == 0) { return {}; }

    constexpr float kMinProbability = 1e-8f;
    std::vector<float> reference(num_actions), q_values(num_actions), radii(num_actions), proposal(num_actions);
    float reference_sum = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        reference[i] = std::max(kMinProbability, root->getChild(i)->getReferencePolicy());
        reference_sum += reference[i];
    }
    for (float& probability : reference) { probability /= reference_sum; }

    const float root_value = root->getMean();
    float baseline = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        const MCTSNode* child = root->getChild(i);
        const float count = child->getCount();
        q_values[i] = count > 0.0f ? child->getNormalizedMean(tree_value_bound_, use_player_value_backup_) : root_value;
        const float variance_numerator = count * child->getVariance() +
                                         config::actor_deviation_prior_count * config::actor_deviation_variance_prior;
        const float effective_count = std::max(1.0f, count + config::actor_deviation_prior_count);
        radii[i] = config::actor_deviation_confidence_scale * std::sqrt(std::max(0.0f, variance_numerator) / (effective_count * effective_count));
        baseline += reference[i] * q_values[i];
    }

    float visit_sum = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        visit_sum += root->getChild(i)->getCount();
    }
    for (int i = 0; i < num_actions; ++i) {
        proposal[i] = visit_sum > 0.0f ? root->getChild(i)->getCount() / visit_sum : reference[i];
    }

    // Certify the policy-level improvement proposed by search, rather than
    // independently accepting noisy action maxima. A soft gate retains a
    // learning signal when the finite-search certificate is inconclusive.
    float expected_improvement = 0.0f, uncertainty_squared = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        const float policy_shift = proposal[i] - reference[i];
        expected_improvement += proposal[i] * (q_values[i] - baseline);
        uncertainty_squared += policy_shift * policy_shift * radii[i] * radii[i];
    }
    const float lower_improvement = expected_improvement - std::sqrt(uncertainty_squared);
    const float temperature = std::max(config::actor_deviation_temperature, 1e-6f);
    const float evidence_gate = 1.0f / (1.0f + std::exp(-lower_improvement / temperature));

    auto calculate_kl = [&reference, &proposal](float mix) {
        float kl = 0.0f;
        for (size_t i = 0; i < reference.size(); ++i) {
            const float probability = (1.0f - mix) * reference[i] + mix * proposal[i];
            if (probability <= 0.0f) { continue; }
            kl += probability * std::log(probability / reference[i]);
        }
        return kl;
    };

    float mix = evidence_gate;
    const float kl_budget = std::max(0.0f, config::actor_deviation_kl_budget);
    if (calculate_kl(mix) > kl_budget) {
        float lower = 0.0f, upper = mix;
        for (int iteration = 0; iteration < 24; ++iteration) {
            mix = (lower + upper) * 0.5f;
            if (calculate_kl(mix) <= kl_budget) {
                lower = mix;
            } else {
                upper = mix;
            }
        }
        mix = lower;
    }
    for (int i = 0; i < num_actions; ++i) { proposal[i] = (1.0f - mix) * reference[i] + mix * proposal[i]; }
    return proposal;
}

std::string MCTS::getCertifiedDeviationPolicyString() const
{
    const MCTSNode* root = getRootNode();
    const std::vector<float> policy = calculateCertifiedDeviationPolicy();
    std::ostringstream oss;
    oss.precision(8);
    for (int i = 0; i < root->getNumChildren(); ++i) {
        oss << (i == 0 ? "" : ",") << root->getChild(i)->getAction().getActionID() << ":" << policy[i];
    }
    return oss.str();
}

float MCTS::getCertifiedDeviationPolicyKL() const
{
    const MCTSNode* root = getRootNode();
    const std::vector<float> policy = calculateCertifiedDeviationPolicy();
    float reference_sum = 0.0f;
    for (int i = 0; i < root->getNumChildren(); ++i) { reference_sum += std::max(1e-8f, root->getChild(i)->getReferencePolicy()); }
    float kl = 0.0f;
    for (int i = 0; i < root->getNumChildren(); ++i) {
        const float reference = std::max(1e-8f, root->getChild(i)->getReferencePolicy()) / reference_sum;
        kl += policy[i] * std::log(policy[i] / reference);
    }
    return kl;
}

std::string MCTS::getDeviationDiagnosticsString() const
{
    const MCTSNode* root = getRootNode();
    float reference_sum = 0.0f, baseline = 0.0f;
    for (int i = 0; i < root->getNumChildren(); ++i) { reference_sum += std::max(1e-8f, root->getChild(i)->getReferencePolicy()); }
    for (int i = 0; i < root->getNumChildren(); ++i) {
        const MCTSNode* child = root->getChild(i);
        const float reference = std::max(1e-8f, child->getReferencePolicy()) / reference_sum;
        baseline += reference * (child->getCount() > 0.0f ? child->getNormalizedMean(tree_value_bound_, use_player_value_backup_) : root->getMean());
    }
    std::ostringstream oss;
    oss.precision(6);
    for (int i = 0; i < root->getNumChildren(); ++i) {
        const MCTSNode* child = root->getChild(i);
        if (i > 0) { oss << ","; }
        oss << child->getAction().getActionID() << ":"
            << (child->getCount() > 0.0f ? child->getNormalizedMean(tree_value_bound_, use_player_value_backup_) - baseline : 0.0f)
            << ":" << child->getCount();
    }
    return oss.str();
}

std::vector<MCTSNode*> MCTS::selectFromNode(MCTSNode* start_node)
{
    assert(start_node);
    MCTSNode* node = start_node;
    std::vector<MCTSNode*> node_path{node};
    while (!node->isLeaf()) {
        node = selectChildByPUCTScore(node);
        node_path.push_back(node);
    }
    return node_path;
}

void MCTS::expand(MCTSNode* leaf_node, const std::vector<ActionCandidate>& action_candidates)
{
    assert(leaf_node && action_candidates.size() > 0);
    leaf_node->setFirstChild(allocateNodes(action_candidates.size()));
    leaf_node->setNumChildren(action_candidates.size());
    for (size_t i = 0; i < action_candidates.size(); ++i) {
        const auto& candidate = action_candidates[i];
        MCTSNode* child = leaf_node->getChild(i);
        child->reset();
        child->setAction(candidate.action_);
        child->setPolicy(candidate.policy_);
        child->setReferencePolicy(candidate.policy_);
        child->setPolicyLogit(candidate.policy_logit_);
    }
}

void MCTS::backup(const std::vector<MCTSNode*>& node_path, const float value, const float reward /* = 0.0f */)
{
    assert(node_path.size() > 0);
    float updated_value = value;
    node_path.back()->setValue(value);
    node_path.back()->setReward(reward);
    for (int i = static_cast<int>(node_path.size() - 1); i >= 0; --i) {
        MCTSNode* node = node_path[i];
        float old_mean = node->getReward() + config::actor_mcts_reward_discount * node->getMean();
        node->add(updated_value);
        updateTreeValueBound(old_mean, node->getReward() + config::actor_mcts_reward_discount * node->getMean());
        updated_value = node->getReward() + config::actor_mcts_reward_discount * updated_value;
    }
}

void MCTS::backup(const std::vector<MCTSNode*>& node_path, const env::PlayerValues& values)
{
    assert(!node_path.empty());
    assert(root_player_ != env::Player::kPlayerNone);
    use_player_value_backup_ = true;

    for (int i = static_cast<int>(node_path.size()) - 1; i >= 0; --i) {
        MCTSNode* node = node_path[i];
        const env::Player action_player = (i == 0 ? root_player_ : node->getAction().getPlayer());
        float value;
        if (multiplayer_search_type_ == MultiplayerSearchType::kParanoid) {
            const float root_value = values[env::playerToIndex(root_player_)];
            value = (action_player == root_player_ ? root_value : -root_value);
        } else {
            value = values[env::playerToIndex(action_player)];
        }
        node->add(value);
        if (i == static_cast<int>(node_path.size()) - 1) { node->setValue(value); }
    }
}

MCTSNode* MCTS::selectChildByPUCTScore(const MCTSNode* node) const
{
    assert(node && !node->isLeaf());
    MCTSNode* selected = nullptr;
    int total_simulation = node->getCountWithVirtualLoss() - 1;
    float init_q_value = calculateInitQValue(node);
    float best_score = std::numeric_limits<float>::lowest(), best_policy = std::numeric_limits<float>::lowest();
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        float score = child->getNormalizedPUCTScore(total_simulation, tree_value_bound_, init_q_value, use_player_value_backup_);
        if (score < best_score || (score == best_score && child->getPolicy() <= best_policy)) { continue; }
        best_score = score;
        best_policy = child->getPolicy();
        selected = child;
    }
    assert(selected != nullptr);
    return selected;
}

float MCTS::calculateInitQValue(const MCTSNode* node) const
{
    // init Q value = avg Q value of all visited children + one loss
    assert(node && !node->isLeaf());
    float sum_of_win = 0.0f, sum = 0.0f;
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        if (child->getCountWithVirtualLoss() == 0) { continue; }
        sum_of_win += child->getNormalizedMean(tree_value_bound_, use_player_value_backup_);
        sum += 1;
    }
#if ATARI
    // explore more in Atari games (TODO: check if this method also performs better in board games)
    return (sum > 0 ? sum_of_win / sum : 1.0f);
#else
    return (sum_of_win - 1) / (sum + 1);
#endif
}

void MCTS::updateTreeValueBound(float old_value, float new_value)
{
    if (!config::actor_mcts_value_rescale) { return; }
    if (tree_value_bound_.count(old_value)) {
        assert(tree_value_bound_[old_value] > 0);
        --tree_value_bound_[old_value];
        if (tree_value_bound_[old_value] == 0) { tree_value_bound_.erase(old_value); }
    }
    ++tree_value_bound_[new_value];
}

} // namespace minizero::actor
