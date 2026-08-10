#include "actor_group.h"
#include "configuration.h"
#include "create_actor.h"
#include "create_network.h"
#include "random.h"
#include "zero_actor.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/cuda.h>
#include <unordered_map>
#include <utility>

namespace minizero::actor {

using namespace network;
using namespace utils;

namespace {

    std::vector<std::string> splitCSV(const std::string& value)
    {
        std::vector<std::string> result;
        std::stringstream stream(value);
        std::string item;
        while (std::getline(stream, item, ',')) { result.push_back(item); }
        return result;
    }

} // namespace

int ThreadSharedData::getAvailableActorIndex()
{
    std::lock_guard lock(mutex_);
    return (actor_index_ < static_cast<int>(actors_.size()) ? actor_index_++ : actors_.size());
}

void ThreadSharedData::outputGame(const std::shared_ptr<BaseActor>& actor)
{
    int game_length = actor->getEnvironment().getActionHistory().size();
    std::pair<int, int> data_range = calculateTrainingDataRange(actor);
    int data_length = 0;
    const auto& action_info_history = actor->getActionInfoHistory();
    for (int pos = data_range.first; pos <= data_range.second; ++pos) {
        bool trainable = true;
        if (pos >= 0 && pos < static_cast<int>(action_info_history.size())) {
            for (const auto& info : action_info_history[pos]) {
                if (info.first == "TR" && info.second == "0") { trainable = false; }
            }
        }
        data_length += trainable ? 1 : 0;
    }
    const Environment& environment = actor->getEnvironment();
    const bool is_resign = !actor->isEnvTerminal();
    std::ostringstream game_return;
    if (environment.getNumPlayer() > 2) {
        game_return << env::playerValuesToString(environment.getEvalScores(is_resign), environment.getNumPlayer());
    } else {
        game_return << environment.getEvalScore(is_resign);
    }

    std::ostringstream oss;
    bool is_terminal = (config::zero_actor_intermediate_sequence_length == 0 || actor->isEnvTerminal());
    oss << "SelfPlay "
        << (is_terminal ? "true" : "false") << " "                                                                         // is terminal
        << data_length << " "                                                                                              // data length
        << game_length << " "                                                                                              // game length
        << game_return.str() << " "                                                                                        // return(s)
        << actor->getRecord({{"DLEN", std::to_string(data_range.first) + "-" + std::to_string(data_range.second)}}) << " " // game record
        << "#";                                                                                                            // end mark for a valid game

    if (!is_terminal) {
        // delete action info history if not complete record to save memory
        auto& action_info_history = actor->getActionInfoHistory();
        for (int i = data_range.first; i <= data_range.second; ++i) {
            action_info_history[i].clear();
            action_info_history[i].shrink_to_fit();
        }
    }

    std::lock_guard lock(mutex_);
    std::cout << oss.str() << std::endl;
}

std::pair<int, int> ThreadSharedData::calculateTrainingDataRange(const std::shared_ptr<BaseActor>& actor)
{
    int game_length = actor->getEnvironment().getActionHistory().size();
    int data_start = 0, data_end = game_length - 1;
    if (config::zero_actor_intermediate_sequence_length > 0) {
        data_end = std::max(0, (actor->getEnvironment().isTerminal() ? data_end : data_end - config::learner_muzero_unrolling_step - config::learner_n_step_return));
        data_start = std::max(0, (actor->getEnvironment().isTerminal() ? data_end - data_end % config::zero_actor_intermediate_sequence_length : data_end + 1 - config::zero_actor_intermediate_sequence_length));
        if (actor->getEnvironment().isTerminal() && (data_end % config::zero_actor_intermediate_sequence_length < config::learner_muzero_unrolling_step + config::learner_n_step_return)) {
            data_start = std::max(0, data_start - config::zero_actor_intermediate_sequence_length);
        }
    }
    return {data_start, data_end};
}

void SlaveThread::initialize()
{
    int seed = config::program_auto_seed ? std::random_device()() : config::program_seed + id_;
    Random::seed(seed);
}

void SlaveThread::runJob()
{
    if (getSharedData()->do_cpu_job_) {
        while (doCPUJob()) {}
    } else {
        doGPUJob();
    }
}

bool SlaveThread::doCPUJob()
{
    size_t actor_id = getSharedData()->getAvailableActorIndex();
    if (actor_id >= getSharedData()->actors_.size()) { return false; }

    std::shared_ptr<BaseActor>& actor = getSharedData()->actors_[actor_id];
    int network_id = actor->getNNEvaluationNetworkID();
    int network_output_id = actor->getNNEvaluationBatchIndex();
    if (network_output_id >= 0) {
        if (network_id < 0 || network_id >= static_cast<int>(getSharedData()->network_outputs_.size()) ||
            network_output_id >= static_cast<int>(getSharedData()->network_outputs_[network_id].size())) {
            throw std::runtime_error("pending inference output does not match its network batch");
        }
        actor->afterNNEvaluation(getSharedData()->network_outputs_[network_id][network_output_id]);
        if (actor->isSearchDone()) { handleSearchDone(actor_id); }
    }
    actor->beforeNNEvaluation();
    return true;
}

void SlaveThread::doGPUJob()
{
    if (id_ >= static_cast<int>(getSharedData()->networks_.size())) { return; }

    std::shared_ptr<Network>& network = getSharedData()->networks_[id_];
    if (network->getNetworkTypeName() == "alphazero") {
        std::shared_ptr<AlphaZeroNetwork> az_network = std::static_pointer_cast<AlphaZeroNetwork>(network);
        if (az_network->getBatchSize() > 0) { getSharedData()->network_outputs_[id_] = az_network->forward(); }
    } else if (network->getNetworkTypeName() == "muzero" || network->getNetworkTypeName() == "muzero_atari") {
        std::shared_ptr<MuZeroNetwork> muzero_network = std::static_pointer_cast<MuZeroNetwork>(network);
        if (muzero_network->getInitialInputBatchSize() > 0) {
            getSharedData()->network_outputs_[id_] = std::static_pointer_cast<MuZeroNetwork>(network)->initialInference();
        } else if (muzero_network->getRecurrentInputBatchSize() > 0) {
            getSharedData()->network_outputs_[id_] = std::static_pointer_cast<MuZeroNetwork>(network)->recurrentInference();
        }
    }
}

void SlaveThread::handleSearchDone(int actor_id)
{
    assert(actor_id >= 0 && actor_id < static_cast<int>(getSharedData()->actors_.size()) && getSharedData()->actors_[actor_id]->isSearchDone());

    std::shared_ptr<BaseActor>& actor = getSharedData()->actors_[actor_id];
    if (!actor->isResign()) { actor->act(actor->getSearchAction()); }
    bool is_endgame = (actor->isResign() || actor->isEnvTerminal());
    bool display_game = (actor_id == 0 && (config::actor_num_simulation >= 50 || (config::actor_num_simulation < 50 && is_endgame)));
    if (display_game) { std::cerr << actor->getEnvironment().toString() << actor->getSearchInfo() << std::endl; }
    if (is_endgame) {
        getSharedData()->outputGame(actor);
        actor->reset();
    } else {
        int game_length = actor->getEnvironment().getActionHistory().size();
        int sequence_length = config::zero_actor_intermediate_sequence_length;
        if (sequence_length > 0 && game_length >= sequence_length && (game_length - config::learner_n_step_return - config::learner_muzero_unrolling_step) % sequence_length == 0) { getSharedData()->outputGame(actor); }
        actor->resetSearch();
    }
}

void ActorGroup::run()
{
    initialize();
    while (true) {
        handleCommand();

        if (!running_) { continue; }
        getSharedData()->actor_index_ = 0;
        for (auto& t : slave_threads_) { t->start(); }
        for (auto& t : slave_threads_) { t->finish(); }
        getSharedData()->do_cpu_job_ = !getSharedData()->do_cpu_job_;
    }
}

void ActorGroup::initialize()
{
    const int num_gpu = std::min(static_cast<int>(torch::cuda::device_count()), config::zero_num_parallel_games);
    const int required_gpu_threads = config::zero_use_jpsro
                                         ? env::kMaxNumPlayers * num_gpu
                                         : (config::zero_use_population ? 2 * num_gpu : num_gpu);
    int num_threads = std::max(required_gpu_threads, config::zero_num_threads);
    createSlaveThreads(num_threads);
    createNeuralNetworks();
    createActors();
    running_ = false;
    getSharedData()->do_cpu_job_ = true;

    // create one thread to handle I/O
    commands_.clear();
    thread_groups_.create_thread(boost::bind(&ActorGroup::handleIO, this));

    // initialize ignored command
    std::vector<std::string> ignored_commands = utils::stringToVector(config::zero_actor_ignored_command);
    for (const auto& command : ignored_commands) { ignored_commands_.insert(command); }
}

void ActorGroup::createNeuralNetworks()
{
    int num_networks = std::min(static_cast<int>(torch::cuda::device_count()), config::zero_num_parallel_games);
    assert(num_networks > 0);
    getSharedData()->num_gpu_ = num_networks;
    getSharedData()->networks_.resize(num_networks);
    getSharedData()->network_model_paths_.resize(num_networks, config::nn_file_name);
    getSharedData()->network_outputs_.resize(num_networks);
    for (int gpu_id = 0; gpu_id < num_networks; ++gpu_id) {
        getSharedData()->networks_[gpu_id] = createNetwork(config::nn_file_name, gpu_id);
    }
}

void ActorGroup::createActors()
{
    assert(getSharedData()->networks_.size() > 0);
    std::shared_ptr<Network>& network = getSharedData()->networks_[0];
    uint64_t tree_node_size = static_cast<uint64_t>(config::actor_num_simulation + 1) * network->getActionSize();
    for (int i = 0; i < config::zero_num_parallel_games; ++i) {
        const int network_id = i % getSharedData()->num_gpu_;
        auto actor = createActor(tree_node_size, getSharedData()->networks_[network_id]);
        auto zero_actor = std::dynamic_pointer_cast<ZeroActor>(actor);
        assert(zero_actor);
        zero_actor->setPopulationNetworks(network_id, getSharedData()->networks_[network_id], -1, nullptr, -1, {});
        getSharedData()->actors_.emplace_back(actor);
    }
    if (config::zero_use_population) {
        const int num_players = getSharedData()->actors_.front()->getEnvironment().getNumPlayer();
        if (num_players <= 2 || config::nn_type_name != "alphazero") {
            throw std::runtime_error("population training requires a multiplayer AlphaZero environment");
        }
        if (config::zero_population_size <= 0 || config::zero_population_snapshot_interval <= 0 ||
            config::zero_population_rotation_interval <= 0 || config::zero_population_temperature <= 0.0f ||
            config::zero_population_hard_ratio < 0.0f || config::zero_population_hard_ratio > 1.0f ||
            config::zero_population_current_seat_min < 1 ||
            config::zero_population_current_seat_max >= num_players ||
            config::zero_population_current_seat_min > config::zero_population_current_seat_max ||
            config::nn_behavior_history_dropout < 0.0f || config::nn_behavior_history_dropout > 1.0f) {
            throw std::runtime_error("invalid multiplayer population configuration");
        }
    }
    if (config::zero_use_jpsro) {
        const int num_players = getSharedData()->actors_.front()->getEnvironment().getNumPlayer();
        if (config::zero_use_population) {
            throw std::runtime_error("zero_use_jpsro and zero_use_population are mutually exclusive");
        }
        if (num_players <= 2 || config::nn_type_name != "alphazero" || config::zero_jpsro_profile_file.empty()) {
            throw std::runtime_error("JPSRO training requires a multiplayer AlphaZero environment and a profile file");
        }
    }
}

void ActorGroup::handleIO()
{
    std::string command;
    const int buffer_size = 10000000;
    command.reserve(buffer_size);
    while (getline(std::cin, command)) {
        std::lock_guard lock(getSharedData()->mutex_);
        commands_.push_back(command);
    }
}

void ActorGroup::handleCommand()
{
    if (commands_.empty() || !getSharedData()->do_cpu_job_) { return; }

    std::lock_guard lock(getSharedData()->mutex_);
    while (!commands_.empty()) {
        const std::string command = commands_.front();
        commands_.pop_front();

        // ignore specific command
        std::string command_prefix = ((command.find(" ") == std::string::npos) ? command : command.substr(0, command.find(" ")));
        if (ignored_commands_.count(command_prefix)) {
            std::cerr << "[ignored command] " << command << std::endl;
            continue;
        }

        // do command
        handleCommand(command_prefix, command);
    }
}

void ActorGroup::handleCommand(const std::string& command_prefix, const std::string& command)
{
    if (command_prefix == "reset_actors") {
        std::cerr << "[command] " << command << std::endl;
        for (auto& actor : getSharedData()->actors_) { actor->reset(); }
        getSharedData()->do_cpu_job_ = true;
    } else if (command_prefix == "load_model") {
        std::cerr << "[command] " << command << std::endl;
        std::vector<std::string> args = utils::stringToVector(command);
        assert(args.size() == 2);
        config::nn_file_name = args[1];
        for (int network_id = 0; network_id < getSharedData()->num_gpu_; ++network_id) {
            auto& network = getSharedData()->networks_[network_id];
            network->loadModel(config::nn_file_name, network->getGPUID());
            getSharedData()->network_model_paths_[network_id] = config::nn_file_name;
        }
    } else if (command_prefix == "load_population") {
        std::cerr << "[command] " << command << std::endl;
        std::vector<std::string> args = utils::stringToVector(command);
        if (args.size() != 4) { throw std::runtime_error("load_population expects: model_path iteration seat_weights"); }
        const std::string& model_path = args[1];
        const int historical_iteration = std::stoi(args[2]);
        std::vector<float> seat_weights;
        std::stringstream weight_stream(args[3]);
        std::string weight;
        while (std::getline(weight_stream, weight, ',')) { seat_weights.push_back(std::stof(weight)); }

        const int history_offset = getSharedData()->num_gpu_;
        if (static_cast<int>(getSharedData()->networks_.size()) == history_offset) {
            for (int gpu_id = 0; gpu_id < getSharedData()->num_gpu_; ++gpu_id) {
                getSharedData()->networks_.push_back(createNetwork(model_path, gpu_id));
                getSharedData()->network_model_paths_.push_back(model_path);
                getSharedData()->network_outputs_.emplace_back();
            }
        } else {
            for (int gpu_id = 0; gpu_id < getSharedData()->num_gpu_; ++gpu_id) {
                getSharedData()->networks_[history_offset + gpu_id]->loadModel(model_path, gpu_id);
                getSharedData()->network_model_paths_[history_offset + gpu_id] = model_path;
            }
        }
        for (size_t actor_id = 0; actor_id < getSharedData()->actors_.size(); ++actor_id) {
            const int gpu_id = actor_id % getSharedData()->num_gpu_;
            auto zero_actor = std::dynamic_pointer_cast<ZeroActor>(getSharedData()->actors_[actor_id]);
            zero_actor->setPopulationNetworks(gpu_id,
                                              getSharedData()->networks_[gpu_id],
                                              history_offset + gpu_id,
                                              getSharedData()->networks_[history_offset + gpu_id],
                                              historical_iteration,
                                              seat_weights);
        }
    } else if (command_prefix == "load_profile") {
        std::cerr << "[command] " << command << std::endl;
        const std::vector<std::string> args = utils::stringToVector(command);
        if (args.size() != 5) {
            throw std::runtime_error("load_profile expects: profile_id policy_ids trainable_mask model_paths");
        }
        const std::string& profile_id = args[1];
        const std::vector<std::string> policy_ids = splitCSV(args[2]);
        const std::vector<std::string> mask_values = splitCSV(args[3]);
        const std::vector<std::string> model_paths = splitCSV(args[4]);
        const int num_players = getSharedData()->actors_.front()->getEnvironment().getNumPlayer();
        if (static_cast<int>(policy_ids.size()) != num_players ||
            static_cast<int>(mask_values.size()) != num_players ||
            static_cast<int>(model_paths.size()) != num_players) {
            throw std::runtime_error("load_profile field counts must equal the environment player count");
        }
        std::vector<bool> trainable_seats(num_players, false);
        std::unordered_map<std::string, int> path_slots;
        std::vector<std::string> frozen_paths;
        for (int player = 0; player < num_players; ++player) {
            if (mask_values[player] != "0" && mask_values[player] != "1") {
                throw std::runtime_error("load_profile trainable mask must contain only 0 or 1");
            }
            trainable_seats[player] = mask_values[player] == "1";
            if (trainable_seats[player] != (model_paths[player] == "CURRENT")) {
                throw std::runtime_error("exactly the trainable responder must use the CURRENT model");
            }
            if (model_paths[player] != "CURRENT" &&
                std::find(frozen_paths.begin(), frozen_paths.end(), model_paths[player]) == frozen_paths.end()) {
                frozen_paths.push_back(model_paths[player]);
            }
        }
        if (static_cast<int>(frozen_paths.size()) >= num_players) {
            throw std::runtime_error("JPSRO profile has more frozen models than available player slots");
        }

        const int num_gpu = getSharedData()->num_gpu_;
        std::vector<bool> used_slots(num_players, false);
        used_slots[0] = true;
        for (const std::string& model_path : frozen_paths) {
            int selected_slot = -1;
            for (int slot = 1; slot < num_players; ++slot) {
                const int network_id = slot * num_gpu;
                if (!used_slots[slot] && network_id < static_cast<int>(getSharedData()->network_model_paths_.size()) &&
                    getSharedData()->network_model_paths_[network_id] == model_path) {
                    selected_slot = slot;
                    break;
                }
            }
            if (selected_slot < 0) {
                selected_slot = static_cast<int>(
                    std::find(used_slots.begin() + 1, used_slots.end(), false) - used_slots.begin());
            }
            if (selected_slot >= num_players) { throw std::runtime_error("no free JPSRO network slot"); }
            used_slots[selected_slot] = true;
            path_slots[model_path] = selected_slot;
        }
        std::vector<std::pair<int, std::string>> slot_paths;
        for (const auto& item : path_slots) { slot_paths.emplace_back(item.second, item.first); }
        std::sort(slot_paths.begin(), slot_paths.end());
        for (const auto& item : slot_paths) {
            const int slot = item.first;
            const std::string& model_path = item.second;
            const int offset = slot * num_gpu;
            for (int gpu_id = 0; gpu_id < num_gpu; ++gpu_id) {
                const int network_id = offset + gpu_id;
                if (network_id == static_cast<int>(getSharedData()->networks_.size())) {
                    getSharedData()->networks_.push_back(createNetwork(model_path, gpu_id));
                    getSharedData()->network_model_paths_.push_back(model_path);
                    getSharedData()->network_outputs_.emplace_back();
                } else if (network_id < static_cast<int>(getSharedData()->networks_.size())) {
                    if (getSharedData()->network_model_paths_[network_id] != model_path) {
                        getSharedData()->networks_[network_id]->loadModel(model_path, gpu_id);
                        getSharedData()->network_model_paths_[network_id] = model_path;
                    }
                } else {
                    throw std::runtime_error("JPSRO network slots are not contiguous");
                }
            }
        }
        for (size_t actor_id = 0; actor_id < getSharedData()->actors_.size(); ++actor_id) {
            const int gpu_id = actor_id % num_gpu;
            std::vector<int> network_ids(num_players);
            std::vector<std::shared_ptr<Network>> networks(num_players);
            for (int player = 0; player < num_players; ++player) {
                const int slot = model_paths[player] == "CURRENT" ? 0 : path_slots.at(model_paths[player]);
                network_ids[player] = slot * num_gpu + gpu_id;
                networks[player] = getSharedData()->networks_[network_ids[player]];
            }
            auto zero_actor = std::dynamic_pointer_cast<ZeroActor>(getSharedData()->actors_[actor_id]);
            zero_actor->setPolicyProfile(profile_id, network_ids, networks, policy_ids, trainable_seats);
        }
    } else if (command_prefix == "clear_population") {
        std::cerr << "[command] " << command << std::endl;
        for (size_t actor_id = 0; actor_id < getSharedData()->actors_.size(); ++actor_id) {
            const int gpu_id = actor_id % getSharedData()->num_gpu_;
            auto zero_actor = std::dynamic_pointer_cast<ZeroActor>(getSharedData()->actors_[actor_id]);
            zero_actor->setPopulationNetworks(gpu_id, getSharedData()->networks_[gpu_id], -1, nullptr, -1, {});
        }
    } else if (command_prefix == "update_config") {
        std::cerr << "[command] " << command << std::endl;
        assert(command.find(" ") != std::string::npos);
        config::ConfigureLoader cl;
        config::setConfiguration(cl);
        if (!cl.loadFromString(command.substr(command.find(" ") + 1))) {
            std::cerr << "Failed to load configuration string." << std::endl;
            exit(0);
        }
    } else if (command_prefix == "start") {
        std::cerr << "[command] " << command << std::endl;
        running_ = true;
    } else if (command_prefix == "stop") {
        std::cerr << "[command] " << command << std::endl;
        running_ = false;
    } else if (command_prefix == "quit") {
        std::cerr << "[command] " << command << std::endl;
        exit(0);
    }
}

} // namespace minizero::actor
