#include "zero_server.h"
#include "environment.h"
#include "git_info.h"
#include "random.h"
#include "utils.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace minizero::zero {

using namespace minizero;
using namespace minizero::utils;

namespace {

void logReturnSummary(ZeroLogger& logger, const std::string& item, const std::vector<float>& returns)
{
    logger.addTrainingLog("[SelfPlay Min. " + item + "] " + std::to_string(*std::min_element(returns.begin(), returns.end())));
    logger.addTrainingLog("[SelfPlay Max. " + item + "] " + std::to_string(*std::max_element(returns.begin(), returns.end())));
    logger.addTrainingLog("[SelfPlay Avg. " + item + "] " + std::to_string(std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size()));
    logger.addTrainingLog("[SelfPlay Std. " + item + "] " + std::to_string(utils::stddev(returns)));
}

void logMultiplayerReturnStatistics(ZeroLogger& logger, const std::vector<std::vector<float>>& game_returns, const std::string& item_prefix = "")
{
    assert(!game_returns.empty() && game_returns.front().size() > 1);
    const int num_players = game_returns.front().size();
    std::vector<std::vector<float>> player_returns(num_players);
    std::vector<int> player_wins(num_players, 0);
    int draws = 0;
    for (const auto& returns : game_returns) {
        if (static_cast<int>(returns.size()) != num_players) { throw std::runtime_error("inconsistent multiplayer return vector size"); }
        for (int player_index = 0; player_index < num_players; ++player_index) { player_returns[player_index].push_back(returns[player_index]); }

        const float best_return = *std::max_element(returns.begin(), returns.end());
        int best_player = -1;
        int num_best_players = 0;
        for (int player_index = 0; player_index < num_players; ++player_index) {
            if (returns[player_index] != best_return) { continue; }
            best_player = player_index;
            ++num_best_players;
        }
        if (num_best_players == 1) {
            ++player_wins[best_player];
        } else {
            ++draws;
        }
    }

    for (int player_index = 0; player_index < num_players; ++player_index) {
        const std::string player = "P" + std::to_string(player_index + 1);
        logReturnSummary(logger, item_prefix + player + " Returns", player_returns[player_index]);
        if (item_prefix.empty()) {
            logger.addTrainingLog("[SelfPlay Avg. " + player + " Win Rate] " + std::to_string(static_cast<float>(player_wins[player_index]) / game_returns.size()));
        }
    }
    if (item_prefix.empty()) { logger.addTrainingLog("[SelfPlay Avg. Draw Rate] " + std::to_string(static_cast<float>(draws) / game_returns.size())); }
}

} // namespace

void ZeroLogger::createLog()
{
    std::string worker_file_name = config::zero_training_directory + "/Worker.log";
    std::string training_file_name = config::zero_training_directory + "/Training.log";
    worker_log_.open(worker_file_name.c_str(), std::ios::out | std::ios::app);
    training_log_.open(training_file_name.c_str(), std::ios::out | std::ios::app);

    for (int i = 0; i < 100; ++i) {
        worker_log_ << "=";
        training_log_ << "=";
    }
    worker_log_ << std::endl;
    training_log_ << std::endl;
    addTrainingLog("[Version] " + std::string(GIT_SHORT_HASH));
}

void ZeroLogger::addLog(const std::string& log_str, std::fstream& log_file)
{
    log_file << TimeSystem::getTimeString("[Y/m/d_H:i:s.f] ") << log_str << std::endl;
    std::cerr << TimeSystem::getTimeString("[Y/m/d_H:i:s.f] ") << log_str << std::endl;
}

ZeroSelfPlayData::ZeroSelfPlayData(std::string input_data)
{
    // format: Selfplay is_terminal data_length game_length return game_record
    input_data = input_data.substr(input_data.find(" ") + 1); // remove Selfplay
    is_terminal_ = (input_data.substr(0, input_data.find(" ")) == "true");
    input_data = input_data.substr(input_data.find(" ") + 1); // remove is_terminal
    data_length_ = std::stoi(input_data.substr(0, input_data.find(" ")));
    input_data = input_data.substr(input_data.find(" ") + 1); // remove data_length
    game_length_ = std::stoi(input_data.substr(0, input_data.find(" ")));
    input_data = input_data.substr(input_data.find(" ") + 1); // remove game_length
    const std::string return_string = input_data.substr(0, input_data.find(" "));
    std::istringstream return_stream(return_string);
    std::string return_token;
    while (std::getline(return_stream, return_token, ',')) { returns_.push_back(std::stof(return_token)); }
    if (returns_.empty()) { throw std::runtime_error("missing self-play return"); }
    return_ = returns_.front();
    input_data = input_data.substr(input_data.find(" ") + 1); // remove return
    game_record_ = input_data.substr(0, input_data.find(" "));
}

bool ZeroWorkerSharedData::getSelfPlayData(ZeroSelfPlayData& sp_data)
{
    if (sp_data_queue_.empty()) { return false; }

    boost::lock_guard<boost::mutex> lock(mutex_);
    if (sp_data_queue_.empty()) { return false; }
    sp_data = sp_data_queue_.front();
    sp_data_queue_.pop();
    return true;
}

bool ZeroWorkerSharedData::isOptimizationPahse()
{
    boost::lock_guard<boost::mutex> lock(mutex_);
    return is_optimization_phase_;
}

int ZeroWorkerSharedData::getModelIetration()
{
    boost::lock_guard<boost::mutex> lock(mutex_);
    return model_iteration_;
}

void ZeroWorkerHandler::handleReceivedMessage(const std::string& message)
{
    std::vector<std::string> args;
    boost::split(args, message, boost::is_any_of(" "), boost::token_compress_on);

    if (args[0] == "Info") {
        name_ = args[1];
        type_ = args[2];
        boost::lock_guard<boost::mutex> lock(shared_data_.worker_mutex_);
        shared_data_.logger_.addWorkerLog("[Worker Connection] " + getName() + " " + getType());
        if (type_ == "sp") {
            std::string job_command = "";
            job_command += "Job_SelfPlay ";
            job_command += config::zero_training_directory + " ";
            job_command += "nn_file_name=" + config::zero_training_directory + "/model/weight_iter_" + std::to_string(shared_data_.getModelIetration()) + ".pt";
            job_command += ":program_auto_seed=false:program_seed=" + std::to_string(utils::Random::randInt());
            write(job_command);
            syncConfig();
        } else if (type_ == "op") {
            if (shared_data_.num_op_worker_ >= 1) {
                shared_data_.logger_.addWorkerLog("[Worker Error] Receive multiple op workers");
                shared_data_.logger_.addWorkerLog("[Worker Disconnection] " + getName() + " " + getType());
                ConnectionHandler::close();
            } else {
                ++shared_data_.num_op_worker_;
                write("Job_Optimization " + config::zero_training_directory);
                syncConfig();
            }
        } else {
            shared_data_.logger_.addWorkerLog("[Worker Disconnection] " + getName() + " " + getType());
            ConnectionHandler::close();
        }
        is_idle_ = true;
    } else if (args[0] == "SelfPlay") {
        if (message.find("SelfPlay", message.find("SelfPlay", 0) + 1) != std::string::npos || message.back() != '#') {
            shared_data_.logger_.addWorkerLog("[Worker Error] Receive broken self-play games");
            return;
        }

        ZeroSelfPlayData sp_data(message); // create data before lock for efficiency
        boost::lock_guard<boost::mutex> lock(shared_data_.mutex_);
        shared_data_.sp_data_queue_.push(sp_data);

        // print number of games if the queue already received many games in buffer
        if (shared_data_.sp_data_queue_.size() % std::max(1, static_cast<int>(config::zero_num_games_per_iteration * 0.25)) == 0) {
            shared_data_.logger_.addTrainingLog("[SelfPlay Game Buffer] " + std::to_string(shared_data_.sp_data_queue_.size()) + " games");
        }
    } else if (args[0] == "Optimization_Done") {
        boost::lock_guard<boost::mutex> lock(shared_data_.mutex_);
        shared_data_.model_iteration_ = stoi(args[1]);
        shared_data_.is_optimization_phase_ = false;
    } else if (args[0] == "Log") {
        shared_data_.logger_.addWorkerLog("[Log] " + getName() + " " + getType() + ": " + message.substr(message.find(" ") + 1));
    } else {
        std::string error_message = message;
        std::replace(error_message.begin(), error_message.end(), '\r', ' ');
        std::replace(error_message.begin(), error_message.end(), '\n', ' ');
        if (getName() != "") {
            shared_data_.logger_.addWorkerLog("[Worker Error] \"" + error_message + "\"");
        } else {
            shared_data_.logger_.addWorkerLog("[Ignore Connection] \"" + error_message + "\"");
        }
        close();
    }
}

void ZeroWorkerHandler::close()
{
    if (isClosed()) { return; }

    boost::lock_guard<boost::mutex> lock(shared_data_.worker_mutex_);
    shared_data_.logger_.addWorkerLog("[Worker Disconnection] " + getName() + " " + getType());
    ConnectionHandler::close();
    if (getType() == "op") { --shared_data_.num_op_worker_; }
}

void ZeroWorkerHandler::syncConfig()
{
    if (shared_data_.updated_conf_str_.empty()) { return; }
    write("update_config " + shared_data_.updated_conf_str_);
}

void ZeroServer::run()
{
    initialize();
    startAccept();
    std::cerr << TimeSystem::getTimeString("[Y/m/d_H:i:s.f] ") << "Server initialize over." << std::endl;

    for (iteration_ = config::zero_start_iteration; iteration_ <= config::zero_end_iteration; ++iteration_) {
        syncConfig();
        selfPlay();
        optimization();
    }

    close();
}

void ZeroServer::initialize()
{
    int seed = config::program_auto_seed ? static_cast<int>(time(NULL)) : config::program_seed;
    utils::Random::seed(seed);
    shared_data_.logger_.createLog();

    std::string nn_file_name = config::nn_file_name;
    nn_file_name = nn_file_name.substr(nn_file_name.find("weight_iter_") + std::string("weight_iter_").size());
    nn_file_name = nn_file_name.substr(0, nn_file_name.find("."));
    shared_data_.num_op_worker_ = 0;
    shared_data_.model_iteration_ = stoi(nn_file_name);
    shared_data_.updated_conf_str_ = getUpdatedConfig();
}

void ZeroServer::selfPlay()
{
    // setup
    std::string self_play_file_name = config::zero_training_directory + "/sgf/" + std::to_string(iteration_) + ".sgf";
    if (config::zero_num_games_per_iteration > 0) { shared_data_.logger_.getSelfPlayFileStream().open(self_play_file_name.c_str(), std::ios::out); }
    shared_data_.logger_.addTrainingLog("[Iteration] =====" + std::to_string(iteration_) + "=====");
    shared_data_.logger_.addTrainingLog("[SelfPlay] Start " + std::to_string(shared_data_.getModelIetration()));

    std::vector<int> game_lengths;
    std::vector<float> game_returns;
    std::vector<std::vector<float>> game_player_returns;
    int num_collect_game = 0, total_data_length = 0;
    while (num_collect_game < config::zero_num_games_per_iteration) {
        broadcastSelfPlayJob();

        // read one selfplay game
        ZeroSelfPlayData sp_data;
        if (!shared_data_.getSelfPlayData(sp_data)) {
            boost::this_thread::sleep(boost::posix_time::milliseconds(100));
            continue;
        } else if (!config::zero_server_accept_different_model_games && sp_data.game_record_.find("weight_iter_" + std::to_string(shared_data_.getModelIetration())) == std::string::npos) {
            // discard previous self-play games
            continue;
        }

        // save record
        shared_data_.logger_.getSelfPlayFileStream() << sp_data.game_record_ << (sp_data.is_terminal_ ? " #" : "") << std::endl;
        ++num_collect_game;
        total_data_length += sp_data.data_length_;
        if (sp_data.is_terminal_ && config::zero_use_population) { recordPopulationResult(sp_data); }
        if (sp_data.is_terminal_) {
            game_lengths.push_back(sp_data.game_length_);
            if (sp_data.returns_.size() > 1) {
                game_player_returns.push_back(sp_data.returns_);
            } else {
                game_returns.push_back(sp_data.return_);
            }
            if (config::zero_display_latest_games > 0) {
                latest_game_lengths_.push_back(sp_data.game_length_);
                if (sp_data.returns_.size() > 1) {
                    latest_game_player_returns_.push_back(sp_data.returns_);
                } else {
                    latest_game_returns_.push_back(sp_data.return_);
                }
                if (static_cast<int>(latest_game_lengths_.size()) > config::zero_display_latest_games) {
                    latest_game_lengths_.erase(latest_game_lengths_.begin());
                    if (!latest_game_player_returns_.empty()) {
                        latest_game_player_returns_.erase(latest_game_player_returns_.begin());
                    } else {
                        latest_game_returns_.erase(latest_game_returns_.begin());
                    }
                }
            }
        }

        // display progress
        if (num_collect_game % std::max(1, static_cast<int>(config::zero_num_games_per_iteration * 0.25)) == 0) {
            shared_data_.logger_.addTrainingLog("[SelfPlay Progress] " +
                                                std::to_string(num_collect_game) + " / " +
                                                std::to_string(config::zero_num_games_per_iteration));
        }
    }

    stopJob("sp");
    if (config::zero_num_games_per_iteration > 0) { shared_data_.logger_.getSelfPlayFileStream().close(); }
    shared_data_.logger_.addTrainingLog("[SelfPlay] Finished.");
    if (config::zero_use_population) {
        for (const auto& item : population_seat_return_stats_) {
            const int historical_iteration = static_cast<int>(item.first >> 32);
            const int num_current = static_cast<int>(item.first & 0xffffffffULL);
            shared_data_.logger_.addTrainingLog(
                "[Population Cumulative Avg. Return] opponent=" + std::to_string(historical_iteration) +
                " current_seats=" + std::to_string(num_current) +
                " games=" + std::to_string(item.second.second) +
                " return=" + std::to_string(item.second.first / item.second.second));
        }
    }
    if (!game_lengths.empty()) {
        shared_data_.logger_.addTrainingLog("[SelfPlay # Finished Games] " + std::to_string(game_lengths.size()));
        shared_data_.logger_.addTrainingLog("[SelfPlay Min. Game Lengths] " + std::to_string(*std::min_element(game_lengths.begin(), game_lengths.end())));
        shared_data_.logger_.addTrainingLog("[SelfPlay Max. Game Lengths] " + std::to_string(*std::max_element(game_lengths.begin(), game_lengths.end())));
        shared_data_.logger_.addTrainingLog("[SelfPlay Avg. Game Lengths] " + std::to_string(std::accumulate(game_lengths.begin(), game_lengths.end(), 0.0f) / game_lengths.size()));
        shared_data_.logger_.addTrainingLog("[SelfPlay Std. Game Lengths] " + std::to_string(utils::stddev(game_lengths)));
        if (!game_player_returns.empty()) {
            logMultiplayerReturnStatistics(shared_data_.logger_, game_player_returns);
        } else {
            shared_data_.logger_.addTrainingLog("[SelfPlay Min. Game Returns] " + std::to_string(*std::min_element(game_returns.begin(), game_returns.end())));
            shared_data_.logger_.addTrainingLog("[SelfPlay Max. Game Returns] " + std::to_string(*std::max_element(game_returns.begin(), game_returns.end())));
            shared_data_.logger_.addTrainingLog("[SelfPlay Avg. Game Returns] " + std::to_string(std::accumulate(game_returns.begin(), game_returns.end(), 0.0f) / game_returns.size()));
            shared_data_.logger_.addTrainingLog("[SelfPlay Std. Game Returns] " + std::to_string(utils::stddev(game_returns)));
        }
        if (config::zero_display_latest_games > 0 && static_cast<int>(latest_game_lengths_.size()) == config::zero_display_latest_games) {
            std::string n = std::to_string(config::zero_display_latest_games);
            shared_data_.logger_.addTrainingLog("[SelfPlay Min. Latest " + n + " Game Lengths] " + std::to_string(*std::min_element(latest_game_lengths_.begin(), latest_game_lengths_.end())));
            shared_data_.logger_.addTrainingLog("[SelfPlay Max. Latest " + n + " Game Lengths] " + std::to_string(*std::max_element(latest_game_lengths_.begin(), latest_game_lengths_.end())));
            shared_data_.logger_.addTrainingLog("[SelfPlay Avg. Latest " + n + " Game Lengths] " + std::to_string(std::accumulate(latest_game_lengths_.begin(), latest_game_lengths_.end(), 0.0f) / latest_game_lengths_.size()));
            shared_data_.logger_.addTrainingLog("[SelfPlay Std. Latest " + n + " Game Lengths] " + std::to_string(utils::stddev(latest_game_lengths_)));
            if (!latest_game_player_returns_.empty()) {
                logMultiplayerReturnStatistics(shared_data_.logger_, latest_game_player_returns_, "Latest " + n + " ");
            } else {
                shared_data_.logger_.addTrainingLog("[SelfPlay Min. Latest " + n + " Game Returns] " + std::to_string(*std::min_element(latest_game_returns_.begin(), latest_game_returns_.end())));
                shared_data_.logger_.addTrainingLog("[SelfPlay Max. Latest " + n + " Game Returns] " + std::to_string(*std::max_element(latest_game_returns_.begin(), latest_game_returns_.end())));
                shared_data_.logger_.addTrainingLog("[SelfPlay Avg. Latest " + n + " Game Returns] " + std::to_string(std::accumulate(latest_game_returns_.begin(), latest_game_returns_.end(), 0.0f) / latest_game_returns_.size()));
                shared_data_.logger_.addTrainingLog("[SelfPlay Std. Latest " + n + " Game Returns] " + std::to_string(utils::stddev(latest_game_returns_)));
            }
        }
    }

    if (static_cast<int>(game_lengths.size()) != num_collect_game) { shared_data_.logger_.addTrainingLog("[SelfPlay Avg. Data Lengths] " + std::to_string(total_data_length * 1.0f / num_collect_game)); }
}

void ZeroServer::broadcastSelfPlayJob()
{
    const std::vector<int> population_iterations = getPopulationIterations();
    boost::lock_guard<boost::mutex> lock(worker_mutex_);
    size_t self_play_worker_index = 0;
    for (auto& worker : connections_) {
        if (!worker->isIdle() || worker->getType() != "sp") { continue; }
        worker->setIdle(false);
        worker->write("load_model " + config::zero_training_directory + "/model/weight_iter_" + std::to_string(shared_data_.getModelIetration()) + ".pt");
        if (!population_iterations.empty()) {
            const size_t rotation = iteration_ / std::max(1, config::zero_population_rotation_interval);
            const size_t opponent_index = (self_play_worker_index + rotation) % population_iterations.size();
            const int opponent_iteration = population_iterations[opponent_index];
            const std::vector<float> seat_weights = getPopulationSeatWeights(opponent_iteration);
            std::ostringstream weight_stream;
            for (size_t i = 0; i < seat_weights.size(); ++i) {
                if (i > 0) { weight_stream << ","; }
                weight_stream << seat_weights[i];
            }
            worker->write("load_population " + config::zero_training_directory + "/model/weight_iter_" +
                          std::to_string(opponent_iteration) + ".pt " + std::to_string(opponent_iteration) + " " + weight_stream.str());
        } else {
            worker->write("clear_population");
        }
        worker->write("reset_actors");
        worker->write("start");
        ++self_play_worker_index;
    }
}

std::vector<int> ZeroServer::getPopulationIterations()
{
    std::vector<int> iterations;
    if (!config::zero_use_population || config::zero_population_size <= 0) { return iterations; }
    const int current = shared_data_.getModelIetration();
    const int interval = std::max(1, config::zero_population_snapshot_interval) *
                         std::max(1, config::learner_training_step);
    for (int candidate = current - interval;
         candidate >= 0 && static_cast<int>(iterations.size()) < config::zero_population_size;
         candidate -= interval) {
        const std::string path = config::zero_training_directory + "/model/weight_iter_" + std::to_string(candidate) + ".pt";
        if (std::filesystem::exists(path)) { iterations.push_back(candidate); }
    }
    return iterations;
}

std::vector<float> ZeroServer::getPopulationSeatWeights(int historical_iteration) const
{
    const int min_seats = config::zero_population_current_seat_min;
    const int max_seats = config::zero_population_current_seat_max;
    const int count = std::max(1, max_seats - min_seats + 1);
    std::vector<float> hard_weights(count, 1.0f);
    bool has_statistics = false;
    for (int i = 0; i < count; ++i) {
        const int64_t key = (static_cast<int64_t>(historical_iteration) << 32) |
                            static_cast<uint32_t>(min_seats + i);
        auto it = population_seat_return_stats_.find(key);
        if (it == population_seat_return_stats_.end() || it->second.second == 0) { continue; }
        const double mean_return = it->second.first / it->second.second;
        hard_weights[i] = std::exp(static_cast<float>(-mean_return / std::max(1e-6f, config::zero_population_temperature)));
        has_statistics = true;
    }
    if (!has_statistics) { return std::vector<float>(count, 1.0f / count); }

    const float hard_sum = std::accumulate(hard_weights.begin(), hard_weights.end(), 0.0f);
    const float hard_ratio = std::clamp(config::zero_population_hard_ratio, 0.0f, 1.0f);
    for (float& weight : hard_weights) {
        weight = (1.0f - hard_ratio) / count + hard_ratio * weight / hard_sum;
    }
    return hard_weights;
}

void ZeroServer::recordPopulationResult(const ZeroSelfPlayData& sp_data)
{
    EnvironmentLoader env_loader;
    if (!env_loader.loadFromString(sp_data.game_record_)) { return; }
    const std::string lineup_string = env_loader.getTag("LM");
    const std::string historical_iteration_string = env_loader.getTag("HI");
    if (lineup_string.empty() || historical_iteration_string.empty()) { return; }

    std::istringstream lineup_stream(lineup_string);
    std::string token;
    int player_index = 0;
    int num_current = 0;
    double current_return_sum = 0.0;
    while (std::getline(lineup_stream, token, ',')) {
        if (token == "0" && player_index < static_cast<int>(sp_data.returns_.size())) {
            ++num_current;
            current_return_sum += sp_data.returns_[player_index];
        }
        ++player_index;
    }
    if (num_current == 0) { return; }
    const int historical_iteration = std::stoi(historical_iteration_string);
    const int64_t key = (static_cast<int64_t>(historical_iteration) << 32) |
                        static_cast<uint32_t>(num_current);
    auto& stat = population_seat_return_stats_[key];
    stat.first += current_return_sum / num_current;
    ++stat.second;
}

void ZeroServer::optimization()
{
    shared_data_.logger_.addTrainingLog("[Optimization] Start.");

    std::string job_command = "train ";
    job_command += "weight_iter_" + std::to_string(shared_data_.getModelIetration()) + ".pkl";
    job_command += " " + std::to_string(std::max(1, iteration_ - config::zero_replay_buffer + 1));
    job_command += " " + std::to_string(iteration_);

    shared_data_.is_optimization_phase_ = true;
    while (shared_data_.isOptimizationPahse()) {
        boost::lock_guard<boost::mutex> lock(worker_mutex_);
        for (auto worker : connections_) {
            if (!worker->isIdle() || worker->getType() != "op") { continue; }
            worker->setIdle(false);
            worker->write(job_command);
        }
    }
    stopJob("op");

    shared_data_.logger_.addTrainingLog("[Optimization] Finished.");
}

std::string ZeroServer::getUpdatedConfig()
{
    std::string job_command = "";
    if (config::learner_use_per && config::learner_per_beta_anneal) {
        float per_beta = std::min(config::learner_per_init_beta + (iteration_ * 1.0f / config::zero_end_iteration) * (1.0f - config::learner_per_init_beta), 1.0f);
        job_command += "learner_per_init_beta=" + std::to_string(per_beta) + ":";
    }
    if (config::actor_select_action_softmax_temperature_decay) {
        float training_progress = iteration_ * 1.0f / config::zero_end_iteration;
        float temperature = (training_progress < 0.5 ? 1.0f : (training_progress < 0.75 ? 0.5f : 0.25f));
        job_command += "actor_select_action_softmax_temperature=" + std::to_string(temperature) + ":";
    }
    if (!job_command.empty()) { job_command.pop_back(); } // remove last ":"
    return job_command;
}

void ZeroServer::syncConfig()
{
    shared_data_.updated_conf_str_ = getUpdatedConfig();

    boost::lock_guard<boost::mutex> lock(worker_mutex_);
    for (auto worker : connections_) { worker->syncConfig(); }
}

void ZeroServer::stopJob(const std::string& job_type)
{
    boost::lock_guard<boost::mutex> lock(worker_mutex_);
    for (auto worker : connections_) {
        if (worker->getType() != job_type) { continue; }
        if (job_type == "sp") { worker->write("stop"); }
        worker->setIdle(true);
    }
}

void ZeroServer::close()
{
    boost::lock_guard<boost::mutex> lock(worker_mutex_);
    for (auto worker : connections_) { worker->write("quit"); }
    exit(0);
}

void ZeroServer::keepAlive()
{
    boost::lock_guard<boost::mutex> lock(worker_mutex_);
    for (auto worker : connections_) {
        worker->write("keep_alive");
    }
    startKeepAlive();
}

void ZeroServer::startKeepAlive()
{
    keep_alive_timer_.expires_from_now(boost::posix_time::minutes(1));
    keep_alive_timer_.async_wait(boost::bind(&ZeroServer::keepAlive, this));
}

} // namespace minizero::zero
