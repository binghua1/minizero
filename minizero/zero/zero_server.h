#pragma once

#include "base_server.h"
#include "configuration.h"
#include "time_system.h"
#include <array>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minizero::zero {

std::vector<float> calculatePopulationSeatBaseWeights(int min_seats, int max_seats, bool balance_seats);
float calculateLeagueUtility(float value, float rank, float rank_weight);
std::vector<float> blendPopulationSeatWeights(const std::vector<float>& base_weights, int target_index, float mixture);

struct RunningStat {
    void add(double value);
    double variance() const;
    double standardError() const;
    double lowerConfidenceBound(double scale) const;

    int count_ = 0;
    double mean_ = 0.0;
    double squared_deviation_sum_ = 0.0;
};

struct PopulationStat {
    RunningStat current_utility_;
    RunningStat historical_utility_;
    int last_seen_iteration_ = -1;
};

enum class LeagueRole {
    kSelf,
    kChampion,
    kFrontier,
    kHard,
    kCoverage,
    kDeviation,
};

class ZeroLogger {
public:
    ZeroLogger() {}
    void createLog();

    inline void addWorkerLog(const std::string& log_str) { addLog(log_str, worker_log_); }
    inline void addTrainingLog(const std::string& log_str) { addLog(log_str, training_log_); }
    inline std::fstream& getSelfPlayFileStream() { return self_play_game_; }

private:
    void addLog(const std::string& log_str, std::fstream& log_file);

    std::fstream worker_log_;
    std::fstream training_log_;
    std::fstream self_play_game_;
};

class ZeroSelfPlayData {
public:
    bool is_terminal_;
    int data_length_;
    int game_length_;
    float return_;
    std::vector<float> returns_;
    std::string game_record_;

    ZeroSelfPlayData() {}
    ZeroSelfPlayData(std::string input_data);
};

class ZeroWorkerSharedData {
public:
    ZeroWorkerSharedData(boost::mutex& worker_mutex)
        : worker_mutex_(worker_mutex)
    {
    }

    bool getSelfPlayData(ZeroSelfPlayData& sp_data);
    bool isOptimizationPahse();
    int getModelIetration();

    bool is_optimization_phase_;
    int num_op_worker_;
    int total_games_;
    int model_iteration_;
    ZeroLogger logger_;
    std::string updated_conf_str_;
    std::queue<ZeroSelfPlayData> sp_data_queue_;
    boost::mutex mutex_;
    boost::mutex& worker_mutex_;
};

class ZeroWorkerHandler : public utils::ConnectionHandler {
public:
    ZeroWorkerHandler(boost::asio::io_service& io_service, ZeroWorkerSharedData& shared_data)
        : ConnectionHandler(io_service),
          is_idle_(false),
          shared_data_(shared_data)
    {
    }

    void handleReceivedMessage(const std::string& message) override;
    void close() override;
    void syncConfig();

    inline bool isIdle() const { return is_idle_; }
    inline std::string getName() const { return name_; }
    inline std::string getType() const { return type_; }
    inline void setIdle(bool is_idle) { is_idle_ = is_idle; }

private:
    bool is_idle_;
    std::string name_;
    std::string type_;
    ZeroWorkerSharedData& shared_data_;
};

class ZeroServer : public utils::BaseServer<ZeroWorkerHandler> {
public:
    ZeroServer()
        : BaseServer(minizero::config::zero_server_port),
          shared_data_(worker_mutex_),
          keep_alive_timer_(io_service_)
    {
        startKeepAlive();
    }

    virtual void run();
    boost::shared_ptr<ZeroWorkerHandler> handleAcceptNewConnection() override { return boost::make_shared<ZeroWorkerHandler>(io_service_, shared_data_); }
    void sendInitialMessage(boost::shared_ptr<ZeroWorkerHandler> connection) override {}

protected:
    virtual void initialize();
    virtual void selfPlay();
    virtual void broadcastSelfPlayJob();
    virtual void optimization();
    virtual std::string getUpdatedConfig();
    std::vector<int> getPopulationIterations();
    std::vector<float> getPopulationSeatWeights(int historical_iteration, LeagueRole role = LeagueRole::kHard) const;
    void refreshLeaguePool();
    LeagueRole chooseLeagueRole();
    int selectLeagueOpponent(LeagueRole role) const;
    int selectRestrictedDeviationOpponent() const;
    void maybePromoteChampion();
    void logPopulationStatistics();
    void recordPopulationResult(const ZeroSelfPlayData& sp_data);
    void syncConfig();
    void stopJob(const std::string& job_type);
    void close();
    void keepAlive();
    void startKeepAlive();

    int iteration_;
    ZeroWorkerSharedData shared_data_;
    boost::asio::deadline_timer keep_alive_timer_;

    std::vector<int> latest_game_lengths_;
    std::vector<float> latest_game_returns_;
    std::vector<std::vector<float>> latest_game_player_returns_;
    std::unordered_map<int64_t, PopulationStat> population_stats_;
    std::unordered_map<int64_t, PopulationStat> iteration_population_stats_;
    std::unordered_map<std::string, int> iteration_role_game_counts_;
    std::vector<int> active_population_iterations_;
    std::array<int64_t, 5> league_role_assignment_counts_{};
    int league_last_refresh_iteration_ = -1;
    int champion_iteration_ = -1;
};

} // namespace minizero::zero
