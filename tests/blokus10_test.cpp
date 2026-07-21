#include "blokus10.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

} // namespace

int main()
{
    using namespace minizero::env;
    using namespace minizero::env::blokus10;

    Blokus10Env initial_score;
    const PlayerValues values = initial_score.getEvalScores();
    require(std::fabs(values[0] + values[1] + values[2] + values[3]) < 1e-6f,
            "initial Blokus10 utilities are not zero-sum");
    require(initial_score.getPlayerScore(Player::kPlayer1) == -kBlokus10TotalSquares,
            "initial Blokus10 score must equal the negative total square count");

    Blokus10Env score_margin;
    require(score_margin.act(score_margin.getLegalActions().front()),
            "generated Blokus10 opening action is illegal");
    const PlayerValues margin_values = score_margin.getEvalScores();
    const float player1_score = static_cast<float>(score_margin.getPlayerScore(Player::kPlayer1));
    const float opponent_mean = (
        static_cast<float>(score_margin.getPlayerScore(Player::kPlayer2)) +
        static_cast<float>(score_margin.getPlayerScore(Player::kPlayer3)) +
        static_cast<float>(score_margin.getPlayerScore(Player::kPlayer4))) /
                                3.0f;
    require(std::fabs(margin_values[0] - (player1_score - opponent_mean) / 49.0f) < 1e-6f,
            "Blokus10 centered score-margin utility is incorrect");
    require(std::fabs(margin_values[0] + margin_values[1] + margin_values[2] + margin_values[3]) < 1e-6f,
            "non-terminal Blokus10 utilities are not zero-sum");

    const PlayerValues resign_values = score_margin.getEvalScores(true);
    require(std::fabs(resign_values[0] + resign_values[1] + resign_values[2] + resign_values[3]) < 1e-6f,
            "Blokus10 resignation utilities are not zero-sum");

    Blokus10EnvLoader loader;
    loader.loadFromEnvironment(score_margin);
    Blokus10EnvLoader loaded;
    require(loaded.loadFromString(loader.toString()), "Blokus10 record round-trip failed");
    require(loaded.getValue(0).size() == kBlokus10NumPlayer,
            "Blokus10 record does not preserve the four-player value vector");

    return 0;
}
