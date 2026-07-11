#include "base_env.h"
#include <sstream>

namespace minizero::env {

char playerToChar(Player p)
{
    switch (p) {
        case Player::kPlayerNone: return 'N';
        case Player::kPlayer1: return 'B';
        case Player::kPlayer2: return 'W';
        case Player::kPlayer3: return 'R';
        case Player::kPlayer4: return 'G';
        case Player::kPlayer5: return 'Y';
        case Player::kPlayer6: return 'P';
        default: return '?';
    }
}

Player charToPlayer(char c)
{
    switch (c) {
        case 'N': return Player::kPlayerNone;
        case 'B':
        case 'b': return Player::kPlayer1;
        case 'W':
        case 'w': return Player::kPlayer2;
        case 'R':
        case 'r': return Player::kPlayer3;
        case 'G':
        case 'g': return Player::kPlayer4;
        case 'Y':
        case 'y': return Player::kPlayer5;
        case 'P':
        case 'p': return Player::kPlayer6;
        default: return Player::kPlayerSize;
    }
}

Player getNextPlayer(Player player, int num_player)
{
    assert(num_player >= 1 && num_player <= kMaxNumPlayers);
    const int player_id = static_cast<int>(player);
    assert(player_id >= 1 && player_id <= num_player);
    return static_cast<Player>(player_id % num_player + 1);
}

Player getPreviousPlayer(Player player, int num_player)
{
    assert(num_player >= 1 && num_player <= kMaxNumPlayers);
    const int player_id = static_cast<int>(player);
    assert(player_id >= 1 && player_id <= num_player);
    return static_cast<Player>((player_id + num_player - 2) % num_player + 1);
}

int playerToIndex(Player player)
{
    const int index = static_cast<int>(player) - 1;
    assert(index >= 0 && index < kMaxNumPlayers);
    return index;
}

Player indexToPlayer(int player_index)
{
    assert(player_index >= 0 && player_index < kMaxNumPlayers);
    return static_cast<Player>(player_index + 1);
}

std::string playerValuesToString(const PlayerValues& values, int num_player)
{
    assert(num_player >= 1 && num_player <= kMaxNumPlayers);
    std::ostringstream oss;
    for (int player_index = 0; player_index < num_player; ++player_index) {
        if (player_index > 0) { oss << ','; }
        oss << values[player_index];
    }
    return oss.str();
}

PlayerValues stringToPlayerValues(const std::string& value_string, int num_player)
{
    assert(num_player >= 1 && num_player <= kMaxNumPlayers);
    PlayerValues values{};
    std::istringstream iss(value_string);
    std::string token;
    int player_index = 0;
    while (std::getline(iss, token, ',')) {
        assert(player_index < num_player);
        values[player_index++] = std::stof(token);
    }
    assert(player_index == num_player);
    return values;
}

std::vector<float> playerValuesToVector(const PlayerValues& values, int num_player)
{
    assert(num_player >= 1 && num_player <= kMaxNumPlayers);
    return std::vector<float>(values.begin(), values.begin() + num_player);
}

} // namespace minizero::env
