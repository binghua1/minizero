#include "connect3x3.h"
#include <cassert>
#include <vector>

namespace {

using namespace minizero::env;
using namespace minizero::env::connect3x3;

void playMoves(Connect3x3Env& env, const std::vector<int>& moves)
{
    for (int move : moves) { assert(env.act(Connect3x3Action(move, env.getTurn()))); }
}

void assertPlayer1Win(const Connect3x3Env& env)
{
    assert(env.isTerminal());
    assert(env.getLegalActions().empty());
    const PlayerValues values = env.getEvalScores();
    assert(values[0] == 1.0f && values[1] == -1.0f && values[2] == -1.0f);
}

} // namespace

int main()
{
    Connect3x3Env horizontal;
    assert(horizontal.getLegalActions().size() == 7);
    assert(horizontal.getFeatures().size() == 6 * 6 * 7);
    playMoves(horizontal, {0, 6, 5, 1, 6, 5, 2});
    assertPlayer1Win(horizontal);

    Connect3x3Env vertical;
    playMoves(vertical, {0, 6, 5, 0, 6, 5, 0});
    assertPlayer1Win(vertical);

    Connect3x3Env diagonal;
    playMoves(diagonal, {0, 1, 2, 1, 2, 6, 2});
    assertPlayer1Win(diagonal);

    Connect3x3Env full_column;
    playMoves(full_column, {0, 0, 0, 0, 0, 0});
    assert(!full_column.isTerminal());
    assert(full_column.getLegalActions().size() == 6);
    assert(!full_column.act(Connect3x3Action(0, full_column.getTurn())));

    Connect3x3Env draw;
    playMoves(draw, {
                        3,
                        4,
                        1,
                        2,
                        3,
                        0,
                        4,
                        6,
                        4,
                        3,
                        3,
                        5,
                        0,
                        2,
                        4,
                        5,
                        3,
                        6,
                        3,
                        0,
                        2,
                        4,
                        0,
                        0,
                        0,
                        5,
                        6,
                        1,
                        6,
                        1,
                        4,
                        6,
                        5,
                        6,
                        1,
                        2,
                        1,
                        1,
                        5,
                        2,
                        5,
                        2,
                    });
    assert(draw.isTerminal());
    const PlayerValues draw_values = draw.getEvalScores();
    assert(draw_values[0] == 0.0f && draw_values[1] == 0.0f && draw_values[2] == 0.0f);

    Connect3x3EnvLoader loader;
    loader.loadFromEnvironment(horizontal);
    Connect3x3EnvLoader loaded;
    assert(loaded.loadFromString(loader.toString()));
    assert(loaded.getActionPairs().size() == 7);
    assert(loaded.getValue(0) == std::vector<float>({1.0f, -1.0f, -1.0f}));
    assert(loaded.getFeatures(7) == horizontal.getFeatures());

    Connect3x3Action console_action({"b", "G"});
    assert(console_action.getActionID() == 6);
    assert(console_action.toConsoleString() == "G");
    assert(console_action.nextPlayer() == Player::kPlayer2);
    return 0;
}
