#include "blokus10.h"
#include <cassert>
#include <cmath>
#include <vector>

namespace {

bool close(float lhs, float rhs)
{
    return std::abs(lhs - rhs) < 1e-6f;
}

} // namespace

int main()
{
    using namespace minizero::env::blokus10;

    Blokus10EnvLoader loader;
    loader.addTag("SC", "40,40,30,20");

    const std::vector<float> distribution = loader.getRankDistribution(0);
    assert(distribution.size() == 16);

    // Players 1 and 2 tie across first and second place.
    assert(close(distribution[0], 0.5f));
    assert(close(distribution[1], 0.5f));
    assert(close(distribution[4], 0.5f));
    assert(close(distribution[5], 0.5f));
    assert(close(distribution[10], 1.0f));
    assert(close(distribution[15], 1.0f));

    const std::vector<float> expected_rank = loader.getRank(0);
    assert(close(expected_rank[0], 2.0f / 3.0f));
    assert(close(expected_rank[1], 2.0f / 3.0f));
    assert(close(expected_rank[2], -1.0f / 3.0f));
    assert(close(expected_rank[3], -1.0f));
    return 0;
}
