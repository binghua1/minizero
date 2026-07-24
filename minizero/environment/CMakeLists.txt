file(GLOB_RECURSE SRCS *.cpp)

add_library(environment ${SRCS})
target_include_directories(
    environment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    base
    amazons
    atari
    blokus
    blokus10
    blokus15
    breakthrough
    clobber
    conhex
    connect3x3
    connect6
    dotsandboxes
    go
    go3
    gomoku
    havannah
    hex
    killallgo
    linesofaction
    nogo
    othello
    rubiks
    santorini
    surakarta
    tictacmo
    tictactoe
    stochastic
    shogi66
    stochastic/puzzle2048
    stochastic/tetrisblockpuzzle
)
target_link_libraries(
    environment
    config
    utils
    ale::ale-lib
    ${OpenCV_LIBS}
)
