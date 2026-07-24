if(GAME_TYPE STREQUAL "TICTACMO")
    add_executable(multiplayer_test multiplayer_test.cpp)
    target_link_libraries(multiplayer_test actor environment zero config utils)
    add_test(NAME multiplayer_test COMMAND multiplayer_test)

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
        NAME multiplayer_eval_test
        COMMAND ${Python3_EXECUTABLE} -m unittest -v tests/test_multiplayer_eval.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
endif()

if(GAME_TYPE STREQUAL "CONNECT3X3")
    add_executable(connect3x3_test connect3x3_test.cpp)
    target_link_libraries(connect3x3_test environment config utils)
    add_test(NAME connect3x3_test COMMAND connect3x3_test)
endif()

if(GAME_TYPE STREQUAL "BLOKUS")
    add_executable(blokus_test blokus_test.cpp)
    target_link_libraries(blokus_test environment config utils)
    add_test(NAME blokus_test COMMAND blokus_test)
endif()

if(GAME_TYPE STREQUAL "BLOKUS15")
    add_executable(blokus15_test blokus15_test.cpp)
    target_link_libraries(blokus15_test environment config utils)
    add_test(NAME blokus15_test COMMAND blokus15_test)
endif()
