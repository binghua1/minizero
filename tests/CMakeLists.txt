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
