cmake_minimum_required(VERSION 3.16)

project(tnie VERSION 0.2.0 LANGUAGES C)

option(TNIE_BUILD_TESTS "Build the TNIE test suite" ON)

add_library(tnie
    src/tnie_nn.c
    src/tnie_activations.c
)

add_library(TNIE::tnie ALIAS tnie)

target_include_directories(tnie
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(tnie PUBLIC c_std_11)
set_target_properties(tnie PROPERTIES C_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(tnie PRIVATE /W4)
else()
    target_compile_options(tnie PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(tnie PUBLIC m)
endif()

add_executable(tnie_xor_demo
    examples/xor/main.c
    examples/xor/model_xor.c
)
target_include_directories(tnie_xor_demo PRIVATE examples/xor)
target_link_libraries(tnie_xor_demo PRIVATE TNIE::tnie)

if(TNIE_BUILD_TESTS)
    enable_testing()

    add_executable(tnie_tests
        tests/test_tnie.c
        examples/xor/model_xor.c
    )
    target_include_directories(tnie_tests PRIVATE examples/xor)
    target_link_libraries(tnie_tests PRIVATE TNIE::tnie)

    add_test(NAME tnie_tests COMMAND tnie_tests)
endif()
