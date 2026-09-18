cmake_minimum_required(VERSION 3.25)

project(zenthrix_core VERSION 0.1.2 LANGUAGES CXX)

option(ENABLE_TESTS "Build unit tests" ON)

add_library(zenthrix_core
  src/core/graph.cpp
  src/core/graph_order.cpp
  src/core/tensor.cpp
  src/passes/pass_manager.cpp
  src/passes/dead_code_elimination.cpp
  src/memory/lifetime_analysis.cpp
  src/memory/memory_planner.cpp
)
target_compile_features(zenthrix_core PUBLIC cxx_std_20)
target_include_directories(zenthrix_core PUBLIC include)
target_compile_options(zenthrix_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(ENABLE_TESTS)
  enable_testing()
  add_executable(zenthrix_core_tests tests/unit/test_graph.cpp)
  target_link_libraries(zenthrix_core_tests PRIVATE zenthrix_core)
  add_test(NAME zenthrix_core_tests COMMAND zenthrix_core_tests)
endif()
