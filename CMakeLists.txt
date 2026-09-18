cmake_minimum_required(VERSION 3.25)

project(zenthrix_kernels VERSION 0.1.1 LANGUAGES CXX)

option(ENABLE_TESTS "Build portable kernel tests" ON)

add_library(zenthrix_kernels
  src/common/kernel_registry.cpp
  src/common/backend_seams.cpp
)
target_compile_features(zenthrix_kernels PUBLIC cxx_std_20)
target_include_directories(zenthrix_kernels PUBLIC include)
target_compile_options(zenthrix_kernels PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(ENABLE_TESTS)
  enable_testing()
  add_executable(zenthrix_kernels_tests tests/unit/test_registry.cpp)
  target_link_libraries(zenthrix_kernels_tests PRIVATE zenthrix_kernels)
  add_test(NAME zenthrix_kernels_tests COMMAND zenthrix_kernels_tests)
endif()
