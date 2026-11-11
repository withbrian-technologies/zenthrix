cmake_minimum_required(VERSION 3.25)

project(zenthrix VERSION 0.2.3 LANGUAGES CXX)

option(ZENTHRIX_BUILD_TESTS "Build native component tests" ON)

set(BUILD_TESTING "${ZENTHRIX_BUILD_TESTS}" CACHE BOOL
    "Enable CTest and native component tests" FORCE)
include(CTest)

set(ENABLE_TESTS "${ZENTHRIX_BUILD_TESTS}" CACHE BOOL
    "Build native component tests" FORCE)

add_subdirectory(core)
add_subdirectory(kernels)
