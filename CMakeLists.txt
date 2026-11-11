cmake_minimum_required(VERSION 3.25)

project(zenthrix VERSION 0.2.3 LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ZENTHRIX_BUILD_TESTS "Build native component tests" ON)

set(BUILD_TESTING "${ZENTHRIX_BUILD_TESTS}" CACHE BOOL
    "Enable CTest and native component tests" FORCE)
include(CTest)

set(ENABLE_TESTS "${ZENTHRIX_BUILD_TESTS}" CACHE BOOL
    "Build native component tests" FORCE)

add_subdirectory(kernels)
add_subdirectory(core)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/zenthrix_export.cpp" "/* zenthrix shared library entry */\n")
add_library(zenthrix SHARED "${CMAKE_CURRENT_BINARY_DIR}/zenthrix_export.cpp")

if(APPLE)
  target_link_libraries(zenthrix PRIVATE -Wl,-all_load zenthrix_core zenthrix_kernels)
elseif(MSVC)
  target_link_libraries(zenthrix PRIVATE /WHOLEARCHIVE:zenthrix_core /WHOLEARCHIVE:zenthrix_kernels)
else()
  target_link_libraries(zenthrix PRIVATE -Wl,--whole-archive zenthrix_core zenthrix_kernels -Wl,--no-whole-archive)
endif()

target_compile_features(zenthrix PUBLIC cxx_std_20)
target_include_directories(zenthrix PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/core/include
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include
)
set_target_properties(zenthrix PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
