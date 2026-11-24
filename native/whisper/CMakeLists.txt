cmake_minimum_required(VERSION 3.16)
project(infera_whisper LANGUAGES C CXX)

add_subdirectory("${WHISPER_SOURCE_DIR}" whisper EXCLUDE_FROM_ALL)
add_executable(infera-whisper-worker worker.cpp)
target_compile_features(infera-whisper-worker PRIVATE cxx_std_17)
target_link_libraries(infera-whisper-worker PRIVATE whisper)
set_target_properties(infera-whisper-worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
