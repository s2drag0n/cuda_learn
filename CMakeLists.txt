cmake_minimum_required(VERSION 3.0)
project(cuda_learn)
enable_language(CUDA)
cmake_policy(SET CMP0104 OLD)

file(GLOB SOURCE_FILES "*.cu")

foreach(SOURCE_FILE ${SOURCE_FILES})
    get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)

    add_executable(${TARGET_NAME} ${SOURCE_FILE})
endforeach()

add_custom_target(clear
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND rm -rf CMakeCache.txt CMakeFiles
)

