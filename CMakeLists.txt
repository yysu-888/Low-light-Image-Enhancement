cmake_minimum_required(VERSION 3.0)
project(test LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")

option(FAST_SMOOTH "option open fast smooth" OFF)
option(OPEN_PYBIND11 "option open python bind"  OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/base.cmake)
FILE(GLOB SRC_LIST src/*.cpp src/*.hpp)

if(FAST_SMOOTH)
    add_definitions(-DFAST_SMOOTH)
endif()

if (OPEN_PYBIND11)
    add_definitions(-DOPEN_PYBIND11)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/include/pybind11
    )
    pybind11_add_module(test ${SRC_LIST})
    add_my_lib_include(test)

    add_custom_command(
        TARGET test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/bin
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/build/*.so ${CMAKE_CURRENT_SOURCE_DIR}/bin/

        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/extra_lib
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_SOURCE_DIR}/extra_lib/__init__.py
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/bin/*.so ${CMAKE_CURRENT_SOURCE_DIR}/extra_lib/
    )

else()
    add_executable(main ${SRC_LIST})
    add_my_lib_include(main)

endif()






