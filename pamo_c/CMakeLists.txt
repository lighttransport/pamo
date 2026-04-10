# Copyright 2024 Light Transport Entertainment Inc.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(pamo C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Options
option(PAMO_USE_PTHREADS   "Enable pthreads parallelism" OFF)
option(PAMO_BUILD_EXAMPLES "Build example app"            ON)
option(PAMO_BUILD_TESTS    "Build unit tests"             ON)
option(PAMO_BUILD_VERIFY   "Build verification harness"   OFF)
option(PAMO_ASAN           "Enable AddressSanitizer"      OFF)

include(cmake/PamoCompilerFlags.cmake)

add_subdirectory(src)

if(PAMO_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

if(PAMO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(PAMO_BUILD_VERIFY)
    add_subdirectory(verify)
endif()
