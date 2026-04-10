# Compiler flags for pamo_c
#
# Copyright 2024 Light Transport Entertainment Inc.
# SPDX-License-Identifier: Apache-2.0

# Warnings
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Werror=implicit-function-declaration
        -Wconversion -Wsign-conversion
        -Wno-unused-parameter
    )
endif()

# AddressSanitizer
if(PAMO_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# pthreads
if(PAMO_USE_PTHREADS)
    add_compile_definitions(PAMO_USE_PTHREADS)
    find_package(Threads REQUIRED)
endif()
