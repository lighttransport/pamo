# Copyright 2024 Light Transport Entertainment Inc.
# SPDX-License-Identifier: Apache-2.0

add_executable(pamo_verify verify_main.c)
target_link_libraries(pamo_verify PRIVATE pamo m)
