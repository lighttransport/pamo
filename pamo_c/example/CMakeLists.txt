# Copyright 2024 Light Transport Entertainment Inc.
# SPDX-License-Identifier: Apache-2.0

add_executable(pamo_example main.c)
target_link_libraries(pamo_example PRIVATE pamo m)
