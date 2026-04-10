# Copyright 2024 Light Transport Entertainment Inc.
# SPDX-License-Identifier: Apache-2.0

# Simple test runner macro
macro(pamo_add_test name)
    add_executable(${name} ${name}.c)
    target_link_libraries(${name} PRIVATE pamo m)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

pamo_add_test(test_alloc)
pamo_add_test(test_mesh)
pamo_add_test(test_bvh)
pamo_add_test(test_tri_tri)
