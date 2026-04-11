# Copyright 2024 Light Transport Entertainment Inc.
# SPDX-License-Identifier: Apache-2.0

add_library(pamo STATIC
    alloc.c
    sort.c
    tri_tri_intersect.c
    mesh/mesh.c
    mesh/geometry.c
    mesh/adjacency.c
    mesh/compact.c
    mesh/topology.c
    bvh/bvh_build.c
    stage1/sdf.c
    stage1/dual_mc.c
    stage1/remesh.c
    stage2/simplify.c
    stage3/spd_project.c
    stage3/cg_solver.c
    stage3/ccd.c
    stage3/energies/distance.c
    stage3/energies/elastic.c
    stage3/energies/hinge.c
    stage3/energies/collision.c
    stage3/safe_project.c
)

target_include_directories(pamo PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(pamo PRIVATE m)

if(PAMO_USE_PTHREADS)
    target_link_libraries(pamo PRIVATE Threads::Threads)
endif()

if(PAMO_USE_LIGHTRT)
    # Add lightrt bridge (C++ file)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    target_sources(pamo PRIVATE bvh/lightrt_bridge.cpp)
    target_include_directories(pamo PRIVATE ${LIGHTRT_DIR})
    target_compile_definitions(pamo PUBLIC PAMO_USE_LIGHTRT)
    # lightrt is header-only with a single .cc implementation
    target_sources(pamo PRIVATE ${LIGHTRT_DIR}/lightrt.cc)
endif()
