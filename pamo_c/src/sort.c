/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Portable introsort for edge cost sorting.
 * O(N log N) worst case.
 */
#include "pamo/pamo_types.h"

#include <string.h>

/* ── Introsort implementation ────────────────────────────────────── */

typedef struct {
    double  cost;
    int32_t edge_idx;
} pamo_edge_cost_entry;

static void swap_entry(pamo_edge_cost_entry *a, pamo_edge_cost_entry *b) {
    pamo_edge_cost_entry tmp = *a;
    *a = *b;
    *b = tmp;
}

/* Insertion sort for small ranges. */
static void insertion_sort(pamo_edge_cost_entry *arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        pamo_edge_cost_entry key = arr[i];
        size_t j = i;
        while (j > 0 && arr[j - 1].cost > key.cost) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

/* Sift down for heapsort. */
static void sift_down(pamo_edge_cost_entry *arr, size_t start, size_t end) {
    size_t root = start;
    while (root * 2 + 1 <= end) {
        size_t child = root * 2 + 1;
        if (child + 1 <= end && arr[child].cost < arr[child + 1].cost)
            child++;
        if (arr[root].cost < arr[child].cost) {
            swap_entry(&arr[root], &arr[child]);
            root = child;
        } else {
            return;
        }
    }
}

static void heapsort(pamo_edge_cost_entry *arr, size_t n) {
    if (n < 2) return;
    /* Build heap. */
    for (size_t i = n / 2; i > 0; i--) {
        sift_down(arr, i - 1, n - 1);
    }
    /* Extract. */
    for (size_t end = n - 1; end > 0; end--) {
        swap_entry(&arr[0], &arr[end]);
        sift_down(arr, 0, end - 1);
    }
}

static size_t partition(pamo_edge_cost_entry *arr, size_t lo, size_t hi) {
    /* Median-of-three pivot. */
    size_t mid = lo + (hi - lo) / 2;
    if (arr[mid].cost < arr[lo].cost) swap_entry(&arr[lo], &arr[mid]);
    if (arr[hi].cost < arr[lo].cost) swap_entry(&arr[lo], &arr[hi]);
    if (arr[mid].cost < arr[hi].cost) swap_entry(&arr[mid], &arr[hi]);
    double pivot = arr[hi].cost;

    size_t i = lo;
    for (size_t j = lo; j < hi; j++) {
        if (arr[j].cost <= pivot) {
            swap_entry(&arr[i], &arr[j]);
            i++;
        }
    }
    swap_entry(&arr[i], &arr[hi]);
    return i;
}

static void introsort_impl(pamo_edge_cost_entry *arr, size_t lo, size_t hi,
                           int depth_limit) {
    while (hi > lo + 16) {
        if (depth_limit == 0) {
            heapsort(arr + lo, hi - lo + 1);
            return;
        }
        depth_limit--;
        size_t p = partition(arr, lo, hi);
        if (p > lo) introsort_impl(arr, lo, p - 1, depth_limit);
        lo = p + 1;
    }
}

void pamo_sort_edge_costs(pamo_edge_cost_entry *arr, size_t n) {
    if (n < 2) return;

    /* Compute depth limit: 2 * floor(log2(n)). */
    int depth = 0;
    for (size_t k = n; k > 1; k >>= 1) depth++;
    depth *= 2;

    introsort_impl(arr, 0, n - 1, depth);
    insertion_sort(arr, n);
}
