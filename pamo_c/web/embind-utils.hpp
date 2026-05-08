// SPDX-License-Identifier: Apache-2.0
//
// Typed-array marshaling helpers for emscripten embind bindings.
//
// Kept in lockstep with ref/pamo/pamo_c/web/embind-utils.hpp. If you
// patch one, patch the other — the file is duplicated across sibling
// submodules because they live in independent repos.

#pragma once

#include <emscripten/bind.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace embind_utils {

using emscripten::typed_memory_view;
using emscripten::val;

// Copy a JS typed array (Float64Array / Int32Array / etc.) into a
// std::vector<T>. Returns empty vector for null/undefined input.
template <typename T>
inline std::vector<T> readTypedArray(const val &arr) {
    if (arr.isUndefined() || arr.isNull()) return {};
    const size_t n = arr["length"].as<size_t>();
    std::vector<T> out(n);
    if (n == 0) return out;
    val dst = val(typed_memory_view(n, out.data()));
    dst.call<void>("set", arr);
    return out;
}

// Read an optional field `k` from `js` into `dst`. No-op if the field
// is missing / null / undefined.
template <typename T>
inline void pick(const val &js, const char *k, T &dst) {
    if (js.isUndefined() || js.isNull()) return;
    val v = js[k];
    if (!v.isUndefined() && !v.isNull()) dst = v.as<T>();
}

// Build a new Float64Array in JS and fill it from a std::vector.
inline val float64ArrayFromVector(const std::vector<double> &v) {
    val F64 = val::global("Float64Array");
    val out = F64.new_(v.size());
    if (!v.empty()) {
        out.call<void>("set", val(typed_memory_view(v.size(), v.data())));
    }
    return out;
}

// Build a new Int32Array in JS and fill it from a std::vector.
inline val int32ArrayFromVector(const std::vector<int32_t> &v) {
    val I32 = val::global("Int32Array");
    val out = I32.new_(v.size());
    if (!v.empty()) {
        out.call<void>("set", val(typed_memory_view(v.size(), v.data())));
    }
    return out;
}

}  // namespace embind_utils
