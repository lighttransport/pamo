"""
Pure warp implementation of spd_project_blocks, replacing the custom
built-in from the Rabbit-Hu/warp fork (warp 1.0.0-beta.2).

Algorithm: Jacobi eigenvalue iteration on a 9x9 symmetric matrix,
clamp negative eigenvalues to zero, reconstruct.
Based on John Burkardt's implementation (GNU LGPL).
"""
import warp as wp

mat99 = wp.types.matrix(shape=(9, 9), dtype=float)
vec9 = wp.types.vector(length=9, dtype=float)


@wp.func
def _jacobi_eigenvalue_9x9(a: mat99, it_max: int):
    n = 9

    v = mat99()
    for i in range(9):
        v[i, i] = 1.0

    d = vec9()
    bw = vec9()
    zw = vec9()
    for i in range(9):
        d[i] = a[i, i]
        bw[i] = d[i]
        zw[i] = 0.0

    for it in range(it_max):
        thresh = float(0.0)
        for j in range(9):
            for i in range(j):
                thresh = thresh + a[i, j] * a[i, j]
        thresh = wp.sqrt(thresh) / float(4 * n)

        if thresh == 0.0:
            break

        for p in range(9):
            for q in range(p + 1, 9):
                gapq = 10.0 * wp.abs(a[p, q])
                termp = gapq + wp.abs(d[p])
                termq = gapq + wp.abs(d[q])

                if it > 3 and termp == wp.abs(d[p]) and termq == wp.abs(d[q]):
                    a[p, q] = 0.0
                elif thresh <= wp.abs(a[p, q]):
                    h = d[q] - d[p]
                    term = wp.abs(h) + gapq

                    if term == wp.abs(h):
                        t = a[p, q] / h
                    else:
                        theta = 0.5 * h / a[p, q]
                        t = 1.0 / (wp.abs(theta) + wp.sqrt(1.0 + theta * theta))
                        if theta < 0.0:
                            t = -t

                    c = 1.0 / wp.sqrt(1.0 + t * t)
                    s = t * c
                    tau = s / (1.0 + c)
                    h = t * a[p, q]

                    zw[p] = zw[p] - h
                    zw[q] = zw[q] + h
                    d[p] = d[p] - h
                    d[q] = d[q] + h

                    a[p, q] = 0.0

                    for j in range(p):
                        g = a[j, p]
                        hh = a[j, q]
                        a[j, p] = g - s * (hh + g * tau)
                        a[j, q] = hh + s * (g - hh * tau)

                    for j in range(p + 1, q):
                        g = a[p, j]
                        hh = a[j, q]
                        a[p, j] = g - s * (hh + g * tau)
                        a[j, q] = hh + s * (g - hh * tau)

                    for j in range(q + 1, 9):
                        g = a[p, j]
                        hh = a[q, j]
                        a[p, j] = g - s * (hh + g * tau)
                        a[q, j] = hh + s * (g - hh * tau)

                    for j in range(9):
                        g = v[j, p]
                        hh = v[j, q]
                        v[j, p] = g - s * (hh + g * tau)
                        v[j, q] = hh + s * (g - hh * tau)

        for i in range(9):
            bw[i] = bw[i] + zw[i]
            d[i] = bw[i]
            zw[i] = 0.0

    return d, v


@wp.func
def spd_project_blocks(blocks: wp.array(dtype=wp.mat33, ndim=2), it_max: int):
    """
    Project the top-left 3x3 sub-grid of mat33 blocks (= 9x9 matrix)
    to positive semi-definite. Operates in-place on the 2D slice.
    Drop-in replacement for wp.spd_project_blocks(3, blocks, it_max).
    """
    # blocks_to_array
    a = mat99()
    for bi in range(3):
        for bj in range(3):
            blk = blocks[bi, bj]
            for k in range(3):
                for l in range(3):
                    a[bi * 3 + k, bj * 3 + l] = blk[k, l]

    d, v = _jacobi_eigenvalue_9x9(a, it_max)

    # Reconstruct with positive eigenvalues only
    result = mat99()
    for kk in range(9):
        if d[kk] > 0.0:
            for i in range(9):
                for j in range(9):
                    result[i, j] = result[i, j] + d[kk] * v[kk, i] * v[kk, j]

    # array_to_blocks with symmetry enforcement
    for bi in range(3):
        for bj in range(3):
            blk = wp.mat33()
            for k in range(3):
                for l in range(3):
                    ia = bi * 3 + k
                    ja = bj * 3 + l
                    blk[k, l] = (result[ia, ja] + result[ja, ia]) * 0.5  # type: ignore
            blocks[bi, bj] = blk
