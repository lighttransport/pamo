# PaMO and SAFE: Pseudocode and Logic

This note summarizes the actual control flow in the source code, mainly:

- `simp_cuda/pamo/__init__.py`
- `simp_cuda/src/cusimp*.{h,cu}`
- `simp_cuda/src/bvh/self_intersect.cuh`
- `simp_cuda/safe_project/src/pamo_safe_project/*.py`

## 1. Full PaMO Pipeline

PaMO is implemented as a 3-stage pipeline:

1. Stage 1: volumetric remeshing
2. Stage 2: GPU parallel simplification
3. Stage 3: SAFE projection (`pamo_safe_project`)

### High-level pseudocode

```text
function PAMO_RUN(input_mesh, ratio, use_stage1, use_stage3):
    store original mesh as gt_mesh
    normalize input to a bounded box

    if use_stage1:
        tris = preprocess_mesh(points, triangles)
        sdf = cumesh2sdf(tris, resolution R)
        (verts, faces) = dual_marching_cubes(sdf)
    else:
        verts = input vertices
        faces = input faces

    target_faces = max(int(ratio * current_face_count), min_verts)
    undo_vertices = empty
    stuck_counter = 0
    is_stuck = false

    repeat up to iter times:
        (verts2, faces2, occ, map, undo_vertices) =
            parallel_safe_simplify_step(
                verts, faces, undo_vertices, scale, threshold, is_stuck, init
            )

        compact vertices using occ
        compact faces using map

        if face_count <= target_faces or face_count <= 10:
            break

        if face_count unchanged:
            stuck_counter += 1
        else:
            stuck_counter = 0
            is_stuck = false

        if stuck_counter >= 2:
            is_stuck = true
        if stuck_counter == tolerance:
            break

    if use_stage3:
        (verts, faces) = SAFE_PROJECT(gt_mesh, stage2_mesh, n_iters=5)

    return verts, faces
```

## 2. Stage 1: Remeshing Logic

The remeshing path in `PaMO.remesh()` does not simplify directly. It first rasterizes the triangle soup into an SDF and then extracts a watertight mesh:

```text
function REMESH(points, triangles):
    center and scale triangles into a unit-like volume
    sdf = torchcumesh2sdf.get_sdf(triangles, R, band)
    sdf = sdf - 0.9 / R
    (V, F) = DMC(sdf, return_quads=false)
    transform V back to original coordinates
    return (V, F)
```

Purpose:

- make the mesh watertight/manifold before simplification
- remove intersections early
- give stage 2 a cleaner topology

## 3. Stage 2: GPU Parallel Simplification

There are two variants:

- `CUSimp`: basic parallel simplifier
- `CUSimp_Free`: intersection-aware version used by `PaMO`

The Python API uses `CUDSP_Free`, so the logic below focuses on that path.

### 3.1 One simplification step

Each call to `CUDSP_Free.forward(...)` rebuilds local adjacency and attempts many independent edge collapses in parallel.

```text
function PARALLEL_SAFE_SIMPLIFY_STEP(points, triangles, undo_vertices_prev, scale, threshold, is_stuck):
    import previous undo vertices
    if not stuck:
        mark previous undo vertices as invalid for current collapses
    else:
        accumulate undo vertices across iterations

    copy input mesh into working buffers
    save original_points and original_tris for undo

    near_tris = vertex -> incident triangle list
    edges = unique directed edges extracted from triangles

    for each vertex:
        Q[v] = sum over incident triangle planes p of (p p^T)

    for each edge (u, v) in parallel:
        if both endpoints were previously invalid:
            cost = INF
            continue
        if non-manifold duplication test fails:
            cost = INF
            continue
        if collapsing to midpoint flips any adjacent face normal:
            cost = INF
            continue

        midpoint = (x[u] + x[v]) / 2
        quadric_cost = [midpoint, 1]^T (Q[u] + Q[v]) [midpoint, 1]
        edge_length_penalty = local_edge_scale + current_edge_length
        skinny_triangle_penalty = penalty from post-collapse triangle quality
        cost = quadric_cost + edge_length_penalty + skinny_triangle_penalty

    for each triangle:
        tri_min_cost[triangle] = minimum encoded (cost, edge_id) over incident edges

    for each edge in parallel:
        collapse only if this edge is the minimum for every triangle touching u or v
        replace both endpoints by midpoint at u
        invalidate v
        delete degenerate/shared triangles
        rewrite references from v -> u
        reject and restore immediately if local triangles become too skinny

    remove triangles with repeated vertex ids
    collapse exact zero-length edges
    run self-intersection detection
    iteratively undo collapses whose neighborhoods touch intersecting triangles
    build prefix-sum map from surviving vertices
    return updated mesh + undo vertex list
```

### 3.2 Cost function logic

The edge cost is a sum of three ideas from the code:

1. Quadric error term:
   fit the collapsed vertex to planes of adjacent triangles using `Q[u] + Q[v]`.
2. Edge-length term:
   discourage overly large local collapses.
3. Triangle-quality term:
   penalize skinny post-collapse triangles and reject normal flips.

The code also rejects collapses that would create non-manifold adjacency by checking how many common neighboring vertices the two endpoints have.

### 3.3 How parallel conflicts are resolved

The simplifier does not use a global priority queue. Instead:

- each edge computes a local cost in parallel
- each triangle stores the cheapest incident edge (`tri_min_cost`)
- an edge collapses only if it is simultaneously the cheapest choice for every triangle in both endpoint neighborhoods

This is the key parallelization idea: many non-conflicting local minima can collapse in one kernel launch.

### 3.4 Self-intersection-free logic

`CUSimp_Free` adds a rollback stage after parallel collapse.

```text
function SELF_INTERSECTION_FILTER(mesh_after_collapse):
    build BVH over triangles
    query overlapping triangle AABBs
    run exact triangle-triangle tests on candidates
    collect intersecting triangle ids

    for each collapsed edge:
        if any incident triangle is in the intersecting set:
            mark this collapse as undo candidate

    while undo candidates exist and retry_count < 5:
        restore original vertices/triangles for those edges
        recompute intersections
        collect new undo candidates
```

Implementation detail:

- broad phase: LBVH over triangle AABBs
- narrow phase: explicit triangle-triangle intersection tests
- output: vertex ids that should be avoided in later iterations when progress stalls

## 4. SAFE / Stage 3 Projection Logic

SAFE is the optimization-based projection stage in `pamo_safe_project`.
In the default config, it optimizes the simplified mesh with four active energies:

- mesh-to-GT distance via BVH
- elastic energy
- hinge bending energy
- collision barrier energy via BVH

GT-to-mesh distance exists in code but is disabled by default.

### High-level pseudocode

```text
function SAFE_PROJECT(gt_mesh, stage2_mesh, n_iters):
    normalize both meshes with the same transform
    register meshes into Stage3System
    sample points on GT mesh
    build edge/triangle BVHs for the current mesh
    preprocess rest-state quantities for all energies

    repeat n_iters times:
        SAFE_STEP()

    denormalize vertices
    return projected_mesh
```

### One SAFE step

`Stage3System.step()` performs one outer iteration, containing several Newton iterations.

```text
function SAFE_STEP():
    update closest-point targets for distance terms
    if configured, detect contacts now

    repeat n_newton_iters times:
        q_prev_newton = q

        if configured, redetect contacts

        energy = sum_i E_i(q)
        grad, hess_diag = sum_i dE_i/dq, diag(H_i)

        p = solve H p = -grad with preconditioned CG

        clamp p so vertices stay inside contact-detection trust region
        alpha = CCD(q, p)    # continuous collision detection step in [0, 1]
        q = line_search(q_prev_newton, p, alpha)
```

## 5. SAFE Energy Terms

### 5.1 Mesh-to-GT distance

For each current vertex:

- query the closest point on the ground-truth mesh using a BVH
- penalize squared distance, weighted by Voronoi area

This pulls the simplified mesh back toward the original surface.

### 5.2 Elastic energy

This is a triangle-based deformation energy around the stage-2 mesh rest state.

Source logic:

- preprocess rest triangle area and inverse rest matrix `inv_Dm`
- build deformation gradient `F`
- use Green strain `E = 0.5 * (F^T F - I)`
- use Saint Venant-Kirchhoff-style energy

Effect:

- prevents uncontrolled stretching during projection

### 5.3 Hinge bending energy

For each interior edge:

- find the two adjacent triangles
- store rest dihedral angle and edge length
- penalize deviation of current hinge angle from the rest angle

Effect:

- preserves sharp features and discourages folding

### 5.4 Collision barrier energy

SAFE detects potential point-triangle and edge-edge contacts, then applies a barrier when distance falls below `d_hat`.

```text
if d < d_hat:
    E_coll += kappa * barrier(d, d_hat)
```

The code uses:

- BVH broad phase for PT and EE queries
- analytic contact classification
- barrier gradient and Hessian-vector product
- CCD to reduce the step length before intersection happens

## 6. SAFE Solver Logic

The linear subproblem is not assembled as a dense matrix.

Instead:

- each energy calculator contributes:
  - energy
  - gradient
  - Hessian diagonal
  - Hessian-vector product
- CG uses only:
  - residual `r`
  - search direction `v`
  - `A v` from summed Hessian-vector products
  - Jacobi preconditioner from the diagonal

So the solver is matrix-free.

### CG pseudocode

```text
function CG_SOLVE():
    p = 0
    r = grad
    z = M^-1 r
    v = z

    repeat n_cg_iters:
        A_v = H(q) * v
        alpha = (z·r) / (v·A_v)
        p = p + alpha * v
        r = r - alpha * A_v
        z_new = M^-1 r
        beta = (z_new·r) / (z·r)
        v = z_new + beta * v
        z = z_new
```

In the system, this solves for the Newton update direction used by CCD and line search.

## 7. Main Design Ideas

The core logic of the repository is:

1. Remesh first so simplification starts from a watertight mesh.
2. Simplify by collapsing many locally independent edges in parallel.
3. Reject unsafe collapses using topology checks, normal-flip checks, skinny-triangle checks, and intersection-driven undo.
4. Project the simplified result back to the original shape using a matrix-free Newton solver with elastic, bending, distance, and collision energies.

That combination is what makes PaMO both fast and intersection-aware in this codebase.

## 8. CPU Reimplementation Guide

If you want to reimplement the whole pipeline on CPU, do not mirror the CUDA code line-by-line. Keep the same algorithmic stages, but replace kernel-oriented buffers with explicit mesh data structures and local work queues.

### 8.1 Recommended module split

```text
mesh/
  mesh.py               # vertices, faces, edges, adjacency, compaction
  topology.py           # incident faces, unique edges, manifold checks
  geometry.py           # normals, area, closest point, triangle quality
  bvh.py                # AABB tree for triangles and edges

stage1/
  sdf.py                # triangle -> signed distance grid
  dual_mc.py            # dual marching cubes wrapper
  remesh.py             # normalize, rasterize, extract, denormalize

stage2/
  quadrics.py           # plane quadrics per vertex
  costs.py              # edge cost and validity tests
  scheduler.py          # independent edge selection on CPU
  collapse.py           # apply collapse, compact mesh, rebuild adjacency
  self_intersection.py  # BVH broad phase + exact tri-tri test
  simplify.py           # outer iterative simplification driver

stage3/
  energies/
    distance.py
    elastic.py
    hinge.py
    collision.py
  cg.py
  ccd.py
  line_search.py
  safe_project.py
```

### 8.2 Core CPU data structures

Prefer these over GPU-style raw buffers:

- `vertices: float64[V][3]`
- `faces: int32[F][3]`
- `vertex_alive: bool[V]`
- `face_alive: bool[F]`
- `incident_faces: list[list[int]]`
- `incident_edges: list[list[int]]`
- `edges: list[tuple[int, int]]` with canonical ordering `(min(u,v), max(u,v))`
- `edge_to_faces: dict[(u,v)] -> list[face_id]`
- `vertex_quadric: float64[V][4][4]`

For a CPU version, rebuilding adjacency after each simplification round is usually simpler and safer than trying to maintain all structures incrementally.

### 8.3 Stage-2 CPU execution strategy

The GPU version collapses many edges in parallel by triangle-local minimum tests. On CPU, the most practical approach is:

```text
function CPU_SIMPLIFY_ROUND(mesh):
    rebuild adjacency
    compute quadrics
    score every valid edge
    sort candidate edges by cost

    locked_vertices = empty set
    accepted = []

    for edge in sorted candidates:
        if either endpoint already locked:
            continue
        if collapse is topologically invalid:
            continue
        if collapse flips normals:
            continue
        if collapse creates skinny triangles:
            continue

        accept edge
        lock both endpoints

    apply accepted collapses sequentially
    remove dead faces
    compact mesh
    run self-intersection filter
```

This differs from the CUDA implementation, but preserves the same intent:

- many non-conflicting low-cost collapses per round
- no shared-vertex conflicts inside one round
- rollback of unsafe results

For a first CPU version, this is easier than reproducing the exact `tri_min_cost` encoding trick.

### 8.4 Practical SAFE CPU execution strategy

The stage-3 code is already modular enough to reimplement directly on CPU.

Use this execution order:

```text
function CPU_SAFE_STEP(system):
    update closest-point targets using triangle BVH
    detect PT and EE contacts using BVHs

    for newton_iter in range(n_newton_iters):
        energy = 0
        grad = zeros(3V)
        hdiag = small positive diagonal

        for each enabled energy:
            accumulate energy
            accumulate gradient
            accumulate Hessian diagonal

        p = preconditioned_CG(hess_vec_product, -grad, hdiag)
        p = clamp_to_contact_trust_region(p)
        alpha = conservative_ccd(system, p)
        q = backtracking_line_search(q, p, alpha)
```

On CPU, you can either:

- stay matrix-free, matching the current code closely, or
- assemble sparse Hessian blocks and use a sparse linear solver

For a faithful port, matrix-free CG is the better first implementation.

## 9. CPU Complexity

Below, `V`, `F`, `E`, and `H` denote current vertices, faces, edges, and hinges. `F_gt` and `S_gt` denote GT faces and GT samples. `K` is the number of broad-phase collision/intersection candidates, and `C` is the number of actual contacts.

### 9.1 Stage 1: Remeshing

```text
normalize mesh:            O(V)
triangle -> SDF grid:      O(F * touched_voxels)   approximate
dual marching cubes:       O(R^3)
```

Practical note:

- if using a dense grid of resolution `R`, memory is `O(R^3)`
- for CPU, this stage is usually the most memory-sensitive part

### 9.2 Stage 2: Simplification

Per simplification round:

```text
build incident triangle lists:         O(F)
build unique edges:                    O(F log F) or O(F) with hashing
compute quadrics:                      O(sum vertex valence) ~= O(F)
score all edges:                       O(sum over edges of local neighborhood work)
apply accepted collapses:              O(number of affected local faces)
compact vertices/faces:                O(V + F)
self-intersection BVH build:           O(F log F)
self-intersection query + narrowphase: O(F log F + K)
```

For edge scoring, the local work is:

- duplicate-neighbor check: `O(deg(u) * deg(v))`
- normal-flip and quality tests: `O(deg(u) + deg(v))`

So:

```text
edge scoring total: O(sum_(u,v in E) deg(u) * deg(v))
```

On well-shaped manifold meshes with bounded valence, this is close to linear in `E`.
Worst case, it can be much larger on highly irregular meshes.

A practical summary for CPU:

- typical round: `O(F log F + K)`
- worst irregular round: `O(E * d_max^2 + F log F + K)`

where `d_max` is maximum vertex valence.

### 9.3 Stage 3: SAFE projection

Per outer SAFE step:

```text
update vertex targets (BVH):       O(V log F_gt)
detect contacts with BVH:          O((V + E) log (F + E) + K)
```

Per Newton iteration:

```text
elastic energy/diff/hess-vec:      O(F)
hinge energy/diff/hess-vec:        O(H)
mesh-to-GT target energy:          O(V)
collision energy/diff/hess-vec:    O(C)
one CG iteration:                  O(F + H + V + C)
all CG iterations:                 O(n_cg * (F + H + V + C))
CCD:                               O(C)
line search:                       O(n_ls * (F + H + V + C))
```

So one SAFE outer step is roughly:

```text
O(
    V log F_gt
    + (V + E) log(F + E)
    + n_newton * (n_cg + n_ls) * (F + H + V + C)
)
```

This is the main CPU bottleneck after remeshing.

## 10. What To Implement First

If the goal is a working CPU clone, implement in this order:

1. `Mesh` container with compaction, adjacency rebuild, unique-edge extraction.
2. Basic QEM simplifier with sequential edge collapses.
3. Local validity checks:
   manifold check, normal-flip check, skinny-triangle rejection.
4. Triangle BVH and self-intersection rollback.
5. Stage-1 remeshing.
6. SAFE distance + elastic energy only.
7. Hinge energy.
8. Collision barrier + CCD.

That order gives a usable system early, before tackling the hardest parts.

## 11. CPU Engineering Notes

For a practical CPU implementation:

- Use `float64` for geometry and solver state first. The CUDA code uses `float32`, but CPU debugability is better in double precision.
- Keep face and vertex ids stable until compaction. Deferred compaction simplifies rollback.
- Rebuild BVHs per outer round, not after every single edge collapse.
- Cache per-face normals, areas, and edge lengths when evaluating many candidates in one round.
- Start with sequential collapse application. Add OpenMP or TBB only after correctness is stable.
- For SAFE, expose each energy as:
  - `update_target(mesh)`
  - `energy(q)`
  - `gradient(q, out_grad, out_diag)`
  - `hess_vec(q, dx, out)`
- For collision handling, implement point-triangle first, then edge-edge.

If performance matters later, the best CPU speedups usually come from:

1. BVH quality and query batching
2. avoiding repeated adjacency rebuilds inside a round
3. parallel edge scoring
4. faster Hessian-vector products in SAFE

## 12. Where Ray Tracing Kernels Help

If by "ray tracing kernel" you mean an RT-style BVH traversal engine such as Embree, OptiX, DXR, Vulkan RT, or hardware-assisted ray queries, then yes: it can help, but mostly for spatial queries, not for the whole algorithm.

### 12.1 Parts that benefit directly

The best matches in this repository are:

- nearest-triangle / closest-surface queries in SAFE
- broad-phase self-intersection candidate generation
- broad-phase point-triangle contact culling
- broad-phase edge-edge proximity culling

These are all query-heavy geometric tasks where BVH traversal dominates.

### 12.2 Parts that do not benefit much

RT kernels do not replace the hard parts of the algorithm:

- quadric edge cost computation
- manifold and duplicate-neighbor checks
- normal-flip and skinny-triangle validation
- actual edge collapse updates
- mesh compaction and adjacency rebuild
- Newton iterations
- CG solve and Hessian-vector products
- analytic collision gradients and CCD updates

So RT support helps the query subsystem, not the mesh optimizer or solver core.

### 12.3 Practical replacement map

For a CPU-first implementation, a good plan is:

```text
custom code:
  mesh topology
  edge collapse scheduler
  QEM cost
  exact tri-tri test
  SAFE energies and solver

RT / BVH backend:
  triangle nearest-neighbor candidates
  triangle overlap candidates
  point-vs-mesh broad phase
  edge-vs-edge broad phase
```

In other words, use RT acceleration as a backend service for "which primitives are nearby?" and keep the algorithmic decisions in your own code.

### 12.4 CPU recommendation

For CPU execution, the most practical choice is usually Embree:

- very strong BVH build and traversal performance
- mature CPU implementation
- good fit for repeated closest-hit / overlap-style queries

Two realistic ways to use it:

1. Closest-point approximation for SAFE
   - query nearby triangles with BVH traversal
   - then run exact point-triangle closest-point on the returned candidates
2. Broad-phase intersection/contact filtering
   - use BVH overlap or ray-like spatial queries to generate candidates
   - then run your own exact PT, EE, or tri-tri narrow phase

### 12.5 Expected impact by stage

#### Stage 2 simplification

Moderate benefit.

Why:

- self-intersection rollback needs broad-phase triangle candidate generation
- but most simplification time is still spent in edge scoring, validity checks, and topology edits

So RT acceleration helps, but usually does not dominate total speedup.

#### Stage 3 SAFE

Higher benefit.

Why:

- nearest-surface queries are repeated every outer step
- contact detection is repeated every step or Newton iteration
- these are exactly the kinds of tasks BVHs accelerate well

If SAFE is run for many iterations, accelerating these queries can matter a lot.

### 12.6 Edge-edge caveat

RT hardware and ray kernels are naturally triangle-oriented. Edge-edge contact is less direct.

You still usually need custom logic:

- use BVH boxes or capsules around edges for broad phase
- then run exact edge-edge distance/classification yourself

So RT can help with edge candidate pruning, but it does not eliminate the need for a custom EE narrow phase.

### 12.7 Bottom-line recommendation

If you are building a CPU version:

- use a strong BVH / RT backend for all nearest-primitive and broad-phase queries
- do not try to express the whole algorithm as ray tracing
- keep simplification, exact validation, and SAFE optimization in normal CPU code

The highest-value insertion point is:

1. SAFE nearest-point and contact queries
2. stage-2 self-intersection broad phase
3. optional edge-edge broad phase

That gives most of the benefit without distorting the implementation.
