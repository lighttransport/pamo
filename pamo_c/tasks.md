# pamo_c — open tasks

## Hole-fill quality on long thin boundary loops

**Status**: deferred. Output is watertight + 2-manifold, but visible artefacts remain.

**Symptom**: long, near-linear boundary loops (e.g. the ~30° seam on the back of
the BirdHouse roof at SDF=256) still show as a visible crack/sliver after the
hole-fill pass. The fill is topologically correct (every edge has 2 incident
faces, V−E+F = 2 per component) but the triangulation lands almost coplanar
with the gap, producing very thin near-degenerate triangles that read as a
crack from grazing angles.

**Pipeline state**: simplify.c hole-fill currently uses min-area DP
triangulation (≤ 96-vertex loops) with a centroid-fan fallback for larger
loops. Both produce flat fills; min-area was an upgrade over centroid fans for
short loops but doesn't help long thin ones.

**Likely root cause**: the boundary loop itself originates upstream — Stage 1
(dual-MC) leaves seams where the SDF isosurface couldn't bridge thin shells,
and Stage 2 collapses can extend those seams. By the time hole-fill runs the
two sides of the seam are already millimetres apart and any in-plane
triangulation will be sliver-shaped.

**Possible fixes (in rough order of payoff vs cost)**:

1. **Pre-fill weld**: before hole-fill, find boundary-edge pairs whose endpoints
   are within ε of each other and zip the loop closed by merging vertex pairs
   (eliminates the seam topologically before any new triangle is needed).
   Cheap; should kill most thin cracks.

2. **Liepa min-max-dihedral fill** instead of min-area DP: weights triangles
   by max dihedral angle to neighbours, so the fill conforms to surrounding
   curvature rather than collapsing to a flat patch.

3. **Stage 1 watertightness**: investigate why DMC leaves boundary edges at
   high SDF resolution. If it's a winding-flip on one MC axis (we fixed two,
   commit `06334ae`) there may be a third edge case.

4. **Boundary-aware QEM cost**: penalise collapses that lengthen an existing
   boundary chain so Stage 2 doesn't grow seams it inherits from Stage 1.

## Other deferred items

- (placeholder — add as encountered)
