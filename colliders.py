# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene objects as colliders: evaluated mesh -> signed distance grid.

Mesh data is read on the main thread (it needs bpy); the distance field is
computed from plain arrays with mathutils.BVHTree, which does not touch
Blender data and can run in the bake thread.

* Normals are made consistent first (bmesh "recalculate outside"), and a
  closed mesh that is inside out (negative volume) is turned around, so
  flipped normals of scans, imports or mirrored objects do not matter.
* A closed mesh is a solid body.
* An open surface (a displaced plane used as ground, a ramp, a wall) is a
  solid layer of a few grain sizes behind its face - like an automatic
  Solidify.  Grains can not slip through it, and the collision normal is
  well defined right at the surface.  Open surfaces are oriented to face
  up (ground), vertical ones keep their orientation."""

import numpy as np

from .solver import SDFCollider

MAX_VOXELS = 6_000_000


def read_collider_mesh(ob, depsgraph):
    """World-space vertices, triangles (consistently oriented) and a
    closed-surface flag, or None."""
    import bmesh
    import bpy

    ev = ob.evaluated_get(depsgraph)
    try:
        me = ev.to_mesh()
    except RuntimeError:
        return None
    if me is None:
        return None
    try:
        if len(me.vertices) == 0 or len(me.polygons) == 0:
            return None
        ne, nl = len(me.edges), len(me.loops)
        ei = np.empty(nl, dtype=np.int32)
        me.loops.foreach_get("edge_index", ei)
        use = np.bincount(ei, minlength=ne) if ne else np.zeros(0)
        closed = bool(ne) and bool(np.all(use == 2))

        # consistent, outward normals (bmesh), then bulk reads from a temp mesh
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        tmp = bpy.data.meshes.new("_sand_collider_tmp")
        try:
            bm.to_mesh(tmp)
            bm.free()
            tmp.calc_loop_triangles()
            nt, nv = len(tmp.loop_triangles), len(tmp.vertices)
            tri = np.empty(nt * 3, dtype=np.int32)
            tmp.loop_triangles.foreach_get("vertices", tri)
            tri = tri.reshape(-1, 3).astype(np.int64)
            co = np.empty(nv * 3, dtype=np.float32)
            tmp.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3).astype(np.float64)
        finally:
            bpy.data.meshes.remove(tmp)
        m = np.array(ev.matrix_world, dtype=np.float64)
        co = co @ m[:3, :3].T + m[:3, 3]
    finally:
        ev.to_mesh_clear()
    if len(tri) == 0:
        return None

    v0, v1, v2 = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    flipped = False
    c = co.mean(axis=0)
    vol = np.einsum("ij,ij->i", v0 - c, np.cross(v1 - c, v2 - c)).sum() / 6.0
    cr = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cr, axis=1).sum()
    up = 0.5 * cr[:, 2].sum()
    ground = (not closed) and area > 0 and abs(up) > 0.3 * area
    if ground:
        flipped = up < 0.0          # open ground-like surface (plane, terrain): face up
    else:
        flipped = vol < 0.0         # closed or almost closed body: normals outward
    if flipped:
        tri = tri[:, [0, 2, 1]]
    return {"name": ob.name, "co": co, "tri": tri, "closed": closed, "flipped": flipped,
            "ground": ground}


def _trilinear(grid, origin, cell, pts):
    dims = np.array(grid.shape)
    g = (pts - origin) / cell
    i0 = np.clip(np.floor(g).astype(np.int64), 0, dims - 2)
    f = np.clip(g - i0, 0.0, 1.0)
    out = np.zeros(len(pts))
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1.0 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                out += wx * wy * wz * grid[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


def _distances(bvh, pts, stop=None):
    """Signed distances (sign from the nearest face's normal)."""
    fn = bvh.find_nearest
    out = np.empty(len(pts))
    for i, p in enumerate(pts.tolist()):
        if stop is not None and (i & 0xFFFF) == 0 and stop():
            raise InterruptedError
        loc, nrm, _idx, dist = fn(p)
        if loc is None:
            out[i] = 1e9
            continue
        if ((p[0] - loc[0]) * nrm[0] + (p[1] - loc[1]) * nrm[1] + (p[2] - loc[2]) * nrm[2]) < 0.0:
            dist = -dist
        out[i] = dist
    return out


def build_sdf(data, diameter, voxel, stop=None):
    """Signed distance grid around the collider (narrow band exact)."""
    from mathutils.bvhtree import BVHTree

    co, tri, closed = data["co"], data["tri"], data["closed"]
    bvh = BVHTree.FromPolygons([tuple(v) for v in co], [tuple(t) for t in tri.tolist()],
                               all_triangles=True)
    cell = max(float(voxel), 1e-5)
    margin = 4.0 * diameter + 3.0 * cell
    lo = co.min(axis=0) - margin
    hi = co.max(axis=0) + margin
    while True:
        dims = np.ceil((hi - lo) / cell).astype(np.int64) + 1
        if int(np.prod(dims)) <= MAX_VOXELS:
            break
        cell *= 1.1
    hi = lo + (dims - 1) * cell

    def grid_points(c, dm):
        axes = [lo[k] + np.arange(dm[k]) * c for k in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # coarse pass everywhere, exact distances only near the surface
    cc = 4.0 * cell
    dims_c = np.ceil((hi - lo) / cc).astype(np.int64) + 2
    coarse = _distances(bvh, grid_points(cc, dims_c), stop).reshape(tuple(dims_c))
    fine_pts = grid_points(cell, dims)
    sdf = _trilinear(coarse, lo, cc, fine_pts)
    band = np.abs(sdf) < 2.0 * cc
    sdf[band] = _distances(bvh, fine_pts[band], stop)
    del fine_pts
    if not closed:
        # open surface: a solid layer of thickness T behind the face
        T = max(3.0 * diameter, 3.0 * cell)
        sdf = np.maximum(sdf, -sdf - T)
    col = SDFCollider(lo, cell, sdf.reshape(tuple(dims)).astype(np.float32), name=data["name"])
    col.closed = closed
    col.flipped = data.get("flipped", False)
    return col
