# SPDX-License-Identifier: GPL-3.0-or-later
"""Points inside a closed mesh ("Distribute Points in Volume").  Pure NumPy.

The inside of the mesh is found with vertical rays on a regular grid of
columns: every triangle that a column passes through gives one crossing,
and between the 1st and 2nd, the 3rd and 4th ... crossing the column is
inside (parity rule).  This works for any closed mesh - concave shapes,
holes, several separate parts - and is exact up to the column spacing."""

import math

import numpy as np

MAX_COLUMNS = 4_000_000
_CHUNK = 4_000_000          # candidate (triangle, column) pairs per batch


def mesh_volume(co, tri):
    v0, v1, v2 = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    c = co.mean(axis=0)
    return abs(float(np.einsum("ij,ij->i", v0 - c, np.cross(v1 - c, v2 - c)).sum()) / 6.0)


def _column_intervals(co, tri, x0, y0, c, nx, ny):
    """Inside intervals of the columns x0 + i c, y0 + j c.
    Returns (column index i * ny + j, z_in, z_out) arrays and the number of
    columns that had an odd number of crossings (holes in the mesh)."""
    a, b, d = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    # 2D signed area: vertical (edge-on) triangles are never crossed
    area = (b[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (d[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    ok = np.abs(area) > 1e-30
    a, b, d, area = a[ok], b[ok], d[ok], area[ok]
    xs = np.stack([a[:, 0], b[:, 0], d[:, 0]], 1)
    ys = np.stack([a[:, 1], b[:, 1], d[:, 1]], 1)
    i0 = np.clip(np.ceil((xs.min(1) - x0) / c), 0, nx).astype(np.int64)
    i1 = np.clip(np.floor((xs.max(1) - x0) / c), -1, nx - 1).astype(np.int64)
    j0 = np.clip(np.ceil((ys.min(1) - y0) / c), 0, ny).astype(np.int64)
    j1 = np.clip(np.floor((ys.max(1) - y0) / c), -1, ny - 1).astype(np.int64)
    wi = np.maximum(i1 - i0 + 1, 0)
    wj = np.maximum(j1 - j0 + 1, 0)
    cnt = wi * wj
    cols, zs = [], []
    t_all = np.flatnonzero(cnt)
    start = 0
    while start < t_all.size:
        # batch of triangles with at most _CHUNK candidate columns
        cs = np.cumsum(cnt[t_all[start:]])
        stop = start + max(int(np.searchsorted(cs, _CHUNK, side="right")), 1)
        t = t_all[start:stop]
        start = stop
        k = cnt[t]
        tt = np.repeat(t, k)
        off = np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)
        ci = i0[tt] + off // wj[tt]
        cj = j0[tt] + off % wj[tt]
        px = x0 + ci * c
        py = y0 + cj * c
        # barycentric coordinates in the xy projection
        ar = area[tt]
        w1 = ((px - a[tt, 0]) * (d[tt, 1] - a[tt, 1]) - (d[tt, 0] - a[tt, 0]) * (py - a[tt, 1])) / ar
        w2 = ((b[tt, 0] - a[tt, 0]) * (py - a[tt, 1]) - (px - a[tt, 0]) * (b[tt, 1] - a[tt, 1])) / ar
        w0 = 1.0 - w1 - w2
        # (the random offset of the grid keeps columns off shared edges)
        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        tt, ci, cj = tt[inside], ci[inside], cj[inside]
        z = (w0[inside] * a[tt, 2] + w1[inside] * b[tt, 2] + w2[inside] * d[tt, 2])
        cols.append(ci * ny + cj)
        zs.append(z)
    if not cols:
        e = np.zeros(0)
        return e.astype(np.int64), e, e, 0
    col = np.concatenate(cols)
    z = np.concatenate(zs)
    order = np.lexsort((z, col))
    col, z = col[order], z[order]
    # rank of every crossing inside its column
    first = np.r_[True, col[1:] != col[:-1]]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, col.size])
    rank = np.arange(col.size) - np.repeat(starts, counts)
    odd_cols = int(np.count_nonzero(counts % 2))
    # columns with an odd number of crossings cross a hole: skip them
    good = np.repeat(counts % 2 == 0, counts)
    enter = good & (rank % 2 == 0)
    idx = np.flatnonzero(enter)
    return col[idx], z[idx], z[idx + 1], odd_cols


def _grid(co, c, rng):
    lo = co.min(axis=0)
    hi = co.max(axis=0)
    # a tiny random offset keeps the columns off vertices and edges
    eps = rng.uniform(0.013, 0.037, size=2) * c
    x0, y0 = lo[0] + 0.5 * c + eps[0], lo[1] + 0.5 * c + eps[1]
    nx = max(int(math.ceil((hi[0] - x0) / c)) + 1, 1)
    ny = max(int(math.ceil((hi[1] - y0) / c)) + 1, 1)
    return x0, y0, nx, ny


def fill(co, tri, count, mode='RANDOM', seed=0):
    """About `count` points inside the closed mesh (co, tri).

    mode 'RANDOM': uniformly random points (exactly `count`),
    mode 'GRID':   a cubic lattice with the spacing that gives about `count`
                   points (the solver's random start offset makes it natural).
    Returns (points (n, 3), spacing, info dict)."""
    co = np.asarray(co, dtype=np.float64)
    tri = np.asarray(tri, dtype=np.int64)
    count = max(int(count), 1)
    rng = np.random.default_rng(seed + 9127)
    vol = mesh_volume(co, tri)
    ext = co.max(axis=0) - co.min(axis=0)
    if vol <= 0.0 or np.any(ext <= 0.0):
        return np.zeros((0, 3)), 0.0, {"volume": vol, "odd": 0}
    s = (vol / count) ** (1.0 / 3.0)

    if mode == 'GRID':
        pts, odd = np.zeros((0, 3)), 0
        for _ in range(3):                       # adjust the spacing to the count
            c = max(s, math.sqrt(ext[0] * ext[1] / MAX_COLUMNS))
            x0, y0, nx, ny = _grid(co, c, rng)
            col, zin, zout, odd = _column_intervals(co, tri, x0, y0, c, nx, ny)
            z0 = co[:, 2].min() + 0.5 * c
            m0 = np.ceil((zin - z0) / c).astype(np.int64)
            m1 = np.floor((zout - z0) / c).astype(np.int64)
            k = np.maximum(m1 - m0 + 1, 0)
            n = int(k.sum())
            if n == 0:
                s *= 0.5
                continue
            rep = np.repeat(np.arange(k.size), k)
            m = m0[rep] + np.arange(n) - np.repeat(np.cumsum(k) - k, k)
            cc = col[rep]
            pts = np.stack([x0 + (cc // ny) * c, y0 + (cc % ny) * c, z0 + m * c], axis=1)
            if abs(n - count) <= 0.02 * count:
                break
            s = c * (n / count) ** (1.0 / 3.0)
        return pts, s, {"volume": vol, "odd": odd}

    # RANDOM: fine columns, points spread over the inside length
    c = max(s / 3.0, math.sqrt(ext[0] * ext[1] / MAX_COLUMNS))
    x0, y0, nx, ny = _grid(co, c, rng)
    col, zin, zout, odd = _column_intervals(co, tri, x0, y0, c, nx, ny)
    length = zout - zin
    total = float(length.sum())
    if total <= 0.0:
        return np.zeros((0, 3)), s, {"volume": vol, "odd": odd}
    pick = np.searchsorted(np.cumsum(length), rng.random(count) * total, side="right")
    pick = np.minimum(pick, length.size - 1)
    cc = col[pick]
    x = x0 + (cc // ny) * c + (rng.random(count) - 0.5) * c
    y = y0 + (cc % ny) * c + (rng.random(count) - 0.5) * c
    z = zin[pick] + rng.random(count) * length[pick]
    return np.stack([x, y, z], axis=1), s, {"volume": vol, "odd": odd}
