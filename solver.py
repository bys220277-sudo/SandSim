# SPDX-License-Identifier: GPL-3.0-or-later
"""
Granular (sand) solver.  Pure NumPy, no bpy.

Every grain is a rigid particle of mass m.  Collisions use a sphere of
diameter d; grains do not roll (angular grains such as cubes have a very high
rolling resistance), so tangential motion is resisted by Coulomb friction
only.  This is what gives a pile a stable angle of repose.

Two contact models:

IMPULSE (default, contact dynamics; Moreau / Jean non-smooth contact dynamics,
the method used for granular materials, with the stacking tricks of rigid
body engines):
* Velocities come from contact impulses, positions from the velocities.
* A contact lets two grains approach only until they touch, never pulls
  (normal impulse >= 0), never bounces (restitution 0) and resists sliding
  by Coulomb friction (|tangential impulse| <= mu * normal impulse).
* Impulses are equal and opposite: momentum is conserved in every
  grain-grain contact, and every impulse can only take kinetic energy away.
  So a pile can not gain energy from the solver - it can not explode.
* Jacobi iterations with the mass of a grain split over its contacts (Tonge
  et al. 2012), warm started with the impulses of the previous substep.
* Stacking: support sweeps from the bottom up (shock propagation, Guendelman
  et al. 2003) resolve what the iterations left, the lower layer being the
  support, again with Coulomb friction.
* Overlap that is left (start positions, grains larger than their space) is
  removed by moving positions only, at a limited rate - it never becomes
  velocity.

PBD (up to version 1.2): Position Based Dynamics (Macklin et al. 2014),
velocities from the position correction, mass scaling, height ordered sweeps,
position-only stabilisation and a depenetration speed limit.

Per grain (impulse model only, both optional): a size factor s (collision
radius r s, mass m s^3 - the same material, a bigger grain is heavier) and a
wetness 0..1 that raises the friction of the grain (pair friction = mean of
the two grains; floor and colliders in the same proportion).  Without them
every grain is the same and the results are exactly those of version 1.3.

Force fields (impulse model only, optional): Blender's Force, Wind, Vortex,
Turbulence, Drag and Harmonic fields, see fields.py.  They change the free
motion of the awake grains and wake sleeping grains they push hard enough.

Common to both:
* Sleeping: a grain that stayed slower than a threshold for some time is
  frozen (treated as static) until a fast neighbour hits it or it loses its
  support.  Standard technique of rigid body engines; it removes numerical
  jitter of large piles and makes settled regions free to compute.
* Colliders: analytic floor plane + static signed distance fields.
* Air drag: quadratic, F = -1/2 rho Cd A |v| v.  With equal masses and no
  drag every grain falls identically (Galileo); drag is where mass matters.

Everything is vectorised with NumPy; there are no Python loops over grains.
"""

import math
import numpy as np

try:
    from . import fields as _fields
except ImportError:                     # loaded as a plain module (tests)
    import fields as _fields

_ZUP = np.array([0.0, 0.0, 1.0])

# 13 neighbour cells of the "half shell" (each unordered cell pair once)
_HALF_SHELL = [
    (ox, oy, oz)
    for oz in (-1, 0, 1)
    for oy in (-1, 0, 1)
    for ox in (-1, 0, 1)
    if (oz > 0) or (oz == 0 and oy > 0) or (oz == 0 and oy == 0 and ox > 0)
]
_FULL_SHELL = [(ox, oy, oz) for oz in (-1, 0, 1) for oy in (-1, 0, 1) for ox in (-1, 0, 1)]


def _cell_keys(p, cell):
    c = np.floor(p * (1.0 / cell)).astype(np.int64)
    c -= c.min(axis=0)
    c += 1
    dims = c.max(axis=0) + 2
    key = c[:, 0] + dims[0] * (c[:, 1] + dims[1] * c[:, 2])
    return key, dims


def find_pairs(p, radius):
    """All unordered pairs (i, j) with |p_i - p_j| < radius (uniform grid)."""
    n = len(p)
    e = np.zeros(0, dtype=np.int64)
    if n < 2:
        return e, e
    key, dims = _cell_keys(p, radius)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    uniq, start, counts = np.unique(skey, return_index=True, return_counts=True)
    end = start + counts
    cell_of = np.repeat(np.arange(len(uniq)), counts)  # sorted particle -> cell

    I_parts, J_parts = [], []
    # same cell, j > i (in sorted order)
    ps = np.arange(n) + 1
    _expand(ps, end[cell_of] - ps, I_parts, J_parts)

    nu = len(uniq)
    for ox, oy, oz in _HALF_SHELL:
        off = ox + dims[0] * (oy + dims[1] * oz)
        nk = uniq + off
        pos = np.minimum(np.searchsorted(uniq, nk), nu - 1)
        found = uniq[pos] == nk
        cs = np.where(found, start[pos], 0)
        cc = np.where(found, counts[pos], 0)
        _expand(cs[cell_of], cc[cell_of], I_parts, J_parts)

    if not I_parts:
        return e, e
    I = order[np.concatenate(I_parts)]
    J = order[np.concatenate(J_parts)]
    dp = p[I] - p[J]
    close = np.einsum("ij,ij->i", dp, dp) < radius * radius
    return I[close], J[close]


def points_near(pts, q, radius):
    """For every query point q: is there a point of pts closer than radius?"""
    out = np.zeros(len(q), dtype=bool)
    if len(pts) == 0 or len(q) == 0:
        return out
    cp = np.floor(pts / radius).astype(np.int64)
    cq = np.floor(q / radius).astype(np.int64)
    lo = np.minimum(cp.min(axis=0), cq.min(axis=0)) - 1
    cp -= lo
    cq -= lo
    dims = np.maximum(cp.max(axis=0), cq.max(axis=0)) + 2
    keyp = cp[:, 0] + dims[0] * (cp[:, 1] + dims[1] * cp[:, 2])
    order = np.argsort(keyp, kind="stable")
    uniq, start, counts = np.unique(keyp[order], return_index=True, return_counts=True)
    nu = len(uniq)
    r2 = radius * radius
    for ox, oy, oz in _FULL_SHELL:
        c = cq + (ox, oy, oz)
        kq = c[:, 0] + dims[0] * (c[:, 1] + dims[1] * c[:, 2])
        pos = np.minimum(np.searchsorted(uniq, kq), nu - 1)
        found = uniq[pos] == kq
        I_parts, J_parts = [], []
        _expand(np.where(found, start[pos], 0), np.where(found, counts[pos], 0), I_parts, J_parts)
        if not I_parts:
            continue
        qi = I_parts[0]
        pj = order[J_parts[0]]
        dd = pts[pj] - q[qi]
        close = np.einsum("ij,ij->i", dd, dd) < r2
        out[qi[close]] = True
    return out


def _expand(ps, pc, I_parts, J_parts):
    """For every sorted particle k, pair it with pc[k] particles starting at ps[k]."""
    pc = np.maximum(pc, 0)
    total = int(pc.sum())
    if total == 0:
        return
    idx = np.repeat(np.arange(len(pc)), pc)
    excl = np.cumsum(pc) - pc
    J = np.repeat(ps - excl, pc) + np.arange(total)
    I_parts.append(idx)
    J_parts.append(J)


class SDFCollider:
    """Static collider stored as a regular grid of signed distances.

    sdf[i, j, k] is the signed distance at origin + (i, j, k) * cell
    (negative = inside the object)."""

    def __init__(self, origin, cell, sdf, name="collider"):
        self.name = name
        self.origin = np.asarray(origin, dtype=np.float64)
        self.cell = float(cell)
        sdf = np.asarray(sdf, dtype=np.float32)
        self.dims = np.array(sdf.shape, dtype=np.int64)
        gx, gy, gz = np.gradient(sdf, self.cell)
        self.data = np.stack([sdf, gx, gy, gz], axis=-1).reshape(-1, 4).astype(np.float32)
        ny, nz = int(self.dims[1]), int(self.dims[2])
        self._stride = np.array([ny * nz, nz, 1], dtype=np.int64)
        bits = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
        self._corner_off = np.array([dx * ny * nz + dy * nz + dz for dx, dy, dz in bits], dtype=np.int64)
        self._corner_bits = np.array(bits, dtype=bool)
        self.bb_min = self.origin
        self.bb_max = self.origin + (self.dims - 1) * self.cell

    def query(self, p, margin, pose=None):
        """Returns (indices into p, phi, unit normal) for points that lie
        inside the grid and closer than `margin` to the surface.  pose (R, t):
        the collider moved rigidly from where its grid was built (a point x of
        the start pose is at R x + t now); None: where it was built."""
        if pose is not None:
            R, t = pose
            p = (p - t) @ R
        inside = np.all((p > self.bb_min) & (p < self.bb_max), axis=1)
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            return idx, None, None
        g = (p[idx] - self.origin) / self.cell
        i0 = np.minimum(np.floor(g).astype(np.int64), self.dims - 2)
        f = g - i0
        base = i0 @ self._stride
        w = np.empty((idx.size, 8))
        for k in range(8):
            b = self._corner_bits[k]
            w[:, k] = ((f[:, 0] if b[0] else 1.0 - f[:, 0])
                       * (f[:, 1] if b[1] else 1.0 - f[:, 1])
                       * (f[:, 2] if b[2] else 1.0 - f[:, 2]))
        vals = self.data[base[:, None] + self._corner_off[None, :]]  # (n, 8, 4)
        res = np.einsum("nk,nkc->nc", w, vals)
        phi = res[:, 0]
        close = phi < margin
        if not np.any(close):
            return idx[:0], None, None
        nrm = res[close, 1:4]
        ln = np.linalg.norm(nrm, axis=1)
        ok = ln > 1e-9
        nrm[ok] /= ln[ok, None]
        nrm[~ok] = (0.0, 0.0, 1.0)
        nrm = nrm.astype(np.float64)
        if pose is not None:
            nrm = nrm @ pose[0].T
        return idx[close], phi[close].astype(np.float64), nrm

    def corners(self):
        """The 8 corners of the grid (start pose)."""
        lo, hi = self.bb_min, self.bb_max
        return np.array([[(lo, hi)[a][0], (lo, hi)[b][1], (lo, hi)[c][2]]
                         for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.float64)


# ------------------------------------------------------------ moving colliders
def _rigid(M):
    """Rotation (proper, 3x3) and position of a 4x4 object matrix; scale and
    mirroring are dropped (the collider's shape is the one of the start frame)."""
    M = np.asarray(M, dtype=np.float64).reshape(4, 4)
    U, _, Vt = np.linalg.svd(M[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        R = -R
    return R, M[:3, 3].copy()


def _quat(R):
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    t = np.trace(R)
    if t > 0.0:
        S = math.sqrt(t + 1.0) * 2.0
        q = (0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = ((R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S)
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = ((R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S)
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = ((R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S)
    q = np.array(q)
    return q / np.linalg.norm(q)


def _quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(qa, qb, s):
    d = float(np.dot(qa, qb))
    if d < 0.0:
        qb, d = -qb, -d
    if d > 0.9995:
        q = qa + s * (qb - qa)
    else:
        th = math.acos(d)
        q = (math.sin((1.0 - s) * th) * qa + math.sin(s * th) * qb) / math.sin(th)
    return q / np.linalg.norm(q)


class ColliderMotion:
    """Where a collider is during one frame: its object matrix at the start
    and at the end of the frame (origin moves on a straight line, rotation by
    slerp - an object turning about its origin turns about its origin), as a
    rigid transform from the start frame of the simulation (where the grid was
    built): pose(s) = (R, t), a point x of the grid is at R x + t."""

    def __init__(self, Ma, Mb, M0):
        self.Ra, self.pa = _rigid(Ma)
        self.Rb, self.pb = _rigid(Mb)
        R0, p0 = _rigid(M0)
        self.R0t, self.p0 = R0.T, p0
        self.qa, self.qb = _quat(self.Ra), _quat(self.Rb)
        self.still = bool(np.allclose(self.Ra, self.Rb, rtol=0.0, atol=1e-12)
                          and np.allclose(self.pa, self.pb, rtol=0.0, atol=1e-12))

    def pose(self, s):
        if self.still or s <= 0.0:
            R, p = self.Ra, self.pa
        elif s >= 1.0:
            R, p = self.Rb, self.pb
        else:
            R = _quat_matrix(_slerp(self.qa, self.qb, s))
            p = self.pa + s * (self.pb - self.pa)
        Rt = R @ self.R0t
        return Rt, p - Rt @ self.p0


class SandParams:
    """All lengths in metres, times in seconds, masses in kilograms."""

    def __init__(self, **kw):
        self.diameter = 0.02              # collision diameter of a grain
        self.mass = 0.02                  # mass of one grain
        self.gravity = (0.0, 0.0, -9.81)
        self.air_drag = True              # quadratic air drag
        self.drag_area = 0.0004           # cross-section A (m^2)
        self.drag_cd = 1.05               # drag coefficient (cube ~ 1.05)
        self.air_density = 1.225
        self.friction_static = 0.25
        self.friction_dynamic = 0.22
        self.collider_friction_static = 0.5
        self.collider_friction_dynamic = 0.42
        self.use_floor = True
        self.floor_height = 0.0
        self.iterations = 3               # Jacobi iterations per substep
        self.stack_sweeps = 1             # height ordered sweeps per substep
        self.stabilization = 1            # position-only sweeps per substep
        self.stab_friction = True         # static friction also in stabilisation
        self.layer_thickness = 0.25       # sweep layer, in diameters
        self.stacking = 10.0              # strength of the mass scaling
        self.relaxation = 1.0
        self.min_substeps = 2
        self.max_substeps = 32
        self.max_travel = 0.35            # max travel per substep, in diameters
        self.use_sleep = True
        self.sleep_velocity = 0.05        # fraction of sqrt(g d)
        self.sleep_time = 0.5             # seconds of calm before sleeping
        self.wake_factor = 2.5            # wake speed = factor * sleep speed
        self.wake_overlap = 0.08          # wake when pressed deeper (diameters)
        self.energy_guard = True
        self.sweep_inner = 4              # support sweep: passes per height layer
        self.warm_start = 0.9             # impulses carried over from the last substep
        self.model = 'IMPULSE'            # 'IMPULSE': contact dynamics with impulses
                                          # (momentum conserving, energy can only
                                          # decrease); 'PBD': position based (<= 1.2)
        self.depenetration = 0.5          # overlap a pair already had at the start
                                          # of a substep may push the grains apart
                                          # at most this fast, in sqrt(g d)
                                          # (<= 0: unlimited, as up to 1.1.3)
        self.wake_grace = 0.1             # seconds a woken grain's overlaps are
                                          # removed without creating velocity
        self.wet_friction_static = None   # friction of a fully wet grain (None: as dry)
        self.wet_friction_dynamic = None
        self.cohesion = 0.0               # wet grains stick together (water bridges): the
                                          # force a bridge of two fully wet grains holds,
                                          # in weights of a grain (0: off, as up to 1.6)
        self.adhesion = 0.0               # wet grains stick to the floor and the
                                          # colliders, same unit (0: off)
        self.bridge = 0.1                 # a bridge breaks when the gap grows beyond this
                                          # fraction of the contact distance
        self.initial_velocity = (0.0, 0.0, 0.0)
        self.jitter = 0.5                 # random start offset, in diameters
        self.tumble = 0.6                 # visual rotation of rolling grains
        self.max_spin = 30.0              # visual rotation at most this many degrees per
                                          # frame (faster spinning cubes only flicker)
        self.random_rotation = True
        self.seed = 0
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError("unknown parameter: " + k)
            setattr(self, k, v)


def _at(mu, idx):
    """Friction coefficient(s) of the contacts idx: a number stays a number."""
    return mu if np.ndim(mu) == 0 else mu[idx]


def grain_friction(params, wet):
    """Per grain friction (static, dynamic, collider static, collider dynamic)
    of grains with wetness `wet` (0 dry .. 1 wet), or None if it changes nothing."""
    ms, mk = float(params.friction_static), float(params.friction_dynamic)
    ws = ms if params.wet_friction_static is None else float(params.wet_friction_static)
    wk = ws * (mk / ms if ms > 1e-9 else 1.0) if params.wet_friction_dynamic is None \
        else float(params.wet_friction_dynamic)
    wk = min(wk, ws)
    if wet is None or (ws == ms and wk == mk):
        return None
    w = np.clip(np.asarray(wet, dtype=np.float64).ravel(), 0.0, 1.0)
    if not w.any():
        return None
    mus = ms + (ws - ms) * w
    muk = mk + (wk - mk) * w
    # the floor and the colliders: a wet grain grips them in the same proportion
    cms, cmk = float(params.collider_friction_static), float(params.collider_friction_dynamic)
    f = (mus / ms) if ms > 1e-9 else None
    cs = cms * f if f is not None else cms + (mus - ms)
    ck = cmk * f if f is not None else cmk + (muk - mk)
    return mus, muk, cs, ck


G_REF = 9.81      # a "weight of a grain" of the cohesion does not depend on the gravity setting


def grain_cohesion(params, wet, scale=None):
    """Per grain strength of the water bridges of grains with wetness `wet`:
    (between grains, to the floor / colliders), each None if off.  Between two
    grains the mean of the two is used; the force of a bridge grows with the
    grain radius (capillary force ~ R), so with size variance a bridge of a
    pair scales with the pair's mean radius, and the pull on a collider acts
    on a grain of mass ~ s^3."""
    c = float(getattr(params, "cohesion", 0.0) or 0.0)
    a = float(getattr(params, "adhesion", 0.0) or 0.0)
    if wet is None or (c <= 0.0 and a <= 0.0):
        return None, None
    w = np.clip(np.asarray(wet, dtype=np.float64).ravel(), 0.0, 1.0)
    if not w.any():
        return None, None
    coh = c * G_REF * w if c > 0.0 else None
    adh = None
    if a > 0.0:
        adh = a * G_REF * w
        if scale is not None:
            adh = adh / np.asarray(scale, dtype=np.float64) ** 2
    return coh, adh


class SandSolver:
    def __init__(self, positions, params, colliders=(), scale=None, wet=None):
        self.p = params
        self.x = np.array(positions, dtype=np.float64).reshape(-1, 3)
        self.n = len(self.x)
        self.d = float(params.diameter)
        self.r = 0.5 * self.d
        # per grain size and friction (impulse model; None = all grains alike)
        self.scale = self.rad = self.winv = self.mu = None
        impulse = params.model != 'PBD'
        if impulse and scale is not None:
            s = np.asarray(scale, dtype=np.float64).ravel()
            if s.size != self.n:
                raise ValueError("scale: %d values for %d grains" % (s.size, self.n))
            if self.n and np.any(s != 1.0):
                s = np.clip(s, 0.05, 20.0)
                self.scale = s
                self.rad = self.r * s                 # collision radius
                self.winv = 1.0 / s ** 3              # inverse mass (relative to `mass`)
        if impulse and wet is not None:
            if np.size(wet) != self.n:
                raise ValueError("wet: %d values for %d grains" % (np.size(wet), self.n))
            self.mu = grain_friction(params, wet)
        # wet sand holds together: per grain strength of its water bridges
        # (velocity change per second of a grain of the nominal mass)
        self.coh = self.adh = None
        if impulse and wet is not None:
            self.coh, self.adh = grain_cohesion(params, wet, self.scale)
        self.r_max = self.r if self.rad is None else float(self.rad.max())
        self.d_min = self.d if self.rad is None else 2.0 * float(self.rad.min())
        self.d_max = self.d if self.rad is None else 2.0 * self.r_max
        self.g = np.array(params.gravity, dtype=np.float64)
        gl = float(np.linalg.norm(self.g))
        self.up = -self.g / gl if gl > 0 else _ZUP.copy()
        self.v_sleep = params.sleep_velocity * math.sqrt(max(gl, 1e-3) * self.d)
        self.v_guard = 0.25 * math.sqrt(max(gl, 1e-3) * self.d)
        self.v_dep = params.depenetration * math.sqrt(max(gl, 1e-3) * self.d)
        self.v = np.tile(np.array(params.initial_velocity, dtype=np.float64), (self.n, 1))
        self.colliders = list(colliders)
        self._cmotion = [None] * len(self.colliders)   # ColliderMotion of this frame
        self._s0, self._s1 = 0.0, 1.0             # substep: fractions of the frame
        self._col_travel = 0.0                    # moving colliders: max travel per substep
        self.asleep = np.zeros(self.n, dtype=bool)
        self._fresh = np.zeros(self.n, dtype=bool)
        self._grace = np.zeros(self.n)
        self._vframe = np.zeros(self.n)       # mean speed over the last frame
        self.calm_time = np.zeros(self.n)
        self._ws_pairs = self._ws_cols = None    # warm start of the impulse model
        self._fields = None                       # force fields: (rows at frame start, at end)
        self._field_t = 0.0                       # fraction of the frame of this substep
        self._has_wind = False
        self._expo = None                         # wind exposure of every grain (wind shadow)
        self.time = 0.0
        self.last_substeps = 0
        self.clamped = 0

        rng = np.random.default_rng(params.seed)
        if params.random_rotation:
            q = rng.normal(size=(self.n, 4))
            q /= np.linalg.norm(q, axis=1, keepdims=True)
        else:
            q = np.zeros((self.n, 4))
            q[:, 0] = 1.0
        self.quat = q
        self.omega = np.zeros((self.n, 3))

    # -------------------------------------------------------- moving colliders
    def set_collider_motion(self, k, Ma=None, Mb=None, M0=None):
        """Collider k moves during the next frame: its object matrix at the
        start (Ma) and at the end (Mb) of the frame, M0: at the start frame
        of the simulation (its grid was built there).  Ma None: static."""
        self._cmotion[k] = None if Ma is None else ColliderMotion(Ma, Mb, M0)

    def _pose(self, k, s):
        m = self._cmotion[k]
        return None if m is None else m.pose(s)

    def _any_motion(self):
        return any(m is not None for m in self._cmotion)

    def _col_move(self, s0, s1):
        """Largest distance a point of a moving collider travels from s0 to s1."""
        best = 0.0
        for k, col in enumerate(self.colliders):
            m = self._cmotion[k]
            if m is None or m.still:
                continue
            c = col.corners()
            (Ra, ta), (Rb, tb) = m.pose(s0), m.pose(s1)
            best = max(best, float(np.max(np.linalg.norm(c @ (Rb - Ra).T + (tb - ta), axis=1))))
        return best

    def _collider_wake(self, dt_frame):
        """Sleeping grains a moving collider reaches (or leaves) during the next
        frame wake up."""
        if not self._any_motion() or not self.asleep.any():
            return
        S = np.flatnonzero(self.asleep)
        reach = self.r_max + 0.25 * self.d + self._col_move(0.0, 1.0)
        wake = np.zeros(S.size, dtype=bool)
        for k, col in enumerate(self.colliders):
            m = self._cmotion[k]
            if m is None or m.still:
                continue
            for sfr in (0.0, 1.0):
                loc, _, _ = col.query(self.x[S], reach, m.pose(sfr))
                wake[loc] = True
        if np.any(wake):
            self._wake(S[wake])

    # ------------------------------------------------------------ force fields
    def set_fields(self, fa, fb=None):
        """Force fields for the next frame: packed rows (fields.pack_field) at
        its start and at its end (None: no fields)."""
        if fa is None or len(fa) == 0:
            self._fields = None
            self._has_wind = False
            return
        fa = np.asarray(fa, dtype=np.float64).reshape(-1, _fields.COLS)
        fb = None if fb is None else np.asarray(fb, dtype=np.float64).reshape(-1, _fields.COLS)
        if fb is not None and fb.shape != fa.shape:
            fb = None
        self._fields = (fa, fb)
        self._has_wind = _fields.has_wind(fa)

    def _field_rows(self, t):
        fa, fb = self._fields
        return _fields.lerp_rows(fa, fb, t)

    def _drag_k(self, idx):
        """Quadratic air drag coefficient of the grains idx (None: no drag)."""
        prm = self.p
        if prm.mass <= 0.0:
            return None
        k = 0.5 * prm.air_density * prm.drag_cd * prm.drag_area / prm.mass
        return k if self.scale is None else k / self.scale[idx]

    def _field_velocity(self, va, A, h):
        """Free motion under the fields: accelerations (a heavier grain less),
        drag fields, and the air drag relative to the wind."""
        prm = self.p
        rows = self._field_rows(self._field_t)
        acc, air, lin, quad = _fields.evaluate(rows, self.x[A], self.v[A])
        if self._expo is not None:
            air = air * self._expo[A][:, None]
        wi = None if self.winv is None else self.winv[A]
        va = va + (acc * h if wi is None else acc * (wi * h)[:, None])
        if lin.any() or quad.any():
            sp = np.sqrt(np.einsum("ij,ij->i", va, va))
            va = va / (1.0 + h * (1.0 if wi is None else wi) * (lin + quad * sp))[:, None]
        if prm.air_drag or self._has_wind:
            k = self._drag_k(A)
            if k is not None:
                rel = va - air
                sr = np.sqrt(np.einsum("ij,ij->i", rel, rel))
                va = air + rel * (1.0 / (1.0 + k * sr * h))[:, None]
        return va, np.sqrt(np.einsum("ij,ij->i", va, va))

    def _field_push(self, rows, idx):
        """Acceleration of grains at rest (idx) by the fields, and the air speed."""
        acc, air, _, _ = _fields.evaluate(rows, self.x[idx], np.zeros((idx.size, 3)))
        if self._expo is not None:
            air = air * self._expo[idx][:, None]
        a = acc if self.winv is None else acc * self.winv[idx][:, None]
        if self._has_wind:
            k = self._drag_k(idx)
            if k is not None:
                sa = np.sqrt(np.einsum("ij,ij->i", air, air))
                a = a + air * (k * sa)[:, None]
        return a, air

    # wind shadow: sample points upwind (and a little up, the air flows over
    # the surface); a grain found there shields from the wind
    SHADOW = ((1, 0.85), (2, 0.5), (3, 0.3))

    def _wind_exposure(self, rows):
        """1 for a grain in the open wind, less for a grain with grains upwind
        of it (inside or behind a pile): the wind erodes the surface."""
        n = self.n
        e = np.ones(n)
        _, air, _, _ = _fields.evaluate(rows, self.x, np.zeros((n, 3)))
        sa = np.sqrt(np.einsum("ij,ij->i", air, air))
        W = np.flatnonzero(sa > 1e-6)
        if W.size == 0:
            return e
        u = air[W] / sa[W, None]
        d = self.d
        for k, wgt in self.SHADOW:
            smp = self.x[W] - u * (k * d) + self.up * (0.35 * k * d)
            b = points_near(self.x, smp, 0.7 * d)
            e[W] *= np.where(b, 1.0 - wgt, 1.0)
        return e

    def _field_wake(self, rows):
        """Sleeping grains that a field pushes sideways or up hard enough wake up."""
        S = np.flatnonzero(self.asleep)
        if S.size == 0:
            return
        a, _ = self._field_push(rows, S)
        gl = float(np.linalg.norm(self.g))
        gref = gl if gl > 1e-3 else 9.81
        up = a @ self.up
        hor = np.linalg.norm(a - up[:, None] * self.up, axis=1)
        wake = (hor > 0.15 * gref) | (up > 0.3 * gref)
        if np.any(wake):
            self._wake(S[wake])

    # ------------------------------------------------------------ utilities
    def positions(self):
        return self.x

    def rotations(self):
        return self.quat

    def compile(self):
        pass

    @property
    def awake_count(self):
        return int(self.n - np.count_nonzero(self.asleep))

    def _active_subset(self, p, awake, radius, rings=1):
        """Indices of awake grains plus sleeping grains up to `rings` cells
        away from them - the only grains that can take part in a contact."""
        if awake.all():
            return np.arange(self.n)
        if not awake.any():
            return np.zeros(0, dtype=np.int64)
        key, dims = _cell_keys(p, radius)
        ak = np.unique(key[awake])
        rg = range(-rings, rings + 1)
        offs = np.array([ox + dims[0] * (oy + dims[1] * oz) for oz in rg for oy in rg for ox in rg],
                        dtype=np.int64)
        near = np.unique((ak[:, None] + offs[None, :]).ravel())
        pos = np.minimum(np.searchsorted(near, key), near.size - 1)
        return np.flatnonzero(near[pos] == key)

    def _pairs(self, p, awake, radius, rings=1, only_awake=True):
        """Pairs within radius among the active subset; with only_awake the
        pairs of two sleeping grains are dropped."""
        S = self._active_subset(p, awake, radius, rings)
        if S.size < 2:
            e = np.zeros(0, dtype=np.int64)
            return e, e
        I, J = find_pairs(p[S], radius)
        I, J = S[I], S[J]
        if only_awake:
            keep = awake[I] | awake[J]
            I, J = I[keep], J[keep]
        return I, J

    # ------------------------------------------------------- initial overlaps
    def resolve_initial_overlaps(self, max_rounds=80, tol=0.02):
        """Optionally shake the start positions (a perfect lattice of points
        would stand like stacked sugar cubes), then push apart grains that
        start closer than one diameter and out of colliders (positions only -
        no velocity is created).
        Returns the remaining worst overlap as a fraction of the diameter."""
        d = self.d
        rng = np.random.default_rng(12345)
        if self.p.jitter > 0.0 and self.n:
            j = np.random.default_rng(self.p.seed + 101).uniform(-1.0, 1.0, size=(self.n, 3))
            self.x += j * (self.p.jitter * d)
        allidx = np.arange(self.n)
        worst = 0.0
        self._project_colliders(self.x, self.x, allidx, friction=False)
        for _ in range(max_rounds):
            I, J = find_pairs(self.x, 1.05 * self.d_max)
            if I.size == 0:
                return 0.0
            dp = self.x[I] - self.x[J]
            dist = np.sqrt(np.einsum("ij,ij->i", dp, dp))
            if self.rad is None:
                target = None
                worst = float(max(0.0, (d - dist.min()) / d))
            else:                                   # grains of different sizes
                target = self.rad[I] + self.rad[J]
                worst = float(max(0.0, ((target - dist) / target).max()))
            if worst < tol:
                break
            same = dist < 1e-6 * d  # coincident points: separate randomly
            if np.any(same):
                k = J[same]
                self.x[k] += rng.normal(scale=0.05 * d, size=(k.size, 3))
            a_eq = np.full(I.size, 0.5)
            for _k in range(4):
                self._jacobi(self.x, self.x, I, J, a_eq, 1.0, friction=False, target=target)
                self._project_colliders(self.x, self.x, allidx, friction=False)
        return worst

    # -------------------------------------------------------------- colliders
    def _project_colliders(self, p, x0, sel, friction=True, shift=None, s=None):
        """Push the grains `sel` out of the floor and the colliders.

        All collider contacts of a grain are resolved together (normal pushes
        summed, friction averaged), so a grain wedged between two surfaces is
        not bounced back and forth between them.  Moving colliders are taken
        where they are at the fraction s of the frame (default: the end of the
        substep); friction acts on the displacement relative to them."""
        if sel.size == 0:
            return
        if s is None:
            s = self._s1
        cds = None
        prm = self.p
        r = self.r if self.rad is None else self.rad[sel]     # radius of every grain of sel
        ss, pens, nrms = [], [], []
        if prm.use_floor:
            hgt = p[sel, 2] - prm.floor_height
            b = hgt < r
            if np.any(b):
                ss.append(sel[b])
                pens.append(_at(r, b) - hgt[b])
                nrms.append(np.broadcast_to(_ZUP, (int(b.sum()), 3)))
        for k, col in enumerate(self.colliders):
            mo = self._cmotion[k]
            pose = None if mo is None else mo.pose(s)
            loc, phi, nrm = col.query(p[sel], self.r_max, pose)
            if loc.size:
                pen = _at(r, loc) - phi
                hit = pen > 0.0
                if np.any(hit):
                    ss.append(sel[loc[hit]])
                    pens.append(pen[hit])
                    nrms.append(nrm[hit])
                    if friction and mo is not None and not mo.still and s != self._s0:
                        # how far the collider carried the contact point
                        if cds is None:
                            cds = [np.zeros((len(a), 3)) for a in ss[:-1]]
                        ph = p[sel[loc[hit]]]
                        R1, t1 = pose
                        R0, t0 = mo.pose(self._s0)
                        q = (ph - t1) @ R1
                        cds.append(ph - (q @ R0.T + t0))
                    elif cds is not None:
                        cds.append(np.zeros((int(hit.sum()), 3)))
        if not ss:
            return
        if cds is not None and len(cds) < len(ss):
            cds = [np.zeros((len(a), 3)) for a in ss[:len(ss) - len(cds)]] + cds
        s = np.concatenate(ss)
        pen = np.concatenate(pens)
        nrm = np.concatenate(nrms)
        corr = pen[:, None] * nrm
        stored = None
        if shift is not None:
            # part of the push only needed because of shifts (see _substep)
            ns = np.einsum("ij,ij->i", shift[s], nrm)
            old = np.clip(np.minimum(pen, -ns), 0.0, None)
            if np.any(old > 0.0):
                stored = old[:, None] * nrm
        fr = None
        if friction:
            disp = p[s] - x0[s]
            if cds is not None:
                disp = disp - np.concatenate(cds)
            dn = np.einsum("ij,ij->i", disp, nrm)
            dt = disp - dn[:, None] * nrm
            lt = np.sqrt(np.einsum("ij,ij->i", dt, dt))
            if self.mu is None:
                cms, cmk = prm.collider_friction_static, prm.collider_friction_dynamic
            else:
                cms, cmk = self.mu[2][s], self.mu[3][s]
            fac = np.minimum(cmk * pen / np.maximum(lt, 1e-12), 1.0)
            fac[lt < cms * pen] = 1.0
            fr = -dt * fac[:, None]
        if len(ss) == 1:       # one source -> every grain appears once
            p[s] += corr if fr is None else corr + fr
            if stored is not None:
                shift[s] += stored
            return
        uniq, inv = np.unique(s, return_inverse=True)
        m = uniq.size
        tot = np.empty((m, 3))
        if stored is not None:
            for k in range(3):
                tot[:, k] = np.bincount(inv, weights=stored[:, k], minlength=m)
            shift[uniq] += tot
        for k in range(3):
            tot[:, k] = np.bincount(inv, weights=corr[:, k], minlength=m)
        if fr is not None:
            w = np.bincount(inv, weights=pen, minlength=m)
            for k in range(3):
                tot[:, k] += np.bincount(inv, weights=fr[:, k] * pen, minlength=m) / np.maximum(w, 1e-300)
        p[uniq] += tot

    # ---------------------------------------------------------- grain contacts
    def _contact_corrections(self, p, x0, ii, jj, a_i, friction, target=None, start=None, shift=None):
        """Position corrections for the grain pairs (ii, jj).
        a_i is the share of the correction taken by grain ii (0..1);
        target is the per-pair contact distance (default: the diameter)."""
        d = self.d
        dp = p[ii] - p[jj]
        dist2 = np.einsum("ij,ij->i", dp, dp)
        if target is None:
            m = np.flatnonzero((dist2 < d * d) & (dist2 > 1e-18))
        else:
            m = np.flatnonzero((dist2 < target * target) & (dist2 > 1e-18))
        if m.size == 0:
            return None
        ii, jj = ii[m], jj[m]
        dist = np.sqrt(dist2[m])
        pen = (d if target is None else target[m]) - dist
        nrm = dp[m] / dist[:, None]
        a_i = a_i[m]
        if friction:
            prm = self.p
            rel = (p[ii] - x0[ii]) - (p[jj] - x0[jj])
            rn = np.einsum("ij,ij->i", rel, nrm)
            rt = rel - rn[:, None] * nrm
            lt = np.sqrt(np.einsum("ij,ij->i", rt, rt))
            fac = np.minimum(prm.friction_dynamic * pen / np.maximum(lt, 1e-12), 1.0)
            fac[lt < prm.friction_static * pen] = 1.0          # static friction
            corr = pen[:, None] * nrm - rt * fac[:, None]
        else:
            corr = pen[:, None] * nrm
        ci = corr * a_i[:, None]
        cj = ci - corr   # = -(1 - a_i) * corr
        if start is None:
            return np.concatenate([ii, jj]), np.concatenate([ci, cj]), np.concatenate([pen, pen])
        # part of the push that is not caused by the real motion of this
        # substep: measured on the unshifted positions against the start distance
        dq = dp[m] - (shift[ii] - shift[jj])
        real = np.clip(np.minimum(start[m], d if target is None else target[m])
                       - np.sqrt(np.einsum("ij,ij->i", dq, dq)), 0.0, None)
        so = np.maximum(pen - real, 0.0)[:, None] * nrm
        si = so * a_i[:, None]
        return (np.concatenate([ii, jj]), np.concatenate([ci, cj]), np.concatenate([pen, pen]),
                np.concatenate([si, si - so]))

    @staticmethod
    def _combine(inv, corr, pen, m):
        """Penetration-weighted mean of the corrections of every grain: plain
        constraint averaging for equal contacts, and a large penetration is
        not diluted by grazing neighbours."""
        s1 = np.bincount(inv, weights=pen, minlength=m)
        acc = np.empty((m, 3))
        for k in range(3):
            acc[:, k] = np.bincount(inv, weights=corr[:, k] * pen, minlength=m)
        return acc / np.maximum(s1, 1e-300)[:, None]

    def _jacobi(self, p, x0, I, J, a_i, omega, friction, target=None, start=None, shift=None):
        if I.size == 0:
            return
        res = self._contact_corrections(p, x0, I, J, a_i, friction, target, start, shift)
        if res is None:
            return
        if start is None:
            idx, corr, pen = res
        else:
            idx, corr, pen, stored = res
            shift += omega * self._combine(idx, stored, pen, self.n)
        p += omega * self._combine(idx, corr, pen, self.n)

    def _sweep_plan(self, h0, I, J):
        """Order the pairs by height layer once per substep, with the unique
        grains of every layer precomputed (used by both height sweeps)."""
        if I.size == 0:
            return None
        n = self.n
        layer = np.floor((h0 - h0.min()) / (self.p.layer_thickness * self.d)).astype(np.int64)
        self._grain_layer = layer
        pl = np.maximum(layer[I], layer[J])
        order = np.argsort(pl, kind="stable")
        pls = pl[order]
        cut = np.flatnonzero(pls[1:] != pls[:-1]) + 1
        starts = np.concatenate([[0], cut])
        ends = np.concatenate([cut, [pls.size]])
        Is, Js = I[order], J[order]
        P = Is.size
        key = np.concatenate([pls * n + Is, pls * n + Js])
        ukey, inv = np.unique(key, return_inverse=True)
        ulayer = ukey // n
        ugrain = ukey - ulayer * n
        ucut = np.searchsorted(ulayer, pls[starts])
        uend = np.append(ucut[1:], ukey.size)
        layers = []
        for k in range(starts.size):
            s0, s1, u0, u1 = int(starts[k]), int(ends[k]), int(ucut[k]), int(uend[k])
            loc = np.concatenate([inv[s0:s1], inv[P + s0:P + s1]]) - u0
            layers.append((s0, s1, ugrain[u0:u1], loc))
        return order, Is, Js, layers

    def _height_sweep(self, p, x0, plan, a_i, friction, target=None, start=None, shift=None,
                      mus=None, muk=None):
        """Gauss-Seidel sweep ordered by height (shock propagation).

        Contacts are processed layer by layer from the bottom up, each layer
        seeing the already corrected positions of the layers below, and the
        upper grain of a pair takes (almost) the whole correction.  One sweep
        therefore carries the support of the ground through the whole pile,
        which a Jacobi iteration can only do one layer at a time.
        mus, muk: per pair friction (default: the parameters)."""
        if plan is None:
            return
        order, Is, Js, layers = plan
        As = a_i[order]
        Ts = None if target is None else target[order]
        S0 = None if start is None else start[order]
        d = self.d
        prm = self.p
        Ms = prm.friction_static if mus is None else mus[order]
        Mk = prm.friction_dynamic if muk is None else muk[order]
        for s0, s1, ug, loc in layers:
            ii, jj = Is[s0:s1], Js[s0:s1]
            dp = p[ii] - p[jj]
            dist = np.sqrt(np.einsum("ij,ij->i", dp, dp))
            pen = (d if Ts is None else Ts[s0:s1]) - dist
            np.maximum(pen, 0.0, out=pen)
            if not pen.any():
                continue
            nrm = dp / np.maximum(dist, 1e-12)[:, None]
            corr = pen[:, None] * nrm
            if S0 is not None:
                dq = dp - (shift[ii] - shift[jj])
                real = np.clip(np.minimum(S0[s0:s1], d if Ts is None else Ts[s0:s1])
                               - np.sqrt(np.einsum("ij,ij->i", dq, dq)), 0.0, None)
                so = np.maximum(pen - real, 0.0)[:, None] * nrm
            if friction:
                rel = (p[ii] - x0[ii]) - (p[jj] - x0[jj])
                rn = np.einsum("ij,ij->i", rel, nrm)
                rt = rel - rn[:, None] * nrm
                lt = np.sqrt(np.einsum("ij,ij->i", rt, rt))
                sl = slice(s0, s1)
                fac = np.minimum(_at(Mk, sl) * pen / np.maximum(lt, 1e-12), 1.0)
                fac[lt < _at(Ms, sl) * pen] = 1.0
                fac[pen <= 0.0] = 0.0
                corr -= rt * fac[:, None]
            ci = corr * As[s0:s1, None]
            w = np.concatenate([pen, pen])
            c = np.concatenate([ci, ci - corr])
            m = ug.size
            s1w = np.bincount(loc, weights=w, minlength=m)
            acc = np.empty((m, 3))
            for k in range(3):
                acc[:, k] = np.bincount(loc, weights=c[:, k] * w, minlength=m)
            p[ug] += acc / np.maximum(s1w, 1e-300)[:, None]
            if S0 is not None:
                si = so * As[s0:s1, None]
                c = np.concatenate([si, si - so])
                for k in range(3):
                    acc[:, k] = np.bincount(loc, weights=c[:, k] * w, minlength=m)
                shift[ug] += acc / np.maximum(s1w, 1e-300)[:, None]

    # -------------------------------------------------------------- stepping
    def step_frame(self, dt_frame):
        """Advance the simulation by one frame of length dt_frame (seconds)."""
        prm = self.p
        use_fields = self._fields is not None and prm.model != 'PBD'
        if use_fields:
            self._expo = self._wind_exposure(self._field_rows(0.0)) if self._has_wind else None
            if prm.use_sleep:
                self._field_wake(self._field_rows(0.0))
        moving = self._any_motion()
        if moving and prm.use_sleep:
            self._collider_wake(dt_frame)
        awake = ~self.asleep
        self.clamped = 0
        if not awake.any():
            self.last_substeps = 0
            self.time += dt_frame
            return
        va = self.v[awake]
        vmax = float(np.sqrt(np.max(np.einsum("ij,ij->i", va, va))))
        vmax += float(np.linalg.norm(self.g)) * dt_frame
        if moving:              # a moving collider must not jump through grains either
            vmax += self._col_move(0.0, 1.0) / dt_frame
        if use_fields:          # the fields may speed the grains up as well
            a, air = self._field_push(self._field_rows(0.0), np.flatnonzero(awake))
            vmax += float(np.sqrt(np.max(np.einsum("ij,ij->i", a, a)))) * dt_frame \
                + float(np.sqrt(np.max(np.einsum("ij,ij->i", air, air))))
        need = math.ceil(vmax * dt_frame / max(prm.max_travel * self.d, 1e-12))
        nsub = int(min(max(need, prm.min_substeps), prm.max_substeps))
        h = dt_frame / nsub
        x_start = self.x.copy()
        sub = self._substep if prm.model == 'PBD' else self._substep_impulse
        for k in range(nsub):
            self._field_t = (k + 0.5) / nsub
            if moving:
                self._s0, self._s1 = k / nsub, (k + 1) / nsub
                self._col_travel = self._col_move(self._s0, self._s1)
            sub(h)
        self._s0, self._s1 = 0.0, 1.0
        self.last_substeps = nsub
        self.time += dt_frame
        self._frame_contacts_and_sleep(dt_frame, x_start)
        self.update_rotation(dt_frame)

    def _substep(self, h):
        prm = self.p
        d = self.d
        awake = ~self.asleep
        A = np.flatnonzero(awake)
        if A.size == 0:
            return
        x0 = self.x

        # 1) predicted velocity of the awake grains ---------------------------
        v = self.v[A] + self.g * h
        sp = np.sqrt(np.einsum("ij,ij->i", v, v))
        if prm.air_drag and prm.mass > 0.0:
            k = 0.5 * prm.air_density * prm.drag_cd * prm.drag_area / prm.mass
            f = 1.0 / (1.0 + k * sp * h)       # semi-implicit quadratic drag
            v *= f[:, None]
            sp *= f
        vcap = 0.9 * d / h                      # never jump through a grain
        fast = sp > vcap
        if np.any(fast):
            v[fast] *= (vcap / sp[fast])[:, None]
            sp[fast] = vcap
            self.clamped = max(self.clamped, int(fast.sum()))
        xp = x0.copy()
        xp[A] += v * h

        # 2) contact pairs: every pair that can touch anywhere during this
        #    substep (start distance minus the travel of both grains; the solve
        #    can push a grain back by up to its own travel).  Two rings of
        #    sleeping grains, so that woken grains see all their neighbours.
        sleeping_exist = not awake.all()
        reach = 1.1 * d + 2.0 * float(sp.max(initial=0.0)) * h
        I, J = self._pairs(x0, awake, reach, rings=2 if sleeping_exist else 1,
                           only_awake=False)
        if prm.use_sleep and I.size and sleeping_exist:
            # sleeping grains hit by a fast grain wake up now (actual velocity,
            # not the gravity prediction, so resting grains wake nothing)
            # fast impacts: substep velocity above the solver noise (~g h);
            # slow pushes: speed averaged over the last frame (noise-free)
            wv = prm.wake_factor * self.v_sleep
            dv = self.v[I] - self.v[J]
            fast_v = max(wv, 2.0 * float(np.linalg.norm(self.g)) * h)
            hit = np.einsum("ij,ij->i", dv, dv) > fast_v * fast_v
            hit |= np.maximum(self._vframe[I], self._vframe[J]) > wv
            hit &= awake[I] ^ awake[J]
            if np.any(hit):
                woke = np.unique(np.concatenate([I[hit], J[hit]]))
                woke = woke[self.asleep[woke]]
                if woke.size:
                    self._wake(woke)
                    vfull = np.zeros((self.n, 3))
                    vfull[A] = v
                    awake = ~self.asleep
                    A = np.flatnonzero(awake)
                    v = vfull[A]
        if I.size:
            keep = awake[I] | awake[J]
            I, J = I[keep], J[keep]
        if I.size:
            hgt = x0 @ self.up
            dz = (hgt[I] - hgt[J]) * (prm.stacking / d)
            a_st = 1.0 / (1.0 + np.exp(-np.clip(dz, -40.0, 40.0)))  # share of I
            a_st[~awake[I]] = 0.0     # sleeping grains are static
            a_st[~awake[J]] = 1.0
        else:
            a_st = np.zeros(0)

        # freshly woken grains may still overlap their (sleeping) neighbours:
        # resolve that by moving positions only, so it can not become speed
        if self._fresh.any() and I.size:
            f = self._fresh[I] | self._fresh[J]
            if np.any(f):
                x_before = x0.copy()
                for _ in range(3):
                    self._jacobi(x0, x0, I[f], J[f], a_st[f], 1.0, friction=False)
                    self._project_colliders(x0, x0, A, friction=False, s=self._s0)
                xp += x0 - x_before
        self._fresh[:] = False

        # 3) constraint solve ---------------------------------------------------
        #    Grains that just woke up may carry overlaps from their sleep: for a
        #    short grace time such a pair is only kept from getting closer, and
        #    the old overlap is removed by the position-only stabilisation
        #    below, so it never turns into velocity ("pops").
        target = None
        if I.size and self._grace.any():
            g = (self._grace[I] > 0.0) | (self._grace[J] > 0.0)
            if np.any(g):
                dp0 = x0[I[g]] - x0[J[g]]
                target = np.full(I.size, d)
                target[g] = np.minimum(d, np.sqrt(np.einsum("ij,ij->i", dp0, dp0)))
        np.maximum(self._grace - h, 0.0, out=self._grace)
        #    Overlap that a pair already has at the start (a pile compressed
        #    by an impact, left-overs of the stabilisation) is pushed out by the
        #    same solve, but that part of the push is booked in `shift` so that
        #    it can be kept out of the velocity (step 4).
        start = shift = None
        if I.size and prm.depenetration > 0.0:
            dp0 = x0[I] - x0[J]
            start = np.sqrt(np.einsum("ij,ij->i", dp0, dp0))
            shift = np.zeros((self.n, 3))
        for _ in range(prm.iterations):
            self._jacobi(xp, x0, I, J, a_st, prm.relaxation, friction=True, target=target,
                         start=start, shift=shift)
            self._project_colliders(xp, x0, A, friction=True, shift=shift)
        plan = self._sweep_plan(x0 @ self.up, I, J) if (prm.stack_sweeps or prm.stabilization) else None
        for _ in range(prm.stack_sweeps):
            self._height_sweep(xp, x0, plan, a_st, friction=True, target=target,
                               start=start, shift=shift)
            self._project_colliders(xp, x0, A, friction=True, shift=shift)

        # 4) velocity update --------------------------------------------------
        vn = (xp[A] - x0[A]) / h
        if shift is not None:
            # removing old overlap may speed a grain up along the push only
            # up to v_dep - and not again every substep: a grain that already
            # moves that fast that way gets nothing more.  So the pressure of
            # a pile still drives its flow, but stored compression can never
            # blow the pile apart (the old full conversion made densely packed
            # sand explode when it landed).
            vs = shift[A] / h
            sv = np.sqrt(np.einsum("ij,ij->i", vs, vs))
            vr = vn - vs                                  # velocity of the real motion
            u = vs / np.maximum(sv, 1e-12)[:, None]
            along = np.einsum("ij,ij->i", vr, u)
            add = np.minimum(sv, np.clip(self.v_dep - along, 0.0, None))
            vn = vr + u * add[:, None]
        if prm.energy_guard:
            # inelastic contacts can not make a grain faster than the fastest
            # thing touching it: cap numerical energy injection ("launches")
            spd = np.zeros(self.n)
            spd[A] = np.sqrt(np.einsum("ij,ij->i", v, v))
            cap = spd.copy()
            if I.size:
                np.maximum.at(cap, I, spd[J])
                np.maximum.at(cap, J, spd[I])
            cap = cap[A] + self.v_guard
            sn = np.sqrt(np.einsum("ij,ij->i", vn, vn))
            over = sn > cap
            if np.any(over):
                vn[over] *= (cap[over] / sn[over])[:, None]
        self.v[A] = vn

        # 5) stabilisation: remove the overlap the solve could not resolve.
        #    Positions only -> stored compression is never turned into speed.
        xs = x0 if prm.stab_friction else xp
        for _ in range(prm.stabilization):
            self._height_sweep(xp, xs, plan, a_st, friction=prm.stab_friction)
            self._project_colliders(xp, xs, A, friction=prm.stab_friction)
        self.x = xp

        # 6) a sleeping grain that an awake grain still presses into is not
        #    immovable in reality: wake it (its overlap is then removed by
        #    position-only pre-stabilisation in the next substep)
        if prm.use_sleep and I.size and not awake.all():
            mixed = awake[I] ^ awake[J]
            if np.any(mixed):
                Im, Jm = I[mixed], J[mixed]
                dp = xp[Im] - xp[Jm]
                deep = np.einsum("ij,ij->i", dp, dp) < ((1.0 - prm.wake_overlap) * d) ** 2
                if np.any(deep):
                    woke = np.concatenate([Im[deep], Jm[deep]])
                    self._wake(np.unique(woke[self.asleep[woke]]))

    # ------------------------------------------------ contact dynamics model
    def _collider_contacts(self, p, sel, margin, h=None):
        """Floor and collider contacts of the grains `sel` closer than `margin`
        (centre to surface): grain index, outward normal, gap (distance of the
        grain's surface to the collider surface), contact key (grain, surface).
        With moving colliders (and h) also the velocity of the surface at every
        contact (None: all surfaces at rest): self.cvel."""
        prm = self.p
        r = self.r if self.rad is None else self.rad[sel]
        nc = len(self.colliders) + 1
        ss, ns, gs, ks = [], [], [], []
        vs = None
        self.cvel = None
        if prm.use_floor:
            hgt = p[sel, 2] - prm.floor_height
            b = hgt < margin
            if np.any(b):
                ss.append(sel[b])
                ns.append(np.broadcast_to(_ZUP, (int(b.sum()), 3)))
                gs.append(hgt[b] - _at(r, b))
                ks.append(sel[b] * nc)
        for k, col in enumerate(self.colliders):
            mo = self._cmotion[k]
            pose = None if mo is None else mo.pose(self._s0)
            loc, phi, nrm = col.query(p[sel], margin, pose)
            if loc.size:
                ss.append(sel[loc])
                ns.append(nrm)
                gs.append(phi - _at(r, loc))
                ks.append(sel[loc] * nc + k + 1)
                if mo is not None and not mo.still and h is not None:
                    if vs is None:
                        vs = [np.zeros((len(a), 3)) for a in ss[:-1]]
                    pc = p[sel[loc]]
                    R0, t0 = pose
                    R1, t1 = mo.pose(self._s1)
                    q = (pc - t0) @ R0
                    vs.append((q @ R1.T + t1 - pc) / h)
                elif vs is not None:
                    vs.append(np.zeros((loc.size, 3)))
        if vs is not None:
            self.cvel = np.concatenate(vs)
        if not ss:
            e = np.zeros(0)
            return e.astype(np.int64), np.zeros((0, 3)), e, e.astype(np.int64)
        return np.concatenate(ss), np.concatenate(ns), np.concatenate(gs), np.concatenate(ks)

    @staticmethod
    def _warm(keys, store, nrm):
        """Impulses of the same contacts in the previous substep (warm start):
        normal impulse and friction impulse projected on the new tangent plane."""
        P = keys.size
        Jn, Jt = np.zeros(P), np.zeros((P, 3))
        if store is None or P == 0 or store[0].size == 0:
            return Jn, Jt
        pk, pn, pt = store
        pos = np.minimum(np.searchsorted(pk, keys), pk.size - 1)
        f = np.flatnonzero(pk[pos] == keys)
        if f.size:
            Jn[f] = pn[pos[f]]
            t = pt[pos[f]]
            Jt[f] = t - np.einsum("ij,ij->i", t, nrm[f])[:, None] * nrm[f]
        return Jn, Jt

    @staticmethod
    def _keep_warm(keys, Jn, Jt, pull=False):
        m = (Jn != 0.0) if pull else (Jn > 0.0)
        k = keys[m]
        o = np.argsort(k)
        return k[o], Jn[m][o], Jt[m][o]

    def _substep_impulse(self, h):
        """One substep of the contact dynamics model.

        Velocities come from contact impulses, positions from the velocities:
          1. free motion: gravity and air drag,
          2. the contacts that can close during the substep (grain pairs, floor,
             colliders) with the gap they still have,
          3. Jacobi iterations of contact impulses.  A contact lets its grains
             approach only until they touch (no penetration), it never pulls
             (normal impulse >= 0; wet grains with cohesion / adhesion: at most
             the strength of their water bridge, while the gap is shorter than
             the bridge) and never bounces (restitution 0), and
             sliding is resisted by Coulomb friction (|tangential impulse| <=
             mu * normal impulse).  Impulses are equal and opposite (the mass of
             a grain is split over its contacts), so momentum is conserved, and
             every impulse can only take kinetic energy away,
          4. support sweeps from the bottom up (shock propagation): what the
             iterations left is resolved layer by layer with the lower grain as
             the support, again with friction - tall piles stand, a falling mass
             is stopped by the ground,
          5. new positions x + h v,
          6. overlap that is left (start positions, unconverged contacts) is
             removed by moving positions only, at most depenetration * sqrt(g d)
             per second: it never becomes velocity and can not blow a pile apart."""
        prm = self.p
        d, r, n = self.d, self.r, self.n
        var = self.rad is not None           # grains of different sizes
        awake = ~self.asleep
        A = np.flatnonzero(awake)
        if A.size == 0:
            return
        x0 = self.x

        # 1) free motion ----------------------------------------------------
        v = np.zeros((n, 3))
        va = self.v[A] + self.g * h
        if self._fields is not None:          # force fields, wind
            va, sp = self._field_velocity(va, A, h)
        else:
            sp = np.sqrt(np.einsum("ij,ij->i", va, va))
            if prm.air_drag and prm.mass > 0.0:
                k = 0.5 * prm.air_density * prm.drag_cd * prm.drag_area / prm.mass
                if var:
                    k = k / self.scale[A]         # area ~ s^2, mass ~ s^3
                f = 1.0 / (1.0 + k * sp * h)
                va *= f[:, None]
                sp *= f
        vcap = 0.9 * self.d_min / h           # never jump through the smallest grain
        fast = sp > vcap
        if np.any(fast):
            va[fast] *= (vcap / sp[fast])[:, None]
            sp[fast] = vcap
            self.clamped = max(self.clamped, int(fast.sum()))
        v[A] = va
        spmax = float(sp.max(initial=0.0))

        # 2) contacts ---------------------------------------------------------
        sleeping_exist = not awake.all()
        reach = 1.1 * self.d_max + 2.0 * spmax * h
        I, J = self._pairs(x0, awake, reach, rings=2 if sleeping_exist else 1,
                           only_awake=False)
        if prm.use_sleep and I.size and sleeping_exist:
            wv = prm.wake_factor * self.v_sleep
            dvv = self.v[I] - self.v[J]
            fast_v = max(wv, 2.0 * float(np.linalg.norm(self.g)) * h)
            hit = np.einsum("ij,ij->i", dvv, dvv) > fast_v * fast_v
            hit |= np.maximum(self._vframe[I], self._vframe[J]) > wv
            hit &= awake[I] ^ awake[J]
            if np.any(hit):
                woke = np.unique(np.concatenate([I[hit], J[hit]]))
                woke = woke[self.asleep[woke]]
                if woke.size:
                    self._wake(woke)            # their velocity is zero
                    awake = ~self.asleep
                    A = np.flatnonzero(awake)
        if I.size:
            keep = awake[I] | awake[J]
            I, J = I[keep], J[keep]
            sw = I > J                  # a pair always as (lower index, higher index)
            I, J = np.where(sw, J, I), np.where(sw, I, J)
        if I.size:
            hgt = x0 @ self.up
            dz = (hgt[I] - hgt[J]) * (prm.stacking / d)
            if var:
                # mass scaling: at equal height the shares conserve momentum
                # (the lighter grain moves more), higher up the upper grain moves
                dz = dz + np.log(self.winv[I] / self.winv[J])
            a_st = 1.0 / (1.0 + np.exp(-np.clip(dz, -40.0, 40.0)))  # share of I
            a_st[~awake[I]] = 0.0
            a_st[~awake[J]] = 1.0
        else:
            a_st = np.zeros(0)
        # freshly woken grains: their old overlap is removed by positions only
        if self._fresh.any() and I.size:
            fr = self._fresh[I] | self._fresh[J]
            if np.any(fr):
                tfr = self.rad[I[fr]] + self.rad[J[fr]] if var else None
                for _ in range(3):
                    self._jacobi(x0, x0, I[fr], J[fr], a_st[fr], 1.0, friction=False, target=tfr)
                    self._project_colliders(x0, x0, A, friction=False, s=self._s0)
        self._fresh[:] = False
        np.maximum(self._grace - h, 0.0, out=self._grace)

        # inverse mass (sleeping grains: static)
        w = awake.astype(np.float64) if not var else np.where(awake, self.winv, 0.0)
        tIJ = d                                    # contact distance of every pair
        if I.size:
            dp = x0[I] - x0[J]
            dist = np.sqrt(np.einsum("ij,ij->i", dp, dp))
            ok = dist > 1e-9 * d
            I, J, a_st, dp, dist = I[ok], J[ok], a_st[ok], dp[ok], dist[ok]
            nrm = dp / dist[:, None]
            if var:
                tIJ = self.rad[I] + self.rad[J]
            slack = np.maximum(dist - tIJ, 0.0) / h      # allowed approach speed
        else:
            nrm, slack = np.zeros((0, 3)), np.zeros(0)
            if var:
                tIJ = np.zeros(0)
        # wet sand: a pair closer than the bridge length may pull (at most cohP)
        cohP = None
        if self.coh is not None:
            if I.size:
                cp = 0.5 * (self.coh[I] + self.coh[J]) * h
                if var:
                    cp = cp * (2.0 / (1.0 / self.scale[I] + 1.0 / self.scale[J]))
                cohP = np.where(dist - tIJ < prm.bridge * tIJ, cp, 0.0)
            else:
                cohP = np.zeros(0)
        cmargin = self.r_max + spmax * h + 0.05 * d
        if self.adh is not None:
            cmargin = max(cmargin, self.r_max + prm.bridge * self.d_max)
        if self._col_travel > 0.0:              # moving colliders reach further
            cmargin += self._col_travel
        cs, cn, cg, ckey = self._collider_contacts(x0, A, cmargin, h)
        cv = self.cvel                          # surface velocities (None: at rest)
        cslack = np.maximum(cg, 0.0) / h
        adhC = None
        if self.adh is not None:
            rc = self.r if not var else self.rad[cs]
            adhC = np.where(cg < prm.bridge * 2.0 * rc, self.adh[cs] * h, 0.0)
        P, C = I.size, cs.size
        if self.mu is None:
            mus, muk = prm.friction_static, prm.friction_dynamic
            cmus, cmuk = prm.collider_friction_static, prm.collider_friction_dynamic
        else:                               # wet grains: per contact friction
            gs_, gk_, gcs, gck = self.mu
            mus, muk = 0.5 * (gs_[I] + gs_[J]), 0.5 * (gk_[I] + gk_[J])
            cmus, cmuk = gcs[cs], gck[cs]

        # 3) Jacobi impulse iterations (mass splitting) -------------------------
        #    accumulated impulses per unit mass, warm started with the impulses
        #    of the same contacts in the previous substep: a resting pile gets
        #    the right support forces at once instead of sinking a little
        pkey = I * n + J
        Jn, Jt = self._warm(pkey, self._ws_pairs, nrm)
        Cn, Ct = self._warm(ckey, self._ws_cols, cn)
        if prm.warm_start < 1.0:
            Jn *= prm.warm_start
            Jt *= prm.warm_start
            Cn *= prm.warm_start
            Ct *= prm.warm_start
        # friction bound: mu * (normal impulse + what the bridge can pull):
        # wet sand also resists shear without load (cohesion of Mohr-Coulomb)
        if cohP is not None:
            np.maximum(Jn, -cohP, out=Jn)
        if adhC is not None:
            np.maximum(Cn, -adhC, out=Cn)
        JnF = Jn if cohP is None else Jn + cohP
        CnF = Cn if adhC is None else Cn + adhC
        tl = np.sqrt(np.einsum("ij,ij->i", Jt, Jt))
        over = tl > mus * JnF
        Jt[over] *= (_at(mus, over) * JnF[over] / np.maximum(tl[over], 1e-300))[:, None]
        tl = np.sqrt(np.einsum("ij,ij->i", Ct, Ct))
        over = tl > cmus * CnF
        Ct[over] *= (_at(cmus, over) * CnF[over] / np.maximum(tl[over], 1e-300))[:, None]
        if P and (Jn.any() or C and Cn.any()):
            imp = Jn[:, None] * nrm + Jt
            cimp = Cn[:, None] * cn + Ct
            for kk in range(3):
                v[:, kk] += np.bincount(I, weights=w[I] * imp[:, kk], minlength=n)
                v[:, kk] -= np.bincount(J, weights=w[J] * imp[:, kk], minlength=n)
                v[:, kk] += np.bincount(cs, weights=cimp[:, kk], minlength=n)
        elif C and Cn.any():
            cimp = Cn[:, None] * cn + Ct
            for kk in range(3):
                v[:, kk] += np.bincount(cs, weights=cimp[:, kk], minlength=n)
        for _ in range(max(prm.iterations, 1)):
            vr = v[I] - v[J]
            vn = np.einsum("ij,ij->i", vr, nrm)
            viol = -slack - vn                  # > 0: the pair would overlap
            act = (Jn > 0.0) | (viol > 0.0)
            if cohP is not None:            # a bridge acts when the pair separates
                act |= (Jn != 0.0) | ((cohP > 0.0) & (vn > 0.0))
            cvn = np.einsum("ij,ij->i", v[cs] if cv is None else v[cs] - cv, cn)
            cviol = -cslack - cvn
            cact = (Cn > 0.0) | (cviol > 0.0)
            if adhC is not None:
                cact |= (Cn != 0.0) | ((adhC > 0.0) & (cvn > 0.0))
            if not act.any() and not cact.any():
                break
            cnt = (np.bincount(I[act], minlength=n) + np.bincount(J[act], minlength=n)
                   + np.bincount(cs[cact], minlength=n)).astype(np.float64)
            dv = np.zeros((n, 3))
            if act.any():
                a = np.flatnonzero(act)
                ia, ja = I[a], J[a]
                D = cnt[ia] * w[ia] + cnt[ja] * w[ja]
                if cohP is None:
                    jn_new = np.maximum(Jn[a] + viol[a] / D, 0.0)
                    jf = jn_new
                else:
                    # a wet pair: pushed apart as always, or else held by its
                    # water bridge - it stops separating (never pulls the
                    # grains together), with at most the bridge's strength
                    jp = Jn[a] + viol[a] / D
                    jq = np.clip(Jn[a] - vn[a] / D, -cohP[a], 0.0)
                    jn_new = np.where(jp > 0.0, jp, jq)
                    jf = jn_new + cohP[a]
                vt = vr[a] - vn[a][:, None] * nrm[a]
                t_new = Jt[a] - vt / D[:, None]
                tl = np.sqrt(np.einsum("ij,ij->i", t_new, t_new))
                over = tl > _at(mus, a) * jf
                if np.any(over):
                    t_new[over] *= (_at(_at(muk, a), over) * jf[over]
                                    / np.maximum(tl[over], 1e-300))[:, None]
                imp = (jn_new - Jn[a])[:, None] * nrm[a] + (t_new - Jt[a])
                Jn[a] = jn_new
                Jt[a] = t_new
                for kk in range(3):
                    dv[:, kk] += np.bincount(ia, weights=w[ia] * imp[:, kk], minlength=n)
                    dv[:, kk] -= np.bincount(ja, weights=w[ja] * imp[:, kk], minlength=n)
            if cact.any():
                c = np.flatnonzero(cact)
                ic = cs[c]
                D = cnt[ic]
                if adhC is None:
                    cn_new = np.maximum(Cn[c] + cviol[c] / D, 0.0)
                    cf = cn_new
                else:               # a wet grain sticks to the surface (as above)
                    jp = Cn[c] + cviol[c] / D
                    jq = np.clip(Cn[c] - cvn[c] / D, -adhC[c], 0.0)
                    cn_new = np.where(jp > 0.0, jp, jq)
                    cf = cn_new + adhC[c]
                vt = (v[ic] if cv is None else v[ic] - cv[c]) - cvn[c][:, None] * cn[c]
                t_new = Ct[c] - vt / D[:, None]
                tl = np.sqrt(np.einsum("ij,ij->i", t_new, t_new))
                over = tl > _at(cmus, c) * cf
                if np.any(over):
                    t_new[over] *= (_at(_at(cmuk, c), over) * cf[over]
                                    / np.maximum(tl[over], 1e-300))[:, None]
                imp = (cn_new - Cn[c])[:, None] * cn[c] + (t_new - Ct[c])
                Cn[c] = cn_new
                Ct[c] = t_new
                for kk in range(3):
                    dv[:, kk] += np.bincount(ic, weights=imp[:, kk], minlength=n)
            v += dv

        self._ws_pairs = self._keep_warm(pkey, Jn, Jt, cohP is not None)
        self._ws_cols = self._keep_warm(ckey, Cn, Ct, adhC is not None)

        # 4) support sweeps from the bottom up (shock propagation) ------------
        plan = self._sweep_plan(x0 @ self.up, I, J) if (prm.stack_sweeps or prm.stabilization) else None
        a_lay = a_st
        if plan is not None:
            # layers below are finished: the upper grain of a pair between two
            # layers takes the whole change (pairs inside a layer share it)
            lay = self._grain_layer
            a_lay = np.where(lay[I] > lay[J], 1.0, np.where(lay[I] < lay[J], 0.0, a_st))
            a_lay[~awake[I]] = 0.0
            a_lay[~awake[J]] = 1.0
            if cohP is not None:
                # a pair that its water bridge holds in tension is no support
                # from below: the lower grain hangs on it.  Its change is
                # shared by mass (momentum conserving), else the sweep would
                # push a hanging clump up in every substep
                b = Jn < 0.0
                if np.any(b):
                    wsum = w[I[b]] + w[J[b]]
                    a_lay[b] = np.where(wsum > 0.0, w[I[b]] / np.maximum(wsum, 1e-300), 0.0)
        if prm.stack_sweeps:
            # impulses so far as relative velocity changes (for the friction cone)
            Nrel = (Jn if cohP is None else Jn + cohP) * (w[I] + w[J])
            Trel = Jt * (w[I] + w[J])[:, None]
            CN = Cn.copy() if adhC is None else Cn + adhC
            CT = Ct.copy()
            col = (cs, cn, cslack, CN, CT, cmus, cmuk, cv)
            for _ in range(prm.stack_sweeps):
                self._support_colliders(v, *col)
                self._support_sweep(v, plan, nrm, slack, a_lay, Nrel, Trel, mus, muk, col)

        # 5) positions from the velocities ------------------------------------
        v[~awake] = 0.0
        self.v[A] = v[A]
        xp = x0 + v * h

        # 6) stabilisation: left-over overlap, positions only, rate limited ----
        if prm.stabilization and plan is not None:
            xb = xp.copy()
            xs = x0 if prm.stab_friction else xp
            wet = self.mu is not None
            for _ in range(prm.stabilization):
                self._height_sweep(xp, xs, plan, a_st, friction=prm.stab_friction,
                                   target=tIJ if var else None,
                                   mus=mus if wet else None, muk=muk if wet else None)
            if prm.depenetration > 0.0:
                step = xp - xb
                sl = np.sqrt(np.einsum("ij,ij->i", step, step))
                lim = self.v_dep * h
                big = sl > lim
                if np.any(big):
                    xp[big] = xb[big] + step[big] * (lim / sl[big])[:, None]
        self._project_colliders(xp, x0 if prm.stab_friction else xp, A, friction=prm.stab_friction)
        self.x = xp

        # 7) a sleeping grain that an awake grain still presses into wakes up
        if prm.use_sleep and I.size and not awake.all():
            mixed = awake[I] ^ awake[J]
            if np.any(mixed):
                Im, Jm = I[mixed], J[mixed]
                dq = xp[Im] - xp[Jm]
                deep = np.einsum("ij,ij->i", dq, dq) < ((1.0 - prm.wake_overlap) * _at(tIJ, mixed)) ** 2
                if np.any(deep):
                    woke = np.concatenate([Im[deep], Jm[deep]])
                    self._wake(np.unique(woke[self.asleep[woke]]))

    @staticmethod
    def _coulomb(T, vt, N, mus, muk):
        """Friction impulse T (relative velocity units, vector) after trying to
        stop the tangential relative velocity vt: stick inside the static cone
        |T| <= mus N, otherwise slide with |T| = muk N."""
        Tn = T - vt
        tl = np.sqrt(np.einsum("ij,ij->i", Tn, Tn))
        over = tl > mus * N
        if np.any(over):
            Tn[over] *= (_at(muk, over) * N[over] / np.maximum(tl[over], 1e-300))[:, None]
        return Tn

    def _support_colliders(self, v, cs, cn, cslack, CN, CT, mus, muk, cv=None, sel=None):
        """The floor and the colliders stop what still moves into them, with
        Coulomb friction against the total normal impulse (contacts `sel`);
        cv: velocities of moving surfaces (relative motion counts)."""
        idx = np.arange(cs.size) if sel is None else sel
        if idx.size == 0:
            return
        g = cs[idx]
        nn = cn[idx]
        vg = v[g] if cv is None else v[g] - cv[idx]
        vn = np.einsum("ij,ij->i", vg, nn)
        push = np.maximum(-cslack[idx] - vn, 0.0)
        if not np.any(push > 0.0):
            return
        N = CN[idx] + push
        vt = vg - vn[:, None] * nn
        Tn = self._coulomb(CT[idx], vt, N, _at(mus, idx), _at(muk, idx))
        corr = push[:, None] * nn + (Tn - CT[idx])
        CN[idx] = N
        CT[idx] = Tn
        uniq, inv = np.unique(g, return_inverse=True)
        m = uniq.size
        tot = np.empty((m, 3))
        for k in range(3):
            tot[:, k] = np.bincount(inv, weights=corr[:, k], minlength=m)
        v[uniq] += tot

    def _support_sweep(self, v, plan, nrm, slack, a_i, Nrel, Trel, mus, muk, col):
        """Velocity version of the height sweep: layer by layer from the bottom,
        a pair that still approaches is stopped, the upper grain taking all of
        the change (pairs inside a layer share it); Coulomb friction against
        the total normal impulse.  The floor and collider contacts of a layer's
        grains are resolved together with the layer."""
        if plan is None:
            return
        order, Is, Js, layers = plan
        As = a_i[order]
        Ns = nrm[order]
        Ss = slack[order]
        lay = self._grain_layer
        cs = col[0]
        corder = np.argsort(lay[cs], kind="stable") if cs.size else np.zeros(0, np.int64)
        clay = lay[cs][corder] if cs.size else np.zeros(0, np.int64)
        for s0, s1, ug, loc in layers:
            ii, jj = Is[s0:s1], Js[s0:s1]
            nn = Ns[s0:s1]
            o = order[s0:s1]
            m = ug.size
            L = max(int(lay[ii[0]]), int(lay[jj[0]]))
            csel = corder[np.searchsorted(clay, L, "left"):np.searchsorted(clay, L, "right")]
            for _ in range(self.p.sweep_inner):     # a few passes per layer
                vr = v[ii] - v[jj]
                vn = np.einsum("ij,ij->i", vr, nn)
                push = np.maximum(-Ss[s0:s1] - vn, 0.0)
                if not push.any():
                    break
                N = Nrel[o] + push
                vt = vr - vn[:, None] * nn
                Tn = self._coulomb(Trel[o], vt, N, _at(mus, o), _at(muk, o))
                corr = push[:, None] * nn + (Tn - Trel[o])
                # a pair that does not approach any more keeps its friction
                keep = push <= 0.0
                corr[keep] = 0.0
                Tn[keep] = Trel[o][keep]
                Nrel[o] = N
                Trel[o] = Tn
                ci = corr * As[s0:s1, None]
                wgt = np.concatenate([push, push])
                c = np.concatenate([ci, ci - corr])
                s1w = np.maximum(np.bincount(loc, weights=wgt, minlength=m), 1e-300)
                acc = np.empty((m, 3))
                for k in range(3):
                    acc[:, k] = np.bincount(loc, weights=c[:, k] * wgt, minlength=m)
                v[ug] += acc / s1w[:, None]
                self._support_colliders(v, *col, sel=csel)

    # ------------------------------------------------------------- sleeping
    def _wake(self, idx):
        self._fresh[idx] = True
        self._grace[idx] = self.p.wake_grace
        self.asleep[idx] = False
        self.calm_time[idx] = 0.0
        self.v[idx] = 0.0

    def _frame_contacts_and_sleep(self, dt, x_start):
        """Once per frame: contact normals (for the visual rotation),
        support test for sleeping grains, and putting calm grains to sleep."""
        prm = self.p
        n, d = self.n, self.d
        var = self.rad is not None
        awake = ~self.asleep
        rad = 1.05 * self.d_max
        # sleeping grains within one cell of an awake grain are checked; the
        # second ring guarantees that all their neighbours are present
        S2 = self._active_subset(self.x, awake, rad, rings=2)
        S1 = self._active_subset(self.x, awake, rad, rings=1)
        in_s1 = np.zeros(n, dtype=bool)
        in_s1[S1] = True
        nsum = np.zeros((n, 3))
        vsum = np.zeros((n, 3))                 # velocities of the contacts (rotation)
        csum = np.zeros(n)
        support = np.zeros(n, dtype=bool)
        if S2.size >= 2:
            I, J = find_pairs(self.x[S2], rad)
            I, J = S2[I], S2[J]
            if I.size:
                dp = self.x[I] - self.x[J]
                ln = np.linalg.norm(dp, axis=1)
                ok = ln > 1e-12
                if var:
                    ok &= ln < 1.05 * (self.rad[I] + self.rad[J])
                I, J, nrm = I[ok], J[ok], dp[ok] / ln[ok, None]
                idx = np.concatenate([I, J])
                nn = np.concatenate([nrm, -nrm])
                other = np.concatenate([J, I])
                for k in range(3):
                    nsum[:, k] = np.bincount(idx, weights=nn[:, k], minlength=n)
                    vsum[:, k] = np.bincount(idx, weights=self.v[other, k], minlength=n)
                csum += np.bincount(idx, minlength=n)
                support[idx[nn @ self.up > 0.2]] = True      # a neighbour below
        if S2.size:
            r2 = self.r if not var else self.rad[S2]
            if prm.use_floor:
                on = S2[self.x[S2, 2] - prm.floor_height < 1.05 * r2]
                nsum[on, 2] += 1.0
                csum[on] += 1.0
                support[on] = True
            for k, col in enumerate(self.colliders):
                loc, phi, nrm = col.query(self.x[S2], 1.05 * self.r_max, self._pose(k, 1.0))
                if loc.size and var:
                    near = phi < 1.05 * r2[loc]
                    loc, nrm = loc[near], nrm[near]
                if loc.size:
                    sel = S2[loc]
                    nsum[sel] += nrm
                    np.add.at(csum, sel, 1.0)
                    support[sel[nrm @ self.up > 0.2]] = True
        self._contact_nsum = nsum
        self._contact_vel = vsum / np.maximum(csum, 1.0)[:, None]
        self._has_contact = np.einsum("ij,ij->i", nsum, nsum) > 0

        if not prm.use_sleep:
            return
        # sleeping grains next to awake ones that lost their support wake up
        lost = self.asleep & in_s1 & ~support
        if np.any(lost):
            self._wake(np.flatnonzero(lost))
            awake = ~self.asleep
        # calm grains fall asleep
        A = np.flatnonzero(awake)
        self._vframe[:] = 0.0
        if A.size:
            disp = np.linalg.norm(self.x[A] - x_start[A], axis=1) / dt
            self._vframe[A] = disp
            calm = disp < self.v_sleep
            self.calm_time[A] = np.where(calm, self.calm_time[A] + dt, 0.0)
            fall = A[self.calm_time[A] >= prm.sleep_time]
            if fall.size:
                self.asleep[fall] = True
                self.v[fall] = 0.0
                self.omega[fall] = 0.0

    # ----------------------------------------------------- visual rotation
    def update_rotation(self, dt):
        """Purely visual tumbling of the cubes (the collision model is a
        non-rolling grain, see the module docstring).

        A grain rolls over what it touches: the spin comes from its velocity
        relative to its contacts (a grain falling together with its
        neighbours does not spin) and from the side it is touched on (a grain
        touched all around does not roll).  At most max_spin degrees per frame:
        a cube turning faster only flickers from frame to frame."""
        prm = self.p
        if prm.tumble <= 0.0 or self.n == 0:
            return
        awake = ~self.asleep
        nsum = getattr(self, "_contact_nsum", None)
        if nsum is None:
            return
        has = awake & self._has_contact
        if np.any(has):
            ns = nsum[has]
            ln = np.linalg.norm(ns, axis=1)
            nh = ns / np.maximum(ln, 1e-12)[:, None]
            r = self.r if self.rad is None else self.rad[has][:, None]
            vrel = self.v[has] - self._contact_vel[has]
            target = np.cross(nh, vrel) / r * (prm.tumble * np.minimum(ln, 1.0))[:, None]
            self.omega[has] = 0.5 * self.omega[has] + 0.5 * target
        free = awake & ~self._has_contact
        self.omega[free] *= 0.98
        ang = np.linalg.norm(self.omega, axis=1) * dt
        cap = math.radians(prm.max_spin)
        fast = ang > cap
        if np.any(fast):
            self.omega[fast] *= (cap / ang[fast])[:, None]
            ang[fast] = cap
        moving = ang > 1e-7
        if not np.any(moving):
            return
        axis = self.omega[moving] / (ang[moving] / dt)[:, None]
        half = 0.5 * ang[moving]
        dq = np.concatenate([np.cos(half)[:, None], axis * np.sin(half)[:, None]], axis=1)
        q = _quat_mul(dq, self.quat[moving])
        self.quat[moving] = q / np.linalg.norm(q, axis=1, keepdims=True)


def _quat_mul(a, b):
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)
