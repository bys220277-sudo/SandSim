# SPDX-License-Identifier: GPL-3.0-or-later
"""
GPU version of the sand solver (NVIDIA Warp, CUDA).

Same algorithm as solver.SandSolver (see its docstring), step by step, so
that both give the same physics:
  predict -> wake by impact -> contact pairs -> pre-stabilise woken grains
  -> Jacobi iterations -> height ordered sweeps (both booking the push that
  only removes old overlap as a shift) -> velocity with the depenetration
  limit + energy guard -> stabilisation -> wake by penetration; once per
  frame: support test, sleeping and the visual rotation.

Differences in form only: float32 instead of float64, pair corrections are
accumulated with atomics instead of bincount, and the height sweep processes
pairs sorted by layer with one kernel launch per layer.
Per grain size and friction (impulse model) come in a GrainData struct; with
its flags off the kernels take exactly the code paths of version 1.3.
Runs on CUDA; the same kernels also run on the CPU (used for testing).
"""

import math

import numpy as np
import warp as wp

from . import fields as _fields
from .solver import SandSolver

if hasattr(wp, "LOG_WARNING"):
    wp.config.log_level = wp.LOG_WARNING      # no init banner
else:
    wp.config.quiet = True

_initialized = False


def init(kernel_cache_dir=None):
    global _initialized
    if not _initialized:
        if kernel_cache_dir:
            wp.config.kernel_cache_dir = kernel_cache_dir
        wp.init()
        _initialized = True


def cuda_available():
    init()
    return wp.get_cuda_device_count() > 0


def device_name(device):
    init()
    d = wp.get_device(device)
    return d.name if d.is_cuda else "CPU"


# --------------------------------------------------------------------- types
@wp.struct
class SimParams:
    d: wp.float32            # collision diameter
    r: wp.float32            # radius
    mu_s: wp.float32
    mu_k: wp.float32
    cmu_s: wp.float32        # colliders
    cmu_k: wp.float32
    stacking: wp.float32
    up: wp.vec3
    use_floor: wp.int32
    floor_h: wp.float32
    layer_h: wp.float32      # layer thickness of the height sweep


@wp.struct
class GrainData:
    """Per grain size and friction (impulse model).  With the flags off the
    kernels use the SimParams values: exactly the code of version 1.3."""
    var_size: wp.int32       # grains of different sizes
    var_mu: wp.int32         # grains of different friction (wet)
    dmax: wp.float32         # largest collision diameter
    rad: wp.array(dtype=wp.float32)      # collision radius
    winv: wp.array(dtype=wp.float32)     # inverse mass relative to the nominal grain
    inv_s: wp.array(dtype=wp.float32)    # 1 / size factor (air drag)
    mu: wp.array(dtype=wp.vec4)          # static, dynamic, collider static, collider dynamic
    use_coh: wp.int32        # wet grains stick together (solver.grain_cohesion)
    use_adh: wp.int32        # wet grains stick to the floor and the colliders
    coh: wp.array(dtype=wp.float32)      # strength of the grain's water bridges (per second)
    adh: wp.array(dtype=wp.float32)      # the same to the floor / colliders
    bridge: wp.float32       # a bridge breaks beyond this fraction of the contact distance
    ch: wp.float32           # length of the current substep


@wp.struct
class ColMotion:
    """Moving colliders (solver.ColliderMotion): pose at the start (0) and at
    the end (1) of the substep; a point x of the grid is at R x + t.  With
    any = 0 every collider is where its grid was built (the code of 1.7)."""
    any: wp.int32
    mov: wp.array(dtype=wp.int32)
    R0: wp.array(dtype=wp.mat33)
    t0: wp.array(dtype=wp.vec3)
    R1: wp.array(dtype=wp.mat33)
    t1: wp.array(dtype=wp.vec3)
    inv_h: wp.float32


# ------------------------------------------------------------------- helpers
@wp.func
def g_rad(i: int, prm: SimParams, gd: GrainData):
    r = prm.r
    if gd.var_size != 0:
        r = gd.rad[i]
    return r


@wp.func
def p_dist(i: int, j: int, prm: SimParams, gd: GrainData):
    """Contact distance of a pair (sum of the radii)."""
    t = prm.d
    if gd.var_size != 0:
        t = gd.rad[i] + gd.rad[j]
    return t


@wp.func
def p_mu(i: int, j: int, prm: SimParams, gd: GrainData):
    """Friction of a pair: mean of the two grains."""
    m = wp.vec2(prm.mu_s, prm.mu_k)
    if gd.var_mu != 0:
        a = gd.mu[i]
        b = gd.mu[j]
        m = wp.vec2(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    return m


@wp.func
def c_mu(i: int, prm: SimParams, gd: GrainData):
    """Friction of a grain on the floor and the colliders."""
    m = wp.vec2(prm.cmu_s, prm.cmu_k)
    if gd.var_mu != 0:
        a = gd.mu[i]
        m = wp.vec2(a[2], a[3])
    return m


@wp.func
def p_coh(i: int, j: int, dist: float, pd: float, gd: GrainData):
    """How much a wet pair may pull in this substep (0: no water bridge)."""
    c = float(0.0)
    if gd.use_coh != 0:
        if dist - pd < gd.bridge * pd:
            c = 0.5 * (gd.coh[i] + gd.coh[j]) * gd.ch
            if gd.var_size != 0:
                c = c * (2.0 / (gd.inv_s[i] + gd.inv_s[j]))
    return c


@wp.func
def c_adh(i: int, gap: float, prm: SimParams, gd: GrainData):
    """How much a wet grain may stick to a surface at distance gap."""
    a = float(0.0)
    if gd.use_adh != 0:
        if gap < gd.bridge * 2.0 * g_rad(i, prm, gd):
            a = gd.adh[i] * gd.ch
    return a


@wp.func
def inv_mass(i: int, asleep: wp.array(dtype=wp.int32), gd: GrainData):
    w = float(0.0)
    if asleep[i] == 0:
        w = 1.0
        if gd.var_size != 0:
            w = gd.winv[i]
    return w


@wp.func
def friction_corr(disp: wp.vec3, n: wp.vec3, pen: float, mus: float, muk: float):
    """Coulomb friction at position level: static cancels the tangential
    displacement, kinetic reduces it by mu_k * pen."""
    dn = wp.dot(disp, n)
    dt = disp - dn * n
    lt = wp.length(dt)
    fac = wp.min(muk * pen / wp.max(lt, 1.0e-12), 1.0)
    if lt < mus * pen:
        fac = 1.0
    return -dt * fac


@wp.func
def share_of_i(i: int, j: int, x0: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
               prm: SimParams, gd: GrainData):
    """Part of a contact correction taken by grain i (mass scaling: the upper
    grain takes almost all; sleeping grains are static; at equal height grains
    of different mass share it conserving momentum)."""
    if asleep[i] != 0:
        return float(0.0)
    if asleep[j] != 0:
        return float(1.0)
    dz = (wp.dot(x0[i], prm.up) - wp.dot(x0[j], prm.up)) * prm.stacking / prm.d
    if gd.var_size != 0:
        dz = dz + wp.log(gd.winv[i] / gd.winv[j])
    dz = wp.clamp(dz, -40.0, 40.0)
    return 1.0 / (1.0 + wp.exp(-dz))


@wp.func
def sdf_sample(p: wp.vec3, c: int,
               col_data: wp.array(dtype=wp.vec4), col_origin: wp.array(dtype=wp.vec3),
               col_cell: wp.array(dtype=wp.float32), col_dims: wp.array(dtype=wp.vec3i),
               col_off: wp.array(dtype=wp.int32)):
    """Trilinear sample (phi, grad) of collider c; phi = 1e9 outside its grid."""
    o = col_origin[c]
    cell = col_cell[c]
    dm = col_dims[c]
    far = wp.vec4(1.0e9, 0.0, 0.0, 1.0)
    hx = o[0] + float(dm[0] - 1) * cell
    hy = o[1] + float(dm[1] - 1) * cell
    hz = o[2] + float(dm[2] - 1) * cell
    if p[0] <= o[0] or p[1] <= o[1] or p[2] <= o[2] or p[0] >= hx or p[1] >= hy or p[2] >= hz:
        return far
    gx = (p[0] - o[0]) / cell
    gy = (p[1] - o[1]) / cell
    gz = (p[2] - o[2]) / cell
    ix = wp.min(int(wp.floor(gx)), dm[0] - 2)
    iy = wp.min(int(wp.floor(gy)), dm[1] - 2)
    iz = wp.min(int(wp.floor(gz)), dm[2] - 2)
    fx = gx - float(ix)
    fy = gy - float(iy)
    fz = gz - float(iz)
    syz = dm[1] * dm[2]
    base = col_off[c] + ix * syz + iy * dm[2] + iz
    res = wp.vec4(0.0, 0.0, 0.0, 0.0)
    for dx in range(2):
        wx = fx * float(dx) + (1.0 - fx) * float(1 - dx)
        for dy in range(2):
            wy = fy * float(dy) + (1.0 - fy) * float(1 - dy)
            for dz in range(2):
                wz = fz * float(dz) + (1.0 - fz) * float(1 - dz)
                res = res + (wx * wy * wz) * col_data[base + dx * syz + dy * dm[2] + dz]
    return res


@wp.func
def col_sample(p: wp.vec3, c: int, which: int, cm: ColMotion,
               col_data: wp.array(dtype=wp.vec4), col_origin: wp.array(dtype=wp.vec3),
               col_cell: wp.array(dtype=wp.float32), col_dims: wp.array(dtype=wp.vec3i),
               col_off: wp.array(dtype=wp.int32)):
    """sdf_sample of collider c where it is now (which 0: start, 1: end of the
    substep); the gradient in world axes."""
    if cm.any != 0:
        if cm.mov[c] != 0:
            R = cm.R0[c]
            t = cm.t0[c]
            if which != 0:
                R = cm.R1[c]
                t = cm.t1[c]
            s = sdf_sample(wp.transpose(R) * (p - t), c, col_data, col_origin, col_cell, col_dims, col_off)
            g = R * wp.vec3(s[1], s[2], s[3])
            return wp.vec4(s[0], g[0], g[1], g[2])
    return sdf_sample(p, c, col_data, col_origin, col_cell, col_dims, col_off)


@wp.func
def col_carry(p: wp.vec3, c: int, cm: ColMotion):
    """How far collider c carried its point now at p (end of the substep)."""
    d = wp.vec3(0.0, 0.0, 0.0)
    if cm.any != 0:
        if cm.mov[c] != 0:
            q = wp.transpose(cm.R1[c]) * (p - cm.t1[c])
            d = p - (cm.R0[c] * q + cm.t0[c])
    return d


@wp.func
def quat_mul(a: wp.vec4, b: wp.vec4):
    # (w, x, y, z) convention, same as solver._quat_mul
    return wp.vec4(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


# ------------------------------------------------------------------- force fields
# (fields.py on the device: same formulas, same noise bits)
_H1 = wp.constant(-1918454973)      # 0x8DA6B343 as int32
_H2 = wp.constant(-669632447)       # 0xD8163841
_H3 = wp.constant(-887442657)       # 0xCB1AB31F
_H4 = wp.constant(374761393)        # 0x165667B1
_H5 = wp.constant(2146121005)       # 0x7FEB352D
_H6 = wp.constant(-2073254261)      # 0x846CA68B
_FC = _fields


@wp.func
def fhash(ix: int, iy: int, iz: int, seed: int):
    h = wp.uint32(ix) * wp.uint32(_H1)
    h = h ^ (wp.uint32(iy) * wp.uint32(_H2))
    h = h ^ (wp.uint32(iz) * wp.uint32(_H3))
    h = h ^ (wp.uint32(seed) * wp.uint32(_H4))
    h = h ^ (h >> wp.uint32(16))
    h = h * wp.uint32(_H5)
    h = h ^ (h >> wp.uint32(15))
    h = h * wp.uint32(_H6)
    h = h ^ (h >> wp.uint32(16))
    return float(h) / 4294967296.0


@wp.func
def vnoise(q: wp.vec3, seed: int):
    fx = wp.floor(q[0])
    fy = wp.floor(q[1])
    fz = wp.floor(q[2])
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    tx = q[0] - fx
    ty = q[1] - fy
    tz = q[2] - fz
    ux = tx * tx * (3.0 - 2.0 * tx)
    uy = ty * ty * (3.0 - 2.0 * ty)
    uz = tz * tz * (3.0 - 2.0 * tz)
    out = float(0.0)
    for dx in range(2):
        wx = ux * float(dx) + (1.0 - ux) * float(1 - dx)
        for dy in range(2):
            wy = uy * float(dy) + (1.0 - uy) * float(1 - dy)
            for dz in range(2):
                wz = uz * float(dz) + (1.0 - uz) * float(1 - dz)
                out = out + wx * wy * wz * fhash(ix + dx, iy + dy, iz + dz, seed)
    return out


@wp.func
def turb1(q: wp.vec3, seed: int):
    return (vnoise(q, seed) + 0.5 * vnoise(2.0 * q, seed + 7)) / 1.5


@wp.func
def falloff_func(fac: float, usemin: float, mind: float, usemax: float, maxd: float, power: float):
    md = mind
    if usemin == 0.0:
        md = 0.0
    out = float(1.0)
    if power != 0.0:
        out = wp.pow(1.0 + wp.max(fac - md, 0.0), -power)
    if fac < md:
        out = 1.0
    if usemax != 0.0 and fac > maxd:
        out = 0.0
    return out


@wp.func
def fcol(fa: wp.array2d(dtype=wp.float32), fb: wp.array2d(dtype=wp.float32), f: int, c: int, t: float):
    a = fa[f, c]
    return a + (fb[f, c] - a) * t


@wp.func
def field_eval(p: wp.vec3, v: wp.vec3, fa: wp.array2d(dtype=wp.float32),
               fb: wp.array2d(dtype=wp.float32), nf: int, t: float):
    """Rows: acceleration of a nominal grain, air velocity, (linear drag,
    quadratic drag, 0) - fields.evaluate for one point."""
    acc = wp.vec3(0.0, 0.0, 0.0)
    air = wp.vec3(0.0, 0.0, 0.0)
    lin = float(0.0)
    quad = float(0.0)
    for f in range(nf):
        typ = int(fa[f, 0])
        shape = int(fa[f, 1])
        fall = int(fa[f, 2])
        zd = int(fa[f, 3])
        z = wp.vec3(fcol(fa, fb, f, 25, t), fcol(fa, fb, f, 26, t), fcol(fa, fb, f, 27, t))
        zl = wp.length(z)
        if zl > 1.0e-12:
            z = z / zl
        else:
            z = wp.vec3(0.0, 0.0, 1.0)
        loc = wp.vec3(fcol(fa, fb, f, 22, t), fcol(fa, fb, f, 23, t), fcol(fa, fb, f, 24, t))
        vec2 = p - loc
        along = wp.dot(vec2, z)
        vec = vec2
        dist = wp.length(vec2)
        if shape == 1:
            vec = vec2 - along * z
            dist = wp.length(vec)
        elif shape == 2:
            vec = along * z
            dist = wp.abs(along)
        fo = float(1.0)
        if zd == 1 and along < 0.0:
            fo = 0.0
        if zd == 2 and along > 0.0:
            fo = 0.0
        power = fcol(fa, fb, f, 11, t)
        dmin = fcol(fa, fb, f, 12, t)
        dmax = fcol(fa, fb, f, 13, t)
        if fall == 0:
            fo = fo * falloff_func(dist, fa[f, 4], dmin, fa[f, 5], dmax, power)
        else:
            fo = fo * falloff_func(wp.abs(along), fa[f, 4], dmin, fa[f, 5], dmax, power)
            rfac = float(0.0)
            if fall == 1:
                rfac = wp.length(vec2 - along * z)
            else:
                l2 = wp.max(wp.length(vec2), 1.0e-12)
                rfac = wp.acos(wp.clamp(wp.abs(along) / l2, 0.0, 1.0)) * 57.29577951308232
            fo = fo * falloff_func(rfac, fa[f, 6], fcol(fa, fb, f, 15, t), fa[f, 7],
                                   fcol(fa, fb, f, 16, t), fcol(fa, fb, f, 14, t))
        s = fcol(fa, fb, f, 10, t)
        sf = s * fo
        if typ == 1:
            vl = wp.length(vec)
            if vl > 1.0e-12:
                acc = acc + vec * (sf / vl)
        elif typ == 2:
            air = air + z * sf
        elif typ == 3:
            if shape == 0:
                c = wp.cross(z, vec2)
                cl = wp.length(c)
                if cl > 1.0e-12:
                    acc = acc + c * (sf * dist / cl)
            else:
                tt = wp.cross(z, vec2) * sf
                acc = acc + wp.cross(z, tt) * sf + tt - v
        elif typ == 4:
            q = vec2
            if fa[f, 8] != 0.0:
                q = p
            q = q / fcol(fa, fb, f, 17, t)
            seed = int(fa[f, 9])
            n0 = turb1(q, seed)
            n1 = turb1(wp.vec3(q[1], q[2], q[0]), seed + 101)
            n2 = turb1(wp.vec3(q[2], q[0], q[1]), seed + 202)
            acc = acc + (2.0 * wp.vec3(n0, n1, n2) - wp.vec3(1.0, 1.0, 1.0)) * sf
        elif typ == 5:
            lin = lin + fcol(fa, fb, f, 18, t) * fo
            quad = quad + fcol(fa, fb, f, 19, t) * fo
        elif typ == 6:
            vv = vec
            rest = fcol(fa, fb, f, 21, t)
            if rest > 0.0:
                vl = wp.max(wp.length(vec), 1.0e-12)
                vv = vec * (1.0 - rest / vl)
            acc = acc - vv * sf - v * (fcol(fa, fb, f, 20, t) * 2.0 * wp.sqrt(wp.abs(s)))
    return wp.mat33(acc[0], acc[1], acc[2], air[0], air[1], air[2], lin, quad, 0.0)


@wp.kernel
def k_wind_exposure(grid: wp.uint64, x: wp.array(dtype=wp.vec3),
                    fa: wp.array2d(dtype=wp.float32), fb: wp.array2d(dtype=wp.float32), nf: int,
                    up: wp.vec3, d: float, expo: wp.array(dtype=wp.float32)):
    """Wind shadow (solver.SandSolver._wind_exposure): grains upwind shield."""
    i = wp.tid()
    m = field_eval(x[i], wp.vec3(0.0, 0.0, 0.0), fa, fb, nf, 0.0)
    air = wp.vec3(m[1, 0], m[1, 1], m[1, 2])
    sa = wp.length(air)
    e = float(1.0)
    if sa > 1.0e-6:
        u = air / sa
        rad = 0.7 * d
        for k in range(1, 4):
            smp = x[i] - u * (float(k) * d) + up * (0.35 * float(k) * d)
            blocked = int(0)
            q = wp.hash_grid_query(grid, smp, rad)
            j = int(0)
            while wp.hash_grid_query_next(q, j):
                if wp.length(x[j] - smp) < rad:
                    blocked = 1
            if blocked != 0:
                wgt = float(0.3)
                if k == 1:
                    wgt = 0.85
                elif k == 2:
                    wgt = 0.5
                e = e * (1.0 - wgt)
    expo[i] = e


@wp.kernel
def k_field_wake(x: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
                 fa: wp.array2d(dtype=wp.float32), fb: wp.array2d(dtype=wp.float32), nf: int,
                 gd: GrainData, drag_k: float, wind: int, up: wp.vec3, gref: float,
                 woke: wp.array(dtype=wp.int32), expo: wp.array(dtype=wp.float32)):
    """A sleeping grain that a field pushes sideways or up hard enough wakes up."""
    i = wp.tid()
    if asleep[i] == 0:
        return
    m = field_eval(x[i], wp.vec3(0.0, 0.0, 0.0), fa, fb, nf, 0.0)
    a = wp.vec3(m[0, 0], m[0, 1], m[0, 2])
    if gd.var_size != 0:
        a = a * gd.winv[i]
    if wind != 0 and drag_k > 0.0:
        air = wp.vec3(m[1, 0], m[1, 1], m[1, 2]) * expo[i]
        k = drag_k
        if gd.var_size != 0:
            k = drag_k * gd.inv_s[i]
        a = a + air * (k * wp.length(air))
    u = wp.dot(a, up)
    if wp.length(a - u * up) > 0.15 * gref or u > 0.3 * gref:
        woke[i] = 1


@wp.kernel
def k_field_max(x: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
                fa: wp.array2d(dtype=wp.float32), fb: wp.array2d(dtype=wp.float32), nf: int,
                gd: GrainData, drag_k: float, wind: int, out: wp.array(dtype=wp.float32),
                expo: wp.array(dtype=wp.float32)):
    """Largest field acceleration and air speed on the awake grains (substeps)."""
    i = wp.tid()
    if asleep[i] != 0:
        return
    m = field_eval(x[i], wp.vec3(0.0, 0.0, 0.0), fa, fb, nf, 0.0)
    a = wp.vec3(m[0, 0], m[0, 1], m[0, 2])
    air = wp.vec3(m[1, 0], m[1, 1], m[1, 2]) * expo[i]
    if gd.var_size != 0:
        a = a * gd.winv[i]
    if wind != 0 and drag_k > 0.0:
        k = drag_k
        if gd.var_size != 0:
            k = drag_k * gd.inv_s[i]
        a = a + air * (k * wp.length(air))
    wp.atomic_max(out, 0, wp.length(a))
    wp.atomic_max(out, 1, wp.length(air))


# ------------------------------------------------------------------- kernels
@wp.kernel
def k_predict(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
              asleep: wp.array(dtype=wp.int32), g: wp.vec3, h: float, drag_k: float, vcap: float,
              xp: wp.array(dtype=wp.vec3), vpred: wp.array(dtype=wp.vec3),
              spd: wp.array(dtype=wp.float32), stat_f: wp.array(dtype=wp.float32),
              stat_i: wp.array(dtype=wp.int32), gd: GrainData,
              fa: wp.array2d(dtype=wp.float32), fb: wp.array2d(dtype=wp.float32), nf: int,
              ft: float, expo: wp.array(dtype=wp.float32)):
    i = wp.tid()
    if asleep[i] != 0:
        xp[i] = x[i]
        vpred[i] = wp.vec3(0.0, 0.0, 0.0)
        spd[i] = 0.0
        return
    vi = v[i] + g * h
    s = wp.length(vi)
    if nf == 0:
        if drag_k > 0.0:
            k = drag_k
            if gd.var_size != 0:
                k = drag_k * gd.inv_s[i]              # area ~ s^2, mass ~ s^3
            f = 1.0 / (1.0 + k * s * h)             # semi-implicit quadratic drag
            vi = vi * f
            s = s * f
    else:
        # force fields: accelerations (a heavier grain less), drag fields and
        # the air drag relative to the wind
        m = field_eval(x[i], v[i], fa, fb, nf, ft)
        wi = float(1.0)
        if gd.var_size != 0:
            wi = gd.winv[i]
        vi = vi + wp.vec3(m[0, 0], m[0, 1], m[0, 2]) * (wi * h)
        lin = m[2, 0]
        quad = m[2, 1]
        if lin != 0.0 or quad != 0.0:
            vi = vi / (1.0 + h * wi * (lin + quad * wp.length(vi)))
        if drag_k > 0.0:
            k = drag_k
            if gd.var_size != 0:
                k = drag_k * gd.inv_s[i]
            air = wp.vec3(m[1, 0], m[1, 1], m[1, 2]) * expo[i]
            rel = vi - air
            vi = air + rel * (1.0 / (1.0 + k * wp.length(rel) * h))
        s = wp.length(vi)
    if s > vcap:
        vi = vi * (vcap / s)
        s = vcap
        wp.atomic_add(stat_i, 0, 1)
    vpred[i] = vi
    spd[i] = s
    xp[i] = x[i] + vi * h
    wp.atomic_max(stat_f, 0, s)


@wp.kernel
def k_wake_hit(grid: wp.uint64, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
               vframe: wp.array(dtype=wp.float32), asleep: wp.array(dtype=wp.int32),
               reach: float, fast_v: float, wv: float, woke: wp.array(dtype=wp.int32)):
    """A sleeping grain touched by a fast awake grain wakes up (impact: actual
    substep velocity above the solver noise; slow push: last frame's mean speed)."""
    i = wp.tid()
    if asleep[i] == 0:
        return
    p = x[i]
    q = wp.hash_grid_query(grid, p, reach)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j != i and asleep[j] == 0:
            if wp.length(x[j] - p) < reach:
                if wp.length(v[j] - v[i]) > fast_v or vframe[j] > wv:
                    woke[i] = 1


@wp.kernel
def k_apply_wake(woke: wp.array(dtype=wp.int32), asleep: wp.array(dtype=wp.int32),
                 fresh: wp.array(dtype=wp.int32), grace: wp.array(dtype=wp.float32),
                 calm: wp.array(dtype=wp.float32), v: wp.array(dtype=wp.vec3),
                 vpred: wp.array(dtype=wp.vec3), grace_time: float, count: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if woke[i] == 0:
        return
    woke[i] = 0
    if asleep[i] == 0:
        return
    asleep[i] = 0
    fresh[i] = 1
    grace[i] = grace_time
    calm[i] = 0.0
    v[i] = wp.vec3(0.0, 0.0, 0.0)
    vpred[i] = wp.vec3(0.0, 0.0, 0.0)
    wp.atomic_add(count, 0, 1)


@wp.kernel
def k_count_pairs(grid: wp.uint64, x: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
                  reach: float, counts: wp.array(dtype=wp.int32)):
    i = wp.tid()
    p = x[i]
    q = wp.hash_grid_query(grid, p, reach)
    j = int(0)
    c = int(0)
    while wp.hash_grid_query_next(q, j):
        if j > i and (asleep[i] == 0 or asleep[j] == 0):
            if wp.length(x[j] - p) < reach:
                c += 1
    counts[i] = c


@wp.kernel
def k_fill_pairs(grid: wp.uint64, x: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
                 reach: float, offsets: wp.array(dtype=wp.int32),
                 pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32)):
    i = wp.tid()
    p = x[i]
    q = wp.hash_grid_query(grid, p, reach)
    j = int(0)
    k = offsets[i]
    while wp.hash_grid_query_next(q, j):
        if j > i and (asleep[i] == 0 or asleep[j] == 0):
            if wp.length(x[j] - p) < reach:
                pi[k] = i
                pj[k] = j
                k += 1


@wp.func
def pair_correction(i: int, j: int, p: wp.array(dtype=wp.vec3), x0: wp.array(dtype=wp.vec3),
                    asleep: wp.array(dtype=wp.int32), grace: wp.array(dtype=wp.float32),
                    prm: SimParams, friction: int, use_target: int,
                    acc: wp.array(dtype=wp.vec3), wsum: wp.array(dtype=wp.float32),
                    split: int, shift: wp.array(dtype=wp.vec3), accs: wp.array(dtype=wp.vec3),
                    gd: GrainData):
    dp = p[i] - p[j]
    dist2 = wp.dot(dp, dp)
    target = p_dist(i, j, prm, gd)
    if use_target != 0:
        if grace[i] > 0.0 or grace[j] > 0.0:
            target = wp.min(target, wp.length(x0[i] - x0[j]))
    if dist2 < target * target and dist2 > 1.0e-18:
        dist = wp.sqrt(dist2)
        pen = target - dist
        n = dp / dist
        corr = pen * n
        if friction != 0:
            rel = (p[i] - x0[i]) - (p[j] - x0[j])
            mu = p_mu(i, j, prm, gd)
            corr = corr + friction_corr(rel, n, pen, mu[0], mu[1])
        a = share_of_i(i, j, x0, asleep, prm, gd)
        ci = corr * a
        cj = ci - corr
        # part of the push that only removes overlap the pair had at the start
        # of the substep (or that shifts caused): measured on the unshifted
        # positions against the start distance, booked as a shift
        so = wp.vec3(0.0, 0.0, 0.0)
        if split != 0:
            dq = dp - (shift[i] - shift[j])
            real = wp.max(wp.min(wp.length(x0[i] - x0[j]), target) - wp.length(dq), 0.0)
            so = wp.max(pen - real, 0.0) * n
        si = so * a
        if asleep[i] == 0:
            wp.atomic_add(acc, i, ci * pen)
            wp.atomic_add(wsum, i, pen)
            if split != 0:
                wp.atomic_add(accs, i, si * pen)
        if asleep[j] == 0:
            wp.atomic_add(acc, j, cj * pen)
            wp.atomic_add(wsum, j, pen)
            if split != 0:
                wp.atomic_add(accs, j, (si - so) * pen)


@wp.kernel
def k_contacts(pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
               p: wp.array(dtype=wp.vec3), x0: wp.array(dtype=wp.vec3),
               asleep: wp.array(dtype=wp.int32), grace: wp.array(dtype=wp.float32),
               fresh: wp.array(dtype=wp.int32), only_fresh: int,
               prm: SimParams, friction: int, use_target: int,
               acc: wp.array(dtype=wp.vec3), wsum: wp.array(dtype=wp.float32),
               split: int, shift: wp.array(dtype=wp.vec3), accs: wp.array(dtype=wp.vec3),
               gd: GrainData):
    k = wp.tid()
    i = pi[k]
    j = pj[k]
    if only_fresh != 0:
        if fresh[i] == 0 and fresh[j] == 0:
            return
    pair_correction(i, j, p, x0, asleep, grace, prm, friction, use_target, acc, wsum,
                    split, shift, accs, gd)


@wp.kernel
def k_apply(p: wp.array(dtype=wp.vec3), acc: wp.array(dtype=wp.vec3),
            wsum: wp.array(dtype=wp.float32), omega: float,
            split: int, shift: wp.array(dtype=wp.vec3), accs: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    w = wsum[i]
    if w > 0.0:
        p[i] = p[i] + acc[i] * (omega / w)
        acc[i] = wp.vec3(0.0, 0.0, 0.0)
        wsum[i] = 0.0
        if split != 0:
            shift[i] = shift[i] + accs[i] * (omega / w)
            accs[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_colliders(p: wp.array(dtype=wp.vec3), x0: wp.array(dtype=wp.vec3),
                asleep: wp.array(dtype=wp.int32), prm: SimParams, friction: int,
                col_data: wp.array(dtype=wp.vec4), col_origin: wp.array(dtype=wp.vec3),
                col_cell: wp.array(dtype=wp.float32), col_dims: wp.array(dtype=wp.vec3i),
                col_off: wp.array(dtype=wp.int32), ncol: int,
                split: int, shift: wp.array(dtype=wp.vec3), gd: GrainData,
                cmo: ColMotion, which: int):
    """All collider contacts of a grain resolved together: normal pushes
    summed, friction averaged (penetration weighted).  With split, the part of
    a push that only undoes what a shift pushed into the surface is booked as
    a shift too."""
    i = wp.tid()
    if asleep[i] != 0:
        return
    pi_ = p[i]
    disp = pi_ - x0[i]
    sh = wp.vec3(0.0, 0.0, 0.0)
    if split != 0:
        sh = shift[i]
    corr = wp.vec3(0.0, 0.0, 0.0)
    frs = wp.vec3(0.0, 0.0, 0.0)
    stored = wp.vec3(0.0, 0.0, 0.0)
    wsum = float(0.0)
    r = g_rad(i, prm, gd)
    cm = c_mu(i, prm, gd)
    if prm.use_floor != 0:
        hgt = pi_[2] - prm.floor_h
        if hgt < r:
            pen = r - hgt
            n = wp.vec3(0.0, 0.0, 1.0)
            corr = corr + pen * n
            stored = stored + wp.max(wp.min(pen, -wp.dot(sh, n)), 0.0) * n
            if friction != 0:
                frs = frs + friction_corr(disp, n, pen, cm[0], cm[1]) * pen
            wsum = wsum + pen
    for c in range(ncol):
        s = col_sample(pi_, c, which, cmo, col_data, col_origin, col_cell, col_dims, col_off)
        if s[0] < r:
            pen = r - s[0]
            gr = wp.vec3(s[1], s[2], s[3])
            ln = wp.length(gr)
            n = wp.vec3(0.0, 0.0, 1.0)
            if ln > 1.0e-9:
                n = gr / ln
            corr = corr + pen * n
            stored = stored + wp.max(wp.min(pen, -wp.dot(sh, n)), 0.0) * n
            if friction != 0:
                dsp = disp
                if cmo.any != 0 and which != 0:     # relative to a moving collider
                    dsp = disp - col_carry(pi_, c, cmo)
                frs = frs + friction_corr(dsp, n, pen, cm[0], cm[1]) * pen
            wsum = wsum + pen
    if wsum > 0.0:
        p[i] = pi_ + corr + frs / wsum
        if split != 0:
            shift[i] = sh + stored


@wp.kernel
def k_height_min(x: wp.array(dtype=wp.vec3), prm: SimParams, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_min(out, 0, wp.dot(x[i], prm.up))


@wp.kernel
def k_pass_keys(pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
                x0: wp.array(dtype=wp.vec3), prm: SimParams, hmin: wp.array(dtype=wp.float32),
                keys: wp.array(dtype=wp.int32), vals: wp.array(dtype=wp.int32),
                lmax: wp.array(dtype=wp.int32)):
    k = wp.tid()
    li = int(wp.floor((wp.dot(x0[pi[k]], prm.up) - hmin[0]) / prm.layer_h))
    lj = int(wp.floor((wp.dot(x0[pj[k]], prm.up) - hmin[0]) / prm.layer_h))
    key = wp.max(li, lj)
    keys[k] = key
    vals[k] = k
    wp.atomic_max(lmax, 0, key)


@wp.kernel
def k_histogram(keys: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32)):
    k = wp.tid()
    wp.atomic_add(counts, keys[k], 1)


@wp.kernel
def k_sweep_pass(order: wp.array(dtype=wp.int32), start: int,
                 pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
                 p: wp.array(dtype=wp.vec3), x0: wp.array(dtype=wp.vec3),
                 asleep: wp.array(dtype=wp.int32), grace: wp.array(dtype=wp.float32),
                 prm: SimParams, friction: int, use_target: int,
                 acc: wp.array(dtype=wp.vec3), wsum: wp.array(dtype=wp.float32),
                 split: int, shift: wp.array(dtype=wp.vec3), accs: wp.array(dtype=wp.vec3),
                 gd: GrainData):
    k = order[start + wp.tid()]
    pair_correction(pi[k], pj[k], p, x0, asleep, grace, prm, friction, use_target, acc, wsum,
                    split, shift, accs, gd)


@wp.kernel
def k_sweep_apply(order: wp.array(dtype=wp.int32), start: int,
                  pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
                  p: wp.array(dtype=wp.vec3), acc: wp.array(dtype=wp.vec3),
                  wsum: wp.array(dtype=wp.float32), stamp: wp.array(dtype=wp.int32), token: int,
                  split: int, shift: wp.array(dtype=wp.vec3), accs: wp.array(dtype=wp.vec3)):
    """Apply the layer's corrections once per grain (first thread wins)."""
    k = order[start + wp.tid()]
    for side in range(2):
        g = pi[k]
        if side == 1:
            g = pj[k]
        if wp.atomic_max(stamp, g, token) < token:
            w = wsum[g]
            if w > 0.0:
                p[g] = p[g] + acc[g] / w
                acc[g] = wp.vec3(0.0, 0.0, 0.0)
                wsum[g] = 0.0
                if split != 0:
                    shift[g] = shift[g] + accs[g] / w
                    accs[g] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_cap_pairs(pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
                spd: wp.array(dtype=wp.float32), cap: wp.array(dtype=wp.float32)):
    k = wp.tid()
    i = pi[k]
    j = pj[k]
    wp.atomic_max(cap, i, spd[j])
    wp.atomic_max(cap, j, spd[i])


@wp.kernel
def k_velocity(xp: wp.array(dtype=wp.vec3), x0: wp.array(dtype=wp.vec3),
               asleep: wp.array(dtype=wp.int32), cap: wp.array(dtype=wp.float32),
               v_guard: float, guard: int, inv_h: float, v: wp.array(dtype=wp.vec3),
               split: int, shift: wp.array(dtype=wp.vec3), v_dep: float):
    i = wp.tid()
    if asleep[i] != 0:
        return
    vn = (xp[i] - x0[i]) * inv_h
    if split != 0:
        # removing old overlap speeds a grain up along the push only up to
        # v_dep (a grain already moving that fast that way gets nothing more)
        vs = shift[i] * inv_h
        sv = wp.length(vs)
        if sv > 1.0e-12:
            vr = vn - vs
            u = vs / sv
            add = wp.min(sv, wp.max(v_dep - wp.dot(vr, u), 0.0))
            vn = vr + u * add
    if guard != 0:
        # inelastic contacts can not make a grain faster than the fastest
        # thing touching it: cap numerical energy injection
        c = cap[i] + v_guard
        s = wp.length(vn)
        if s > c:
            vn = vn * (c / s)
    v[i] = vn


@wp.kernel
def k_shift(xp: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3),
            x_before: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    xp[i] = xp[i] + (x[i] - x_before[i])


@wp.kernel
def k_wake_pen(pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
               xp: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
               thr2: float, woke: wp.array(dtype=wp.int32), frac: float, prm: SimParams,
               gd: GrainData):
    """A sleeping grain that an awake grain still presses into wakes up."""
    k = wp.tid()
    i = pi[k]
    j = pj[k]
    if (asleep[i] != 0) == (asleep[j] != 0):
        return
    dp = xp[i] - xp[j]
    t2 = thr2
    if gd.var_size != 0:
        t = frac * p_dist(i, j, prm, gd)
        t2 = t * t
    if wp.dot(dp, dp) < t2:
        if asleep[i] != 0:
            woke[i] = 1
        else:
            woke[j] = 1


@wp.kernel
def k_grace(grace: wp.array(dtype=wp.float32), h: float, fresh: wp.array(dtype=wp.int32)):
    i = wp.tid()
    grace[i] = wp.max(grace[i] - h, 0.0)
    fresh[i] = 0


@wp.kernel
def k_frame_contacts(grid: wp.uint64, x: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
                     prm: SimParams,
                     col_data: wp.array(dtype=wp.vec4), col_origin: wp.array(dtype=wp.vec3),
                     col_cell: wp.array(dtype=wp.float32), col_dims: wp.array(dtype=wp.vec3i),
                     col_off: wp.array(dtype=wp.int32), ncol: int, use_sleep: int,
                     nsum: wp.array(dtype=wp.vec3), woke: wp.array(dtype=wp.int32),
                     gd: GrainData, v: wp.array(dtype=wp.vec3), vbar: wp.array(dtype=wp.vec3),
                     cmo: ColMotion):
    """Contact normals and the mean velocity of the contacts (visual rotation)
    and the support test: a sleeping grain next to awake grains that has
    nothing below it any more wakes up."""
    i = wp.tid()
    p = x[i]
    rad = 1.05 * prm.d
    near = 2.0 * rad
    ri = g_rad(i, prm, gd)
    if gd.var_size != 0:
        near = 2.0 * (1.05 * gd.dmax)
    ns = wp.vec3(0.0, 0.0, 0.0)
    vs = wp.vec3(0.0, 0.0, 0.0)
    cs = float(0.0)
    support = int(0)
    near_awake = int(0)
    q = wp.hash_grid_query(grid, p, near)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j != i:
            dp = p - x[j]
            dist = wp.length(dp)
            if dist < near and asleep[j] == 0:
                near_awake = 1
            cr = rad
            if gd.var_size != 0:
                cr = 1.05 * (ri + gd.rad[j])
            if dist < cr and dist > 1.0e-12:
                n = dp / dist
                ns = ns + n
                vs = vs + v[j]
                cs = cs + 1.0
                if wp.dot(n, prm.up) > 0.2:
                    support = 1
    if prm.use_floor != 0:
        if p[2] - prm.floor_h < 1.05 * ri:
            ns = ns + wp.vec3(0.0, 0.0, 1.0)
            cs = cs + 1.0
            support = 1
    for c in range(ncol):
        s = col_sample(p, c, 1, cmo, col_data, col_origin, col_cell, col_dims, col_off)
        if s[0] < 1.05 * ri:
            gr = wp.vec3(s[1], s[2], s[3])
            ln = wp.length(gr)
            n = wp.vec3(0.0, 0.0, 1.0)
            if ln > 1.0e-9:
                n = gr / ln
            ns = ns + n
            cs = cs + 1.0
            if wp.dot(n, prm.up) > 0.2:
                support = 1
    nsum[i] = ns
    vbar[i] = vs / wp.max(cs, 1.0)
    if use_sleep != 0 and asleep[i] != 0 and near_awake != 0 and support == 0:
        woke[i] = 1


@wp.kernel
def k_sleep(x: wp.array(dtype=wp.vec3), x_start: wp.array(dtype=wp.vec3),
            asleep: wp.array(dtype=wp.int32), v: wp.array(dtype=wp.vec3),
            omega: wp.array(dtype=wp.vec3), calm: wp.array(dtype=wp.float32),
            vframe: wp.array(dtype=wp.float32), dt: float, v_sleep: float, sleep_time: float,
            use_sleep: int):
    i = wp.tid()
    vframe[i] = 0.0
    if asleep[i] != 0:
        return
    s = wp.length(x[i] - x_start[i]) / dt
    vframe[i] = s
    if use_sleep == 0:
        return
    if s < v_sleep:
        calm[i] = calm[i] + dt
    else:
        calm[i] = 0.0
    if calm[i] >= sleep_time:
        asleep[i] = 1
        v[i] = wp.vec3(0.0, 0.0, 0.0)
        omega[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_rotation(asleep: wp.array(dtype=wp.int32), nsum: wp.array(dtype=wp.vec3),
               v: wp.array(dtype=wp.vec3), omega: wp.array(dtype=wp.vec3),
               quat: wp.array(dtype=wp.vec4), r: float, tumble: float, dt: float,
               gd: GrainData, vbar: wp.array(dtype=wp.vec3), cap: float):
    """Purely visual tumbling of the cubes (solver.SandSolver.update_rotation):
    rolling relative to the contacts, at most `cap` radians per frame."""
    i = wp.tid()
    if asleep[i] == 0:
        ns = nsum[i]
        ln = wp.length(ns)
        if ln > 0.0:
            nh = ns / wp.max(ln, 1.0e-12)
            rr = r
            if gd.var_size != 0:
                rr = gd.rad[i]
            vrel = v[i] - vbar[i]
            omega[i] = 0.5 * omega[i] + 0.5 * (wp.cross(nh, vrel) / rr * (tumble * wp.min(ln, 1.0)))
        else:
            omega[i] = omega[i] * 0.98
    om = omega[i]
    w = wp.length(om)
    ang = w * dt
    if ang > cap:
        om = om * (cap / ang)
        omega[i] = om
        w = wp.length(om)
        ang = cap
    if ang > 1.0e-7:
        axis = om / w
        half = 0.5 * ang
        sh = wp.sin(half)
        dq = wp.vec4(wp.cos(half), axis[0] * sh, axis[1] * sh, axis[2] * sh)
        q = quat_mul(dq, quat[i])
        quat[i] = q / wp.length(q)


@wp.kernel
def k_frame_stats(v: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
                  stat_f: wp.array(dtype=wp.float32), stat_i: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if asleep[i] == 0:
        wp.atomic_add(stat_i, 0, 1)
        wp.atomic_max(stat_f, 0, wp.length(v[i]))


# ------------------------------------------------------------------ contact dynamics
# (the impulse model, solver.SandSolver._substep_impulse)
@wp.func
def coulomb(t: wp.vec3, vt: wp.vec3, nimp: float, mus: float, muk: float):
    """Friction impulse after trying to stop the tangential velocity vt:
    stick inside the static cone, otherwise slide with |T| = muk N."""
    tn = t - vt
    tl = wp.length(tn)
    if tl > mus * nimp:
        tn = tn * (muk * nimp / wp.max(tl, 1.0e-30))
    return tn


@wp.func
def layer_of(p: wp.vec3, prm: SimParams, hmin: float):
    return int(wp.floor((wp.dot(p, prm.up) - hmin) / prm.layer_h))


@wp.kernel
def k_col_contacts(x: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32), prm: SimParams,
                   margin: float,
                   col_data: wp.array(dtype=wp.vec4), col_origin: wp.array(dtype=wp.vec3),
                   col_cell: wp.array(dtype=wp.float32), col_dims: wp.array(dtype=wp.vec3i),
                   col_off: wp.array(dtype=wp.int32), ncol: int, nc: int,
                   cnr: wp.array(dtype=wp.vec3), cgap: wp.array(dtype=wp.float32),
                   gd: GrainData, cmo: ColMotion, cvel: wp.array(dtype=wp.vec3)):
    """Floor (slot 0) and collider contacts of every awake grain; gap >= 1e8: none.
    Moving colliders: cvel = velocity of the surface at the contact."""
    i = wp.tid()
    for q in range(nc):
        cgap[i * nc + q] = 1.0e9
        if cmo.any != 0:
            cvel[i * nc + q] = wp.vec3(0.0, 0.0, 0.0)
    if asleep[i] != 0:
        return
    p = x[i]
    r = g_rad(i, prm, gd)
    if prm.use_floor != 0:
        hgt = p[2] - prm.floor_h
        if hgt < margin:
            cnr[i * nc] = wp.vec3(0.0, 0.0, 1.0)
            cgap[i * nc] = hgt - r
    for c in range(ncol):
        s = col_sample(p, c, 0, cmo, col_data, col_origin, col_cell, col_dims, col_off)
        if s[0] < margin:
            gr = wp.vec3(s[1], s[2], s[3])
            ln = wp.length(gr)
            n = wp.vec3(0.0, 0.0, 1.0)
            if ln > 1.0e-9:
                n = gr / ln
            cnr[i * nc + c + 1] = n
            cgap[i * nc + c + 1] = s[0] - r
            if cmo.any != 0:
                if cmo.mov[c] != 0:
                    ql = wp.transpose(cmo.R0[c]) * (p - cmo.t0[c])
                    cvel[i * nc + c + 1] = (cmo.R1[c] * ql + cmo.t1[c] - p) * cmo.inv_h


@wp.kernel
def k_col_wake(x: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
               col_data: wp.array(dtype=wp.vec4), col_origin: wp.array(dtype=wp.vec3),
               col_cell: wp.array(dtype=wp.float32), col_dims: wp.array(dtype=wp.vec3i),
               col_off: wp.array(dtype=wp.int32), ncol: int, cmo: ColMotion, reach: float,
               woke: wp.array(dtype=wp.int32)):
    """Sleeping grains a moving collider reaches (or leaves) this frame wake up
    (poses: start and end of the frame)."""
    i = wp.tid()
    if asleep[i] == 0:
        return
    for c in range(ncol):
        if cmo.mov[c] != 0:
            for w in range(2):
                s = col_sample(x[i], c, w, cmo, col_data, col_origin, col_cell, col_dims, col_off)
                if s[0] < reach:
                    woke[i] = 1


@wp.kernel
def k_warm_pairs(pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
                 x: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
                 poff: wp.array(dtype=wp.int32), pcnt: wp.array(dtype=wp.int32),
                 ppj: wp.array(dtype=wp.int32), pjn: wp.array(dtype=wp.float32),
                 pjt: wp.array(dtype=wp.vec3), warm: float, prm: SimParams,
                 jn: wp.array(dtype=wp.float32), jt: wp.array(dtype=wp.vec3),
                 dv: wp.array(dtype=wp.vec3), gd: GrainData):
    """Impulses of the same pair in the previous substep, applied at once."""
    k = wp.tid()
    i = pi[k]
    j = pj[k]
    mus = p_mu(i, j, prm, gd)[0]
    a = float(0.0)
    t = wp.vec3(0.0, 0.0, 0.0)
    o = poff[i]
    for q in range(pcnt[i]):
        if ppj[o + q] == j:
            a = pjn[o + q]
            t = pjt[o + q]
    dp = x[i] - x[j]
    n = dp / wp.max(wp.length(dp), 1.0e-30)
    t = (t - wp.dot(t, n) * n) * warm
    a = a * warm
    af = a
    if gd.use_coh != 0:
        cp = p_coh(i, j, wp.length(dp), p_dist(i, j, prm, gd), gd)
        a = wp.max(a, -cp)
        af = a + cp
    tl = wp.length(t)
    if tl > mus * af:
        t = t * (mus * af / wp.max(tl, 1.0e-30))
    jn[k] = a
    jt[k] = t
    imp = a * n + t
    if asleep[i] == 0:
        wp.atomic_add(dv, i, imp * inv_mass(i, asleep, gd))
    if asleep[j] == 0:
        wp.atomic_sub(dv, j, imp * inv_mass(j, asleep, gd))


@wp.kernel
def k_warm_cols(asleep: wp.array(dtype=wp.int32), cnr: wp.array(dtype=wp.vec3),
                cgap: wp.array(dtype=wp.float32), cn_: wp.array(dtype=wp.float32),
                ct_: wp.array(dtype=wp.vec3), warm: float, prm: SimParams, nc: int,
                dv: wp.array(dtype=wp.vec3), gd: GrainData):
    i = wp.tid()
    acc = wp.vec3(0.0, 0.0, 0.0)
    cmus = c_mu(i, prm, gd)[0]
    for s in range(nc):
        idx = i * nc + s
        if cgap[idx] > 1.0e8 or asleep[i] != 0:
            cn_[idx] = 0.0
            ct_[idx] = wp.vec3(0.0, 0.0, 0.0)
        else:
            n = cnr[idx]
            a = cn_[idx] * warm
            af = a
            if gd.use_adh != 0:
                ca = c_adh(i, cgap[idx], prm, gd)
                a = wp.max(a, -ca)
                af = a + ca
            t = ct_[idx]
            t = (t - wp.dot(t, n) * n) * warm
            tl = wp.length(t)
            if tl > cmus * af:
                t = t * (cmus * af / wp.max(tl, 1.0e-30))
            cn_[idx] = a
            ct_[idx] = t
            acc = acc + a * n + t
    if asleep[i] == 0:
        wp.atomic_add(dv, i, acc)


@wp.kernel
def k_add_dv(v: wp.array(dtype=wp.vec3), dv: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    v[i] = v[i] + dv[i]
    dv[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_imp_count_cols(v: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
                     cnr: wp.array(dtype=wp.vec3), cgap: wp.array(dtype=wp.float32),
                     cn_: wp.array(dtype=wp.float32), nc: int, inv_h: float,
                     cnt: wp.array(dtype=wp.int32), prm: SimParams, gd: GrainData,
                     cmo: ColMotion, cvel: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    c = int(0)
    if asleep[i] == 0:
        for s in range(nc):
            idx = i * nc + s
            if cgap[idx] < 1.0e8:
                vi = v[i]
                if cmo.any != 0:            # relative to a moving surface
                    vi = vi - cvel[idx]
                viol = -wp.max(cgap[idx], 0.0) * inv_h - wp.dot(vi, cnr[idx])
                if cn_[idx] > 0.0 or viol > 0.0:
                    c += 1
                elif gd.use_adh != 0:
                    if cn_[idx] != 0.0 or (c_adh(i, cgap[idx], prm, gd) > 0.0
                                           and wp.dot(vi, cnr[idx]) > 0.0):
                        c += 1
    cnt[i] = c


@wp.kernel
def k_imp_count_pairs(pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
                      x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                      jn: wp.array(dtype=wp.float32), prm: SimParams, inv_h: float,
                      cnt: wp.array(dtype=wp.int32), act: wp.array(dtype=wp.int32),
                      gd: GrainData):
    k = wp.tid()
    i = pi[k]
    j = pj[k]
    dp = x[i] - x[j]
    dist = wp.length(dp)
    n = dp / wp.max(dist, 1.0e-30)
    viol = -wp.max(dist - p_dist(i, j, prm, gd), 0.0) * inv_h - wp.dot(v[i] - v[j], n)
    a = int(0)
    if jn[k] > 0.0 or viol > 0.0:
        a = 1
    elif gd.use_coh != 0:           # a water bridge acts when the pair separates
        if jn[k] != 0.0 or (p_coh(i, j, dist, p_dist(i, j, prm, gd), gd) > 0.0
                            and wp.dot(v[i] - v[j], n) > 0.0):
            a = 1
    if a != 0:
        wp.atomic_add(cnt, i, 1)
        wp.atomic_add(cnt, j, 1)
    act[k] = a


@wp.kernel
def k_imp_pairs(pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
                x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                asleep: wp.array(dtype=wp.int32), jn: wp.array(dtype=wp.float32),
                jt: wp.array(dtype=wp.vec3), act: wp.array(dtype=wp.int32),
                cnt: wp.array(dtype=wp.int32), prm: SimParams, inv_h: float,
                dv: wp.array(dtype=wp.vec3), gd: GrainData):
    """One Jacobi impulse on a pair: equal and opposite, the grain masses split
    over their contacts; no penetration, no pulling, Coulomb friction."""
    k = wp.tid()
    if act[k] == 0:
        return
    i = pi[k]
    j = pj[k]
    wi = inv_mass(i, asleep, gd)
    wj = inv_mass(j, asleep, gd)
    den = float(cnt[i]) * wi + float(cnt[j]) * wj
    dp = x[i] - x[j]
    dist = wp.length(dp)
    n = dp / wp.max(dist, 1.0e-30)
    vr = v[i] - v[j]
    vn = wp.dot(vr, n)
    pd = p_dist(i, j, prm, gd)
    viol = -wp.max(dist - pd, 0.0) * inv_h - vn
    jn_new = wp.max(jn[k] + viol / den, 0.0)
    jf = jn_new
    if gd.use_coh != 0:
        # a wet pair: pushed apart as always, or else held by its water
        # bridge - it stops separating, with at most the bridge's strength
        cp = p_coh(i, j, dist, pd, gd)
        jp = jn[k] + viol / den
        if jp > 0.0:
            jn_new = jp
        else:
            jn_new = wp.clamp(jn[k] - vn / den, -cp, 0.0)
        jf = jn_new + cp
    vt = vr - vn * n
    mu = p_mu(i, j, prm, gd)
    t_new = coulomb(jt[k], vt / den, jf, mu[0], mu[1])
    imp = (jn_new - jn[k]) * n + (t_new - jt[k])
    jn[k] = jn_new
    jt[k] = t_new
    if wi > 0.0:
        wp.atomic_add(dv, i, imp * wi)
    if wj > 0.0:
        wp.atomic_sub(dv, j, imp * wj)


@wp.kernel
def k_imp_cols(v: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
               cnr: wp.array(dtype=wp.vec3), cgap: wp.array(dtype=wp.float32),
               cn_: wp.array(dtype=wp.float32), ct_: wp.array(dtype=wp.vec3),
               cnt: wp.array(dtype=wp.int32), prm: SimParams, nc: int, inv_h: float,
               dv: wp.array(dtype=wp.vec3), gd: GrainData,
               cmo: ColMotion, cvel: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if asleep[i] != 0:
        return
    den = float(cnt[i])
    cm = c_mu(i, prm, gd)
    acc = wp.vec3(0.0, 0.0, 0.0)
    for s in range(nc):
        idx = i * nc + s
        vi = v[i]
        if cmo.any != 0:                    # relative to a moving surface
            vi = vi - cvel[idx]
        if cgap[idx] < 1.0e8:
            n = cnr[idx]
            vn = wp.dot(vi, n)
            viol = -wp.max(cgap[idx], 0.0) * inv_h - vn
            ca = float(0.0)
            on = int(0)
            if cn_[idx] > 0.0 or viol > 0.0:
                on = 1
            elif gd.use_adh != 0:
                ca = c_adh(i, cgap[idx], prm, gd)
                if cn_[idx] != 0.0 or (ca > 0.0 and vn > 0.0):
                    on = 1
            if on != 0:
                a_new = wp.max(cn_[idx] + viol / den, 0.0)
                af = a_new
                if gd.use_adh != 0:     # a wet grain sticks to the surface
                    ca = c_adh(i, cgap[idx], prm, gd)
                    jp = cn_[idx] + viol / den
                    if jp > 0.0:
                        a_new = jp
                    else:
                        a_new = wp.clamp(cn_[idx] - vn / den, -ca, 0.0)
                    af = a_new + ca
                vt = vi - vn * n
                t_new = coulomb(ct_[idx], vt / den, af, cm[0], cm[1])
                acc = acc + (a_new - cn_[idx]) * n + (t_new - ct_[idx])
                cn_[idx] = a_new
                ct_[idx] = t_new
    wp.atomic_add(dv, i, acc)


@wp.func
def support_cols(i: int, v: wp.array(dtype=wp.vec3), cnr: wp.array(dtype=wp.vec3),
                 cgap: wp.array(dtype=wp.float32), cN: wp.array(dtype=wp.float32),
                 cT: wp.array(dtype=wp.vec3), prm: SimParams, nc: int, inv_h: float,
                 gd: GrainData, cmo: ColMotion, cvel: wp.array(dtype=wp.vec3)):
    """The floor and colliders stop what still moves into them (all contacts of
    the grain from the same velocity, summed), Coulomb friction against the
    total normal impulse."""
    vi0 = v[i]
    acc = wp.vec3(0.0, 0.0, 0.0)
    cm = c_mu(i, prm, gd)
    for s in range(nc):
        idx = i * nc + s
        if cgap[idx] < 1.0e8:
            n = cnr[idx]
            vi = vi0
            if cmo.any != 0:                # relative to a moving surface
                vi = vi0 - cvel[idx]
            vn = wp.dot(vi, n)
            push = wp.max(-wp.max(cgap[idx], 0.0) * inv_h - vn, 0.0)
            if push > 0.0:
                nimp = cN[idx] + push
                vt = vi - vn * n
                t_new = coulomb(cT[idx], vt, nimp, cm[0], cm[1])
                acc = acc + push * n + (t_new - cT[idx])
                cN[idx] = nimp
                cT[idx] = t_new
    v[i] = vi0 + acc


@wp.kernel
def k_sup_cols_all(v: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=wp.int32),
                   cnr: wp.array(dtype=wp.vec3), cgap: wp.array(dtype=wp.float32),
                   cN: wp.array(dtype=wp.float32), cT: wp.array(dtype=wp.vec3),
                   prm: SimParams, nc: int, inv_h: float, gd: GrainData,
                   cmo: ColMotion, cvel: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if asleep[i] == 0:
        support_cols(i, v, cnr, cgap, cN, cT, prm, nc, inv_h, gd, cmo, cvel)


@wp.kernel
def k_sup_init(pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
               asleep: wp.array(dtype=wp.int32), jn: wp.array(dtype=wp.float32),
               jt: wp.array(dtype=wp.vec3), nrel: wp.array(dtype=wp.float32),
               trel: wp.array(dtype=wp.vec3), gd: GrainData,
               x: wp.array(dtype=wp.vec3), prm: SimParams):
    k = wp.tid()
    w = inv_mass(pi[k], asleep, gd) + inv_mass(pj[k], asleep, gd)
    a = jn[k]
    if gd.use_coh != 0:     # friction cone: normal impulse + what the bridge holds
        i = pi[k]
        j = pj[k]
        a = a + p_coh(i, j, wp.length(x[i] - x[j]), p_dist(i, j, prm, gd), gd)
    nrel[k] = a * w
    trel[k] = jt[k] * w


@wp.kernel
def k_sup_adh(asleep: wp.array(dtype=wp.int32), cgap: wp.array(dtype=wp.float32),
              cN: wp.array(dtype=wp.float32), prm: SimParams, nc: int, gd: GrainData):
    """Friction cone of the surfaces: normal impulse + what adhesion holds."""
    i = wp.tid()
    if asleep[i] != 0:
        return
    for s in range(nc):
        idx = i * nc + s
        if cgap[idx] < 1.0e8:
            cN[idx] = cN[idx] + c_adh(i, cgap[idx], prm, gd)


@wp.kernel
def k_sup_pass(order: wp.array(dtype=wp.int32), start: int,
               pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
               x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
               asleep: wp.array(dtype=wp.int32), prm: SimParams, hmin: wp.array(dtype=wp.float32),
               inv_h: float, nrel: wp.array(dtype=wp.float32), trel: wp.array(dtype=wp.vec3),
               acc: wp.array(dtype=wp.vec3), wsum: wp.array(dtype=wp.float32),
               gd: GrainData, jn: wp.array(dtype=wp.float32)):
    """Support sweep, one pair of the layer: stop what still approaches, the
    upper grain taking the change (inside a layer: shared), with friction."""
    k = order[start + wp.tid()]
    i = pi[k]
    j = pj[k]
    dp = x[i] - x[j]
    dist = wp.length(dp)
    n = dp / wp.max(dist, 1.0e-30)
    vr = v[i] - v[j]
    vn = wp.dot(vr, n)
    push = wp.max(-wp.max(dist - p_dist(i, j, prm, gd), 0.0) * inv_h - vn, 0.0)
    if push <= 0.0:
        return
    nimp = nrel[k] + push
    vt = vr - vn * n
    mu = p_mu(i, j, prm, gd)
    t_new = coulomb(trel[k], vt, nimp, mu[0], mu[1])
    corr = push * n + (t_new - trel[k])
    nrel[k] = nimp
    trel[k] = t_new
    li = layer_of(x[i], prm, hmin[0])
    lj = layer_of(x[j], prm, hmin[0])
    a = share_of_i(i, j, x, asleep, prm, gd)
    if asleep[i] == 0 and asleep[j] == 0:
        if li > lj:
            a = 1.0
        elif li < lj:
            a = 0.0
    if gd.use_coh != 0:
        # a pair that its water bridge holds in tension is no support from
        # below (the lower grain hangs on it): shared by mass, momentum conserving
        if jn[k] < 0.0:
            wi = inv_mass(i, asleep, gd)
            wj = inv_mass(j, asleep, gd)
            a = 0.0
            if wi + wj > 0.0:
                a = wi / (wi + wj)
    ci = corr * a
    wp.atomic_add(acc, i, ci * push)
    wp.atomic_add(wsum, i, push)
    wp.atomic_add(acc, j, (ci - corr) * push)
    wp.atomic_add(wsum, j, push)


@wp.kernel
def k_sup_apply(order: wp.array(dtype=wp.int32), start: int,
                pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
                x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                acc: wp.array(dtype=wp.vec3), wsum: wp.array(dtype=wp.float32),
                stamp: wp.array(dtype=wp.int32), token: int, layer: int,
                asleep: wp.array(dtype=wp.int32), prm: SimParams, hmin: wp.array(dtype=wp.float32),
                cnr: wp.array(dtype=wp.vec3), cgap: wp.array(dtype=wp.float32),
                cN: wp.array(dtype=wp.float32), cT: wp.array(dtype=wp.vec3),
                nc: int, inv_h: float, gd: GrainData,
                cmo: ColMotion, cvel: wp.array(dtype=wp.vec3)):
    """Apply the layer's pair corrections once per grain, then the floor and
    collider contacts of the layer's own grains."""
    k = order[start + wp.tid()]
    for side in range(2):
        g = pi[k]
        if side == 1:
            g = pj[k]
        if wp.atomic_max(stamp, g, token) < token:
            w = wsum[g]
            if w > 0.0:
                v[g] = v[g] + acc[g] / w
                acc[g] = wp.vec3(0.0, 0.0, 0.0)
                wsum[g] = 0.0
            if asleep[g] == 0 and layer_of(x[g], prm, hmin[0]) == layer:
                support_cols(g, v, cnr, cgap, cN, cT, prm, nc, inv_h, gd, cmo, cvel)


@wp.kernel
def k_positions(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                asleep: wp.array(dtype=wp.int32), h: float,
                xp: wp.array(dtype=wp.vec3), vout: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if asleep[i] != 0:
        xp[i] = x[i]
        return
    xp[i] = x[i] + v[i] * h
    vout[i] = v[i]


@wp.kernel
def k_rate_limit(xp: wp.array(dtype=wp.vec3), xb: wp.array(dtype=wp.vec3), lim: float):
    i = wp.tid()
    step = xp[i] - xb[i]
    sl = wp.length(step)
    if sl > lim:
        xp[i] = xb[i] + step * (lim / sl)


# -------------------------------------------------------------------- solver
class SandSolverGPU:
    """Drop-in replacement of solver.SandSolver running on a Warp device."""

    def __init__(self, positions, params, colliders=(), device="cuda:0", scale=None, wet=None):
        init()
        self.p = params
        self.device = wp.get_device(device)
        # the NumPy solver prepares the start state (same seeds, same code)
        # and the per grain size / friction
        self._host = SandSolver(positions, params, colliders, scale=scale, wet=wet)
        self.n = self._host.n
        self.d = self._host.d
        self.r = self._host.r
        self.g = self._host.g
        self.up = self._host.up
        self.v_sleep = self._host.v_sleep
        self.v_guard = self._host.v_guard
        self.v_dep = self._host.v_dep
        self.r_max = self._host.r_max
        self.d_min = self._host.d_min
        self.d_max = self._host.d_max
        self._split = 1 if params.depenetration > 0.0 else 0
        self._impulse = params.model != "PBD"
        self.colliders = list(colliders)
        self.last_substeps = 0
        self.clamped = 0
        self.time = 0.0
        self._awake = self.n
        self._uploaded = False
        self._grace_left = 0.0
        self._any_fresh = False
        self._col_travel = 0.0            # moving colliders: max travel per substep
        self._token = 0
        self._fnf = 0                   # number of force fields this frame
        self._field_rows = None
        self._field_t = 0.0

        prm = SimParams()
        prm.d = float(self.d)
        prm.r = float(self.r)
        prm.mu_s = float(params.friction_static)
        prm.mu_k = float(params.friction_dynamic)
        prm.cmu_s = float(params.collider_friction_static)
        prm.cmu_k = float(params.collider_friction_dynamic)
        prm.stacking = float(params.stacking)
        prm.up = wp.vec3(*[float(c) for c in self.up])
        prm.use_floor = 1 if params.use_floor else 0
        prm.floor_h = float(params.floor_height)
        prm.layer_h = float(params.layer_thickness * self.d)
        self.prm = prm

    # ------------------------------------------------------------ helpers
    def _arr(self, data, dtype):
        return wp.array(data, dtype=dtype, device=self.device)

    def _zeros(self, n, dtype):
        return wp.zeros(max(int(n), 1), dtype=dtype, device=self.device)

    def _launch(self, kernel, dim, inputs):
        if dim > 0:
            wp.launch(kernel, dim=int(dim), inputs=inputs, device=self.device)

    def compile(self):
        """Compile the kernels now (first use on a machine takes a while)."""
        with wp.ScopedDevice(self.device):
            wp.load_module(module=__name__, device=self.device)

    def resolve_initial_overlaps(self, **kw):
        return self._host.resolve_initial_overlaps(**kw)

    def _upload(self):
        h = self._host
        n = self.n
        with wp.ScopedDevice(self.device):
            self.x = self._arr(h.x.astype(np.float32), wp.vec3)
            self.v = self._arr(h.v.astype(np.float32), wp.vec3)
            self.quat = self._arr(h.quat.astype(np.float32), wp.vec4)
            self.omega = self._zeros(n, wp.vec3)
            self.asleep = self._zeros(n, wp.int32)
            self.fresh = self._zeros(n, wp.int32)
            self.woke = self._zeros(n, wp.int32)
            self.grace = self._zeros(n, wp.float32)
            self.calm = self._zeros(n, wp.float32)
            self.vframe = self._zeros(n, wp.float32)
            self.xp = self._zeros(n, wp.vec3)
            self.vpred = self._zeros(n, wp.vec3)
            self.spd = self._zeros(n, wp.float32)
            self.cap = self._zeros(n, wp.float32)
            self.acc = self._zeros(n, wp.vec3)
            self.wsum = self._zeros(n, wp.float32)
            self.shift = self._zeros(n, wp.vec3)
            self.accs = self._zeros(n, wp.vec3)
            # impulse model: floor / collider contacts per grain slot, velocity
            # changes, contact counts, warm start of the pairs
            nc = len(self.colliders) + 1
            self.nc = nc
            self.cnr = self._zeros(n * nc, wp.vec3)
            self.cgap = self._zeros(n * nc, wp.float32)
            self.cimp_n = self._zeros(n * nc, wp.float32)
            self.cimp_t = self._zeros(n * nc, wp.vec3)
            self.csup_n = self._zeros(n * nc, wp.float32)
            self.csup_t = self._zeros(n * nc, wp.vec3)
            self.dv = self._zeros(n, wp.vec3)
            self.cnt = self._zeros(n, wp.int32)
            self.xb = self._zeros(n, wp.vec3)
            self.poff = self._zeros(n, wp.int32)
            self.pcnt = self._zeros(n, wp.int32)
            self._prev_cap = 0
            self.x_before = self._zeros(n, wp.vec3)
            self.x_start = self._zeros(n, wp.vec3)
            self.nsum = self._zeros(n, wp.vec3)
            self.vbar = self._zeros(n, wp.vec3)
            self.stamp = self._zeros(n, wp.int32)
            self.counts = self._zeros(n, wp.int32)
            self.offsets = self._zeros(n, wp.int32)
            self.stat_f = self._zeros(1, wp.float32)
            self.stat_i = self._zeros(1, wp.int32)
            self.wake_count = self._zeros(1, wp.int32)
            self.hmin = self._zeros(1, wp.float32)
            self.lmax = self._zeros(1, wp.int32)
            self._pair_cap = 0
            self._layer_cap = 0
            gd = GrainData()
            if h.rad is not None:
                gd.var_size = 1
                gd.rad = self._arr(h.rad.astype(np.float32), wp.float32)
                gd.winv = self._arr(h.winv.astype(np.float32), wp.float32)
                gd.inv_s = self._arr((1.0 / h.scale).astype(np.float32), wp.float32)
            else:
                gd.var_size = 0
                gd.rad = gd.winv = gd.inv_s = self._zeros(1, wp.float32)
            gd.dmax = float(self.d_max)
            if h.mu is not None:
                gd.var_mu = 1
                gd.mu = self._arr(np.stack(h.mu, axis=1).astype(np.float32), wp.vec4)
            else:
                gd.var_mu = 0
                gd.mu = self._zeros(1, wp.vec4)
            gd.use_coh = 1 if h.coh is not None else 0
            gd.coh = self._arr(h.coh.astype(np.float32), wp.float32) if h.coh is not None \
                else self._zeros(1, wp.float32)
            gd.use_adh = 1 if h.adh is not None else 0
            gd.adh = self._arr(h.adh.astype(np.float32), wp.float32) if h.adh is not None \
                else self._zeros(1, wp.float32)
            gd.bridge = float(self.p.bridge)
            gd.ch = 0.0
            self.gd = gd
            self.fa = self.fb = wp.zeros((1, _fields.COLS), dtype=wp.float32, device=self.device)
            self.fstat = self._zeros(2, wp.float32)
            self.expo = wp.ones(max(n, 1), dtype=wp.float32, device=self.device)
            self.grid = wp.HashGrid(128, 128, 128, device=self.device)
            self.grid2 = wp.HashGrid(128, 128, 128, device=self.device)
            # colliders
            cols = self.colliders
            self.ncol = len(cols)
            if cols:
                data = np.concatenate([c.data for c in cols]).astype(np.float32)
                offs = np.cumsum([0] + [len(c.data) for c in cols[:-1]]).astype(np.int32)
                self.col_data = self._arr(data, wp.vec4)
                self.col_origin = self._arr(np.array([c.origin for c in cols], np.float32), wp.vec3)
                self.col_cell = self._arr(np.array([c.cell for c in cols], np.float32), wp.float32)
                self.col_dims = self._arr(np.array([c.dims for c in cols], np.int32), wp.vec3i)
                self.col_off = self._arr(offs, wp.int32)
            else:
                self.col_data = self._zeros(1, wp.vec4)
                self.col_origin = self._zeros(1, wp.vec3)
                self.col_cell = self._zeros(1, wp.float32)
                self.col_dims = self._zeros(1, wp.vec3i)
                self.col_off = self._zeros(1, wp.int32)
            # moving colliders (poses are uploaded per substep while any moves)
            m = max(self.ncol, 1)
            cm = ColMotion()
            cm.any = 0
            cm.mov = self._zeros(m, wp.int32)
            eye = np.tile(np.eye(3, dtype=np.float32), (m, 1, 1))
            cm.R0 = self._arr(eye, wp.mat33)
            cm.R1 = self._arr(eye, wp.mat33)
            cm.t0 = self._zeros(m, wp.vec3)
            cm.t1 = self._zeros(m, wp.vec3)
            cm.inv_h = 0.0
            self.cm = cm
            self.cvel = self._zeros(1, wp.vec3)
        self._uploaded = True

    def set_collider_motion(self, k, Ma=None, Mb=None, M0=None):
        """See solver.SandSolver.set_collider_motion."""
        self._host.set_collider_motion(k, Ma, Mb, M0)

    def _set_poses(self, s0, s1, h=None):
        """Upload the poses of the colliders at s0 and s1 of the frame."""
        h_ = self._host
        m = max(self.ncol, 1)
        mov = np.zeros(m, np.int32)
        R0 = np.tile(np.eye(3), (m, 1, 1))
        R1 = R0.copy()
        t0 = np.zeros((m, 3))
        t1 = np.zeros((m, 3))
        for k in range(self.ncol):
            mo = h_._cmotion[k]
            if mo is not None:
                mov[k] = 1
                R0[k], t0[k] = mo.pose(s0)
                R1[k], t1[k] = mo.pose(s1)
        cm = self.cm
        cm.mov.assign(mov)
        cm.R0.assign(R0.astype(np.float32))
        cm.R1.assign(R1.astype(np.float32))
        cm.t0.assign(t0.astype(np.float32))
        cm.t1.assign(t1.astype(np.float32))
        cm.inv_h = float(1.0 / h) if h else 0.0
        cm.any = 1

    def _col_args(self):
        return [self.col_data, self.col_origin, self.col_cell, self.col_dims, self.col_off, self.ncol]

    def _ensure_pairs(self, P):
        P = max(int(P), 1)
        if P > self._pair_cap:
            cap = int(P * 1.5) + 1024
            self.pi = self._zeros(cap, wp.int32)
            self.pj = self._zeros(cap, wp.int32)
            self.keys = self._zeros(2 * cap, wp.int32)
            self.vals = self._zeros(2 * cap, wp.int32)
            if self._impulse:
                self.jn = self._zeros(cap, wp.float32)
                self.jt = self._zeros(cap, wp.vec3)
                self.act = self._zeros(cap, wp.int32)
                self.nrel = self._zeros(cap, wp.float32)
                self.trel = self._zeros(cap, wp.vec3)
            self._pair_cap = cap

    def _ensure_prev(self, P):
        if P > self._prev_cap:
            cap = int(P * 1.5) + 1024
            self.ppj = self._zeros(cap, wp.int32)
            self.pjn = self._zeros(cap, wp.float32)
            self.pjt = self._zeros(cap, wp.vec3)
            self._prev_cap = cap

    def _ensure_layers(self, L):
        if L > self._layer_cap:
            cap = int(L * 1.5) + 64
            self.lcounts = self._zeros(cap, wp.int32)
            self.loffsets = self._zeros(cap, wp.int32)
            self._layer_cap = cap

    # ------------------------------------------------------------ public
    def set_fields(self, fa, fb=None):
        """Force fields for the next frame (see solver.SandSolver.set_fields)."""
        h = self._host
        h.set_fields(fa, fb)
        if h._fields is None:
            self._fnf = 0
            self._field_rows = None
            return
        a, b = h._fields
        self._field_rows = (a.astype(np.float32), (a if b is None else b).astype(np.float32))
        self._fnf = len(a)

    def _upload_fields(self):
        a, b = self._field_rows
        self.fa = wp.array(a, dtype=wp.float32, device=self.device)
        self.fb = wp.array(b, dtype=wp.float32, device=self.device)

    def _drag_k(self, fields):
        prm = self.p
        if (prm.air_drag or (fields and self._host._has_wind)) and prm.mass > 0.0:
            return 0.5 * prm.air_density * prm.drag_cd * prm.drag_area / prm.mass
        return 0.0

    @property
    def awake_count(self):
        return int(self._awake)

    def positions(self):
        if not self._uploaded:
            return self._host.x
        return self.x.numpy().astype(np.float64)

    def rotations(self):
        if not self._uploaded:
            return self._host.quat
        return self.quat.numpy().astype(np.float64)

    def step_frame(self, dt_frame):
        if not self._uploaded:
            self._upload()
        with wp.ScopedDevice(self.device):
            self._step_frame(dt_frame)

    def _read_stats(self):
        self.stat_f.zero_()
        self.stat_i.zero_()
        self._launch(k_frame_stats, self.n, [self.v, self.asleep, self.stat_f, self.stat_i])
        return float(self.stat_f.numpy()[0]), int(self.stat_i.numpy()[0])

    def _step_frame(self, dt_frame):
        prm = self.p
        self.clamped = 0
        use_fields = self._fnf > 0 and self._impulse
        n = self.n
        if use_fields:
            self._upload_fields()
            wind = 1 if self._host._has_wind else 0
            dk = float(self._drag_k(True))
            upv = wp.vec3(*[float(c) for c in self.up])
            if wind:
                self.grid2.build(self.x, 0.7 * self.d)
                self._launch(k_wind_exposure, n, [self.grid2.id, self.x, self.fa, self.fb, self._fnf,
                                                  upv, float(self.d), self.expo])
            else:
                self.expo.fill_(1.0)
            if prm.use_sleep:
                gl = float(np.linalg.norm(self.g))
                self._launch(k_field_wake, n, [self.x, self.asleep, self.fa, self.fb, self._fnf, self.gd,
                                               dk, wind, upv, float(gl if gl > 1e-3 else 9.81), self.woke,
                                               self.expo])
                self._apply_wakes()
        moving = self._host._any_motion()
        if moving:
            if self.cvel.shape[0] != n * self.nc and self._impulse:
                self.cvel = self._zeros(n * self.nc, wp.vec3)
            self._set_poses(0.0, 1.0)
            if prm.use_sleep:           # sleeping grains a collider reaches wake up
                reach = self.r_max + 0.25 * self.d + self._host._col_move(0.0, 1.0)
                self._launch(k_col_wake, n, [self.x, self.asleep] + self._col_args()
                             + [self.cm, float(reach), self.woke])
                self._apply_wakes()
        else:
            self.cm.any = 0
        self._col_travel = 0.0
        vmax, awake = self._read_stats()
        self._awake = awake
        if awake == 0:
            self.last_substeps = 0
            self.time += dt_frame
            return
        vmax += float(np.linalg.norm(self.g)) * dt_frame
        if moving:
            vmax += self._host._col_move(0.0, 1.0) / dt_frame
        if use_fields:          # the fields may speed the grains up as well
            self.fstat.zero_()
            self._launch(k_field_max, n, [self.x, self.asleep, self.fa, self.fb, self._fnf, self.gd,
                                          dk, wind, self.fstat, self.expo])
            amax, umax = [float(c) for c in self.fstat.numpy()[:2]]
            vmax += amax * dt_frame + umax
        need = math.ceil(vmax * dt_frame / max(prm.max_travel * self.d, 1e-12))
        nsub = int(min(max(need, prm.min_substeps), prm.max_substeps))
        h = dt_frame / nsub
        wp.copy(self.x_start, self.x)
        sub = self._substep_impulse if self._impulse else self._substep
        for k in range(nsub):
            self._field_t = (k + 0.5) / nsub
            if moving:
                s0, s1 = k / nsub, (k + 1) / nsub
                self._set_poses(s0, s1, h)
                self._col_travel = self._host._col_move(s0, s1)
            sub(h)
        self.last_substeps = nsub
        self.time += dt_frame
        self._frame_end(dt_frame)
        _, self._awake = self._read_stats()

    def _apply_wakes(self):
        self.wake_count.zero_()
        self._launch(k_apply_wake, self.n, [self.woke, self.asleep, self.fresh, self.grace,
                                            self.calm, self.v, self.vpred,
                                            float(self.p.wake_grace), self.wake_count])
        woke = int(self.wake_count.numpy()[0])
        if woke:
            self._any_fresh = True
            self._grace_left = self.p.wake_grace
        return woke

    def _sweep(self, p, x0, friction, use_target, offs, L, split=0):
        a = [self.vals, 0, self.pi, self.pj, p, x0, self.asleep, self.grace, self.prm,
             friction, use_target, self.acc, self.wsum, split, self.shift, self.accs, self.gd]
        b = [self.vals, 0, self.pi, self.pj, p, self.acc, self.wsum, self.stamp, 0,
             split, self.shift, self.accs]
        # one pair of launches per layer: recorded once, then only the layer
        # range changes (much less launch overhead)
        ca = wp.launch(k_sweep_pass, dim=1, inputs=a, device=self.device, record_cmd=True)
        cb = wp.launch(k_sweep_apply, dim=1, inputs=b, device=self.device, record_cmd=True)
        for layer in range(L):
            s0, s1 = int(offs[layer]), int(offs[layer + 1])
            if s1 <= s0:
                continue
            self._token += 1
            ca.set_param_at_index(1, s0)
            ca.set_dim(s1 - s0)
            ca.launch()
            cb.set_param_at_index(1, s0)
            cb.set_param_at_index(8, self._token)
            cb.set_dim(s1 - s0)
            cb.launch()

    def _substep(self, h):
        prm = self.p
        n = self.n
        d = self.d
        g = wp.vec3(*[float(c) for c in self.g])
        drag_k = 0.0
        if prm.air_drag and prm.mass > 0.0:
            drag_k = 0.5 * prm.air_density * prm.drag_cd * prm.drag_area / prm.mass
        vcap = 0.9 * d / h

        # 1) prediction
        self.stat_f.zero_()
        self.stat_i.zero_()
        self._launch(k_predict, n, [self.x, self.v, self.asleep, g, float(h), float(drag_k),
                                    float(vcap), self.xp, self.vpred, self.spd,
                                    self.stat_f, self.stat_i, self.gd, self.fa, self.fb, 0, 0.0,
                                    self.expo])
        spmax = float(self.stat_f.numpy()[0])
        self.clamped = max(self.clamped, int(self.stat_i.numpy()[0]))
        reach = 1.1 * d + 2.0 * spmax * h
        self.grid.build(self.x, reach)

        # 2) sleeping grains hit by a fast grain wake up (before the pairs,
        #    so that the pair list already sees them awake)
        if prm.use_sleep and self._awake < n:
            wv = prm.wake_factor * self.v_sleep
            fast_v = max(wv, 2.0 * float(np.linalg.norm(self.g)) * h)
            self._launch(k_wake_hit, n, [self.grid.id, self.x, self.v, self.vframe, self.asleep,
                                         float(reach), float(fast_v), float(wv), self.woke])
            self._apply_wakes()

        # contact pairs: every pair that can touch during this substep
        self._launch(k_count_pairs, n, [self.grid.id, self.x, self.asleep, float(reach), self.counts])
        wp.utils.array_scan(self.counts, self.offsets, inclusive=False)
        tail = np.concatenate([self.offsets[n - 1:n].numpy(), self.counts[n - 1:n].numpy()])
        P = int(tail[0] + tail[1])
        self._ensure_pairs(P)
        self._launch(k_fill_pairs, n, [self.grid.id, self.x, self.asleep, float(reach),
                                       self.offsets, self.pi, self.pj])

        # freshly woken grains: remove their old overlaps by moving positions
        # only (applied to the start and the predicted positions alike)
        if self._any_fresh and P:
            wp.copy(self.x_before, self.x)
            for _ in range(3):
                self._launch(k_contacts, P, [self.pi, self.pj, self.x, self.x, self.asleep,
                                             self.grace, self.fresh, 1, self.prm, 0, 0,
                                             self.acc, self.wsum, 0, self.shift, self.accs,
                                             self.gd])
                self._launch(k_apply, n, [self.x, self.acc, self.wsum, 1.0, 0, self.shift, self.accs])
                self._launch(k_colliders, n, [self.x, self.x, self.asleep, self.prm, 0]
                             + self._col_args() + [0, self.shift, self.gd, self.cm, 0])
            self._launch(k_shift, n, [self.xp, self.x, self.x_before])
        self._any_fresh = False
        use_target = 1 if self._grace_left > 0.0 else 0

        # 3) constraint solve: Jacobi iterations ...
        #    (the push that only removes overlap a pair already had is booked
        #    in `shift`, see solver.SandSolver._substep)
        sp = self._split if P else 0
        if sp:
            self.shift.zero_()
        for _ in range(prm.iterations):
            if P:
                self._launch(k_contacts, P, [self.pi, self.pj, self.xp, self.x, self.asleep,
                                             self.grace, self.fresh, 0, self.prm, 1, use_target,
                                             self.acc, self.wsum, sp, self.shift, self.accs,
                                             self.gd])
                self._launch(k_apply, n, [self.xp, self.acc, self.wsum, float(prm.relaxation),
                                          sp, self.shift, self.accs])
            self._launch(k_colliders, n, [self.xp, self.x, self.asleep, self.prm, 1]
                         + self._col_args() + [sp, self.shift, self.gd, self.cm, 1])

        # ... and height ordered sweeps (shock propagation)
        L = 0
        offs = None
        if P and (prm.stack_sweeps or prm.stabilization):
            self.hmin.fill_(1.0e30)
            self.lmax.zero_()
            self._launch(k_height_min, n, [self.x, self.prm, self.hmin])
            self._launch(k_pass_keys, P, [self.pi, self.pj, self.x, self.prm, self.hmin,
                                          self.keys, self.vals, self.lmax])
            L = int(self.lmax.numpy()[0]) + 1
            self._ensure_layers(L + 1)
            wp.utils.radix_sort_pairs(self.keys, self.vals, P)
            self.lcounts.zero_()
            self._launch(k_histogram, P, [self.keys, self.lcounts])
            wp.utils.array_scan(self.lcounts, self.loffsets, inclusive=False)
            offs = self.loffsets.numpy()[:L + 1].copy()
            offs[L] = P
        for _ in range(prm.stack_sweeps):
            if L:
                self._sweep(self.xp, self.x, 1, use_target, offs, L, sp)
            self._launch(k_colliders, n, [self.xp, self.x, self.asleep, self.prm, 1]
                         + self._col_args() + [sp, self.shift, self.gd, self.cm, 1])

        # 4) velocity update: depenetration limit and energy guard
        wp.copy(self.cap, self.spd)
        if P and prm.energy_guard:
            self._launch(k_cap_pairs, P, [self.pi, self.pj, self.spd, self.cap])
        self._launch(k_velocity, n, [self.xp, self.x, self.asleep, self.cap, float(self.v_guard),
                                     1 if prm.energy_guard else 0, float(1.0 / h), self.v,
                                     sp, self.shift, float(self.v_dep)])

        # 5) stabilisation: positions only (static friction relative to the start)
        fr = 1 if prm.stab_friction else 0
        xs = self.x if prm.stab_friction else self.xp
        for _ in range(prm.stabilization):
            if L:
                self._sweep(self.xp, xs, fr, 0, offs, L)
            self._launch(k_colliders, n, [self.xp, xs, self.asleep, self.prm, fr]
                         + self._col_args() + [0, self.shift, self.gd, self.cm, 1])
        self.x, self.xp = self.xp, self.x

        # grace time runs out; the fresh flags have been used
        self._launch(k_grace, n, [self.grace, float(h), self.fresh])
        self._grace_left = max(self._grace_left - h, 0.0)

        # 6) sleeping grains an awake grain presses deeply into wake up
        if prm.use_sleep and P and self._awake < n:
            thr = (1.0 - prm.wake_overlap) * d
            self._launch(k_wake_pen, P, [self.pi, self.pj, self.x, self.asleep, float(thr * thr),
                                         self.woke, float(1.0 - prm.wake_overlap), self.prm,
                                         self.gd])
            self._apply_wakes()

    # ------------------------------------------------ contact dynamics model
    def _substep_impulse(self, h):
        """solver.SandSolver._substep_impulse on the device, step by step."""
        prm = self.p
        n = self.n
        d = self.d
        inv_h = 1.0 / h
        nc = self.nc
        g = wp.vec3(*[float(c) for c in self.g])
        drag_k = self._drag_k(self._fnf > 0)
        vcap = 0.9 * self.d_min / h           # never jump through the smallest grain
        self.gd.ch = float(h)                 # water bridges: impulse per substep

        # 1) free motion (vpred is the working velocity of this substep)
        self.stat_f.zero_()
        self.stat_i.zero_()
        self._launch(k_predict, n, [self.x, self.v, self.asleep, g, float(h), float(drag_k),
                                    float(vcap), self.xp, self.vpred, self.spd,
                                    self.stat_f, self.stat_i, self.gd, self.fa, self.fb, self._fnf,
                                    float(self._field_t), self.expo])
        spmax = float(self.stat_f.numpy()[0])
        self.clamped = max(self.clamped, int(self.stat_i.numpy()[0]))
        reach = 1.1 * self.d_max + 2.0 * spmax * h
        self.grid.build(self.x, reach)

        # 2) contacts
        if prm.use_sleep and self._awake < n:
            wv = prm.wake_factor * self.v_sleep
            fast_v = max(wv, 2.0 * float(np.linalg.norm(self.g)) * h)
            self._launch(k_wake_hit, n, [self.grid.id, self.x, self.v, self.vframe, self.asleep,
                                         float(reach), float(fast_v), float(wv), self.woke])
            self._apply_wakes()
        self._launch(k_count_pairs, n, [self.grid.id, self.x, self.asleep, float(reach), self.counts])
        wp.utils.array_scan(self.counts, self.offsets, inclusive=False)
        tail = np.concatenate([self.offsets[n - 1:n].numpy(), self.counts[n - 1:n].numpy()])
        P = int(tail[0] + tail[1])
        self._ensure_pairs(P)
        self._launch(k_fill_pairs, n, [self.grid.id, self.x, self.asleep, float(reach),
                                       self.offsets, self.pi, self.pj])
        if self._any_fresh and P:
            for _ in range(3):
                self._launch(k_contacts, P, [self.pi, self.pj, self.x, self.x, self.asleep,
                                             self.grace, self.fresh, 1, self.prm, 0, 0,
                                             self.acc, self.wsum, 0, self.shift, self.accs,
                                             self.gd])
                self._launch(k_apply, n, [self.x, self.acc, self.wsum, 1.0, 0, self.shift, self.accs])
                self._launch(k_colliders, n, [self.x, self.x, self.asleep, self.prm, 0]
                             + self._col_args() + [0, self.shift, self.gd, self.cm, 0])
        self._any_fresh = False
        margin = self.r_max + spmax * h + 0.05 * d
        if self.gd.use_adh != 0:
            margin = max(margin, self.r_max + prm.bridge * self.d_max)
        if self._col_travel > 0.0:              # moving colliders reach further
            margin += self._col_travel
        self._launch(k_col_contacts, n, [self.x, self.asleep, self.prm, float(margin)]
                     + self._col_args() + [nc, self.cnr, self.cgap, self.gd, self.cm, self.cvel])

        # 3) Jacobi impulses, warm started with the impulses of the last substep
        warm = float(prm.warm_start)
        if P:
            if self._prev_cap == 0:
                self._ensure_prev(1)
            self._launch(k_warm_pairs, P, [self.pi, self.pj, self.x, self.asleep, self.poff,
                                           self.pcnt, self.ppj, self.pjn, self.pjt,
                                           warm, self.prm, self.jn, self.jt, self.dv, self.gd])
        self._launch(k_warm_cols, n, [self.asleep, self.cnr, self.cgap, self.cimp_n, self.cimp_t,
                                      warm, self.prm, nc, self.dv, self.gd])
        self._launch(k_add_dv, n, [self.vpred, self.dv])
        for _ in range(max(prm.iterations, 1)):
            self._launch(k_imp_count_cols, n, [self.vpred, self.asleep, self.cnr, self.cgap,
                                               self.cimp_n, nc, float(inv_h), self.cnt,
                                               self.prm, self.gd, self.cm, self.cvel])
            if P:
                self._launch(k_imp_count_pairs, P, [self.pi, self.pj, self.x, self.vpred, self.jn,
                                                    self.prm, float(inv_h), self.cnt, self.act,
                                                    self.gd])
                self._launch(k_imp_pairs, P, [self.pi, self.pj, self.x, self.vpred, self.asleep,
                                              self.jn, self.jt, self.act, self.cnt, self.prm,
                                              float(inv_h), self.dv, self.gd])
            self._launch(k_imp_cols, n, [self.vpred, self.asleep, self.cnr, self.cgap, self.cimp_n,
                                         self.cimp_t, self.cnt, self.prm, nc, float(inv_h), self.dv,
                                         self.gd, self.cm, self.cvel])
            self._launch(k_add_dv, n, [self.vpred, self.dv])
        # the pairs' impulses for the next substep
        wp.copy(self.poff, self.offsets)
        wp.copy(self.pcnt, self.counts)
        if P:
            self._ensure_prev(P)
            wp.copy(self.ppj, self.pj, count=P)
            wp.copy(self.pjn, self.jn, count=P)
            wp.copy(self.pjt, self.jt, count=P)

        # 4) support sweeps (shock propagation)
        L = 0
        offs = None
        if P and (prm.stack_sweeps or prm.stabilization):
            self.hmin.fill_(1.0e30)
            self.lmax.zero_()
            self._launch(k_height_min, n, [self.x, self.prm, self.hmin])
            self._launch(k_pass_keys, P, [self.pi, self.pj, self.x, self.prm, self.hmin,
                                          self.keys, self.vals, self.lmax])
            L = int(self.lmax.numpy()[0]) + 1
            self._ensure_layers(L + 1)
            wp.utils.radix_sort_pairs(self.keys, self.vals, P)
            self.lcounts.zero_()
            self._launch(k_histogram, P, [self.keys, self.lcounts])
            wp.utils.array_scan(self.lcounts, self.loffsets, inclusive=False)
            offs = self.loffsets.numpy()[:L + 1].copy()
            offs[L] = P
        if prm.stack_sweeps:
            wp.copy(self.csup_n, self.cimp_n)
            wp.copy(self.csup_t, self.cimp_t)
            if self.gd.use_adh != 0:
                self._launch(k_sup_adh, n, [self.asleep, self.cgap, self.csup_n, self.prm, nc, self.gd])
            if P:
                self._launch(k_sup_init, P, [self.pi, self.pj, self.asleep, self.jn, self.jt,
                                             self.nrel, self.trel, self.gd, self.x, self.prm])
            for _ in range(prm.stack_sweeps):
                self._launch(k_sup_cols_all, n, [self.vpred, self.asleep, self.cnr, self.cgap,
                                                 self.csup_n, self.csup_t, self.prm, nc, float(inv_h),
                                                 self.gd, self.cm, self.cvel])
                if L:
                    self._support_sweep(offs, L, inv_h)

        # 5) positions from the velocities
        self._launch(k_positions, n, [self.x, self.vpred, self.asleep, float(h), self.xp, self.v])

        # 6) left-over overlap: positions only, rate limited
        fr = 1 if prm.stab_friction else 0
        xs = self.x if prm.stab_friction else self.xp
        if prm.stabilization and L:
            wp.copy(self.xb, self.xp)
            for _ in range(prm.stabilization):
                self._sweep(self.xp, xs, fr, 0, offs, L)
            if prm.depenetration > 0.0:
                self._launch(k_rate_limit, n, [self.xp, self.xb, float(self.v_dep * h)])
        self._launch(k_colliders, n, [self.xp, xs, self.asleep, self.prm, fr]
                     + self._col_args() + [0, self.shift, self.gd, self.cm, 1])
        self.x, self.xp = self.xp, self.x
        self._launch(k_grace, n, [self.grace, float(h), self.fresh])
        self._grace_left = max(self._grace_left - h, 0.0)

        # 7) sleeping grains an awake grain presses deeply into wake up
        if prm.use_sleep and P and self._awake < n:
            thr = (1.0 - prm.wake_overlap) * d
            self._launch(k_wake_pen, P, [self.pi, self.pj, self.x, self.asleep, float(thr * thr),
                                         self.woke, float(1.0 - prm.wake_overlap), self.prm,
                                         self.gd])
            self._apply_wakes()

    def _support_sweep(self, offs, L, inv_h):
        nc = self.nc
        a = [self.vals, 0, self.pi, self.pj, self.x, self.vpred, self.asleep, self.prm, self.hmin,
             float(inv_h), self.nrel, self.trel, self.acc, self.wsum, self.gd, self.jn]
        b = [self.vals, 0, self.pi, self.pj, self.x, self.vpred, self.acc, self.wsum, self.stamp, 0, 0,
             self.asleep, self.prm, self.hmin, self.cnr, self.cgap, self.csup_n, self.csup_t, nc,
             float(inv_h), self.gd, self.cm, self.cvel]
        ca = wp.launch(k_sup_pass, dim=1, inputs=a, device=self.device, record_cmd=True)
        cb = wp.launch(k_sup_apply, dim=1, inputs=b, device=self.device, record_cmd=True)
        for layer in range(L):
            s0, s1 = int(offs[layer]), int(offs[layer + 1])
            if s1 <= s0:
                continue
            ca.set_param_at_index(1, s0)
            ca.set_dim(s1 - s0)
            cb.set_param_at_index(1, s0)
            cb.set_param_at_index(10, layer)
            cb.set_dim(s1 - s0)
            for _ in range(self.p.sweep_inner):
                self._token += 1
                ca.launch()
                cb.set_param_at_index(9, self._token)
                cb.launch()

    def _frame_end(self, dt):
        prm = self.p
        n = self.n
        self.grid2.build(self.x, 2.1 * self.d_max)
        self._launch(k_frame_contacts, n, [self.grid2.id, self.x, self.asleep, self.prm]
                     + self._col_args() + [1 if prm.use_sleep else 0, self.nsum, self.woke,
                                           self.gd, self.v, self.vbar, self.cm])
        if prm.use_sleep:
            self._apply_wakes()
        self._launch(k_sleep, n, [self.x, self.x_start, self.asleep, self.v, self.omega, self.calm,
                                  self.vframe, float(dt), float(self.v_sleep),
                                  float(prm.sleep_time), 1 if prm.use_sleep else 0])
        if prm.tumble > 0.0:
            self._launch(k_rotation, n, [self.asleep, self.nsum, self.v, self.omega, self.quat,
                                         float(self.r), float(prm.tumble), float(dt), self.gd,
                                         self.vbar, float(math.radians(prm.max_spin))])
