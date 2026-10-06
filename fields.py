# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender force fields acting on the sand (NumPy; no bpy needed except in
pack_field, which only reads object attributes).

A field is packed into one row of floats (see the F_* columns) per frame;
during a frame the rows of its start and its end are interpolated, so a
moving or animated field acts smoothly.

The formulas follow Blender's effectors (source/blender/blenkernel/intern/
effect.cc) for particles, with a particle of the nominal grain mass taken as
1 kg (Strength 10 of a Force field ~ gravity, as for Blender particles);
heavier grains (size variance) are accelerated less.  Wind is the exception:
its Strength is the air speed (m/s), and it acts through the air drag of the
grain - small grains are blown away more easily than big ones.

Supported: Force, Wind, Vortex, Turbulence, Drag, Harmonic; shapes Point,
Line, Plane (Surface and Points count as Point); falloff Sphere, Tube, Cone
with min / max distance and power, Z direction."""

import math

import numpy as np

try:
    from .i18n import iface as _
except ImportError:                     # plain module (tests)
    def _(s):
        return s

FIELD_TYPES = {'FORCE': 1, 'WIND': 2, 'VORTEX': 3, 'TURBULENCE': 4, 'DRAG': 5, 'HARMONIC': 6}
TYPE_NAMES = {v: k for k, v in FIELD_TYPES.items()}
SHAPES = {'POINT': 0, 'LINE': 1, 'PLANE': 2}
FALLOFFS = {'SPHERE': 0, 'TUBE': 1, 'CONE': 2}
ZDIRS = {'BOTH': 0, 'POSITIVE': 1, 'NEGATIVE': 2}

# integer / flag columns (not interpolated)
F_TYPE, F_SHAPE, F_FALL, F_ZDIR, F_USEMIN, F_USEMAX, F_USEMINR, F_USEMAXR, F_GLOBAL, F_SEED = range(10)
F_LERP = 10                     # the columns from here on are interpolated
(F_STRENGTH, F_POWER, F_MIN, F_MAX, F_POWR, F_MINR, F_MAXR, F_SIZE, F_LIN, F_QUAD,
 F_DAMP, F_REST) = range(10, 22)
F_LOC = 22                      # 22, 23, 24
F_Z = 25                        # 25, 26, 27: the field's Z axis (unit)
COLS = 32


# ------------------------------------------------------------------ packing
def pack_field(ob):
    """Row of a Blender object with a force field (evaluated state), or
    (None, reason) if it is not one we simulate."""
    fs = getattr(ob, "field", None)
    if fs is None or fs.type == 'NONE':
        return None, None
    if fs.type not in FIELD_TYPES:
        return None, _("%s: field “%s” is not supported") % (ob.name, _ui_name(fs, "type"))
    if not fs.apply_to_location:
        return None, None
    row = np.zeros(COLS)
    row[F_TYPE] = FIELD_TYPES[fs.type]
    row[F_SHAPE] = SHAPES.get(fs.shape, 0)
    row[F_FALL] = FALLOFFS.get(fs.falloff_type, 0)
    row[F_ZDIR] = ZDIRS.get(fs.z_direction, 0)
    row[F_USEMIN] = fs.use_min_distance
    row[F_USEMAX] = fs.use_max_distance
    row[F_USEMINR] = fs.use_radial_min
    row[F_USEMAXR] = fs.use_radial_max
    row[F_GLOBAL] = fs.use_global_coords
    row[F_SEED] = fs.seed
    row[F_STRENGTH] = fs.strength
    row[F_POWER] = fs.falloff_power
    row[F_MIN] = fs.distance_min
    row[F_MAX] = fs.distance_max
    row[F_POWR] = fs.radial_falloff
    row[F_MINR] = fs.radial_min
    row[F_MAXR] = fs.radial_max
    row[F_SIZE] = fs.size if fs.size > 0.0 else 1.0
    row[F_LIN] = fs.linear_drag
    row[F_QUAD] = fs.quadratic_drag
    row[F_DAMP] = fs.harmonic_damping
    row[F_REST] = fs.rest_length
    m = np.array(ob.matrix_world, dtype=np.float64)
    row[F_LOC:F_LOC + 3] = m[:3, 3]
    z = m[:3, 2]
    ln = float(np.linalg.norm(z))
    row[F_Z:F_Z + 3] = z / ln if ln > 1e-12 else (0.0, 0.0, 1.0)
    note = None
    if fs.shape not in SHAPES:
        note = _("%s: field shape “%s” is treated as a point") % (ob.name, _ui_name(fs, "shape"))
    return row, note


def _ui_name(fs, prop):
    """Name of an enum value as the interface shows it."""
    value = getattr(fs, prop)
    try:
        return _(fs.bl_rna.properties[prop].enum_items[value].name)
    except Exception:
        return str(value).title()


def lerp_rows(fa, fb, t):
    """Fields at the fraction t of the frame between the rows fa and fb."""
    if fb is None or t <= 0.0:
        return fa
    f = fa.copy()
    f[:, F_LERP:] = fa[:, F_LERP:] + (fb[:, F_LERP:] - fa[:, F_LERP:]) * t
    z = f[:, F_Z:F_Z + 3]
    ln = np.linalg.norm(z, axis=1, keepdims=True)
    f[:, F_Z:F_Z + 3] = np.where(ln > 1e-12, z / np.maximum(ln, 1e-12), (0.0, 0.0, 1.0))
    return f


def has_wind(rows):
    return rows is not None and bool(np.any(rows[:, F_TYPE] == FIELD_TYPES['WIND']))


# ------------------------------------------------------------------ noise
_M32 = np.uint64(0xFFFFFFFF)


def _hash(ix, iy, iz, seed):
    """Integer hash of a lattice point -> [0, 1) (same bits as the GPU)."""
    def u(a):
        return (np.asarray(a, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint64)
    h = (u(ix) * np.uint64(0x8DA6B343)) & _M32
    h ^= (u(iy) * np.uint64(0xD8163841)) & _M32
    h ^= (u(iz) * np.uint64(0xCB1AB31F)) & _M32
    h ^= (u(seed) * np.uint64(0x165667B1)) & _M32
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x7FEB352D)) & _M32
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(0x846CA68B)) & _M32
    h ^= h >> np.uint64(16)
    return h.astype(np.float64) / 4294967296.0


def value_noise(q, seed):
    """Smooth value noise in [0, 1] at the points q (m, 3)."""
    f = np.floor(q)
    i = f.astype(np.int64)
    t = q - f
    u = t * t * (3.0 - 2.0 * t)
    out = np.zeros(len(q))
    for dx in (0, 1):
        wx = u[:, 0] if dx else 1.0 - u[:, 0]
        for dy in (0, 1):
            wy = u[:, 1] if dy else 1.0 - u[:, 1]
            for dz in (0, 1):
                wz = u[:, 2] if dz else 1.0 - u[:, 2]
                out += wx * wy * wz * _hash(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out


def turbulence(q, seed):
    """Three components of 2-octave noise in [0, 1] (as Blender's turbulence
    field, each component with its own permutation of the coordinates)."""
    out = np.empty((len(q), 3))
    for k in range(3):
        qk = np.roll(q, -k, axis=1)
        s = int(seed) + 101 * k
        out[:, k] = (value_noise(qk, s) + 0.5 * value_noise(2.0 * qk, s + 7)) / 1.5
    return out


# ------------------------------------------------------------------ evaluation
def _falloff_func(fac, usemin, mind, usemax, maxd, power):
    if not usemin:
        mind = 0.0
    out = np.power(1.0 + np.maximum(fac - mind, 0.0), -power) if power != 0.0 else np.ones_like(fac)
    out = np.where(fac < mind, 1.0, out)
    if usemax:
        out = np.where(fac > maxd, 0.0, out)
    return out


def _unit(a):
    ln = np.linalg.norm(a, axis=1, keepdims=True)
    return np.where(ln > 1e-12, a / np.maximum(ln, 1e-12), 0.0)


def evaluate(rows, p, v):
    """Effect of the fields on points p with velocities v:
    (acceleration of a grain of the nominal mass, air velocity of the wind,
    linear drag, quadratic drag)."""
    m = len(p)
    acc = np.zeros((m, 3))
    air = np.zeros((m, 3))
    lin = np.zeros(m)
    quad = np.zeros(m)
    if rows is None or m == 0:
        return acc, air, lin, quad
    for row in rows:
        typ = int(row[F_TYPE])
        shape = int(row[F_SHAPE])
        z = row[F_Z:F_Z + 3]
        vec2 = p - row[F_LOC:F_LOC + 3]
        along = vec2 @ z
        if shape == 1:                                  # line: distance to the Z axis
            vec = vec2 - along[:, None] * z
            dist = np.linalg.norm(vec, axis=1)
        elif shape == 2:                                # plane: distance to the XY plane
            vec = along[:, None] * z
            dist = np.abs(along)
        else:
            vec = vec2
            dist = np.linalg.norm(vec2, axis=1)
        # falloff (Blender's effector_falloff)
        f = np.ones(m)
        zd = int(row[F_ZDIR])
        if zd == 1:
            f[along < 0.0] = 0.0
        elif zd == 2:
            f[along > 0.0] = 0.0
        fall = int(row[F_FALL])
        power = float(row[F_POWER])
        if fall == 0:
            f *= _falloff_func(dist, row[F_USEMIN], row[F_MIN], row[F_USEMAX], row[F_MAX], power)
        else:
            f *= _falloff_func(np.abs(along), row[F_USEMIN], row[F_MIN], row[F_USEMAX], row[F_MAX], power)
            if fall == 1:
                rfac = np.linalg.norm(vec2 - along[:, None] * z, axis=1)
            else:
                l2 = np.linalg.norm(vec2, axis=1)
                rfac = np.degrees(np.arccos(np.clip(np.abs(along) / np.maximum(l2, 1e-12), 0.0, 1.0)))
            f *= _falloff_func(rfac, row[F_USEMINR], row[F_MINR], row[F_USEMAXR], row[F_MAXR],
                               float(row[F_POWR]))
        s = float(row[F_STRENGTH])
        sf = s * f
        if typ == 1:                                    # Force: along the direction from the field
            acc += _unit(vec) * sf[:, None]
        elif typ == 2:                                  # Wind: air flow along Z (m/s)
            air += np.outer(sf, z)
        elif typ == 3:                                  # Vortex
            if shape == 0:
                acc += _unit(np.cross(z, vec2)) * (sf * dist)[:, None]
            else:                                       # flow of a rotation with omega = s f
                t = np.cross(z, vec2) * sf[:, None]
                acc += np.cross(z, t) * sf[:, None] + t - v
        elif typ == 4:                                  # Turbulence
            q = p if row[F_GLOBAL] else vec2
            acc += (2.0 * turbulence(q / float(row[F_SIZE]), row[F_SEED]) - 1.0) * sf[:, None]
        elif typ == 5:                                  # Drag
            lin += float(row[F_LIN]) * f
            quad += float(row[F_QUAD]) * f
        elif typ == 6:                                  # Harmonic: spring to the field
            vv = vec
            rest = float(row[F_REST])
            if rest > 0.0:
                ln = np.maximum(np.linalg.norm(vec, axis=1), 1e-12)
                vv = vec * (1.0 - rest / ln)[:, None]
            acc += -vv * sf[:, None] - v * (float(row[F_DAMP]) * 2.0 * math.sqrt(abs(s)))
    return acc, air, lin, quad
