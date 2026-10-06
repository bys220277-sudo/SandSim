# SPDX-License-Identifier: GPL-3.0-or-later
"""Disk cache of the simulation and playback through a frame handler.

One file per frame: float32 array (n, 7) = position xyz + quaternion wxyz.
Files are written to a temporary name and renamed, so a frame that is being
read during baking is never half written."""

import json
import os
import tempfile

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import nodes

_loaded = {}   # object name -> (cache dir, frame, first vertex) of the shown frame


# ------------------------------------------------------------------ paths
def cache_root(settings):
    path = settings.cache_dir or "//sand_cache"
    if path.startswith("//") and not bpy.data.filepath:
        # unsaved file: keep the cache in the system temp folder
        return os.path.join(tempfile.gettempdir(), "blender_sand_cache")
    return os.path.normpath(bpy.path.abspath(path))


def cache_dir(ob):
    """Folder of this object's frames. The folder name is fixed at bake time,
    so renaming the object keeps its cache."""
    st = ob.sand_sim
    folder = st.cache_folder or (bpy.path.clean_name(ob.name) + "_" + st.cache_id)
    return os.path.join(cache_root(st), folder)


def folder_shared(ob):
    """True if another sand object (e.g. a duplicate) uses the same cache."""
    me = cache_dir(ob)
    for other in bpy.data.objects:
        if other is not ob and other.type == 'MESH' and other.sand_sim.is_sand \
                and other.sand_sim.is_baked and cache_dir(other) == me:
            return True
    return False


def frame_file(directory, frame):
    return os.path.join(directory, "frame_%06d.npy" % frame)


def write_frame(directory, frame, x, quat):
    arr = np.empty((len(x), 7), dtype=np.float32)
    arr[:, :3] = x
    arr[:, 3:] = quat
    path = frame_file(directory, frame)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def read_frame(directory, frame):
    try:
        return np.load(frame_file(directory, frame))
    except (OSError, ValueError):
        return None


def write_meta(directory, meta):
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)


def clear_dir(directory):
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if (name.startswith("frame_") and (name.endswith(".npy") or name.endswith(".tmp"))) \
                or name == "meta.json":
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                pass
    try:
        os.rmdir(directory)
    except OSError:
        pass


def forget(ob_name=None):
    if ob_name is None:
        _loaded.clear()
    else:
        _loaded.pop(ob_name, None)


# ------------------------------------------------------------------ mesh
def set_mesh_points(me, positions, quat=None):
    """Write grain positions (and rotations) into the point mesh,
    resizing it when the number of grains changed."""
    n = len(positions)
    if len(me.vertices) != n:
        me.clear_geometry()
        me.vertices.add(n)
    pos = np.ascontiguousarray(positions, dtype=np.float32).ravel()
    me.attributes["position"].data.foreach_set("vector", pos)
    if quat is not None:
        attr = me.attributes.get(nodes.ROT_ATTR)
        if attr is None or attr.data_type != 'QUATERNION' or attr.domain != 'POINT':
            if attr is not None:
                me.attributes.remove(attr)
            attr = me.attributes.new(nodes.ROT_ATTR, 'QUATERNION', 'POINT')
        attr.data.foreach_set("value", np.ascontiguousarray(quat, dtype=np.float32).ravel())
    me.update()


def world_to_local(ob, pos, quat=None):
    """Simulation (world space) -> the sand object's local space, so the grains
    are shown where they are simulated whatever the object's transform."""
    m = np.array(ob.matrix_world, dtype=np.float64)
    if np.allclose(m, np.eye(4), atol=1e-12):
        return pos, quat
    inv = np.linalg.inv(m)
    loc = np.asarray(pos, dtype=np.float64) @ inv[:3, :3].T + inv[:3, 3]
    if quat is not None:
        from .solver import _quat_mul
        qo = ob.matrix_world.to_quaternion()
        qi = np.array([[qo.w, -qo.x, -qo.y, -qo.z]])
        quat = _quat_mul(np.repeat(qi, len(quat), axis=0), np.asarray(quat, dtype=np.float64))
    return loc, quat


def local_to_world(ob, pos):
    m = np.array(ob.matrix_world, dtype=np.float64)
    return np.asarray(pos, dtype=np.float64) @ m[:3, :3].T + m[:3, 3]


def set_static_attributes(me, rest, seed=0):
    """Rest positions and a random tint per grain (for the material)."""
    n = len(me.vertices)
    for name, dtype in ((nodes.REST_ATTR, 'FLOAT_VECTOR'), (nodes.TINT_ATTR, 'FLOAT')):
        attr = me.attributes.get(name)
        if attr is not None and (attr.data_type != dtype or attr.domain != 'POINT'):
            me.attributes.remove(attr)
            attr = None
        if attr is None:
            me.attributes.new(name, dtype, 'POINT')
    me.attributes[nodes.REST_ATTR].data.foreach_set(
        "vector", np.ascontiguousarray(rest, dtype=np.float32).ravel())
    tint = np.random.default_rng(seed + 7).random(n).astype(np.float32)
    me.attributes[nodes.TINT_ATTR].data.foreach_set("value", tint)


def get_rest_positions(me):
    attr = me.attributes.get(nodes.REST_ATTR)
    n = len(me.vertices)
    out = np.empty(n * 3, dtype=np.float32)
    if attr is not None and attr.data_type == 'FLOAT_VECTOR' and attr.domain == 'POINT':
        attr.data.foreach_get("vector", out)
    else:
        me.attributes["position"].data.foreach_get("vector", out)
    return out.reshape(-1, 3).astype(np.float64)


# ------------------------------------------------------------------ playback
def show_frame(ob, frame):
    """Load the cached state of `frame` (clamped to the baked range)."""
    st = ob.sand_sim
    if not st.is_baked or st.baked_end < st.baked_start:
        return False
    f = min(max(int(frame), st.baked_start), st.baked_end)
    directory = cache_dir(ob)
    me = ob.data
    first = tuple(me.vertices[0].co) if len(me.vertices) else None
    if _loaded.get(ob.name) == (directory, f, first):
        return True
    arr = read_frame(directory, f)
    if arr is None:
        return False
    pos, quat = world_to_local(ob, arr[:, :3], arr[:, 3:7])
    set_mesh_points(me, pos, quat)
    nodes.sync_cube_size(ob, st.grain_size)
    _loaded[ob.name] = (directory, f, tuple(me.vertices[0].co) if len(me.vertices) else None)
    return True


@persistent
def on_frame_change(scene, depsgraph=None):
    frame = scene.frame_current
    for ob in scene.objects:
        if ob.type != 'MESH':
            continue
        st = ob.sand_sim
        if st.is_sand and st.is_baked:
            try:
                show_frame(ob, frame)
            except Exception as exc:  # never break playback / rendering
                print("Sand Simulation: cannot load frame", frame, "of", ob.name, ":", exc)


@persistent
def on_load_post(*_args):
    forget()
