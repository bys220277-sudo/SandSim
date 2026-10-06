# SPDX-License-Identifier: GPL-3.0-or-later
"""Static per grain data besides the rest position and the tint:

* size factor (attribute sand_scale): random, uniform in [1 - v, 1 + v],
  normalised to a mean volume of one grain of the nominal size - the total
  volume and mass of the sand do not change with the variance,
* paint weight (attribute sand_paint): a vertex group (Weight Paint) or an
  attribute of the source object, per start point; for a filled volume the
  weights of the nearest surface vertices."""

import math

import bpy
import numpy as np

from . import cache, nodes
from .i18n import iface as _

_LUMA = np.array([0.2126, 0.7152, 0.0722])


def grain_scales(n, variance_pct, seed):
    """Size factor of every grain, None when all grains are alike."""
    v = min(max(float(variance_pct), 0.0), 95.0) / 100.0
    if v <= 0.0 or n == 0:
        return None
    u = np.random.default_rng(int(seed) + 13).uniform(1.0 - v, 1.0 + v, int(n))
    return u / (1.0 + v * v) ** (1.0 / 3.0)       # E[u^3] = 1 + v^2


# ---------------------------------------------------------------- paint
def _count(data):
    attr = data.attributes.get("position") if data is not None else None
    return 0 if attr is None else len(attr.data)


def _attr_values(data, name):
    """Float per point of the attribute `name` (float, integer, boolean or
    colour - its luminance) of a mesh / point cloud / curves; None if absent."""
    attr = data.attributes.get(name)
    if attr is None:
        return None
    m = len(attr.data)
    types = {'FLOAT': np.float32, 'INT': np.int32, 'INT8': np.int8, 'BOOLEAN': bool}
    if attr.data_type in types:
        buf = np.empty(m, dtype=types[attr.data_type])
        attr.data.foreach_get("value", buf)
        vals = buf.astype(np.float64)
    elif attr.data_type in {'FLOAT_COLOR', 'BYTE_COLOR'}:
        buf = np.empty(m * 4, dtype=np.float32)
        attr.data.foreach_get("color", buf)
        vals = buf.reshape(-1, 4)[:, :3] @ _LUMA
    else:
        return None
    if attr.domain == 'POINT':
        return vals
    if attr.domain == 'CORNER' and data.attributes.get(".corner_vert") is not None:
        cv = np.empty(m, dtype=np.int32)
        data.attributes[".corner_vert"].data.foreach_get("value", cv)
        npt = _count(data)
        s = np.bincount(cv, weights=vals, minlength=npt)
        c = np.bincount(cv, minlength=npt)
        return s / np.maximum(c, 1)
    return None


def _vertex_group(ev, name, nverts):
    """Weights of the vertex group `name` on the evaluated mesh of `ev`."""
    vg = ev.vertex_groups.get(name) if hasattr(ev, "vertex_groups") else None
    if vg is None:
        return None
    gi = vg.index
    me = ev.to_mesh()
    try:
        if len(me.vertices) != nverts:
            return None
        w = np.zeros(nverts)
        for v in me.vertices:
            for g in v.groups:
                if g.group == gi:
                    w[v.index] = g.weight
                    break
        return w
    finally:
        ev.to_mesh_clear()


def source_weights(src, depsgraph, name):
    """(weights of the source points in the order of read_source_points,
    world positions of the mesh vertices with their weights, found)."""
    ev = src.evaluated_get(depsgraph)
    gs = ev.evaluated_geometry()
    out, found = [], False
    mesh_part = None
    for k, data in enumerate((gs.mesh, gs.pointcloud, gs.curves)):
        c = _count(data)
        if c == 0:
            continue
        vals = _attr_values(data, name)
        if vals is None and k == 0:
            vals = _vertex_group(ev, name, c)
        if vals is None:
            vals = np.zeros(c)
        else:
            found = True
        out.append(vals)
        if k == 0:
            co = np.empty(c * 3, dtype=np.float32)
            data.attributes["position"].data.foreach_get("vector", co)
            m = np.array(ev.matrix_world, dtype=np.float64)
            mesh_part = (co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3], vals)
    try:
        inst = gs.instances_pointcloud()
        c = _count(inst)
        if c:
            vals = _attr_values(inst, name)
            found = found or vals is not None
            out.append(np.zeros(c) if vals is None else vals)
    except Exception:
        pass
    w = np.concatenate(out) if out else np.zeros(0)
    return np.clip(w, 0.0, 1.0), mesh_part, found


def _nearest_weights(points, verts, weights, k=6):
    """Inverse distance weighted paint of the k nearest surface vertices."""
    from mathutils.kdtree import KDTree
    kd = KDTree(len(verts))
    for i, co in enumerate(verts):
        kd.insert(co, i)
    kd.balance()
    k = min(k, len(verts))
    out = np.empty(len(points))
    for i, p in enumerate(points):
        res = kd.find_n(p, k)
        idx = [r[1] for r in res]
        dist = np.array([r[2] for r in res])
        wt = 1.0 / (dist * dist + 1e-12)
        out[i] = float((weights[idx] * wt).sum() / wt.sum())
    return out


def grain_paint(st, rest, depsgraph):
    """(paint weight of every grain or None, message)."""
    src = st.source
    name = st.wet_paint_name.strip()
    if not name:
        return None, _("Choose a vertex group or an attribute")
    if src is None:
        return None, _("No source: nowhere to read the paint from")
    w, mesh_part, found = source_weights(src, depsgraph, name)
    if not found:
        return None, _("“%s” has no “%s”") % (src.name, name)
    if st.source_mode == 'VOLUME':
        if mesh_part is None or len(mesh_part[0]) == 0:
            return None, _("The source has no vertices")
        return np.clip(_nearest_weights(rest, mesh_part[0], mesh_part[1]), 0.0, 1.0), ""
    if len(w) != len(rest):
        return None, _("The number of source points has changed: press “Reset”")
    return w, ""


# ---------------------------------------------------------------- attributes
def write_scale(ob):
    st = ob.sand_sim
    me = ob.data
    nodes.set_float_attribute(me, nodes.SCALE_ATTR,
                              grain_scales(len(me.vertices), st.size_variance, st.seed))


def write_paint(ob, depsgraph, rest=None):
    """Read the paint of the source into sand_paint (if the paint is used)."""
    st = ob.sand_sim
    if not st.use_wet_paint:
        return
    if rest is None:
        rest = cache.get_rest_positions(ob.data)
    try:
        w, msg = grain_paint(st, rest, depsgraph)
    except Exception as exc:          # never break creating / baking
        w, msg = None, _("Cannot read the paint: %s") % exc
    if w is not None:
        nodes.set_float_attribute(ob.data, nodes.PAINT_ATTR, w)
        msg = _("Paint: %d of %d grains") % (int(np.count_nonzero(w > 0.01)), len(w))
    if st.wet_info != msg:
        st.wet_info = msg
    if st.wet_ok != (w is not None):
        st.wet_ok = w is not None


def write_attributes(ob, depsgraph, rest=None, project=True):
    """Size factors, paint weights and projected colours of the grains (after
    the points were (re)written; call again whenever the number of grains
    changes)."""
    write_scale(ob)
    write_paint(ob, depsgraph, rest)
    if project:
        write_projection(ob, depsgraph, rest)


def valid_shape(ob):
    """The grain shape object if it is usable, else None (cube)."""
    shape = ob.sand_sim.shape_object
    if shape is None or shape == ob or shape.type != 'MESH' or shape.name not in bpy.data.objects:
        return None
    return shape


def sync_nodes(ob):
    """Node group settings from the object's sand settings."""
    st = ob.sand_sim
    nodes.set_shape(ob, valid_shape(ob), st.shape_material)
    wet = st.use_wet_noise or st.use_wet_paint
    nodes.set_wet_sources(ob, st.use_wet_noise, st.use_wet_paint)
    if wet:
        nodes.ensure_wet_shading(bpy.data.materials.get(nodes.MATERIAL_NAME))
    nodes.set_materials(ob, st.dry_material, st.wet_material, wet and st.wet_material is not None)
    if st.use_projection:
        for mat in (st.dry_material or bpy.data.materials.get(nodes.MATERIAL_NAME), st.wet_material):
            if mat is not None and mat.name == nodes.MATERIAL_NAME:
                nodes.ensure_projection_shading(mat)


# ---------------------------------------------------------------- grain shape size
_shape_cache = {}


def _matrix_key(m):
    return tuple(round(v, 7) for row in m for v in row)


def shape_volume_ratio(shape, sand_ob, depsgraph=None, use_cache=True):
    """Volume of the grain shape as the node group shows it, divided by the
    cube of its largest side (1 for a cube, pi/6 for a sphere); None if the
    shape has no faces.  An open or flat mesh counts as the ellipsoid of its
    bounding box."""
    me0 = shape.data
    key = (shape.name, me0.name if me0 else "", len(me0.vertices) if me0 else 0,
           len(shape.modifiers), _matrix_key(shape.matrix_world), _matrix_key(sand_ob.matrix_world))
    if use_cache and key in _shape_cache:
        return _shape_cache[key]
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = shape.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        nv = len(me.vertices)
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        if nv == 0:
            ratio = None
        else:
            co = np.empty(nv * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            tri = np.empty(nt * 3, dtype=np.int32)
            if nt:
                me.loop_triangles.foreach_get("vertices", tri)
            # the shape in the sand object's space (Object Info, Relative)
            m = np.linalg.inv(np.array(sand_ob.matrix_world, dtype=np.float64)) \
                @ np.array(shape.matrix_world, dtype=np.float64)
            p = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T
            ext = p.max(axis=0) - p.min(axis=0)
            big = float(ext.max())
            if big <= 0.0:
                ratio = None
            else:
                t = tri.reshape(-1, 3)
                vol = abs(float(np.einsum("ij,ij->i", p[t[:, 0]],
                                          np.cross(p[t[:, 1]], p[t[:, 2]])).sum())) / 6.0 if nt else 0.0
                box = float(np.prod(np.maximum(ext, 0.05 * big)))
                if vol < 0.02 * box:              # open / flat mesh: its bounding ellipsoid
                    vol = math.pi / 6.0 * box
                ratio = min(vol / big ** 3, 1.0)
    finally:
        ev.to_mesh_clear()
    if len(_shape_cache) > 64:
        _shape_cache.clear()
    _shape_cache[key] = ratio
    return ratio


def size_factor(ob, depsgraph=None, use_cache=True):
    """Physical size of a grain relative to the cube of the Grain Size: the
    edge of the cube with the volume of the grain shape as it is shown
    (1 for the cube)."""
    st = ob.sand_sim
    shape = valid_shape(ob)
    if shape is None or nodes.get_node(ob, nodes.SHAPE_INFO) is None:
        return 1.0
    try:
        ratio = shape_volume_ratio(shape, ob, depsgraph, use_cache)
    except Exception:
        ratio = None
    if ratio is None:                       # no geometry: the cube is shown
        return 1.0
    return float(st.shape_scale) * max(ratio, 0.008) ** (1.0 / 3.0)


# ---------------------------------------------------------------- camera projection
def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def image_pixels(img):
    """(h, w, 4) float array of an image in scene linear colours, or None."""
    w, h = img.size
    if w == 0 or h == 0:
        return None
    ch = img.channels
    buf = np.empty(w * h * ch, dtype=np.float32)
    img.pixels.foreach_get(buf)
    px = buf.reshape(h, w, ch).astype(np.float64)
    if ch == 1:
        px = np.repeat(px, 3, axis=2)
    if px.shape[2] == 3:
        px = np.concatenate([px, np.ones((h, w, 1))], axis=2)
    if px.shape[2] == 2:
        px = np.concatenate([np.repeat(px[:, :, :1], 3, axis=2), px[:, :, 1:2]], axis=2)
    px = px[:, :, :4]
    cs = img.colorspace_settings.name.lower()
    if not img.is_float and "srgb" in cs and "linear" not in cs:
        px[:, :, :3] = _srgb_to_linear(px[:, :, :3])
    return px


def camera_project(points, cam, scene, depsgraph):
    """Normalised frame coordinates (u, v) of world points seen by the
    camera, and a mask of the points in front of it."""
    r = scene.render
    proj = np.array(cam.calc_matrix_camera(depsgraph, x=r.resolution_x, y=r.resolution_y,
                                           scale_x=r.pixel_aspect_x, scale_y=r.pixel_aspect_y),
                    dtype=np.float64)
    view = np.linalg.inv(np.array(cam.matrix_world, dtype=np.float64))
    ph = np.c_[points, np.ones(len(points))]
    pc = ph @ view.T
    clip = pc @ proj.T
    w = clip[:, 3]
    ok = np.abs(w) > 1e-12
    ndc = np.zeros((len(points), 2))
    ndc[ok] = clip[ok, :2] / w[ok, None]
    front = pc[:, 2] < 0.0
    return 0.5 * (ndc[:, 0] + 1.0), 0.5 * (ndc[:, 1] + 1.0), front & ok


def projected_colors(ob, depsgraph, rest=None):
    """(RGBA per grain - alpha = strength inside the camera frame, 0 outside -
    or None, message)."""
    st = ob.sand_sim
    img = st.project_image
    if img is None:
        return None, _("Choose an image")
    scene = bpy.context.scene
    cam = st.project_camera or scene.camera
    if cam is None or cam.type != 'CAMERA':
        return None, _("The scene has no active camera")
    try:
        px = image_pixels(img)
    except Exception as exc:
        return None, _("Cannot read the image: %s") % exc
    if px is None:
        return None, _("Image “%s” is empty or not found") % img.name
    if rest is None:
        rest = cache.get_rest_positions(ob.data)
    pts = cache.local_to_world(ob, rest)
    u, v, front = camera_project(pts, cam, scene, depsgraph)
    h, w = px.shape[:2]
    inside = front & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    # bilinear sample
    x = np.clip(u * w - 0.5, 0.0, w - 1.0)
    y = np.clip(v * h - 0.5, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    col = (px[y0, x0] * (1 - fx) * (1 - fy) + px[y0, x1] * fx * (1 - fy)
           + px[y1, x0] * (1 - fx) * fy + px[y1, x1] * fx * fy)
    out = np.zeros((len(pts), 4))
    out[:, :3] = col[:, :3]
    out[:, 3] = np.where(inside, col[:, 3] * float(st.project_strength), 0.0)
    return out, _("Image on %d of %d grains") % (int(np.count_nonzero(out[:, 3] > 0.01)), len(pts))


def write_projection(ob, depsgraph, rest=None):
    """Colours of the image seen through the camera, stuck to the grains at
    their start positions (attribute sand_color); removed when off."""
    st = ob.sand_sim
    if not st.use_projection:
        nodes.set_color_attribute(ob.data, nodes.COLOR_ATTR, None)
        return
    try:
        col, msg = projected_colors(ob, depsgraph, rest)
    except Exception as exc:          # never break creating / baking
        col, msg = None, _("Projection error: %s") % exc
    if col is not None:
        nodes.set_color_attribute(ob.data, nodes.COLOR_ATTR, col)
    if st.project_info != msg:
        st.project_info = msg
    if st.project_ok != (col is not None):
        st.project_ok = col is not None
