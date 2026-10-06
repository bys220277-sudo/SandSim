# SPDX-License-Identifier: GPL-3.0-or-later
"""Geometry Nodes set-up that turns the grain points into cubes (or the
user's own shape), and the default sand material.

The node group ("Sand Cubes"):
  points --[wet zones: store sand_wet]--> Instance on Points <-- grain shape
  grain shape = cube, or the chosen object scaled to the grain size
  instance scale = per grain size factor (attribute sand_scale, 1 if missing)
  sand_wet = ColorRamp(Noise(rest position)) * paint (attribute sand_paint)

Groups made by older versions are upgraded in place (nodes are added, the
existing ones and their settings are kept)."""

import bpy
import numpy as np

from .i18n import iface as _

ROT_ATTR = "sand_rotation"   # quaternion per grain (cached, animated)
TINT_ATTR = "sand_tint"      # random 0..1 per grain (static, for shading)
REST_ATTR = "sand_rest"      # initial positions (static)
SCALE_ATTR = "sand_scale"    # size factor per grain (static, optional)
WET_ATTR = "sand_wet"        # wetness 0..1 per grain (computed by the node group)
PAINT_ATTR = "sand_paint"    # paint weight 0..1 per grain, read from the source
COLOR_ATTR = "sand_color"    # projected image colour per grain (RGBA, A = strength)
CUBE_NODE = "SandCube"
MODIFIER_NAME = "Sand Cubes"
MATERIAL_NAME = "Sand"

# nodes added in version 1.4
SHAPE_INFO = "SandShapeInfo"
SHAPE_SIZE = "SandShapeSize"
SHAPE_MAT = "SandShapeMaterial"
SHAPE_SWITCH = "SandShapeSwitch"
SCALE_SWITCH = "SandScaleSwitch"
WET_NOISE_ON = "SandWetNoiseOn"
WET_PAINT_ON = "SandWetPaintOn"
WET_NOISE = "SandWetNoise"
WET_RAMP = "SandWetRamp"
WET_OFFSET = "SandWetOffset"
WET_SWITCH = "SandWetSwitch"
WET_SHADER_NODE = "SandWetAttr"
# nodes added in version 1.5
DRY_MAT = "SandDryMaterial"
WET_MAT = "SandWetMaterial"
MAT_ON = "SandUseWetMaterial"
MAT_SWITCH = "SandMaterialSwitch"
PROJ_SHADER_NODE = "SandColorAttr"


# ---------------------------------------------------------------- material
def ensure_material():
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(MATERIAL_NAME)
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        return mat
    bsdf.inputs["Roughness"].default_value = 0.85
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = 'INSTANCER'
    attr.attribute_name = TINT_ATTR
    attr.location = (bsdf.location.x - 520, bsdf.location.y)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (bsdf.location.x - 300, bsdf.location.y)
    el = ramp.color_ramp.elements
    el[0].color = (0.46, 0.32, 0.18, 1.0)
    el[1].color = (0.80, 0.66, 0.45, 1.0)
    el.new(0.55).color = (0.68, 0.52, 0.32, 1.0)
    nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    mat.diffuse_color = (0.68, 0.52, 0.32, 1.0)
    return mat


def _socket(sockets, name, stype):
    return next(s for s in sockets if s.name == name and s.type == stype)


def ensure_wet_shading(mat):
    """Wet grains darker and a little glossier (attribute sand_wet of the
    instances; without it the look is exactly as before).  Only added to a
    material with the add-on's own layout."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return False
    nt = mat.node_tree
    if nt.nodes.get(WET_SHADER_NODE) is not None:
        return True
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        return False
    base = bsdf.inputs["Base Color"]
    rough = bsdf.inputs["Roughness"]
    if rough.is_linked:
        return False
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.name = WET_SHADER_NODE
    attr.label = _("Wetness")
    attr.attribute_type = 'INSTANCER'
    attr.attribute_name = WET_ATTR
    attr.location = (bsdf.location.x - 520, bsdf.location.y - 260)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.name = "SandWetDarken"
    mix.label = _("Wet Is Darker")
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.location = (bsdf.location.x - 200, bsdf.location.y + 40)
    a = _socket(mix.inputs, "A", 'RGBA')
    b = _socket(mix.inputs, "B", 'RGBA')
    b.default_value = (0.5, 0.5, 0.5, 1.0)
    if base.is_linked:
        src = base.links[0].from_socket
        nt.links.remove(base.links[0])
        nt.links.new(src, a)
    else:
        a.default_value = base.default_value
    nt.links.new(attr.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(_socket(mix.outputs, "Result", 'RGBA'), base)
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.name = "SandWetGloss"
    mr.label = _("Wet Is Glossier")
    mr.location = (bsdf.location.x - 200, bsdf.location.y - 260)
    mr.inputs["To Min"].default_value = rough.default_value
    mr.inputs["To Max"].default_value = min(rough.default_value, 0.45)
    nt.links.new(attr.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], rough)
    return True


def ensure_projection_shading(mat):
    """Projected image colour (attribute sand_color of the instances, its
    alpha = strength) over the sand colour, with a little per grain variation.
    Without the attribute the look is exactly as before."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return False
    nt = mat.node_tree
    if nt.nodes.get(PROJ_SHADER_NODE) is not None:
        return True
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        return False
    dark = nt.nodes.get("SandWetDarken")
    target = _socket(dark.inputs, "A", 'RGBA') if dark is not None else bsdf.inputs["Base Color"]
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.name = PROJ_SHADER_NODE
    attr.label = _("Projected Color")
    attr.attribute_type = 'INSTANCER'
    attr.attribute_name = COLOR_ATTR
    attr.location = (bsdf.location.x - 760, bsdf.location.y + 300)
    tint = nt.nodes.new("ShaderNodeAttribute")
    tint.attribute_type = 'INSTANCER'
    tint.attribute_name = TINT_ATTR
    tint.location = (bsdf.location.x - 760, bsdf.location.y + 480)
    var = nt.nodes.new("ShaderNodeMapRange")
    var.location = (bsdf.location.x - 560, bsdf.location.y + 480)
    var.inputs["To Min"].default_value = 0.85
    var.inputs["To Max"].default_value = 1.15
    nt.links.new(tint.outputs["Fac"], var.inputs["Value"])
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    mul.location = (bsdf.location.x - 560, bsdf.location.y + 300)
    mul.inputs["Factor"].default_value = 1.0
    nt.links.new(attr.outputs["Color"], _socket(mul.inputs, "A", 'RGBA'))
    nt.links.new(var.outputs["Result"], _socket(mul.inputs, "B", 'RGBA'))
    mix = nt.nodes.new("ShaderNodeMix")
    mix.name = "SandColorMix"
    mix.label = _("Image")
    mix.data_type = 'RGBA'
    mix.location = (bsdf.location.x - 360, bsdf.location.y + 200)
    a = _socket(mix.inputs, "A", 'RGBA')
    if target.is_linked:
        src = target.links[0].from_socket
        nt.links.remove(target.links[0])
        nt.links.new(src, a)
    else:
        a.default_value = target.default_value
    nt.links.new(attr.outputs["Alpha"], mix.inputs["Factor"])
    nt.links.new(_socket(mul.outputs, "Result", 'RGBA'), _socket(mix.inputs, "B", 'RGBA'))
    nt.links.new(_socket(mix.outputs, "Result", 'RGBA'), target)
    return True


# ---------------------------------------------------------------- node group
def create_node_group(size, material):
    ng = bpy.data.node_groups.new("Sand Cubes", 'GeometryNodeTree')
    ng.is_modifier = True
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    n, l = ng.nodes, ng.links

    gi = n.new('NodeGroupInput')
    gi.location = (-600, 0)
    cube = n.new('GeometryNodeMeshCube')
    cube.name = CUBE_NODE
    cube.label = _("Grain")
    cube.location = (-600, -180)
    cube.inputs["Size"].default_value = (size, size, size)
    setmat = n.new('GeometryNodeSetMaterial')
    setmat.location = (-380, -180)
    setmat.inputs["Material"].default_value = material
    rot = n.new('GeometryNodeInputNamedAttribute')
    rot.data_type = 'QUATERNION'
    rot.inputs["Name"].default_value = ROT_ATTR
    rot.location = (-380, -380)
    iop = n.new('GeometryNodeInstanceOnPoints')
    iop.location = (-120, 0)
    go = n.new('NodeGroupOutput')
    go.location = (120, 0)

    l.new(gi.outputs["Geometry"], iop.inputs["Points"])
    l.new(cube.outputs["Mesh"], setmat.inputs["Geometry"])
    l.new(setmat.outputs["Geometry"], iop.inputs["Instance"])
    l.new(rot.outputs["Attribute"], iop.inputs["Rotation"])
    l.new(iop.outputs["Instances"], go.inputs["Geometry"])
    upgrade_node_group(ng, size)
    return ng


def _parts(ng):
    """(group input, cube, set material, instance on points) of an add-on
    node group, or None if the user rebuilt it."""
    n = ng.nodes
    cube = n.get(CUBE_NODE)
    iop = next((x for x in n if x.bl_idname == 'GeometryNodeInstanceOnPoints'), None)
    gi = next((x for x in n if x.bl_idname == 'NodeGroupInput'), None)
    if cube is None or iop is None or gi is None:
        return None
    setmat = n.get(DRY_MAT)
    for lk in ([] if setmat is not None else cube.outputs["Mesh"].links):
        if lk.to_node.bl_idname == 'GeometryNodeSetMaterial':
            setmat = lk.to_node
    if setmat is None and n.get(SHAPE_SWITCH) is not None:
        for lk in n[SHAPE_SWITCH].outputs[0].links:
            if lk.to_node.bl_idname == 'GeometryNodeSetMaterial' and lk.to_node.name != WET_MAT:
                setmat = lk.to_node
    if setmat is None:
        return None
    return gi, cube, setmat, iop


def upgrade_node_group(ng, size=None):
    """Bring a group made by an older version up to date in place (nodes are
    added, the existing ones and their settings are kept; with the new
    features off the grains are exactly the old ones)."""
    if ng.nodes.get(WET_SWITCH) is None and not _upgrade_14(ng, size):
        return False
    if ng.nodes.get(MAT_SWITCH) is None and not _upgrade_15(ng):
        return False
    return True


def _upgrade_15(ng):
    """Separate materials for dry and wet grains: two variants of the grain,
    every grain picks the wet one with the probability of its wetness."""
    parts = _parts(ng)
    sw = ng.nodes.get(SHAPE_SWITCH)
    if parts is None or sw is None:
        return False
    gi, cube, dry, iop = parts
    n, l = ng.nodes, ng.links
    x0, y0 = dry.location.x, dry.location.y
    dry.name = DRY_MAT
    dry.label = _("Dry Sand Material")
    wet = n.new('GeometryNodeSetMaterial')
    wet.name = WET_MAT
    wet.label = _("Wet Sand Material")
    wet.location = (x0, y0 - 220)
    wet.inputs["Material"].default_value = dry.inputs["Material"].default_value
    l.new(sw.outputs[0], wet.inputs["Geometry"])
    sel = dry.inputs["Selection"]
    if sel.is_linked:
        l.new(sel.links[0].from_socket, wet.inputs["Selection"])
    else:
        wet.inputs["Selection"].default_value = sel.default_value
    g2i = n.new('GeometryNodeGeometryToInstance')
    g2i.location = (x0 + 220, y0 - 120)
    # instance 0 = dry, 1 = wet (the order of a multi-input is checked below)
    l.new(wet.outputs["Geometry"], g2i.inputs[0])
    l.new(dry.outputs["Geometry"], g2i.inputs[0])
    wet_first = g2i.inputs[0].links[0].from_node == wet
    on = n.new('FunctionNodeInputBool')
    on.name = MAT_ON
    on.label = _("Separate Wet Material")
    on.location = (x0 + 220, y0 + 160)
    msw = n.new('GeometryNodeSwitch')
    msw.input_type = 'GEOMETRY'
    msw.name = MAT_SWITCH
    msw.label = _("Dry / Dry + Wet")
    msw.location = (x0 + 420, y0)
    l.new(on.outputs[0], msw.inputs["Switch"])
    for lk in list(iop.inputs["Instance"].links):
        l.remove(lk)
    l.new(dry.outputs["Geometry"], msw.inputs["False"])
    l.new(g2i.outputs[0], msw.inputs["True"])
    l.new(msw.outputs[0], iop.inputs["Instance"])
    l.new(on.outputs[0], iop.inputs["Pick Instance"])
    rnd = n.new('FunctionNodeRandomValue')
    rnd.data_type = 'FLOAT'
    rnd.inputs["Seed"].default_value = 11
    rnd.location = (x0 + 220, y0 + 420)
    wa = n.new('GeometryNodeInputNamedAttribute')
    wa.data_type = 'FLOAT'
    wa.inputs["Name"].default_value = WET_ATTR
    wa.location = (x0 + 220, y0 + 300)
    cmp = n.new('FunctionNodeCompare')
    cmp.operation = 'LESS_THAN'
    cmp.location = (x0 + 420, y0 + 360)
    l.new(next(o for o in rnd.outputs if o.enabled), cmp.inputs[0])
    l.new(wa.outputs["Attribute"], cmp.inputs[1])
    pick = cmp.outputs[0]
    if wet_first:
        inv = n.new('FunctionNodeBooleanMath')
        inv.operation = 'NOT'
        inv.location = (x0 + 560, y0 + 360)
        l.new(pick, inv.inputs[0])
        pick = inv.outputs[0]
    l.new(pick, iop.inputs["Instance Index"])
    return True


def _upgrade_14(ng, size=None):
    parts = _parts(ng)
    if parts is None:
        return False
    gi, cube, setmat, iop = parts
    n, l = ng.nodes, ng.links
    x0, y0 = cube.location.x, cube.location.y
    if size is None:
        size = cube.inputs["Size"].default_value[0]

    def node(kind, loc, name=None, label=None, **props):
        nd = n.new(kind)
        nd.location = (x0 + loc[0], y0 + loc[1])
        if name:
            nd.name = name
        if label:
            nd.label = _(label)
        for k, v in props.items():
            setattr(nd, k, v)
        return nd

    def math(op, loc, a=None, b=None, vector=False):
        nd = node('ShaderNodeVectorMath' if vector else 'ShaderNodeMath', loc, operation=op)
        for k, s in ((0, a), (1, b)):
            if s is None:
                continue
            if isinstance(s, bpy.types.NodeSocket):
                l.new(s, nd.inputs[k])
            else:
                nd.inputs[k].default_value = s
        return nd

    # --- custom grain shape: the object, scaled so that its largest side is
    #     the grain size, centred; cube when there is no (valid) object
    info = node('GeometryNodeObjectInfo', (-200, -700), SHAPE_INFO, "Grain Shape",
                transform_space='RELATIVE')
    val = node('ShaderNodeValue', (-200, -950), SHAPE_SIZE, "Shape Size")
    val.outputs[0].default_value = size
    bb = node('GeometryNodeBoundBox', (0, -700))
    l.new(info.outputs["Geometry"], bb.inputs["Geometry"])
    ext = math('SUBTRACT', (200, -650), bb.outputs["Max"], bb.outputs["Min"], vector=True)
    sep = node('ShaderNodeSeparateXYZ', (400, -650))
    l.new(ext.outputs["Vector"], sep.inputs["Vector"])
    m1 = math('MAXIMUM', (600, -650), sep.outputs["X"], sep.outputs["Y"])
    m2 = math('MAXIMUM', (800, -650), m1.outputs[0], sep.outputs["Z"])
    fac = math('DIVIDE', (1000, -700), val.outputs[0], m2.outputs[0])
    ctr = math('ADD', (200, -800), bb.outputs["Min"], bb.outputs["Max"], vector=True)
    neg = math('MULTIPLY', (1200, -760), fac.outputs[0], -0.5)
    trn = node('ShaderNodeVectorMath', (1400, -800), operation='SCALE')
    l.new(ctr.outputs["Vector"], trn.inputs[0])
    l.new(neg.outputs[0], trn.inputs["Scale"])
    xf = node('GeometryNodeTransform', (1600, -700))
    l.new(info.outputs["Geometry"], xf.inputs["Geometry"])
    l.new(trn.outputs["Vector"], xf.inputs["Translation"])
    l.new(fac.outputs[0], xf.inputs["Scale"])
    cnt = node('GeometryNodeAttributeDomainSize', (0, -560))
    l.new(info.outputs["Geometry"], cnt.inputs["Geometry"])
    has = node('FunctionNodeCompare', (200, -520), operation='GREATER_THAN')
    l.new(cnt.outputs["Point Count"], has.inputs[0])
    has.inputs[1].default_value = 0.5
    sw = node('GeometryNodeSwitch', (1800, -300), SHAPE_SWITCH, "Cube / Custom Shape",
              input_type='GEOMETRY')
    l.new(has.outputs[0], sw.inputs["Switch"])
    for lk in list(cube.outputs["Mesh"].links):
        l.remove(lk)
    l.new(cube.outputs["Mesh"], sw.inputs["False"])
    l.new(xf.outputs["Geometry"], sw.inputs["True"])
    l.new(sw.outputs[0], setmat.inputs["Geometry"])
    own = node('FunctionNodeInputBool', (1800, -520), SHAPE_MAT, "Sand Material on the Shape")
    own.boolean = True
    imp = node('FunctionNodeBooleanMath', (2000, -480), operation='IMPLY')
    l.new(has.outputs[0], imp.inputs[0])
    l.new(own.outputs[0], imp.inputs[1])
    l.new(imp.outputs[0], setmat.inputs["Selection"])
    setmat.location = (x0 + 2200, y0)

    # --- per grain size factor (1 if the attribute does not exist)
    sc = node('GeometryNodeInputNamedAttribute', (1900, 300), label="sand_scale", data_type='FLOAT')
    sc.inputs["Name"].default_value = SCALE_ATTR
    ssw = node('GeometryNodeSwitch', (2150, 300), SCALE_SWITCH, "Grain Size", input_type='FLOAT')
    l.new(sc.outputs["Exists"], ssw.inputs["Switch"])
    ssw.inputs["False"].default_value = 1.0
    l.new(sc.outputs["Attribute"], ssw.inputs["True"])
    l.new(ssw.outputs[0], iop.inputs["Scale"])

    # --- wet zones: Noise (in the bounding box of the rest positions, like
    #     the Generated coordinates of a material) through a ColorRamp, times
    #     the paint weight; stored as sand_wet (read by the material and the
    #     solver)
    rest = node('GeometryNodeInputNamedAttribute', (-400, 900), label="sand_rest",
                data_type='FLOAT_VECTOR')
    rest.inputs["Name"].default_value = REST_ATTR
    stat = node('GeometryNodeAttributeStatistic', (-200, 1100), data_type='FLOAT_VECTOR', domain='POINT')
    l.new(gi.outputs["Geometry"], stat.inputs["Geometry"])
    l.new(rest.outputs["Attribute"], stat.inputs["Attribute"])
    wext = math('SUBTRACT', (0, 1100), stat.outputs["Max"], stat.outputs["Min"], vector=True)
    wsep = node('ShaderNodeSeparateXYZ', (200, 1100))
    l.new(wext.outputs["Vector"], wsep.inputs["Vector"])
    w1 = math('MAXIMUM', (400, 1100), wsep.outputs["X"], wsep.outputs["Y"])
    w2 = math('MAXIMUM', (600, 1100), w1.outputs[0], wsep.outputs["Z"])
    winv = math('DIVIDE', (800, 1100), 1.0, w2.outputs[0])
    rel = math('SUBTRACT', (0, 900), rest.outputs["Attribute"], stat.outputs["Min"], vector=True)
    gen = node('ShaderNodeVectorMath', (1000, 950), operation='SCALE')
    l.new(rel.outputs["Vector"], gen.inputs[0])
    l.new(winv.outputs[0], gen.inputs["Scale"])
    off = node('ShaderNodeVectorMath', (1200, 950), WET_OFFSET, "Noise Offset", operation='ADD')
    l.new(gen.outputs["Vector"], off.inputs[0])
    noise = node('ShaderNodeTexNoise', (1400, 950), WET_NOISE, "Wetness Noise")
    noise.inputs["Scale"].default_value = 5.0
    noise.inputs["Detail"].default_value = 2.0
    noise.inputs["Roughness"].default_value = 0.5
    l.new(off.outputs["Vector"], noise.inputs["Vector"])
    ramp = node('ShaderNodeValToRGB', (1600, 950), WET_RAMP, "Wetness Ramp")
    el = ramp.color_ramp.elements
    el[0].position, el[0].color = 0.45, (0.0, 0.0, 0.0, 1.0)
    el[1].position, el[1].color = 0.55, (1.0, 1.0, 1.0, 1.0)
    l.new(noise.outputs[0], ramp.inputs["Fac"])
    non = node('FunctionNodeInputBool', (1600, 700), WET_NOISE_ON, "Noise")
    pon = node('FunctionNodeInputBool', (1600, 600), WET_PAINT_ON, "Paint")
    nsw = node('GeometryNodeSwitch', (1900, 900), label="Noise On", input_type='FLOAT')
    l.new(non.outputs[0], nsw.inputs["Switch"])
    nsw.inputs["False"].default_value = 1.0
    l.new(ramp.outputs["Color"], nsw.inputs["True"])
    paint = node('GeometryNodeInputNamedAttribute', (1600, 500), label="sand_paint", data_type='FLOAT')
    paint.inputs["Name"].default_value = PAINT_ATTR
    psw = node('GeometryNodeSwitch', (1900, 650), label="Paint On", input_type='FLOAT')
    l.new(pon.outputs[0], psw.inputs["Switch"])
    psw.inputs["False"].default_value = 1.0
    l.new(paint.outputs["Attribute"], psw.inputs["True"])
    mul = math('MULTIPLY', (2100, 800), nsw.outputs[0], psw.outputs[0])
    mul.use_clamp = True
    store = node('GeometryNodeStoreNamedAttribute', (2300, 700), data_type='FLOAT', domain='POINT')
    store.inputs["Name"].default_value = WET_ATTR
    anyw = node('FunctionNodeBooleanMath', (2100, 600), operation='OR')
    l.new(non.outputs[0], anyw.inputs[0])
    l.new(pon.outputs[0], anyw.inputs[1])
    gsw = node('GeometryNodeSwitch', (2500, 500), WET_SWITCH, "Wetness", input_type='GEOMETRY')
    l.new(anyw.outputs[0], gsw.inputs["Switch"])
    for lk in list(iop.inputs["Points"].links):
        l.remove(lk)
    l.new(gi.outputs["Geometry"], gsw.inputs["False"])
    l.new(gi.outputs["Geometry"], store.inputs["Geometry"])
    l.new(mul.outputs[0], store.inputs["Value"])
    l.new(store.outputs["Geometry"], gsw.inputs["True"])
    l.new(gsw.outputs[0], iop.inputs["Points"])
    iop.location = (x0 + 2700, y0 + 180)
    go = next((x for x in n if x.bl_idname == 'NodeGroupOutput'), None)
    if go is not None:
        go.location = (x0 + 2950, y0 + 180)
    return True


def node_group(ob):
    for mod in ob.modifiers:
        if mod.type == 'NODES' and mod.node_group is not None \
                and mod.node_group.nodes.get(CUBE_NODE) is not None:
            return mod
    return None


def find_cube_node(ob):
    mod = node_group(ob)
    return None if mod is None else mod.node_group.nodes.get(CUBE_NODE)


def own_group(ob):
    """The object's node group, copied first if another object shares it
    (a duplicate): settings of one sand object never change another."""
    mod = node_group(ob)
    if mod is None:
        return None
    if mod.node_group.users > 1:
        mod.node_group = mod.node_group.copy()
    return mod.node_group


def get_node(ob, name):
    mod = node_group(ob)
    return None if mod is None else mod.node_group.nodes.get(name)


def _shape_size(ob, size):
    st = getattr(ob, "sand_sim", None)
    return size * (st.shape_scale if st is not None else 1.0)


def set_cube_size(ob, size):
    ng = own_group(ob)
    if ng is None:
        return
    ng.nodes[CUBE_NODE].inputs["Size"].default_value = (size, size, size)
    val = ng.nodes.get(SHAPE_SIZE)
    if val is not None:
        val.outputs[0].default_value = _shape_size(ob, size)


def _local_size(ob, grain_size):
    s = sum(ob.matrix_world.to_scale()) / 3.0
    return grain_size / s if s > 1e-9 else grain_size


def sync_cube_size(ob, grain_size):
    """Cubes are built in the object's local space: undo the object's scale
    so that a grain always has its real size."""
    size = _local_size(ob, grain_size)
    node = find_cube_node(ob)
    if node is None:
        return
    tol = 1e-7 * max(size, 1e-9)
    val = get_node(ob, SHAPE_SIZE)
    if abs(node.inputs["Size"].default_value[0] - size) > tol \
            or (val is not None and abs(val.outputs[0].default_value - _shape_size(ob, size)) > tol):
        set_cube_size(ob, size)


def ensure_modifier(ob, size):
    mod = node_group(ob)
    if mod is not None:
        upgrade_node_group(mod.node_group)
        set_cube_size(ob, size)
        return
    mod = ob.modifiers.new(MODIFIER_NAME, 'NODES')
    mod.node_group = create_node_group(size, ensure_material())


# ---------------------------------------------------------------- settings
def set_shape(ob, shape, sand_material=True):
    """Grain shape: an object (shown instead of the cube) or None (cube)."""
    mod = node_group(ob)
    if mod is None:
        return False
    if mod.node_group.nodes.get(SHAPE_INFO) is None and not upgrade_node_group(mod.node_group):
        return False
    info = mod.node_group.nodes[SHAPE_INFO]
    own = mod.node_group.nodes[SHAPE_MAT]
    if info.inputs["Object"].default_value != shape or own.boolean != bool(sand_material):
        ng = own_group(ob)
        ng.nodes[SHAPE_INFO].inputs["Object"].default_value = shape
        ng.nodes[SHAPE_MAT].boolean = bool(sand_material)
    return True


def set_wet_sources(ob, noise, paint):
    """Switch the wet zone sources on or off (both off: no wetness)."""
    mod = node_group(ob)
    if mod is None:
        return False
    if mod.node_group.nodes.get(WET_SWITCH) is None and not upgrade_node_group(mod.node_group):
        return False
    n = mod.node_group.nodes
    if n[WET_NOISE_ON].boolean != bool(noise) or n[WET_PAINT_ON].boolean != bool(paint):
        ng = own_group(ob)
        ng.nodes[WET_NOISE_ON].boolean = bool(noise)
        ng.nodes[WET_PAINT_ON].boolean = bool(paint)
    return True


def set_materials(ob, dry, wet, use_wet):
    """Material of the dry grains (None: keep the one in the node group) and of
    the wet ones (used when use_wet)."""
    mod = node_group(ob)
    if mod is None:
        return False
    if mod.node_group.nodes.get(MAT_SWITCH) is None and not upgrade_node_group(mod.node_group):
        return False
    n = mod.node_group.nodes
    dry_now = n[DRY_MAT].inputs["Material"].default_value
    dry_new = dry if dry is not None else dry_now
    wet_new = wet if wet is not None else dry_new
    if dry_now != dry_new or n[WET_MAT].inputs["Material"].default_value != wet_new \
            or n[MAT_ON].boolean != bool(use_wet):
        ng = own_group(ob)
        ng.nodes[DRY_MAT].inputs["Material"].default_value = dry_new
        ng.nodes[WET_MAT].inputs["Material"].default_value = wet_new
        ng.nodes[MAT_ON].boolean = bool(use_wet)
    return True


def set_color_attribute(me, name, values):
    """Static per grain RGBA attribute (None removes it)."""
    attr = me.attributes.get(name)
    if values is None:
        if attr is not None:
            me.attributes.remove(attr)
            me.update()
        return
    if attr is not None and (attr.data_type != 'FLOAT_COLOR' or attr.domain != 'POINT'):
        me.attributes.remove(attr)
        attr = None
    if attr is None:
        attr = me.attributes.new(name, 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set("color", np.ascontiguousarray(values, dtype=np.float32).ravel())
    me.update()


def set_float_attribute(me, name, values):
    """Static per grain float attribute (None removes it)."""
    attr = me.attributes.get(name)
    if values is None:
        if attr is not None:
            me.attributes.remove(attr)
        return
    if attr is not None and (attr.data_type != 'FLOAT' or attr.domain != 'POINT'):
        me.attributes.remove(attr)
        attr = None
    if attr is None:
        attr = me.attributes.new(name, 'FLOAT', 'POINT')
    attr.data.foreach_set("value", np.ascontiguousarray(values, dtype=np.float32).ravel())
    me.update()


def read_float_attribute(me, name):
    attr = me.attributes.get(name)
    if attr is None or attr.data_type != 'FLOAT' or attr.domain != 'POINT':
        return None
    out = np.empty(len(attr.data), dtype=np.float32)
    attr.data.foreach_get("value", out)
    return out


def evaluated_wetness(ob, depsgraph):
    """Wetness of every grain as the node group computes it (the values the
    material shows), or None if the wet zones are off."""
    mod = node_group(ob)
    if mod is None:
        return None
    n = mod.node_group.nodes
    if n.get(WET_SWITCH) is None or not (n[WET_NOISE_ON].boolean or n[WET_PAINT_ON].boolean):
        return None
    # a disabled modifier or an object disabled in the viewport is not
    # evaluated: switch them on for the moment of reading
    shown, hidden = mod.show_viewport, ob.hide_viewport
    if not shown:
        mod.show_viewport = True
    if hidden:
        ob.hide_viewport = False
    try:
        depsgraph.update()
        gs = ob.evaluated_get(depsgraph).evaluated_geometry()
        pc = gs.instances_pointcloud()
        attr = pc.attributes.get(WET_ATTR) if pc is not None else None
        if attr is None or len(attr.data) != len(ob.data.vertices):
            return None
        out = np.empty(len(attr.data), dtype=np.float32)
        attr.data.foreach_get("value", out)
        return np.clip(out.astype(np.float64), 0.0, 1.0)
    except (TypeError, RuntimeError, ReferenceError):
        return None
    finally:
        if not shown:
            mod.show_viewport = False
        if hidden:
            ob.hide_viewport = True
