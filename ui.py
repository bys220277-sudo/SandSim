# SPDX-License-Identifier: GPL-3.0-or-later
import bpy

import time

from . import cache, deps, nodes, ops
from .i18n import iface as _
from .props import SOURCE_TYPES, VOLUME_TYPES


class SandPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Sand"

    @classmethod
    def poll(cls, context):
        return ops.sand_object(context) is not None


class SAND_PT_main(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Sand"
    bl_label = "Sand Simulation"

    def draw(self, context):
        layout = self.layout
        ob = ops.sand_object(context)
        if ob is None:
            self.draw_create(context, layout)
            return
        st = ob.sand_sim
        job = ops.running_job(ob)

        col = layout.column(align=True)
        col.prop(st, "source")
        draw_source_mode(col, st)
        if st.applied_source and st.applied_source != ops.source_signature(st):
            col.label(text="Applied by “Reset” or at baking", icon='INFO')
        col.label(text=_("Grains: %d") % st.grain_count, icon='PARTICLES')

        box = layout.box()
        if job is not None:
            s = job.status
            frac = s["done"] / max(s["total"], 1) if s["stage"] == "Simulating" else 0.0
            box.progress(factor=frac, type='BAR',
                         text="%s  %d%%" % (ops.stage_text(s), round(frac * 100)))
            box.label(text=job.status_line(), translate=False)
            box.operator("sand_sim.bake_stop", icon='CANCEL')
        else:
            row = box.row(align=True)
            row.prop(st, "device", expand=True)
            if deps.state["running"]:
                box.label(text=_("%s %d s") % (_(deps.state["log"] or "Installing NVIDIA Warp…"),
                                            time.time() - deps.state["t0"]), icon='SORTTIME')
            elif not deps.warp_available():
                col = box.column(align=True)
                col.label(text="GPU computation needs NVIDIA Warp (~150 MB)", icon='INFO')
                if not deps.online_allowed():
                    col.label(text="Allow network access: Preferences → System → Allow Online Access",
                              icon='ERROR')
                col.operator("sand_sim.install_warp", icon='IMPORT')
            row = box.row(align=True)
            info = ops._gpu_info
            if info["checked"]:
                row.label(text=("GPU: " if info["ok"] else _("GPU not available: ")) + info["text"], translate=False,
                          icon='CHECKMARK' if info["ok"] else 'ERROR')
            row.operator("sand_sim.check_gpu", text="" if info["checked"] else "Check GPU",
                         icon='VIEWZOOM')
            row = box.row()
            row.scale_y = 1.5
            row.operator("sand_sim.bake", icon='PHYSICS')
            box.operator("sand_sim.free", icon='TRASH')
            if st.bake_info:
                box.label(text=st.bake_info, icon='INFO', translate=False)
            changed = (st.baked_sig != st.grain_signature()) if st.baked_sig \
                else abs(st.baked_size - st.grain_size) > 1e-9
            if st.is_baked and changed:
                box.label(text="Grains changed after baking (size, shape, variance): bake again", icon='ERROR')
            elif not st.is_baked:
                box.label(text="Not baked", icon='INFO')

    def draw_create(self, context, layout):
        ob = context.active_object
        opts = context.scene.sand_sim_create
        col = layout.column()
        if ob is not None and (ob.type in SOURCE_TYPES or ob.type in VOLUME_TYPES):
            col.label(text=_("Source: ") + ob.name, icon='OBJECT_DATA', translate=False)
        else:
            col.label(text="Select an object with points", icon='INFO')
            col.label(text="(mesh, point cloud or Geometry Nodes)")
        draw_source_mode(col.column(align=True), opts)
        row = col.row()
        row.scale_y = 1.5
        row.operator("sand_sim.create", icon='ADD')
        sands = [o for o in context.scene.objects if o.type == 'MESH' and o.sand_sim.is_sand]
        if sands:
            col.separator()
            col.label(text="Sand in the scene:")
            for o in sands:
                col.operator("sand_sim.select", text=o.name, icon='RESTRICT_SELECT_OFF', translate=False).name = o.name


def draw_source_mode(col, opts):
    row = col.row(align=True)
    row.prop(opts, "source_mode", expand=True)
    if opts.source_mode == 'VOLUME':
        col.prop(opts, "volume_count")
        row = col.row(align=True)
        row.prop(opts, "volume_fill", expand=True)


class SAND_PT_grains(SandPanel, bpy.types.Panel):
    bl_label = "Grains"
    bl_parent_id = "SAND_PT_main"

    def draw(self, context):
        st = ops.sand_object(context).sand_sim
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.prop(st, "grain_size")
        col.operator("sand_sim.auto_size", icon='DRIVER_DISTANCE')
        col.prop(st, "collision_scale")
        f = st.size_factor()
        col.label(text=_("Collisions: sphere Ø %.1f mm%s") % (
            1000.0 * st.grain_size * st.collision_scale * f,
            _(" (by the shape volume)") if st.shape_object is not None else ""), icon='SPHERE')
        col.prop(st, "jitter")
        col.prop(st, "mass")
        vol = st.grain_size ** 3
        if vol > 0:
            col.label(text=_("Material density ≈ %.0f kg/m³ (quartz ≈ 2650)") % (st.mass / vol))
        col.separator()
        col.prop(st, "size_variance")
        if st.size_variance > 0:
            if st.model == 'PBD':
                col.label(text="With “Positions (1.2)” physics the variance is only visual", icon='ERROR')
            elif st.is_baked:
                col.label(text="Applied at baking", icon='INFO')
        col.separator()
        col.prop(st, "shape_object")
        if st.shape_object is not None:
            col.prop(st, "shape_scale")
            col.prop(st, "shape_material")
            if nodes.get_node(ops.sand_object(context), nodes.SHAPE_INFO) is None:
                col.label(text="Old sand nodes: press “Reset”", icon='ERROR')


class SAND_PT_physics(SandPanel, bpy.types.Panel):
    bl_label = "Physics"
    bl_parent_id = "SAND_PT_main"

    def draw(self, context):
        st = ops.sand_object(context).sand_sim
        scene = context.scene
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.prop(scene, "use_gravity", text="Scene Gravity")
        sub = col.column()
        sub.active = scene.use_gravity
        sub.prop(scene, "gravity", text="Acceleration")
        col.separator()
        col.prop(st, "friction")
        col.prop(st, "friction_dynamic")
        col.separator()
        col.prop(st, "use_air_drag")
        sub = col.column()
        sub.active = st.use_air_drag
        sub.prop(st, "drag_cd")
        sub.prop(st, "air_density")
        col.separator()
        col.prop(st, "initial_velocity")


class SAND_PT_wet(SandPanel, bpy.types.Panel):
    bl_label = "Wetness"
    bl_parent_id = "SAND_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        ob = ops.sand_object(context)
        st = ob.sand_sim
        layout = self.layout
        row = layout.row(align=True)
        row.prop(st, "use_wet_noise", toggle=True, icon='TEXTURE')
        row.prop(st, "use_wet_paint", toggle=True, icon='BRUSH_DATA')
        on = st.use_wet_noise or st.use_wet_paint
        noise = nodes.get_node(ob, nodes.WET_NOISE)
        if on and noise is None:
            layout.label(text="Old sand nodes: press “Reset”", icon='ERROR')
            return
        if st.use_wet_noise:
            box = layout.box()
            col = box.column()
            col.use_property_split = True
            col.use_property_decorate = False
            col.prop(noise.inputs["Scale"], "default_value", text="Scale")
            col.prop(noise.inputs["Detail"], "default_value", text="Detail")
            col.prop(noise.inputs["Roughness"], "default_value", text="Roughness")
            off = nodes.get_node(ob, nodes.WET_OFFSET)
            if off is not None:
                col.prop(off.inputs[1], "default_value", text="Offset")
            ramp = nodes.get_node(ob, nodes.WET_RAMP)
            if ramp is not None:
                box.label(text="Ramp: black is dry, white is wet")
                box.template_color_ramp(ramp, "color_ramp", expand=True)
        if st.use_wet_paint:
            box = layout.box()
            col = box.column()
            src = st.source
            if src is not None and src.type == 'MESH':
                col.prop_search(st, "wet_paint_name", src, "vertex_groups", text="Group")
            else:
                col.prop(st, "wet_paint_name", text="Attribute")
            col.operator("sand_sim.refresh_paint", icon='FILE_REFRESH')
            if st.wet_info:
                col.label(text=st.wet_info, translate=False,
                          icon='CHECKMARK' if st.wet_ok else 'ERROR')
            if st.use_wet_noise:
                col.label(text="Paint is a mask: noise only where painted", icon='INFO')
        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        sub = col.column()
        sub.active = on
        sub.prop(st, "wet_friction")
        sub.prop(st, "use_cohesion")
        coh = sub.column()
        coh.active = on and st.use_cohesion
        coh.prop(st, "cohesion")
        coh.prop(st, "adhesion")
        if st.use_cohesion and not on:
            col.label(text="Clumps need wet grains: Noise or Paint Map", icon='INFO')
        if st.use_wet_noise:
            col.operator("sand_sim.wet_all", icon='MOD_FLUIDSIM')
        if on:
            col.label(text="Wet sand is darker (Material Preview / Render)", icon='SHADING_TEXTURE')
            if st.model == 'PBD':
                col.label(text="With “Positions (1.2)” physics wetness is ignored", icon='ERROR')
            elif st.is_baked:
                col.label(text="Changes are applied at baking", icon='INFO')


class SAND_PT_look(SandPanel, bpy.types.Panel):
    bl_label = "Material and Image"
    bl_parent_id = "SAND_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        ob = ops.sand_object(context)
        st = ob.sand_sim
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.prop(st, "dry_material")
        sub = col.column()
        wet = st.use_wet_noise or st.use_wet_paint
        sub.active = wet
        sub.prop(st, "wet_material")
        if st.wet_material is not None and not wet:
            sub.label(text="Needs wetness (“Wetness” panel)", icon='INFO')
        layout.separator()
        col = layout.column()
        col.prop(st, "use_projection")
        if st.use_projection:
            col.template_ID(st, "project_image", new="image.new", open="image.open")
            col.prop(st, "project_camera")
            if st.project_camera is None:
                cam = context.scene.camera
                col.label(text=_("Scene camera: ") + (cam.name if cam else _("none")), translate=False,
                          icon='CAMERA_DATA' if cam else 'ERROR')
            col.prop(st, "project_strength")
            col.operator("sand_sim.project", icon='IMAGE_DATA')
            if st.project_info:
                col.label(text=st.project_info, translate=False,
                          icon='CHECKMARK' if st.project_ok else 'ERROR')
            col.label(text="Color: attribute sand_color (Attribute, Instancer)", icon='INFO')


class SAND_PT_collisions(SandPanel, bpy.types.Panel):
    bl_label = "Collisions"
    bl_parent_id = "SAND_PT_main"

    def draw(self, context):
        st = ops.sand_object(context).sand_sim
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.prop(st, "use_floor")
        sub = col.column()
        sub.active = st.use_floor
        sub.prop(st, "floor_height")
        col.separator()
        col.prop(st, "colliders")
        sub = col.column()
        sub.active = st.colliders is not None
        sub.prop(st, "collider_friction")
        sub.prop(st, "collider_voxel")
        sub.operator("sand_sim.preview_colliders", icon='POINTCLOUD_DATA')


FIELD_ICONS = {'FORCE': 'FORCE_FORCE', 'WIND': 'FORCE_WIND', 'VORTEX': 'FORCE_VORTEX',
               'TURBULENCE': 'FORCE_TURBULENCE', 'DRAG': 'FORCE_DRAG', 'HARMONIC': 'FORCE_HARMONIC'}


class SAND_PT_fields(SandPanel, bpy.types.Panel):
    bl_label = "Force Fields"
    bl_parent_id = "SAND_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        st = ops.sand_object(context).sand_sim
        layout = self.layout
        col = layout.column()
        col.prop(st, "force_fields", text="Collection")
        if st.force_fields is None:
            col.label(text="Put objects with a Force Field into the collection", icon='INFO')
            return
        objs, notes = ops.field_objects(st)
        if not objs:
            col.label(text="The collection has no Force, Wind, Vortex,", icon='ERROR')
            col.label(text="Turbulence, Drag, Harmonic")
        box = col.box() if objs else col
        for o in objs:
            f = o.field
            row = box.row()
            row.label(text=o.name, icon=FIELD_ICONS.get(f.type, 'FORCE_FORCE'), translate=False)
            if f.type == 'WIND':
                row.label(text=_("wind %.1f m/s") % f.strength)
            elif f.type == 'DRAG':
                row.label(text=_("drag %.2f / %.2f") % (f.linear_drag, f.quadratic_drag))
            else:
                row.label(text=_("strength %.2f") % f.strength)
        for note in notes:
            col.label(text=note, icon='ERROR', translate=False)
        if objs:
            col.label(text="Strength as for Blender particles (10 ≈ gravity);", icon='INFO')
            col.label(text="Wind: Strength = wind speed, m/s")
            if st.model == 'PBD':
                col.label(text="With “Positions (1.2)” physics fields have no effect", icon='ERROR')


class SAND_PT_bake(SandPanel, bpy.types.Panel):
    bl_label = "Frames and Cache"
    bl_parent_id = "SAND_PT_main"

    def draw(self, context):
        ob = ops.sand_object(context)
        st = ob.sand_sim
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column(align=True)
        col.prop(st, "frame_start")
        col.prop(st, "frame_end")
        col = layout.column()
        col.prop(st, "time_scale")
        col.prop(st, "cache_dir")
        if not bpy.data.filepath and st.cache_dir.startswith("//"):
            col.label(text="File not saved: the cache is in a temporary folder", icon='ERROR')
        col.prop(st, "live_preview")
        col.label(text=cache.cache_dir(ob), icon='FILE_FOLDER', translate=False)


class SAND_PT_accuracy(SandPanel, bpy.types.Panel):
    bl_label = "Accuracy and Speed"
    bl_parent_id = "SAND_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        st = ops.sand_object(context).sand_sim
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.prop(st, "model")
        col.prop(st, "iterations")
        col.prop(st, "stack_sweeps")
        col.prop(st, "min_substeps")
        col.prop(st, "max_substeps")
        col.separator()
        col.prop(st, "use_sleep")
        sub = col.column()
        sub.active = st.use_sleep
        sub.prop(st, "sleep_velocity")
        sub.prop(st, "sleep_time")
        col.separator()
        col.prop(st, "tumble")
        col.prop(st, "seed")


classes = (SAND_PT_main, SAND_PT_grains, SAND_PT_physics, SAND_PT_wet, SAND_PT_look,
           SAND_PT_collisions, SAND_PT_fields, SAND_PT_bake, SAND_PT_accuracy)
