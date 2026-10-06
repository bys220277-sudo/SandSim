# SPDX-License-Identifier: GPL-3.0-or-later
"""Sand Simulation: granular simulation of cube grains for Blender 5.2.

Each input point becomes a cube grain with mass, gravity, friction and
collisions with the other grains, the floor and collider objects.
UI: 3D Viewport > Sidebar (N) > Sand."""

import bpy
from bpy.app.handlers import persistent

from . import cache, deps, i18n, ops, props, ui

_classes = (props.SandSettings, props.SandCreateSettings) + ops.classes + ui.classes


@persistent
def _on_load_pre(*_args):
    ops.stop_all_jobs()


def register():
    i18n.register()
    deps.add_to_path()
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.sand_sim = bpy.props.PointerProperty(type=props.SandSettings)
    bpy.types.Scene.sand_sim_create = bpy.props.PointerProperty(type=props.SandCreateSettings)
    handlers = bpy.app.handlers
    if cache.on_frame_change not in handlers.frame_change_pre:
        handlers.frame_change_pre.append(cache.on_frame_change)
    if cache.on_load_post not in handlers.load_post:
        handlers.load_post.append(cache.on_load_post)
    if _on_load_pre not in handlers.load_pre:
        handlers.load_pre.append(_on_load_pre)


def unregister():
    ops.stop_all_jobs()
    handlers = bpy.app.handlers
    for lst, fn in ((handlers.frame_change_pre, cache.on_frame_change),
                    (handlers.load_post, cache.on_load_post),
                    (handlers.load_pre, _on_load_pre)):
        if fn in lst:
            lst.remove(fn)
    del bpy.types.Object.sand_sim
    del bpy.types.Scene.sand_sim_create
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    cache.forget()
    i18n.unregister()
