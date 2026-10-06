# SPDX-License-Identifier: GPL-3.0-or-later
"""Interface language: the add-on is written in English and follows the
language of Blender's interface (Preferences > Interface > Translation).
The Russian translation is in i18n_ru.py."""

try:
    import bpy
except ImportError:          # the solver modules are also used without Blender (tests)
    bpy = None


def iface(msg):
    """The interface text msg in the current language (for texts built at run
    time: formatted labels, reports, messages)."""
    if bpy is None or not msg:
        return msg
    try:
        return bpy.app.translations.pgettext_iface(msg)
    except Exception:
        return msg


def _translations():
    from .i18n_ru import RU
    table = {}
    for en, ru in RU.items():
        # "*": properties, panels, labels; "Operator": operator names
        table[("*", en)] = ru
        table[("Operator", en)] = ru
    return {"ru_RU": table}


def register():
    try:
        bpy.app.translations.register(__package__, _translations())
    except ValueError:              # already registered (reload)
        bpy.app.translations.unregister(__package__)
        bpy.app.translations.register(__package__, _translations())


def unregister():
    try:
        bpy.app.translations.unregister(__package__)
    except Exception:
        pass
