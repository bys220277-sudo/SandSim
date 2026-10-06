# SPDX-License-Identifier: GPL-3.0-or-later
import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty,
                       PointerProperty, StringProperty)

from . import nodes

SOURCE_TYPES = {'MESH', 'POINTCLOUD', 'CURVES'}           # points: vertices, points
VOLUME_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}  # volume: closed surfaces

SOURCE_MODES = [
    ('POINTS', "Points", "A grain at every point of the source: mesh vertices, point cloud, "
                        "Geometry Nodes result"),
    ('VOLUME', "Volume", "Fill the volume of a closed mesh with grains (like Distribute Points in "
                        "Volume in Geometry Nodes)"),
]
VOLUME_FILLS = [
    ('RANDOM', "Random", "Random points in the whole volume (Random mode)"),
    ('GRID', "Grid", "Cubic lattice (Grid mode). The random offset of the grains at baking "
                       "makes the packing natural"),
]
_COUNT_DESC = ("How many grains to place in the volume. The grain size is chosen so that they "
               "fill the volume")


def _poll_source(self, ob):
    return (ob.type in SOURCE_TYPES or ob.type in VOLUME_TYPES) and not ob.sand_sim.is_sand


class SandCreateSettings(bpy.types.PropertyGroup):
    """Options of Create Sand (on the scene, shown before the sand exists)."""
    source_mode: EnumProperty(name="Grains From", items=SOURCE_MODES, default='POINTS')
    volume_count: IntProperty(name="Count", default=20000, min=1, soft_max=2_000_000,
                              description=_COUNT_DESC)
    volume_fill: EnumProperty(name="Placement", items=VOLUME_FILLS, default='RANDOM')


def _update_size(self, context):
    ob = self.id_data
    if isinstance(ob, bpy.types.Object):
        nodes.sync_cube_size(ob, self.grain_size)


def _sand(self):
    ob = self.id_data
    return ob if isinstance(ob, bpy.types.Object) and ob.type == 'MESH' and self.is_sand else None


def _update_shape(self, context):
    ob = _sand(self)
    if ob is not None:
        from . import grains
        grains.sync_nodes(ob)
        nodes.sync_cube_size(ob, self.grain_size)


def _update_variance(self, context):
    ob = _sand(self)
    if ob is not None:
        from . import grains
        grains.write_scale(ob)


def _update_wet(self, context):
    ob = _sand(self)
    if ob is not None:
        from . import grains
        grains.sync_nodes(ob)


def _update_wet_paint(self, context):
    ob = _sand(self)
    if ob is not None:
        from . import grains
        grains.sync_nodes(ob)
        if self.use_wet_paint:
            grains.write_paint(ob, context.evaluated_depsgraph_get())


def _poll_shape(self, ob):
    return ob.type == 'MESH' and not ob.sand_sim.is_sand and ob != self.id_data


def _update_materials(self, context):
    ob = _sand(self)
    if ob is not None:
        from . import grains
        if self.dry_material is None:        # cleared: back to the add-on's material
            nodes.set_materials(ob, nodes.ensure_material(), self.wet_material, False)
        grains.sync_nodes(ob)


def _update_projection(self, context):
    ob = _sand(self)
    if ob is not None:
        from . import grains
        grains.sync_nodes(ob)
        grains.write_projection(ob, context.evaluated_depsgraph_get())


def _poll_camera(self, ob):
    return ob.type == 'CAMERA'


class SandSettings(bpy.types.PropertyGroup):
    # --- identity / state (set by the add-on) -------------------------------
    is_sand: BoolProperty(default=False, options={'HIDDEN'})
    cache_id: StringProperty(default="", options={'HIDDEN'})
    is_baked: BoolProperty(default=False, options={'HIDDEN'})
    baked_start: IntProperty(default=1, options={'HIDDEN'})
    baked_end: IntProperty(default=0, options={'HIDDEN'})
    grain_count: IntProperty(default=0, options={'HIDDEN'})
    bake_info: StringProperty(default="", options={'HIDDEN'})
    cache_folder: StringProperty(default="", options={'HIDDEN'})
    applied_source: StringProperty(default="", options={'HIDDEN'})   # source settings last read
    baked_size: FloatProperty(default=0.0, options={'HIDDEN'})

    source: PointerProperty(
        name="Point Source", type=bpy.types.Object, poll=_poll_source,
        description="Object whose points give the start positions of the grains (mesh, point "
                    "cloud or Geometry Nodes result). Read at every bake")
    source_mode: EnumProperty(
        name="Grains From", items=SOURCE_MODES, default='POINTS',
        description="The points of the source or the filling of its volume. Applied by “Reset” or "
                    "at baking")
    volume_count: IntProperty(name="Count", default=20000, min=1, soft_max=2_000_000,
                              description=_COUNT_DESC)
    volume_fill: EnumProperty(name="Placement", items=VOLUME_FILLS, default='RANDOM')

    # --- grains ---------------------------------------------------------------
    grain_size: FloatProperty(
        name="Grain Size", default=0.02, min=0.0002, soft_max=0.5,
        unit='LENGTH', precision=4, update=_update_size,
        description="Edge length of the cube")
    collision_scale: FloatProperty(
        name="Collision Size", default=1.15, min=0.5, max=2.0,
        description="Diameter of the collision sphere in cube edges. 1.0: inscribed sphere "
                    "(corners of neighboring cubes may intersect), 1.24: sphere of the same "
                    "volume as the cube")
    mass: FloatProperty(
        name="Grain Mass", default=0.02, min=1e-7, soft_max=10.0,
        unit='MASS', precision=5,
        description="Mass of a cube of the Grain Size (with size variance: of the mean volume). "
                    "Set like quartz (2650 kg/m³) when the sand is created. With a custom shape "
                    "the mass follows its volume. Together with the air drag it decides how much "
                    "the air slows the fall and how the wind blows")
    size_variance: FloatProperty(
        name="Size Variance", default=0.0, min=0.0, max=70.0, soft_max=60.0,
        subtype='PERCENTAGE', precision=0, update=_update_variance,
        description="Grains of different sizes: from −N% to +N% (random, by the seed). The size "
                    "really affects the physics: collision radius and mass (a bigger grain is "
                    "heavier). The mean volume stays that of the Grain Size, so the total volume "
                    "and mass of the sand do not change. Only for the “Impulses” physics")
    shape_object: PointerProperty(
        name="Shape", type=bpy.types.Object, poll=_poll_shape, update=_update_shape,
        description="Your own mesh object instead of the cube (a pebble, a grain…). Scaled so "
                    "that its largest side equals the grain size. Empty: cube. In the physics the "
                    "grain is a sphere of the same volume as the shape (× Collision Size / 1.24, "
                    "as for the cube)")
    shape_scale: FloatProperty(
        name="Shape Scale", default=1.0, min=0.05, soft_max=4.0, update=_update_size,
        description="Extra size factor of the custom shape. Changes both the look and the "
                    "physics: the collision sphere follows the volume of the shape")
    shape_material: BoolProperty(
        name="Sand Material", default=True, update=_update_shape,
        description="Give the shape the sand material (with tints and wetness). Off: the shape "
                    "keeps its own materials")

    # --- physics --------------------------------------------------------------
    friction: FloatProperty(
        name="Static Friction", default=0.25, min=0.0, soft_max=1.5,
        description="Static friction coefficient between grains (Coulomb's law). The main "
                    "parameter of the angle of repose. The grains do not roll (angular), so the "
                    "angle is larger than arctan(μ): 0.2 ≈ 33°, 0.25 ≈ 36° (dry sand), 0.3 ≈ 38°, "
                    "0.4 ≈ 41°")
    friction_dynamic: FloatProperty(
        name="Sliding Friction", default=0.22, min=0.0, soft_max=1.5,
        description="Sliding (kinetic) friction coefficient between grains")

    # --- wet zones -----------------------------------------------------------------
    use_wet_noise: BoolProperty(
        name="Noise", default=False, update=_update_wet,
        description="Wet zones from noise (Noise Texture, as in materials) through a color ramp")
    use_wet_paint: BoolProperty(
        name="Paint Map", default=False, update=_update_wet_paint,
        description="Wet zones painted on the source: vertex group (Weight Paint) or attribute. "
                    "Together with the noise it works as a mask: noise only where painted")
    wet_paint_name: StringProperty(
        name="Vertex Group", default="", update=_update_wet_paint,
        description="Vertex group (Weight Paint) or attribute of the source (float, color). In "
                    "Volume mode taken from the nearest surface vertices")
    wet_friction: FloatProperty(
        name="Wet Friction", default=0.8, min=0.0, soft_max=2.0,
        description="Static friction of a fully wet grain (dry: Static Friction). Wet sand holds "
                    "steeper slopes: 0.5 ≈ 48°, 0.8 ≈ 60°, above 1 almost vertical walls. The "
                    "friction of a pair is the mean of the two grains; on the floor and colliders "
                    "a wet grain grips as many times harder")
    wet_info: StringProperty(default="", options={'HIDDEN'})
    wet_ok: BoolProperty(default=False, options={'HIDDEN'})
    use_cohesion: BoolProperty(
        name="Clumps", default=False,
        description="Wet grains hold together with water bridges: wet sand keeps its shape, "
                    "cracks into plates and breaks into clumps. A bridge that is stretched too "
                    "far breaks for good. Needs wetness and the “Impulses” physics")
    cohesion: FloatProperty(
        name="Cohesion", default=10.0, min=0.0, soft_max=50.0,
        description="Strength of the water bridge between two fully wet grains, in weights of a "
                    "grain (1: a grain holds one grain hanging on it). More: bigger clumps, "
                    "steeper walls, fewer cracks")
    adhesion: FloatProperty(
        name="Stickiness", default=3.0, min=0.0, soft_max=30.0,
        description="How strongly wet grains stick to the floor and the colliders, in weights of "
                    "a grain: wet sand stays on the objects and slides off them in sheets")

    # --- look: materials and image projection --------------------------------------
    dry_material: PointerProperty(
        name="Dry Sand", type=bpy.types.Material, update=_update_materials,
        description="Material of the dry grains. Empty: the add-on's “Sand” material")
    wet_material: PointerProperty(
        name="Wet Sand", type=bpy.types.Material, update=_update_materials,
        description="Material of the wet grains (needs wetness). A grain gets it with a "
                    "probability equal to its wetness: at the border of the zones dry and wet "
                    "grains are mixed. Empty: all grains use the dry material (wet ones are "
                    "darker)")
    use_projection: BoolProperty(
        name="Image From Camera", default=False, update=_update_projection,
        description="Color the grains with an image projected from the camera onto their start "
                    "positions. The color sticks to the grain and moves with it")
    project_image: PointerProperty(
        name="Image", type=bpy.types.Image, update=_update_projection,
        description="Image seen from the camera, stretched over the frame")
    project_camera: PointerProperty(
        name="Camera", type=bpy.types.Object, poll=_poll_camera, update=_update_projection,
        description="Projection camera. Empty: the active scene camera (at the first simulation frame)")
    project_strength: FloatProperty(
        name="Strength", default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=_update_projection,
        description="1: image color, 0: sand color")
    project_info: StringProperty(default="", options={'HIDDEN'})
    project_ok: BoolProperty(default=False, options={'HIDDEN'})

    # --- force fields ------------------------------------------------------------------
    force_fields: PointerProperty(
        name="Force Fields", type=bpy.types.Collection,
        description="Collection of objects with a Force Field (Force, Wind, Vortex, Turbulence, "
                    "Drag, Harmonic). Animated strength and position of the fields are used")
    baked_sig: StringProperty(default="", options={'HIDDEN'})
    use_air_drag: BoolProperty(
        name="Air Drag", default=True,
        description="Quadratic drag F = ½·ρ·Cd·A·v² (A: face of the cube)")
    drag_cd: FloatProperty(
        name="Cd", default=1.05, min=0.0, soft_max=3.0,
        description="Drag coefficient (cube ≈ 1.05)")
    air_density: FloatProperty(
        name="Air Density", default=1.225, min=0.0, soft_max=5.0,
        description="kg/m³")
    jitter: FloatProperty(
        name="Random Offset", default=0.5, min=0.0, max=1.0,
        description="Randomly offset the start positions by this fraction of the grain size. "
                    "Points on a perfect lattice otherwise land on top of each other in columns "
                    "and are held by friction, like a stack of sugar cubes")
    initial_velocity: FloatVectorProperty(
        name="Initial Velocity", size=3, default=(0.0, 0.0, 0.0), unit='VELOCITY',
        subtype='VELOCITY')

    # --- collisions -------------------------------------------------------------
    use_floor: BoolProperty(name="Floor", default=True,
                            description="Infinite horizontal plane")
    floor_height: FloatProperty(name="Floor Height", default=0.0, unit='LENGTH')
    colliders: PointerProperty(
        name="Colliders", type=bpy.types.Collection,
        description="Collection of obstacle objects (meshes). Their shape at the first simulation "
                    "frame is used, with modifiers. Closed meshes are solid bodies, open ones "
                    "(planes) thin two-sided surfaces. Animated position and rotation (keys, parent, "
                    "constraints) move the collider and push the sand")
    collider_friction: FloatProperty(
        name="Collider Friction", default=0.5, min=0.0, soft_max=1.5,
        description="Static friction coefficient on the floor and colliders (sliding friction = "
                    "85% of it)")
    collider_voxel: FloatProperty(
        name="Collider Accuracy", default=0.5, min=0.1, max=4.0,
        description="Cell size of the distance field in grain sizes. Smaller is more accurate, "
                    "but slower to prepare")

    # --- baking -----------------------------------------------------------------
    frame_start: IntProperty(name="Start", default=1, min=0)
    frame_end: IntProperty(name="End", default=250, min=1)
    time_scale: FloatProperty(
        name="Simulation Speed", default=1.0, min=0.01, soft_max=4.0,
        description="Time multiplier of the simulation (1: real time)")
    cache_dir: StringProperty(
        name="Cache Folder", default="//sand_cache", subtype='DIR_PATH',
        description="Where to save the simulation frames (//: next to the .blend file)")
    live_preview: BoolProperty(
        name="Show While Baking", default=True,
        description="Switch to the last computed frame while baking")

    device: EnumProperty(
        name="Device",
        items=[('AUTO', "Auto", "NVIDIA graphics card (CUDA) if there is one, otherwise the processor"),
               ('GPU', "GPU (CUDA)", "Only an NVIDIA graphics card. Without one: error"),
               ('CPU', "CPU", "Processor (compiled NVIDIA Warp kernels, if not available: NumPy)")],
        default='AUTO',
        description="Where to compute the simulation. The physics is the same, only the speed differs")

    # --- accuracy -----------------------------------------------------------------
    model: EnumProperty(
        name="Physics",
        items=[('IMPULSE', "Impulses",
                "Contact dynamics: grains exchange equal and opposite impulses (momentum is "
                "conserved), impacts are inelastic, Coulomb friction: energy can only decrease, "
                "so the sand cannot explode"),
               ('PBD', "Positions (1.2)",
                "The previous position based model (Position Based Dynamics), as in versions before 1.3")],
        default='IMPULSE',
        description="Contact model")
    iterations: IntProperty(
        name="Iterations", default=3, min=1, max=20,
        description="Contact solver iterations per substep")
    stack_sweeps: IntProperty(
        name="Height Sweeps", default=1, min=0, max=5,
        description="Bottom-up sweeps that carry the support of the floor through the whole pile. "
                    "Usually 1 is enough; more is more accurate for tall piles, but slower")
    min_substeps: IntProperty(name="Min Substeps", default=2, min=1, max=64)
    max_substeps: IntProperty(
        name="Max Substeps", default=32, min=1, max=256,
        description="A grain moves at most ~⅓ of its size per substep. If there are not enough "
                    "substeps, the speed is limited")
    use_sleep: BoolProperty(
        name="Sleeping", default=True,
        description="Grains at rest “fall asleep” (are not computed) until something touches "
                    "them. Removes jitter of a settled pile and speeds up the computation")
    sleep_velocity: FloatProperty(
        name="Rest Threshold", default=0.05, min=0.001, max=1.0,
        description="Sleep speed in units of √(g·d)")
    sleep_time: FloatProperty(
        name="Time to Sleep", default=0.5, min=0.02, max=5.0, unit='TIME_ABSOLUTE',
        description="How many seconds a grain must be at rest")
    tumble: FloatProperty(
        name="Tumbling", default=0.6, min=0.0, max=2.0,
        description="Visual rotation of the cubes when they roll over what they touch (by the "
                    "speed relative to the neighbors and the floor). Grains flying together do "
                    "not spin; at most 30° per frame so that the cubes do not flicker")
    seed: IntProperty(name="Seed", default=0, min=0,
                      description="Random start rotations and tints")

    def size_factor(self, use_cache=True):
        """Physical size of a grain relative to the cube of the Grain Size
        (1 for the cube; the volume of a custom shape as it is shown)."""
        from . import grains
        ob = self.id_data
        return grains.size_factor(ob, use_cache=use_cache) if isinstance(ob, bpy.types.Object) else 1.0

    def physical_diameter(self, use_cache=True):
        return self.grain_size * self.collision_scale * self.size_factor(use_cache)

    def grain_signature(self):
        """What the grains of a bake are made of (a change asks for a new bake)."""
        return "%.6g|%.6g|%.6g|%.6g|%d|%s" % (self.grain_size, self.collision_scale,
                                              self.size_factor(), self.size_variance,
                                              self.seed, self.model)

    def solver_params(self, scene):
        from .solver import SandParams
        g = tuple(scene.gravity) if scene.use_gravity else (0.0, 0.0, 0.0)
        f = self.size_factor(use_cache=False)      # 1.0 for the cube
        return SandParams(
            model=self.model,
            diameter=self.grain_size * self.collision_scale * f,
            mass=self.mass * f ** 3,
            gravity=g,
            air_drag=self.use_air_drag,
            drag_area=(self.grain_size * f) ** 2,
            drag_cd=self.drag_cd,
            air_density=self.air_density,
            friction_static=self.friction,
            friction_dynamic=min(self.friction_dynamic, self.friction),
            wet_friction_static=self.wet_friction if (self.use_wet_noise or self.use_wet_paint) else None,
            cohesion=self.cohesion if self.use_cohesion else 0.0,
            adhesion=self.adhesion if self.use_cohesion else 0.0,
            collider_friction_static=self.collider_friction,
            collider_friction_dynamic=0.85 * self.collider_friction,
            use_floor=self.use_floor,
            floor_height=self.floor_height,
            iterations=self.iterations,
            stack_sweeps=self.stack_sweeps,
            min_substeps=self.min_substeps,
            max_substeps=max(self.max_substeps, self.min_substeps),
            use_sleep=self.use_sleep,
            sleep_velocity=self.sleep_velocity,
            sleep_time=self.sleep_time,
            initial_velocity=tuple(self.initial_velocity),
            jitter=self.jitter,
            tumble=self.tumble,
            seed=self.seed,
        )
