# SPDX-License-Identifier: GPL-3.0-or-later
import os
import threading
import time
import traceback
import uuid

import bpy
import numpy as np
from bpy.props import BoolProperty, StringProperty

from . import cache, colliders, deps, fields, grains, nodes, volume
from .i18n import iface as _
from .props import SOURCE_TYPES, VOLUME_TYPES
from .solver import SandSolver, find_pairs

_jobs = {}   # sand object name -> running BakeJob
_gpu_info = {"checked": False, "ok": False, "text": ""}


def _warp_cache_dir():
    try:
        return bpy.utils.extension_path_user(__package__, path="warp_cache", create=True)
    except Exception:
        return None


def collider_meshes(ob, dg):
    st = ob.sand_sim
    out = []
    if st.colliders is not None:
        # a copy of the list first: reading a mesh adds and removes a temporary
        # mesh, which makes Blender rebuild the collection's object list - the
        # iteration would go on in freed memory (crash with 2+ colliders)
        for c in list(st.colliders.all_objects):
            if c is ob or c is st.source or c.type not in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}:
                continue
            data = colliders.read_collider_mesh(c, dg)
            if data is not None:
                out.append(data)
    return out


def field_objects(st):
    """(objects of the force field collection whose field is simulated, notes)."""
    objs, notes = [], []
    col = st.force_fields
    if col is None:
        return objs, notes
    for o in list(col.all_objects):
        row, note = fields.pack_field(o)
        if note:
            notes.append(note)
        if row is not None:
            objs.append(o)
    return objs, notes


def _animated(o):
    while o is not None:
        ad = o.animation_data
        if ad is not None and (ad.action is not None or len(ad.drivers) or len(ad.nla_tracks)):
            return True
        if len(o.constraints):
            return True
        o = o.parent
    return False


def _pack_all(objs):
    rows = []
    for o in objs:
        row, _ = fields.pack_field(o)
        rows.append(row if row is not None else np.zeros(fields.COLS))
    return np.array(rows)


def sample_fields(scene, objs, start, end):
    """Packed force fields: {frame: rows} for every frame of the bake, or
    {"static": rows} when no field moves or changes."""
    if not objs:
        return None
    if not any(_animated(o) for o in objs):
        return {"static": _pack_all(objs)}
    # sand of the scene is not needed for this: switch its nodes off meanwhile
    muted = []
    for o in scene.objects:
        if o.type == 'MESH' and o.sand_sim.is_sand:
            m = nodes.node_group(o)
            if m is not None and m.show_viewport:
                m.show_viewport = False
                muted.append(m)
    table = {}
    try:
        for f in range(start, end + 1):
            scene.frame_set(f)
            table[f] = _pack_all(objs)
    finally:
        for m in muted:
            m.show_viewport = True
    return table


def _may_move(o):
    """Can the object's transform change over time (animation, drivers,
    constraints - also of a parent)?"""
    while o is not None:
        if _animated(o) or len(o.constraints):
            return True
        o = o.parent
    return False


def sample_collider_motion(scene, objs, start, end):
    """Object matrices of moving colliders: a list aligned with objs, None for
    a collider that does not move, else {frame: 4x4 matrix} for every frame of
    the bake.  Only the transform counts (the mesh shape is the one of the
    start frame)."""
    out = [None] * len(objs)
    cand = [k for k, o in enumerate(objs) if o is not None and _may_move(o)]
    if not cand:
        return out
    muted = []
    for o in scene.objects:
        if o.type == 'MESH' and o.sand_sim.is_sand:
            m = nodes.node_group(o)
            if m is not None and m.show_viewport:
                m.show_viewport = False
                muted.append(m)
    table = {k: {} for k in cand}
    try:
        for f in range(start, end + 1):
            scene.frame_set(f)
            for k in cand:
                table[k][f] = np.array(objs[k].matrix_world, dtype=np.float64)
    finally:
        for m in muted:
            m.show_viewport = True
        scene.frame_set(start)
    for k in cand:
        t = table[k]
        m0 = t[start]
        if any(not np.allclose(t[f], m0, rtol=0.0, atol=1e-9) for f in t):
            out[k] = t
    return out


def check_gpu(cache_dir=None):
    """(ok, text): is a CUDA device usable through NVIDIA Warp?"""
    if not deps.warp_available():
        return False, _("NVIDIA Warp is not installed (button “Install NVIDIA Warp”)")
    try:
        from . import solver_gpu
    except Exception as exc:                      # warp not installed
        return False, _("NVIDIA Warp is not installed (%s)") % exc.__class__.__name__
    try:
        solver_gpu.init(cache_dir)
        if not solver_gpu.cuda_available():
            return False, _("no NVIDIA graphics card (CUDA) found")
        return True, solver_gpu.device_name("cuda:0")
    except Exception as exc:
        return False, _("CUDA error: %s") % exc


# ---------------------------------------------------------------- helpers
def running_job(ob):
    job = _jobs.get(ob.name) if ob is not None else None
    return job if job is not None and not job.status["finished"] else None


def stop_all_jobs():
    for job in list(_jobs.values()):
        job.stop_event.set()


def _positions(data):
    if data is None:
        return None
    attr = data.attributes.get("position")
    if attr is None or attr.domain != 'POINT' or len(attr.data) == 0:
        return None
    arr = np.empty(len(attr.data) * 3, dtype=np.float32)
    attr.data.foreach_get("vector", arr)
    return arr.reshape(-1, 3)


def read_source_points(src, depsgraph):
    """World-space points of the evaluated source object: mesh vertices,
    point cloud points, curve points and instance positions."""
    ev = src.evaluated_get(depsgraph)
    gs = ev.evaluated_geometry()
    parts = [_positions(gs.mesh), _positions(gs.pointcloud), _positions(gs.curves)]
    try:
        parts.append(_positions(gs.instances_pointcloud()))
    except Exception:
        pass
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return np.zeros((0, 3))
    pts = np.concatenate(parts).astype(np.float64)
    m = np.array(ev.matrix_world, dtype=np.float64)
    return pts @ m[:3, :3].T + m[:3, 3]


# collision diameter of a volume fill in lattice spacings (V / N)^(1/3):
# touching on a grid, a little smaller for random points (less to push apart)
VOLUME_SIZE = {'RANDOM': 0.9, 'GRID': 1.0}


def source_signature(st):
    return "%s|%d|%s|%d" % (st.source_mode, st.volume_count, st.volume_fill, st.seed) \
        if st.source_mode == 'VOLUME' else "POINTS"


def start_points(opts, src, depsgraph):
    """Start points of the sand from the source object (world space):
    its points, or points filling its volume.
    Returns (points, spacing for the grain size or None, warning text)."""
    if opts.source_mode != 'VOLUME':
        if src.type not in SOURCE_TYPES:
            raise ValueError(_("“%s” has no points: choose “Volume” to fill it with sand")
                             % src.name)
        return read_source_points(src, depsgraph), None, ""
    if src.type not in VOLUME_TYPES:
        raise ValueError(_("Only meshes (and curves, text with a surface) can be filled"))
    data = colliders.read_collider_mesh(src, depsgraph)
    if data is None:
        raise ValueError(_("“%s” has no surface, its volume cannot be filled") % src.name)
    pts, sp, info = volume.fill(data["co"], data["tri"], opts.volume_count, opts.volume_fill,
                                opts.seed if hasattr(opts, "seed") else 0)
    if len(pts) == 0:
        raise ValueError(_("“%s” is not closed: the surface has holes, its volume cannot be filled")
                         % src.name)
    warn = _("the surface is not closed, part of the volume was skipped") if info["odd"] else ""
    return pts, sp * VOLUME_SIZE.get(opts.volume_fill, 1.0), warn


def fit_volume_size(st, spacing):
    """Grain size that fills the volume; the mass keeps the material density."""
    old = st.grain_size
    st.grain_size = max(spacing / st.collision_scale, 0.0002)
    if old > 0:
        st.mass = max(st.mass * (st.grain_size / old) ** 3, 1e-7)


def estimate_spacing(pts):
    """Median distance to the nearest neighbour."""
    n = len(pts)
    if n < 2:
        return None
    ext = pts.max(axis=0) - pts.min(axis=0)
    ext = np.maximum(ext, max(float(ext.max()), 1e-6) * 1e-3)
    r = float((np.prod(ext) / n) ** (1.0 / 3.0))
    for _ in range(12):
        I, J = find_pairs(pts, r)
        if I.size >= 0.6 * n:
            break
        r *= 2.0
    if I.size == 0:
        return None
    dist = np.linalg.norm(pts[I] - pts[J], axis=1)
    nn = np.full(n, np.inf)
    np.minimum.at(nn, I, dist)
    np.minimum.at(nn, J, dist)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if nn.size else None


def random_quats(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def sand_object(context):
    ob = context.active_object
    return ob if ob is not None and ob.type == 'MESH' and ob.sand_sim.is_sand else None


def stage_text(status):
    """The stage of a bake in the interface language."""
    st = status["stage"]
    if isinstance(st, tuple):
        return _(st[0]) % st[1]
    return _(st)


def _fmt_time(sec):
    sec = int(round(sec))
    if sec < 60:
        return _("%d s") % sec
    if sec < 3600:
        return _("%d min %02d s") % (sec // 60, sec % 60)
    return _("%d h %02d min") % (sec // 3600, (sec % 3600) // 60)


def _redraw(context):
    wm = context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type in {'VIEW_3D', 'PROPERTIES', 'DOPESHEET_EDITOR', 'TIMELINE'}:
                area.tag_redraw()


# ---------------------------------------------------------------- bake job
class BakeJob:
    """Everything that needs bpy is gathered in __init__ (main thread);
    run() only uses NumPy / mathutils and may run in a worker thread."""

    def __init__(self, context, ob):
        scene = context.scene
        st = ob.sand_sim
        self.ob_name = ob.name
        self.start, self.end = st.frame_start, st.frame_end
        if self.end <= self.start:
            raise ValueError(_("The end frame must be after the start frame"))
        scene.frame_set(self.start)
        dg = context.evaluated_depsgraph_get()
        self.warning = ""
        if st.source is not None:
            local, spacing, self.warning = start_points(st, st.source, dg)
            if spacing and st.applied_source != source_signature(st):
                fit_volume_size(st, spacing)      # the volume settings changed
            st.applied_source = source_signature(st)
        else:
            local = cache.get_rest_positions(ob.data)
        if len(local) == 0:
            raise ValueError(_("The source has no points"))
        # the sand object's transform moves / turns the start cloud: the
        # simulation starts exactly where the grains are shown
        pts = cache.local_to_world(ob, local)
        self.pts = pts
        self.params = st.solver_params(scene)
        self.device = st.device
        self.kernel_cache = _warp_cache_dir()
        self.device_used = "CPU"
        fps = scene.render.fps / scene.render.fps_base
        self.dt = st.time_scale / fps
        self.voxel = st.collider_voxel * self.params.diameter

        self.meshes = collider_meshes(ob, dg)
        # colliders that move: their matrix in every frame
        self.col_motion = sample_collider_motion(
            scene, [bpy.data.objects.get(m["name"]) for m in self.meshes], self.start, self.end)
        dg = context.evaluated_depsgraph_get()

        # fresh cache (never delete frames a duplicate still shows)
        if not cache.folder_shared(ob):
            cache.clear_dir(cache.cache_dir(ob))
        st.cache_id = uuid.uuid4().hex[:8]
        st.cache_folder = bpy.path.clean_name(ob.name) + "_" + st.cache_id
        self.dir = cache.cache_dir(ob)
        os.makedirs(self.dir, exist_ok=True)
        st.is_baked = False
        st.grain_count = len(pts)
        cache.forget(ob.name)
        pos_l, q_l = cache.world_to_local(ob, pts, random_quats(len(pts), st.seed))
        cache.set_mesh_points(ob.data, pos_l, q_l)
        cache.set_static_attributes(ob.data, local, st.seed)
        grains.write_attributes(ob, dg, local)
        nodes.ensure_modifier(ob, st.grain_size)
        nodes.sync_cube_size(ob, st.grain_size)
        grains.sync_nodes(ob)
        # the grain as it is shown now (an upgraded node group may show the shape)
        self.params = st.solver_params(scene)
        self.voxel = st.collider_voxel * self.params.diameter
        cache.write_meta(self.dir, {
            "object": ob.name, "grains": int(len(pts)), "frame_start": self.start,
            "frame_end": self.end, "grain_size": st.grain_size,
            "diameter": self.params.diameter, "mass": self.params.mass, "dt": self.dt,
            "size_variance": st.size_variance,
            "format": "float32 (n, 7): x y z qw qx qy qz",
        })
        # per grain size and wetness (the values the viewport shows)
        self.scale = grains.grain_scales(len(pts), st.size_variance, st.seed)
        self.wet = None
        if st.use_wet_noise or st.use_wet_paint:
            self.wet = nodes.evaluated_wetness(ob, dg)
            if self.wet is None:
                self.warning = ((self.warning + " · ") if self.warning else "") + \
                    _("wetness not read: the sand object is hidden in the view layer, computed as dry")
        self.notes = []
        if self.params.model == 'PBD' and (self.scale is not None or self.wet is not None):
            self.notes.append(_("size variance and wetness work only with the “Impulses” physics"))
        else:
            if self.scale is not None:
                self.notes.append(_("variance ±%.0f%%") % st.size_variance)
            if self.wet is not None:
                self.notes.append(_("wet %.0f%%") % (100.0 * float(np.mean(self.wet > 0.5))))
                if st.use_cohesion and (st.cohesion > 0.0 or st.adhesion > 0.0):
                    self.notes.append(_("clumps"))
        # force fields, frame by frame when they are animated
        fobjs, fnotes = field_objects(st)
        self.fields = None
        if fobjs:
            if self.params.model == 'PBD':
                self.notes.append(_("force fields work only with the “Impulses” physics"))
            else:
                self.fields = sample_fields(scene, fobjs, self.start, self.end)
                scene.frame_set(self.start)
                names = sorted({fields.TYPE_NAMES[int(r[fields.F_TYPE])].title()
                                for r in next(iter(self.fields.values())) if r[fields.F_TYPE] > 0})
                self.notes.append(_("fields: %d (%s)") % (len(fobjs), ", ".join(names)))
        self.notes.extend(fnotes)
        nmov = sum(1 for m in self.col_motion if m is not None)
        if nmov:
            self.notes.append(_("moving colliders: %d") % nmov)

        self.stop_event = threading.Event()
        self.status = {
            "stage": "Preparing", "frame": self.start - 1, "done": 0,
            "total": self.end - self.start, "awake": len(pts), "substeps": 0,
            "clamped": 0, "sec_per_frame": 0.0, "elapsed": 0.0, "overlap": 0.0,
            "error": None, "finished": False,
        }

    def run(self):
        s = self.status
        t0 = time.perf_counter()
        try:
            cols = []
            for i, m in enumerate(self.meshes):
                s["stage"] = ("Collider “%s” (%d/%d)", (m["name"], i + 1, len(self.meshes)))
                col = colliders.build_sdf(m, self.params.diameter, self.voxel,
                                          stop=self.stop_event.is_set)
                cols.append(col)
                if col.cell > 1.5 * self.params.diameter:
                    s.setdefault("warnings", []).append(
                        _("collider “%s” is very large: cell %.0f mm (%.1f × grain)")
                        % (m["name"], col.cell * 1000, col.cell / self.params.diameter))
            solver = self._make_solver(cols)
            s["stage"] = "Pushing apart overlapping grains"
            s["overlap"] = solver.resolve_initial_overlaps()
            cache.write_frame(self.dir, self.start, solver.positions(), solver.rotations())
            s["frame"] = self.start
            s["stage"] = "Simulating"
            for f in range(self.start + 1, self.end + 1):
                if self.stop_event.is_set():
                    break
                t = time.perf_counter()
                if self.fields is not None:
                    st_ = self.fields.get("static")
                    solver.set_fields(st_ if st_ is not None else self.fields[f - 1],
                                      None if st_ is not None else self.fields[f])
                for k, mo in enumerate(self.col_motion):
                    if mo is not None:
                        solver.set_collider_motion(k, mo[f - 1], mo[f], mo[self.start])
                solver.step_frame(self.dt)
                cache.write_frame(self.dir, f, solver.positions(), solver.rotations())
                s["sec_per_frame"] = time.perf_counter() - t
                s["awake"] = solver.awake_count
                s["substeps"] = solver.last_substeps
                s["clamped"] = max(s["clamped"], solver.clamped)
                s["done"] = f - self.start
                s["elapsed"] = time.perf_counter() - t0
                s["frame"] = f
        except InterruptedError:
            pass
        except Exception:
            s["error"] = traceback.format_exc()
        finally:
            s["elapsed"] = time.perf_counter() - t0
            s["finished"] = True

    def _make_solver(self, cols):
        """GPU (NVIDIA Warp / CUDA) when requested and available; on the CPU the
        same Warp kernels (compiled, faster than NumPy); NumPy as the last resort."""
        s = self.status
        warp_err = ""
        solver_gpu = None
        try:
            if not deps.warp_available():
                raise ImportError("not installed")
            from . import solver_gpu
            solver_gpu.init(self.kernel_cache)
        except Exception as exc:
            solver_gpu = None
            warp_err = _("NVIDIA Warp is not installed (button “Install NVIDIA Warp”)") \
                if isinstance(exc, ImportError) and str(exc) == "not installed" \
                else _("NVIDIA Warp is not available (%s: %s)") % (exc.__class__.__name__, exc)
        if self.device in {'AUTO', 'GPU'}:
            s["stage"] = "Looking for a graphics card"
            if solver_gpu is not None and solver_gpu.cuda_available():
                solver = solver_gpu.SandSolverGPU(self.pts, self.params, cols, device="cuda:0",
                                                  scale=self.scale, wet=self.wet)
                s["stage"] = "Compiling GPU kernels (slow only the first time)"
                solver.compile()
                self.device_used = "GPU " + solver_gpu.device_name("cuda:0")
                s["device"] = self.device_used
                return solver
            text = warp_err or _("no NVIDIA graphics card (CUDA) found")
            if self.device == 'GPU':
                raise RuntimeError(_("GPU computation is not possible: ") + text)
            s["device_note"] = text
        if solver_gpu is not None:
            try:
                solver = solver_gpu.SandSolverGPU(self.pts, self.params, cols, device="cpu",
                                                  scale=self.scale, wet=self.wet)
                s["stage"] = "Compiling CPU kernels (slow only the first time)"
                solver.compile()
                self.device_used = "CPU (Warp)"
                s["device"] = self.device_used
                return solver
            except Exception:
                traceback.print_exc()
        self.device_used = "CPU (NumPy)"
        s["device"] = self.device_used
        return SandSolver(self.pts, self.params, cols, scale=self.scale, wet=self.wet)

    def error_text(self):
        err = self.status.get("error") or ""
        lines = [ln.strip() for ln in err.strip().splitlines() if ln.strip()]
        last = lines[-1] if lines else _("unknown error")
        return last.split(": ", 1)[1] if last.startswith(("RuntimeError: ", "ValueError: ")) else last

    def status_line(self):
        s = self.status
        if s["stage"] != "Simulating":
            return stage_text(s) + "…"
        done, total = s["done"], max(s["total"], 1)
        eta = s["elapsed"] / max(done, 1) * (total - done)
        return (_("%s · frame %d (%d/%d) · moving %d · substeps %d · %.2f s/frame · ~%s left")
                % (s.get("device", "CPU"), s["frame"], done, total, s["awake"], s["substeps"],
                   s["sec_per_frame"], _fmt_time(eta)))

    def apply_progress(self, ob):
        st = ob.sand_sim
        if self.status["frame"] >= self.start:
            st.is_baked = True
            st.baked_start = self.start
            st.baked_end = self.status["frame"]

    def finalize(self, context):
        ob = bpy.data.objects.get(self.ob_name)
        if ob is None:
            return
        st = ob.sand_sim
        s = self.status
        self.apply_progress(ob)
        n = len(self.pts)
        st.baked_size = st.grain_size
        st.baked_sig = st.grain_signature()
        if s["error"]:
            st.bake_info = _("Error: ") + self.error_text()
        else:
            st.bake_info = _("Frames %d–%d · %d grains · %s · %s") % (
                st.baked_start, st.baked_end, n, _fmt_time(s["elapsed"]), self.device_used)
            for note in self.notes:
                st.bake_info += " · " + note
            if st.baked_end < self.end:
                st.bake_info = _("Stopped · ") + st.bake_info
            for w in s.get("warnings", []):
                st.bake_info += " · " + w
            if self.warning:
                st.bake_info += " · " + self.warning
            if s["clamped"]:
                st.bake_info += _(" · speed was limited: increase “Max Substeps”")
        context.scene.render.use_lock_interface = True   # safe rendering with the handler
        cache.forget(ob.name)
        cache.show_frame(ob, context.scene.frame_current)


# ---------------------------------------------------------------- operators
class SAND_OT_create(bpy.types.Operator):
    bl_idname = "sand_sim.create"
    bl_label = "Create Sand"
    bl_description = ("Create a sand object: a cube grain appears at every point of the active "
                      "object")
    bl_options = {'REGISTER', 'UNDO'}

    auto_size: BoolProperty(
        name="Size From Point Spacing", default=True,
        description="Choose the grain size so that neighboring grains touch")
    hide_source: BoolProperty(name="Hide Source", default=True)

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and (ob.type in SOURCE_TYPES or ob.type in VOLUME_TYPES) \
            and not ob.sand_sim.is_sand

    def execute(self, context):
        src = context.active_object
        opts = context.scene.sand_sim_create
        try:
            pts, spacing, warn = start_points(opts, src, context.evaluated_depsgraph_get())
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        if len(pts) == 0:
            self.report({'ERROR'}, _("Object “%s” has no points") % src.name)
            return {'CANCELLED'}
        name = src.name + " Sand"
        me = bpy.data.meshes.new(name)
        ob = bpy.data.objects.new(name, me)
        coll = src.users_collection[0] if src.users_collection else context.scene.collection
        coll.objects.link(ob)
        st = ob.sand_sim
        st.is_sand = True
        st.source = src
        st.cache_id = uuid.uuid4().hex[:8]
        st.frame_start = context.scene.frame_start
        st.frame_end = context.scene.frame_end
        st.source_mode = opts.source_mode
        st.volume_count = opts.volume_count
        st.volume_fill = opts.volume_fill
        st.applied_source = source_signature(st)
        if self.auto_size:
            sp = spacing or estimate_spacing(pts)
            if sp:
                st.grain_size = max(sp / st.collision_scale, 0.0002)
        st.mass = 2650.0 * st.grain_size ** 3        # quartz by default
        st.grain_count = len(pts)
        cache.set_mesh_points(me, pts, random_quats(len(pts), st.seed))
        cache.set_static_attributes(me, pts, st.seed)
        nodes.ensure_modifier(ob, st.grain_size)
        grains.write_attributes(ob, context.evaluated_depsgraph_get(), pts)
        if self.hide_source:
            src.hide_set(True)
            src.hide_render = True
        for o in context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        self.report({'WARNING'} if warn else {'INFO'},
                    _("Created %d grains") % len(pts) + (" (%s)" % warn if warn else ""))
        return {'FINISHED'}


class SAND_OT_bake(bpy.types.Operator):
    bl_idname = "sand_sim.bake"
    bl_label = "Bake Simulation"
    bl_description = "Compute the simulation and save the frames to the cache (Esc to stop)"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        ob = sand_object(context)
        return ob is not None and running_job(ob) is None

    def _make_job(self, context):
        try:
            return BakeJob(context, sand_object(context))
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return None

    def execute(self, context):
        # synchronous bake (scripts, command line, background mode)
        job = self._make_job(context)
        if job is None:
            return {'CANCELLED'}
        job.run()
        job.finalize(context)
        if job.status["error"]:
            print(job.status["error"])
            self.report({'ERROR'}, job.error_text())
            return {'CANCELLED'}
        return {'FINISHED'}

    def invoke(self, context, event):
        if bpy.app.background or context.window is None:
            return self.execute(context)
        job = self._make_job(context)
        if job is None:
            return {'CANCELLED'}
        self.job = job
        self.shown = None
        _jobs[job.ob_name] = job
        self.thread = threading.Thread(target=job.run, name="sand_bake", daemon=True)
        self.thread.start()
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        job = self.job
        if event.type == 'ESC' and event.value == 'PRESS':
            job.stop_event.set()
            return {'RUNNING_MODAL'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        ob = bpy.data.objects.get(job.ob_name)
        if ob is None:
            job.stop_event.set()
        else:
            job.apply_progress(ob)
            f = job.status["frame"]
            if ob.sand_sim.live_preview and f >= job.start and f != self.shown:
                self.shown = f
                context.scene.frame_set(f)
        if context.workspace is not None:
            context.workspace.status_text_set(_("Sand: ") + job.status_line() + _("   (Esc to stop)"))
        _redraw(context)
        if not job.status["finished"]:
            return {'PASS_THROUGH'}
        self.thread.join()
        context.window_manager.event_timer_remove(self.timer)
        if context.workspace is not None:
            context.workspace.status_text_set(None)
        job.finalize(context)
        _jobs.pop(job.ob_name, None)
        _redraw(context)
        if job.status["error"]:
            print(job.status["error"])
            self.report({'ERROR'}, job.error_text())
            return {'CANCELLED'}
        self.report({'INFO'}, _("Sand baked: ") + (ob.sand_sim.bake_info if ob else ""))
        return {'FINISHED'}


class SAND_OT_bake_stop(bpy.types.Operator):
    bl_idname = "sand_sim.bake_stop"
    bl_label = "Stop"
    bl_description = "Stop baking (the computed frames are kept)"

    @classmethod
    def poll(cls, context):
        return running_job(sand_object(context)) is not None

    def execute(self, context):
        running_job(sand_object(context)).stop_event.set()
        return {'FINISHED'}


class SAND_OT_free(bpy.types.Operator):
    bl_idname = "sand_sim.free"
    bl_label = "Reset"
    bl_description = "Delete the cache and put the grains back at their start positions (read the source again)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = sand_object(context)
        return ob is not None and running_job(ob) is None

    def execute(self, context):
        ob = sand_object(context)
        st = ob.sand_sim
        if not cache.folder_shared(ob):
            cache.clear_dir(cache.cache_dir(ob))
        st.is_baked = False
        st.bake_info = ""
        st.cache_folder = ""
        cache.forget(ob.name)
        if st.source is not None:
            try:
                pts, spacing, warn = start_points(st, st.source, context.evaluated_depsgraph_get())
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            if spacing and st.applied_source != source_signature(st):
                fit_volume_size(st, spacing)      # the volume settings changed
                nodes.sync_cube_size(ob, st.grain_size)
            st.applied_source = source_signature(st)
            if warn:
                self.report({'WARNING'}, warn)
        else:
            pts = cache.get_rest_positions(ob.data)
        if len(pts):
            cache.set_mesh_points(ob.data, pts, random_quats(len(pts), st.seed))
            cache.set_static_attributes(ob.data, pts, st.seed)
            grains.write_attributes(ob, context.evaluated_depsgraph_get(), pts)
            st.grain_count = len(pts)
        if nodes.node_group(ob) is not None:
            nodes.upgrade_node_group(nodes.node_group(ob).node_group)
            grains.sync_nodes(ob)
        return {'FINISHED'}


class SAND_OT_auto_size(bpy.types.Operator):
    bl_idname = "sand_sim.auto_size"
    bl_label = "From Point Spacing"
    bl_description = "Choose the grain size so that neighboring grains touch"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = sand_object(context)
        return ob is not None and running_job(ob) is None

    def execute(self, context):
        ob = sand_object(context)
        st = ob.sand_sim
        sp = None
        if st.source is not None:
            try:
                pts, sp, _warn = start_points(st, st.source, context.evaluated_depsgraph_get())
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
        else:
            pts = cache.get_rest_positions(ob.data)
        sp = sp or estimate_spacing(pts)
        if not sp:
            self.report({'WARNING'}, _("Not enough points"))
            return {'CANCELLED'}
        st.grain_size = max(sp / st.collision_scale, 0.0002)
        return {'FINISHED'}


class SAND_OT_preview_colliders(bpy.types.Operator):
    bl_idname = "sand_sim.preview_colliders"
    bl_label = "Show Colliders"
    bl_description = ("Show the collider surfaces as a point cloud, the way the simulation sees "
                      "them (at the first frame). Helps to find holes, extra or shifted surfaces")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = sand_object(context)
        return ob is not None and ob.sand_sim.colliders is not None

    def execute(self, context):
        ob = sand_object(context)
        st = ob.sand_sim
        scene = context.scene
        frame = scene.frame_current
        scene.frame_set(st.frame_start)
        dg = context.evaluated_depsgraph_get()
        d = st.physical_diameter(use_cache=False)
        pts, notes = [], []
        for data in collider_meshes(ob, dg):
            col = colliders.build_sdf(data, d, st.collider_voxel * d)
            phi = col.data[:, 0].reshape(tuple(col.dims))
            # grid points next to the zero level (the collision surface), moved onto it
            near = np.abs(phi) < 0.5 * col.cell
            idx = np.argwhere(near)
            p = col.origin + idx * col.cell
            g = col.data[:, 1:4].reshape(tuple(col.dims) + (3,))[near]
            gl = np.linalg.norm(g, axis=1, keepdims=True)
            p = p - phi[near][:, None] * g / np.maximum(gl, 1e-9)
            pts.append(p)
            kind = _("closed") if data["closed"] else _("open surface (+%.0f mm layer)") % (
                1000 * max(3 * d, 3 * col.cell))
            notes.append(_("%s: %s, cell %.1f mm (%.1f × grain)%s") % (
                data["name"], kind, 1000 * col.cell, col.cell / d,
                _(", flipped normals were fixed") if data.get("flipped") and data["closed"] else ""))
        scene.frame_set(frame)
        if not pts:
            self.report({'WARNING'}, _("The collider collection has no meshes"))
            return {'CANCELLED'}
        p = np.concatenate(pts)
        if len(p) > 600000:
            p = p[np.random.default_rng(0).choice(len(p), 600000, replace=False)]
        name = "Sand Collider Preview"
        pc_ob = bpy.data.objects.get(name)
        pc = bpy.data.pointclouds.new(name)
        pc.resize(len(p))
        pc.attributes["position"].data.foreach_set("vector", p.astype(np.float32).ravel())
        rad = pc.attributes.get("radius") or pc.attributes.new("radius", 'FLOAT', 'POINT')
        rad.data.foreach_set("value", np.full(len(p), 0.15 * d, np.float32))
        if pc_ob is None:
            pc_ob = bpy.data.objects.new(name, pc)
            context.scene.collection.objects.link(pc_ob)
        else:
            old = pc_ob.data
            pc_ob.data = pc
            if old.users == 0:
                bpy.data.pointclouds.remove(old)
        pc_ob.hide_render = True
        pc_ob.hide_select = True
        for line in notes:
            print("Sand Simulation collider:", line)
        self.report({'INFO'}, " | ".join(notes))
        return {'FINISHED'}


class SAND_OT_check_gpu(bpy.types.Operator):
    bl_idname = "sand_sim.check_gpu"
    bl_label = "Check GPU"
    bl_description = "Check whether an NVIDIA graphics card can be used (NVIDIA Warp / CUDA)"

    def execute(self, context):
        ok, text = check_gpu(_warp_cache_dir())
        _gpu_info.update(checked=True, ok=ok, text=text)
        self.report({'INFO'} if ok else {'WARNING'}, ("GPU: " if ok else _("GPU not available: ")) + text)
        _redraw(context)
        return {'FINISHED'}


class SAND_OT_install_warp(bpy.types.Operator):
    bl_idname = "sand_sim.install_warp"
    bl_label = "Install NVIDIA Warp"
    bl_description = ("Download the NVIDIA Warp library (~150 MB, from PyPI) into the add-on "
                      "folder. Needed to compute on an NVIDIA graphics card; installed once")

    @classmethod
    def poll(cls, context):
        return not deps.state["running"]

    def _finish(self, context):
        _gpu_info.update(checked=False, ok=False, text="")
        if deps.state["ok"]:
            ok, text = check_gpu(_warp_cache_dir())
            _gpu_info.update(checked=True, ok=ok, text=text)
            self.report({'INFO'}, _("NVIDIA Warp installed. ") + ("GPU: " if ok else _("GPU not available: ")) + text)
            return {'FINISHED'}
        print("Sand Simulation: NVIDIA Warp install failed:", deps.state["error"])
        self.report({'ERROR'}, _("Could not install NVIDIA Warp: ") + deps.state["error"][-200:])
        return {'CANCELLED'}

    def execute(self, context):
        if not deps.online_allowed():
            self.report({'ERROR'}, _("Online access is disabled: Preferences → System → Network → Allow Online Access"))
            return {'CANCELLED'}
        deps.install_warp(deps.site_dir(create=True))
        return self._finish(context)

    def invoke(self, context, event):
        if bpy.app.background or context.window is None:
            return self.execute(context)
        if not deps.online_allowed():
            self.report({'ERROR'}, _("Online access is disabled: Preferences → System → Network → Allow Online Access"))
            return {'CANCELLED'}
        self.thread = deps.start_install()
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        _redraw(context)
        if deps.state["running"]:
            return {'PASS_THROUGH'}
        context.window_manager.event_timer_remove(self.timer)
        _redraw(context)
        return self._finish(context)


class SAND_OT_refresh_paint(bpy.types.Operator):
    bl_idname = "sand_sim.refresh_paint"
    bl_label = "Refresh Paint"
    bl_description = ("Read the paint of the source (vertex group or attribute) again, for "
                      "example after painting in Weight Paint")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = sand_object(context)
        return ob is not None and ob.sand_sim.use_wet_paint and running_job(ob) is None

    def execute(self, context):
        ob = sand_object(context)
        grains.sync_nodes(ob)
        grains.write_paint(ob, context.evaluated_depsgraph_get())
        info = ob.sand_sim.wet_info
        self.report({'INFO'} if ob.sand_sim.wet_ok else {'WARNING'}, info or _("Done"))
        return {'FINISHED'}


class SAND_OT_wet_all(bpy.types.Operator):
    bl_idname = "sand_sim.wet_all"
    bl_label = "All Wet"
    bl_description = ("Make the noise wet everywhere (both colors of the ramp white). With the "
                      "Paint Map on: wet wherever painted")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = sand_object(context)
        return ob is not None and running_job(ob) is None

    def execute(self, context):
        ob = sand_object(context)
        st = ob.sand_sim
        if not st.use_wet_noise:
            st.use_wet_noise = True
        ramp = nodes.get_node(ob, nodes.WET_RAMP)
        if ramp is None:
            self.report({'WARNING'}, _("Old sand nodes: press “Reset”"))
            return {'CANCELLED'}
        for el in ramp.color_ramp.elements:
            el.color = (1.0, 1.0, 1.0, 1.0)
        ob.update_tag()
        return {'FINISHED'}


class SAND_OT_project(bpy.types.Operator):
    bl_idname = "sand_sim.project"
    bl_label = "Project"
    bl_description = ("Color the grains with the image seen from the camera (camera at the first "
                      "simulation frame, grains at their start positions)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = sand_object(context)
        return ob is not None and ob.sand_sim.use_projection and running_job(ob) is None

    def execute(self, context):
        ob = sand_object(context)
        st = ob.sand_sim
        scene = context.scene
        frame = scene.frame_current
        if frame != st.frame_start:
            scene.frame_set(st.frame_start)
        try:
            grains.sync_nodes(ob)
            grains.write_projection(ob, context.evaluated_depsgraph_get())
        finally:
            if scene.frame_current != frame:
                scene.frame_set(frame)
        info = st.project_info
        self.report({'INFO'} if st.project_ok else {'WARNING'}, info or _("Done"))
        return {'FINISHED'}


class SAND_OT_select(bpy.types.Operator):
    bl_idname = "sand_sim.select"
    bl_label = "Select"
    bl_options = {'REGISTER', 'UNDO'}
    name: StringProperty()

    def execute(self, context):
        ob = context.scene.objects.get(self.name)
        if ob is None:
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        if ob.hide_get():
            ob.hide_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        return {'FINISHED'}


classes = (SAND_OT_create, SAND_OT_bake, SAND_OT_bake_stop, SAND_OT_free, SAND_OT_check_gpu,
           SAND_OT_install_warp, SAND_OT_preview_colliders,
           SAND_OT_auto_size, SAND_OT_refresh_paint, SAND_OT_wet_all, SAND_OT_project, SAND_OT_select)
