import bpy, bmesh, gpu, heapq, time, traceback
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, kdtree

bl_info = {
    "name": "SoftViz",
    "author": "Niels Couvreur, WXP",
    "version": (1, 5, 0),
    "blender": (4, 2, 0),
    "location": "View3D > N-Panel > SoftViz",
    "description": "Advanced, GPU-accelerated heatmap visualizer for Proportional Editing.",
    "doc_url": "", 
    "tracker_url": "", 
    "category": "3D View",
}

NG_NAME = "SoftViz_ColorRamp_NG"
DRAW_HANDLE = None
SHADER = None
# Set True if gpu.types.GPUShader failed; cleared when heatmap is toggled off.
SOFTVIZ_SHADER_FAILED = False

_QUAD_CORNERS = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

# Scroll-spy state — set/cleared by VIEW3D_OT_softviz_transform_spy
_SV_MODAL_RADIUS = None
# obj_name -> list[Vector|None] indexed by mesh vertex index; world coords at transform start
_SV_TRANSFORM_SNAPSHOT = None
_SV_KEYMAPS = []
# Transform spy: print start/end/scroll radius to the system console.
debug = False

# Legacy full GLSL (Blender < 5 still allows gpu.types.GPUShader(vertex, frag)).
_SOFTVIZ_VERT_LEGACY = """
uniform mat4 u_mvp;
uniform mat4 u_view_mat;
uniform vec3 u_right;
uniform vec3 u_up;
uniform float u_dot_size;
uniform float u_ortho_half;
uniform int u_screen_mode;

in vec3 pos;
in vec2 corner;
in vec4 color;

out vec4 frag_color;

void main() {
    float factor;
    if (u_screen_mode == 0) {
        factor = u_dot_size * 0.005;
    } else if (u_screen_mode == 2) {
        factor = u_ortho_half;
    } else {
        float depth = max(0.01, -(u_view_mat * vec4(pos, 1.0)).z);
        factor = depth * u_dot_size * 0.0005;
    }
    vec3 world_pos = pos
        + u_right * (corner.x * factor)
        + u_up    * (corner.y * factor);
    gl_Position = u_mvp * vec4(world_pos, 1.0);
    frag_color = color;
}
"""

_SOFTVIZ_FRAG_LEGACY = """
in vec4 frag_color;
out vec4 FragColor;

void main() {
    FragColor = frag_color;
}
"""

# Blender 5+ (Vulkan-compatible): uniforms must be declared via push_constant
# in GPUShaderCreateInfo — NOT as bare 'uniform' in GLSL source.
# Two mat4s = 128 bytes exactly, so we replace the full view matrix with just
# its Z row (vec4, 16 bytes) since that's all we need for depth.
# Pack dot_size, ortho_half, and screen_mode into u_params.xyz so total stays
# at 128 bytes (64 + 16*4); some backends reject >128-byte push-constant blocks.
_SOFTVIZ_VERT_NEW = """
void main() {
    int mode = int(u_params.z + 0.5);
    float factor;
    if (mode == 0) {
        factor = u_params.x * 0.005;
    } else if (mode == 2) {
        factor = u_params.y;
    } else {
        float depth = max(0.01, -dot(u_view_z_row, vec4(pos, 1.0)));
        factor = depth * u_params.x * 0.0005;
    }
    vec3 world_pos = pos
        + u_right.xyz * (corner.x * factor)
        + u_up.xyz    * (corner.y * factor);
    gl_Position = u_mvp * vec4(world_pos, 1.0);
    frag_color = color;
}
"""

_SOFTVIZ_FRAG_NEW = """
void main() {
    FragColor = frag_color;
}
"""


def create_softviz_shader():
    create_fn = getattr(gpu.shader, "create_from_info", None)
    if create_fn is None:
        return gpu.types.GPUShader(_SOFTVIZ_VERT_LEGACY, _SOFTVIZ_FRAG_LEGACY)

    iface = gpu.types.GPUStageInterfaceInfo("softviz_v2f")
    iface.smooth("VEC4", "frag_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, "VEC3", "pos")
    info.vertex_in(1, "VEC2", "corner")
    info.vertex_in(2, "VEC4", "color")
    info.push_constant("MAT4", "u_mvp")
    info.push_constant("VEC4", "u_view_z_row")
    info.push_constant("VEC4", "u_right")
    info.push_constant("VEC4", "u_up")
    info.push_constant("VEC4", "u_params")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "FragColor")
    info.vertex_source(_SOFTVIZ_VERT_NEW)
    info.fragment_source(_SOFTVIZ_FRAG_NEW)

    return create_fn(info)

# -------------------------------------------------
# CACHE SYSTEM
# -------------------------------------------------
class SoftVizCache:
    def __init__(self):
        self.hash = None
        self.coord_hash = None
        self.last_change_time = 0
        self.is_dirty = False
        self.weights = {}
        self.vert_weights = None
        self.batch = None
        self.batch_hash = None
        self.ramp_lut = None
        self.ramp_lut_key = None
        self.draw_error_logged = False

VIZ_CACHE = SoftVizCache()

# -------------------------------------------------
# DRAW HANDLER SAFETY
# -------------------------------------------------
def _bpy_scenes():
    """bpy.data can be _RestrictData during add-on register — no .scenes until file is available."""
    return getattr(bpy.data, "scenes", None)

def remove_draw_handler():
    global DRAW_HANDLE, SHADER, SOFTVIZ_SHADER_FAILED
    if DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(DRAW_HANDLE, 'WINDOW')
        except Exception:
            pass
        DRAW_HANDLE = None
    VIZ_CACHE.batch = None
    VIZ_CACHE.batch_hash = None
    VIZ_CACHE.vert_weights = None
    VIZ_CACHE.draw_error_logged = False
    SHADER = None
    SOFTVIZ_SHADER_FAILED = False

def sync_softviz_draw_handler():
    """Match POST_VIEW draw handler to scene softviz_running (handlers are not saved in .blend)."""
    global DRAW_HANDLE
    scenes = _bpy_scenes()
    any_on = any(s.softviz_running for s in scenes) if scenes is not None else False
    remove_draw_handler()
    if not any_on:
        return
    ensure_nodegroup()
    DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        draw_callback, (), 'WINDOW', 'POST_VIEW')

@bpy.app.handlers.persistent
def softviz_load_post(dummy):
    scenes = _bpy_scenes()
    if scenes is not None:
        for scene in scenes:
            # Force native Connected Only to default to OFF on load
            scene.tool_settings.use_proportional_connected = False

    sync_softviz_draw_handler()
    if not bpy.app.timers.is_registered(init_nodegroup_timer):
        bpy.app.timers.register(init_nodegroup_timer, first_interval=0.1)

# -------------------------------------------------
# DEBOUNCE TIMER
# -------------------------------------------------
def softviz_cache_timer():
    scene = bpy.context.scene
    if scene.softviz_running:
        if VIZ_CACHE.is_dirty and (time.time() - VIZ_CACHE.last_change_time) > 0.2:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
    return 0.1

# -------------------------------------------------
# RAMP
# -------------------------------------------------
def init_nodegroup_timer():
    ensure_nodegroup()
    # register() can run with restricted bpy.data; resync draw handler once open.
    sync_softviz_draw_handler()
    return None 

def build_default_ramp(r):
    while len(r.elements) > 1:
        r.elements.remove(r.elements[0])

    r.interpolation = 'LINEAR'

    r.elements[0].position = 0.0
    r.elements[0].color = (0, 0, 1, 1)

    e = r.elements.new(0.25)
    e.color = (0, 1, 1, 1)

    e = r.elements.new(0.5)
    e.color = (1, 1, 0, 1)

    e = r.elements.new(0.75)
    e.color = (1, 0.5, 0, 1)

    e = r.elements.new(1.0)
    e.color = (1, 0, 0, 1)

def ensure_nodegroup():
    ng = bpy.data.node_groups.get(NG_NAME)
    if ng:
        ng.use_fake_user = True
        return ng

    ng = bpy.data.node_groups.new(NG_NAME, 'ShaderNodeTree')
    ng.use_fake_user = True
    node = ng.nodes.new("ShaderNodeValToRGB")
    build_default_ramp(node.color_ramp)

    ng.interface_update(bpy.context)
    ng.update_tag()
    return ng

def get_ramp_node():
    ng = bpy.data.node_groups.get(NG_NAME)
    if not ng:
        return None
    for n in ng.nodes:
        if n.type == 'VALTORGB':
            return n
    return None

def remove_softviz_ramp_nodegroup():
    """Remove addon-owned node group (e.g. on unregister / disable add-on)."""
    ng = bpy.data.node_groups.get(NG_NAME)
    if ng is not None:
        ng.use_fake_user = False
        bpy.data.node_groups.remove(ng)
    VIZ_CACHE.ramp_lut = None
    VIZ_CACHE.ramp_lut_key = None

def get_or_bake_lut(ramp_node):
    if ramp_node is None:
        return None
    ramp = ramp_node.color_ramp
    key = tuple((e.position, tuple(e.color)) for e in ramp.elements)
    if key != VIZ_CACHE.ramp_lut_key:
        VIZ_CACHE.ramp_lut = [tuple(ramp.evaluate(i / 255.0)) for i in range(256)]
        VIZ_CACHE.ramp_lut_key = key
    return VIZ_CACHE.ramp_lut

# -------------------------------------------------
# RESET RAMP
# -------------------------------------------------
class VIEW3D_OT_softviz_reset_ramp(bpy.types.Operator):
    bl_idname = "view3d.softviz_reset_ramp"
    bl_label = "Reset Ramp"
    def execute(self, context):
        ramp_node = get_ramp_node()
        if ramp_node:
            build_default_ramp(ramp_node.color_ramp)
            ramp_node.id_data.update_tag()
        return {'FINISHED'}

# -------------------------------------------------
# FALLOFF
# -------------------------------------------------
def falloff(d, mode):
    if mode == 'CONSTANT': return 1
    if mode == 'LINEAR': return 1-d
    if mode == 'SMOOTH': return 1-(3*d**2-2*d**3)
    if mode == 'SPHERE': return (1-d*d)**0.5
    if mode == 'ROOT': return 1-d**0.5
    if mode == 'SHARP': return (1-d)**2
    return 1-d

# -------------------------------------------------
# MODIFIER CAGE / EVALUATED MESH (align with edit-mode display)
# -------------------------------------------------
# Deform modifiers that keep a 1:1 mapping to mesh verts — safe to read from
# evaluated mesh positions. Subdivision / Multires / etc. are NOT listed here
# so enabling "Display in Edit Mode" on them alone does not turn on cage eval
# (Blender Python cannot match cage verts to original indices).
_DEFORM_MODIFIER_EDIT_TYPES = frozenset({
    'ARMATURE', 'CAST', 'CURVE', 'DISPLACE', 'HOOK', 'LAPLACIANDEFORM',
    'LATTICE', 'MESH_DEFORM', 'SHRINKWRAP', 'SIMPLE_DEFORM', 'SMOOTH',
    'CORRECTIVE_SMOOTH', 'SURFACE_DEFORM', 'WARP', 'WAVE',
})

# Modifiers that usually change vert count when stacked — temporarily turn off
# "show in edit mode" only while sampling evaluated coords so Armature etc.
# still resolve (API evaluates full stack; without this, count mismatch → no cage).
_TOPOLOGY_EDITDISPLAY_MODS = frozenset({
    'SUBSURF', 'MULTIRES', 'MIRROR', 'ARRAY', 'BOOLEAN', 'BUILD',
    'DECIMATE', 'REMESH', 'NODES', 'WELD', 'WIREFRAME', 'SKIN',
    'BEVEL', 'SCREW', 'SOLIDIFY',
})

def modifier_edit_display_signature(obj):
    return tuple(
        (m.name, m.type, m.show_viewport, m.show_in_editmode)
        for m in obj.modifiers
    )

def object_needs_evaluated_deform_cage(obj, shape_key_visualization=False):
    if obj.type == 'MESH' and obj.data.shape_keys:
        if obj.use_shape_key_edit_mode:
            return True
        # Edit-mode bmesh is the basis cage; viewport shows evaluated shape keys.
        if shape_key_visualization:
            return True
    for mod in obj.modifiers:
        if mod.show_viewport and mod.show_in_editmode:
            if mod.type in _DEFORM_MODIFIER_EDIT_TYPES:
                return True
    return False

def topology_edit_display_warning_lines(context):
    lines = []
    for obj in context.objects_in_mode:
        if obj.type != 'MESH':
            continue
        for mod in obj.modifiers:
            if (mod.show_viewport and mod.show_in_editmode
                    and mod.type in _TOPOLOGY_EDITDISPLAY_MODS):
                lines.append(f"{obj.name}: {mod.name} ({mod.type})")
    return lines

def proportional_mirror_world_positions(obj, mat, wp, epsilon=1e-4):
    """World positions mirrored in object-local space per mesh symmetry flags."""
    if not (obj.use_mesh_mirror_x or obj.use_mesh_mirror_y or obj.use_mesh_mirror_z):
        return (wp.copy(),)
    inv = mat.inverted()
    local = inv @ wp
    opts_x = (-1.0, 1.0) if obj.use_mesh_mirror_x else (1.0,)
    opts_y = (-1.0, 1.0) if obj.use_mesh_mirror_y else (1.0,)
    opts_z = (-1.0, 1.0) if obj.use_mesh_mirror_z else (1.0,)
    out = []
    for sx in opts_x:
        for sy in opts_y:
            for sz in opts_z:
                wps = mat @ Vector((local.x * sx, local.y * sy, local.z * sz))
                dup = False
                for p in out:
                    if (wps - p).length <= epsilon:
                        dup = True
                        break
                if not dup:
                    out.append(wps.copy())
    return tuple(out)

def eval_vert_world_coords(obj, depsgraph, expected_vert_count):
    eval_obj = obj.evaluated_get(depsgraph)
    try:
        me = eval_obj.to_mesh()
    except Exception:
        return None
    try:
        if len(me.vertices) != expected_vert_count:
            return None
        mw = eval_obj.matrix_world
        return [mw @ me.vertices[i].co.copy() for i in range(len(me.vertices))]
    finally:
        eval_obj.to_mesh_clear()

def eval_vert_world_coords_for_draw_cage(obj, ctx, expected_vert_count):
    restored = []
    try:
        for mod in obj.modifiers:
            if mod.show_viewport and mod.show_in_editmode and mod.type in _TOPOLOGY_EDITDISPLAY_MODS:
                restored.append((mod, mod.show_in_editmode))
                mod.show_in_editmode = False
        if restored:
            ctx.view_layer.update()
        depsgraph = ctx.evaluated_depsgraph_get()
        return eval_vert_world_coords(obj, depsgraph, expected_vert_count)
    finally:
        for mod, prev in restored:
            mod.show_in_editmode = prev

def vert_world_pos(mat, v, cage_coords):
    if cage_coords is not None and v.index < len(cage_coords):
        return cage_coords[v.index]
    return mat @ v.co

def _capture_softviz_transform_snapshot(context):
    """World position per vertex index at G/R/S start — matches Blender proportional falloff basis."""
    edit_objs = [o for o in context.objects_in_mode if o.type == 'MESH']
    if not edit_objs:
        return {}
    s = context.scene.softviz_settings
    sk_viz = s.viz_mode == 'SHAPE_KEY'
    cage_coords_by_obj = {}
    for obj in edit_objs:
        if object_needs_evaluated_deform_cage(obj, shape_key_visualization=sk_viz):
            c = eval_vert_world_coords_for_draw_cage(
                obj, context, len(obj.data.vertices))
            if c is not None:
                cage_coords_by_obj[obj] = c
    snap = {}
    for obj in edit_objs:
        bm = bmesh.from_edit_mesh(obj.data)
        mat = obj.matrix_world
        cage = cage_coords_by_obj.get(obj)
        n = len(obj.data.vertices)
        coords = [None] * n
        bm.verts.ensure_lookup_table()
        for v in bm.verts:
            if v.index < n:
                coords[v.index] = vert_world_pos(mat, v, cage).copy()
        snap[obj.name] = coords
    return snap

def _snap_vert_world(snapshot, obj, vert_index, mat, v, cage):
    """World pos for weight math during modal: snapshot if valid, else live cage pos."""
    if snapshot:
        row = snapshot.get(obj.name)
        if row and vert_index < len(row):
            p = row[vert_index]
            if p is not None:
                return p
    return vert_world_pos(mat, v, cage)

# -------------------------------------------------
# DRAW
# -------------------------------------------------
def draw_callback():
    ctx = bpy.context
    if not ctx.scene.softviz_running:
        return
    ts = ctx.tool_settings
    s = ctx.scene.softviz_settings

    if s.viz_mode == 'PROPORTIONAL' and not ts.use_proportional_edit:
        return

    if s.viz_mode in ('SHAPE_KEY', 'VERTEX_GROUP'):
        if ctx.mode == 'EDIT_MESH':
            edit_objs = [o for o in ctx.objects_in_mode if o.type == 'MESH']
        else:
            edit_objs = [o for o in ctx.selected_objects if o.type == 'MESH']
            if not edit_objs and ctx.active_object and ctx.active_object.type == 'MESH':
                edit_objs = [ctx.active_object]
    else:
        edit_objs = [o for o in ctx.objects_in_mode if o.type == 'MESH']

    if not edit_objs:
        return

    ramp_node = get_ramp_node()
    ramp = ramp_node.color_ramp if ramp_node else None

    bms = {}
    if s.viz_mode == 'PROPORTIONAL':
        for obj in edit_objs:
            bms[obj] = bmesh.from_edit_mesh(obj.data)
    elif ctx.mode == 'EDIT_MESH' and s.viz_mode in ('SHAPE_KEY', 'VERTEX_GROUP'):
        for obj in edit_objs:
            bms[obj] = bmesh.from_edit_mesh(obj.data)

    cage_coords_by_obj = {}
    sk_viz = s.viz_mode == 'SHAPE_KEY'
    for obj in edit_objs:
        if object_needs_evaluated_deform_cage(obj, shape_key_visualization=sk_viz):
            c = eval_vert_world_coords_for_draw_cage(
                obj, ctx, len(obj.data.vertices))
            if c is not None:
                cage_coords_by_obj[obj] = c

    # ------- VERTEX GROUP mode -------
    if s.viz_mode == 'VERTEX_GROUP':
        vert_weights = []
        for obj in edit_objs:
            mat = obj.matrix_world
            cage = cage_coords_by_obj.get(obj)
            mesh = obj.data

            vg = obj.vertex_groups.get(s.vgroup_name) if s.vgroup_name else None
            if not vg:
                vg = obj.vertex_groups.active
            if not vg:
                continue

            bm = bms.get(obj)
            if bm is not None:
                dvert_layer = bm.verts.layers.deform.verify()
                vg_idx = vg.index
                for v in bm.verts:
                    w = v[dvert_layer].get(vg_idx, 0.0)
                    if w > 0.0:
                        wp = vert_world_pos(mat, v, cage)
                        vert_weights.append((wp, w))
            else:
                vg_idx = vg.index
                for vert in mesh.vertices:
                    w = 0.0
                    for ge in vert.groups:
                        if ge.group == vg_idx:
                            w = ge.weight
                            break
                    if w > 0.0:
                        wp = vert_world_pos(mat, vert, cage)
                        vert_weights.append((wp, w))

        if not vert_weights: return

    # ------- SHAPE KEY mode -------
    elif s.viz_mode == 'SHAPE_KEY':
        vert_weights = []
        for obj in edit_objs:
            mesh = obj.data
            if not mesh.shape_keys or not mesh.shape_keys.key_blocks:
                continue

            sk = mesh.shape_keys.key_blocks.get(s.shape_key_name) if s.shape_key_name else None
            if not sk:
                idx = obj.active_shape_key_index
                if 0 <= idx < len(mesh.shape_keys.key_blocks):
                    sk = mesh.shape_keys.key_blocks[idx]
            if not sk:
                continue

            basis = mesh.shape_keys.reference_key
            bm = bms.get(obj)
            # In mesh edit + shape key edit mode, bmesh only tracks the *active* key.
            idx_act = obj.active_shape_key_index
            sk_is_active = (
                0 <= idx_act < len(mesh.shape_keys.key_blocks)
                and mesh.shape_keys.key_blocks[idx_act] == sk
            )
            in_sk_edit = (
                bm is not None
                and obj.use_shape_key_edit_mode
                and sk is not basis
                and sk_is_active
            )
            if in_sk_edit:
                bm.verts.ensure_lookup_table()
                displacements = [
                    (bm.verts[i].co - basis.data[i].co).length
                    for i in range(len(mesh.vertices))
                ]
            else:
                displacements = [
                    (sk.data[i].co - basis.data[i].co).length
                    for i in range(len(mesh.vertices))
                ]
            max_d = max(displacements) if displacements else 0.0
            if max_d < 1e-6:
                continue

            mat = obj.matrix_world
            cage = cage_coords_by_obj.get(obj)
            if bm is not None:
                bm.verts.ensure_lookup_table()
                for v in bm.verts:
                    d = displacements[v.index]
                    if d > 0.0:
                        wp = vert_world_pos(mat, v, cage)
                        vert_weights.append((wp, d / max_d))
            else:
                for vert in mesh.vertices:
                    d = displacements[vert.index]
                    if d > 0.0:
                        wp = vert_world_pos(mat, vert, cage)
                        vert_weights.append((wp, d / max_d))

        if not vert_weights: return

    # ------- PROPORTIONAL mode -------
    else:
        # During a live transform, use the spy-tracked estimate if available;
        # ts.proportional_size is stale inside the modal until the operator exits.
        rad = _SV_MODAL_RADIUS if (_SV_MODAL_RADIUS is not None) else ts.proportional_size

        cache_key_elements = [
            rad,
            ts.proportional_edit_falloff,
            ts.use_proportional_connected,
        ]

        for obj in edit_objs:
            bm = bms[obj]
            mat = obj.matrix_world
            sel_indices = tuple(v.index for v in bm.verts if v.select)
            cache_key_elements.extend([
                obj.name,
                len(bm.verts),
                len(bm.edges),
                sel_indices,
                tuple(mat.col[3]),
                modifier_edit_display_signature(obj),
                obj.use_shape_key_edit_mode,
                obj.use_mesh_mirror_x,
                obj.use_mesh_mirror_y,
                obj.use_mesh_mirror_z,
            ])

        current_hash = hash(tuple(cache_key_elements))

        coord_elements = []
        for obj in edit_objs:
            bm = bms[obj]
            mat = obj.matrix_world
            cage = cage_coords_by_obj.get(obj)
            for v in bm.verts:
                if v.select:
                    wp = vert_world_pos(mat, v, cage)
                    coord_elements.extend([round(wp.x, 3), round(wp.y, 3), round(wp.z, 3)])
        current_coord_hash = hash(tuple(coord_elements))

        if current_coord_hash != VIZ_CACHE.coord_hash:
            VIZ_CACHE.coord_hash = current_coord_hash
            VIZ_CACHE.last_change_time = time.time()
            VIZ_CACHE.is_dirty = True

        rebuild = False

        if _SV_MODAL_RADIUS is not None:
            # Spy-keyed G/R/S session: recalc every frame while dragging/scrolling.
            rebuild = True
            VIZ_CACHE.is_dirty = False
        elif current_hash != VIZ_CACHE.hash:
            rebuild = True
            VIZ_CACHE.hash = current_hash
            VIZ_CACHE.is_dirty = False

        elif VIZ_CACHE.is_dirty and (time.time() - VIZ_CACHE.last_change_time) > 0.2:
            rebuild = True
            VIZ_CACHE.is_dirty = False

        if rebuild:
            VIZ_CACHE.weights.clear()
            snap = _SV_TRANSFORM_SNAPSHOT

            global_centers = []
            for obj in edit_objs:
                bm = bms[obj]
                mat = obj.matrix_world
                cage = cage_coords_by_obj.get(obj)
                sel = [v for v in bm.verts if v.select]
                for v in sel:
                    global_centers.append(
                        _snap_vert_world(snap, obj, v.index, mat, v, cage))
                VIZ_CACHE.weights[obj.name] = []

            if global_centers:
                if ts.use_proportional_connected:
                    for obj in edit_objs:
                        bm = bms[obj]
                        mat = obj.matrix_world
                        cage = cage_coords_by_obj.get(obj)
                        sel = [v for v in bm.verts if v.select]
                        if not sel: continue

                        distances = {v: 0.0 for v in sel}
                        pq = [(0.0, v.index, v) for v in sel]
                        heapq.heapify(pq)

                        obj_weights = []
                        while pq:
                            dist, _, v = heapq.heappop(pq)

                            if dist > distances.get(v, float('inf')): continue

                            w = falloff(dist / rad, ts.proportional_edit_falloff)
                            if w > 0.0:
                                obj_weights.append((v.index, w))

                            for edge in v.link_edges:
                                neighbor = edge.other_vert(v)
                                p_v = _snap_vert_world(snap, obj, v.index, mat, v, cage)
                                p_n = _snap_vert_world(
                                    snap, obj, neighbor.index, mat, neighbor, cage)
                                edge_len = (p_v - p_n).length
                                new_dist = dist + edge_len

                                if new_dist <= rad:
                                    if new_dist < distances.get(neighbor, float('inf')):
                                        distances[neighbor] = new_dist
                                        heapq.heappush(pq, (new_dist, neighbor.index, neighbor))

                        VIZ_CACHE.weights[obj.name] = obj_weights

                else:
                    kd = kdtree.KDTree(len(global_centers))
                    for i, c in enumerate(global_centers):
                        kd.insert(c, i)
                    kd.balance()

                    for obj in edit_objs:
                        bm = bms[obj]
                        mat = obj.matrix_world
                        cage = cage_coords_by_obj.get(obj)
                        obj_weights = []
                        for v in bm.verts:
                            wp = _snap_vert_world(snap, obj, v.index, mat, v, cage)
                            _, _, dist = kd.find(wp)
                            if dist <= rad:
                                w = falloff(dist / rad, ts.proportional_edit_falloff)
                                if w > 0.0:
                                    obj_weights.append((v.index, w))

                        VIZ_CACHE.weights[obj.name] = obj_weights

        if rebuild:
            cached_vw = []
            for obj in edit_objs:
                if obj.name not in VIZ_CACHE.weights: continue
                bm = bms[obj]
                mat = obj.matrix_world
                cage = cage_coords_by_obj.get(obj)
                bm.verts.ensure_lookup_table()
                for v_idx, w in VIZ_CACHE.weights[obj.name]:
                    try:
                        v = bm.verts[v_idx]
                        wp = vert_world_pos(mat, v, cage)
                        for wp_sym in proportional_mirror_world_positions(obj, mat, wp):
                            cached_vw.append((wp_sym, w))
                    except IndexError:
                        pass
            VIZ_CACHE.vert_weights = cached_vw

        vert_weights = VIZ_CACHE.vert_weights if VIZ_CACHE.vert_weights is not None else []
        if not vert_weights: return

    global SHADER, SOFTVIZ_SHADER_FAILED
    if SHADER is None and not SOFTVIZ_SHADER_FAILED:
        try:
            SHADER = create_softviz_shader()
        except Exception as ex:
            SOFTVIZ_SHADER_FAILED = True
            if not VIZ_CACHE.draw_error_logged:
                print("SoftViz: GPUShader creation failed:", ex)
                traceback.print_exc()
                VIZ_CACHE.draw_error_logged = True
            return

    if SHADER is None:
        return

    rv3d = ctx.region_data
    if not rv3d: return

    view_mat = rv3d.view_matrix
    view_inv = view_mat.inverted().to_3x3()
    right_base = view_inv @ Vector((1, 0, 0))
    up_base = view_inv @ Vector((0, 1, 0))

    lut = get_or_bake_lut(ramp_node)

    # Batch stores center positions + corner offsets + pre-baked colors.
    # Camera orientation and dot_size are passed as uniforms every frame,
    # so the batch only needs rebuilding when the underlying weights or
    # colors change — not on every camera move or mode switch.
    if s.viz_mode == 'PROPORTIONAL':
        data_hash = VIZ_CACHE.hash
        # VIZ_CACHE.hash omits per-vertex coords and is not advanced every modal frame
        # during the spy session. Fingerprint pos + influence so the batch rebakes on
        # grab (moved verts), scroll radius (weights change at fixed positions), etc.
        pos_fp = hash(tuple(
            (round(wp.x, 3), round(wp.y, 3), round(wp.z, 3), round(w, 6))
            for wp, w in vert_weights
        ))
    elif s.viz_mode == 'VERTEX_GROUP':
        data_hash = ('VG', s.vgroup_name, len(vert_weights))
        pos_fp = hash(tuple((round(wp.x, 3), round(wp.y, 3), round(wp.z, 3)) for wp, _ in vert_weights))
    else:
        data_hash = ('SK', s.shape_key_name, len(vert_weights))
        pos_fp = hash(tuple((round(wp.x, 3), round(wp.y, 3), round(wp.z, 3)) for wp, _ in vert_weights))
    batch_key = (data_hash, pos_fp, VIZ_CACHE.ramp_lut_key, round(s.alpha_fade, 4))

    if VIZ_CACHE.batch is None or VIZ_CACHE.batch_hash != batch_key:
        positions = []
        corner_list = []
        color_list = []
        indices = []
        vc = 0
        alpha_fade = s.alpha_fade

        for wp, w in vert_weights:
            if lut is not None:
                r, g, b, a = lut[min(255, int(w * 255))]
            else:
                r, g, b, a = (1.0, 0.0, 0.0, 1.0)
            a = (a * (1.0 - alpha_fade)) + (w * alpha_fade)
            col = (r, g, b, a)

            p = (wp.x, wp.y, wp.z)
            positions.extend([p, p, p, p])
            corner_list.extend(_QUAD_CORNERS)
            color_list.extend([col, col, col, col])
            indices.extend([(vc, vc + 1, vc + 2), (vc, vc + 2, vc + 3)])
            vc += 4

        if not positions: return

        try:
            VIZ_CACHE.batch = batch_for_shader(
                SHADER, 'TRIS',
                {"pos": positions, "corner": corner_list, "color": color_list},
                indices=indices,
            )
            VIZ_CACHE.batch_hash = batch_key
        except Exception as ex:
            VIZ_CACHE.batch = None
            VIZ_CACHE.batch_hash = None
            if not VIZ_CACHE.draw_error_logged:
                print("SoftViz: batch_for_shader failed:", ex)
                traceback.print_exc()
                VIZ_CACHE.draw_error_logged = True
            return

    if VIZ_CACHE.batch is None:
        return

    ortho_half = 0.0
    if s.use_screen_space:
        if rv3d.is_perspective:
            screen_mode = 1
        else:
            # Same world-space factor as perspective (depth * dot * 0.0005 in the
            # shader), using view_distance so size matches when toggling ortho — the
            # old pixel-based ortho size was visually smaller than this tuned formula.
            screen_mode = 2
            depth_ref = max(0.01, float(rv3d.view_distance))
            ortho_half = depth_ref * s.dot_size * 0.0005
    else:
        screen_mode = 0

    try:
        view_z_row = tuple(rv3d.view_matrix[2])
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS' if s.use_xray else 'LESS_EQUAL')
        SHADER.bind()
        SHADER.uniform_float("u_mvp", rv3d.perspective_matrix)
        SHADER.uniform_float("u_view_z_row", view_z_row)
        SHADER.uniform_float("u_right", (*right_base, 0.0))
        SHADER.uniform_float("u_up", (*up_base, 0.0))
        SHADER.uniform_float(
            "u_params",
            (float(s.dot_size), float(ortho_half), float(screen_mode), 0.0),
        )
        VIZ_CACHE.batch.draw(SHADER)
    except Exception as ex:
        if not VIZ_CACHE.draw_error_logged:
            print("SoftViz: GPU draw/uniforms failed:", ex)
            traceback.print_exc()
            VIZ_CACHE.draw_error_logged = True

# -------------------------------------------------
# TRANSFORM SPY (scroll-wheel radius tracker)
# -------------------------------------------------
_TRANSFORM_OPS = {
    'TRANSLATE': lambda: bpy.ops.transform.translate('INVOKE_DEFAULT'),
    'ROTATE':    lambda: bpy.ops.transform.rotate('INVOKE_DEFAULT'),
    'RESIZE':    lambda: bpy.ops.transform.resize('INVOKE_DEFAULT'),
}

def _softviz_spy_should_run(context):
    """Spy only when heatmap is on and we're visualizing proportional influence (same as draw_callback)."""
    if not context.scene.softviz_running:
        return False
    if context.scene.softviz_settings.viz_mode != 'PROPORTIONAL':
        return False
    if not context.tool_settings.use_proportional_edit:
        return False
    return True

class VIEW3D_OT_softviz_transform_spy(bpy.types.Operator):
    """Intercepts G/R/S to track scroll-wheel radius changes during modal transform."""
    bl_idname = "view3d.softviz_transform_spy"
    bl_label = "SoftViz Transform Spy"
    bl_options = {'MODAL_PRIORITY'}

    transform_type: bpy.props.EnumProperty(
        items=[
            ('TRANSLATE', "Translate", ""),
            ('ROTATE',    "Rotate",    ""),
            ('RESIZE',    "Scale",     ""),
        ],
        default='TRANSLATE',
    )

    def invoke(self, context, event):
        global _SV_MODAL_RADIUS, _SV_TRANSFORM_SNAPSHOT
        if not _softviz_spy_should_run(context):
            result = _TRANSFORM_OPS[self.transform_type]()
            # Transform has already registered its own modal handler; spy must not
            # return RUNNING_MODAL without modal_handler_add(self).
            return {'CANCELLED'} if 'CANCELLED' in result else {'FINISHED'}
        snap = _capture_softviz_transform_snapshot(context)
        result = _TRANSFORM_OPS[self.transform_type]()
        # If nothing was selected / transform cancelled immediately, don't go modal.
        if 'RUNNING_MODAL' not in result and 'FINISHED' not in result:
            return {'CANCELLED'}
        _SV_TRANSFORM_SNAPSHOT = snap
        _SV_MODAL_RADIUS = context.scene.tool_settings.proportional_size
        self._ending = False
        if debug:
            print(f"[SoftVizSpy] transform started  | initial radius = {_SV_MODAL_RADIUS:.4f}")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _SV_MODAL_RADIUS, _SV_TRANSFORM_SNAPSHOT

        # Previous event was a confirm/cancel — transform has now exited and
        # ts.proportional_size reflects the committed value.
        if self._ending:
            final = context.scene.tool_settings.proportional_size
            if debug:
                print(f"[SoftVizSpy] transform ended    | final radius   = {final:.4f}")
            _SV_MODAL_RADIUS = None
            _SV_TRANSFORM_SNAPSHOT = None
            return {'FINISHED'}

        # Detect confirm / cancel events; let transform consume them too.
        if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER',
                          'RIGHTMOUSE', 'ESC'} and event.value in {'PRESS', 'CLICK'}:
            self._ending = True
            return {'PASS_THROUGH'}

        if event.type == 'WHEELUPMOUSE' and _SV_MODAL_RADIUS is not None:
            _SV_MODAL_RADIUS /= 1.1
            if debug:
                print(f"[SoftVizSpy] SCROLL UP          | est. radius    = {_SV_MODAL_RADIUS:.4f}")

        elif event.type == 'WHEELDOWNMOUSE' and _SV_MODAL_RADIUS is not None:
            _SV_MODAL_RADIUS *= 1.1
            if debug:
                print(f"[SoftVizSpy] SCROLL DOWN        | est. radius    = {_SV_MODAL_RADIUS:.4f}")

        return {'PASS_THROUGH'}

# -------------------------------------------------
# TOGGLE
# -------------------------------------------------
class VIEW3D_OT_softviz_toggle(bpy.types.Operator):
    bl_idname = "view3d.softviz_toggle"
    bl_label = "SoftViz Heatmap"

    def execute(self, context):
        global DRAW_HANDLE
        scene = context.scene

        if scene.softviz_running:
            remove_draw_handler()
            scene.softviz_running = False
        else:
            remove_draw_handler()
            ensure_nodegroup()

            DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
                draw_callback, (), 'WINDOW', 'POST_VIEW')

            scene.softviz_running = True

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        return {'FINISHED'}

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
class SoftVizSettings(bpy.types.PropertyGroup):
    use_xray: bpy.props.BoolProperty(name="X-Ray (Show Through)", default=True)
    use_screen_space: bpy.props.BoolProperty(name="Constant Screen Size", default=False)
    dot_size: bpy.props.FloatProperty(default=5.0, min=0.01, max=100.0)
    alpha_fade: bpy.props.FloatProperty(default=0.5, min=0.0, max=1.0)
    viz_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('PROPORTIONAL', "Proportional", "Visualize proportional editing influence"),
            ('VERTEX_GROUP', "Vertex Group", "Visualize vertex group weights"),
            ('SHAPE_KEY', "Shape Key", "Visualize shape key displacement per vertex"),
        ],
        default='PROPORTIONAL',
    )
    vgroup_name: bpy.props.StringProperty(name="Vertex Group", default="")
    shape_key_name: bpy.props.StringProperty(name="Shape Key", default="")

# -------------------------------------------------
# UI
# -------------------------------------------------
class VIEW3D_PT_softviz(bpy.types.Panel):
    bl_label = "SoftViz for blender v4.2+"
    bl_idname = "VIEW3D_PT_softviz"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SoftViz'

    def draw(self, context):
        l = self.layout
        l.operator(
            "view3d.softviz_toggle",
            depress=context.scene.softviz_running,
            icon='PARTICLES',
        )
        if context.mode == 'EDIT_MESH':
            warn = topology_edit_display_warning_lines(context)
            if warn:
                box = l.box()
                box.alert = True
                box.label(
                    text='Topology modifier uses "Display Modifier in Edit Mode":',
                    icon='INFO',
                )
                for line in warn[:6]:
                    box.label(text=line)
                if len(warn) > 6:
                    box.label(text=f"… +{len(warn) - 6} more")
                box.label(text="SoftViz cannot match subdiv/cage verts to mesh indices.")


class VIEW3D_PT_softviz_display(bpy.types.Panel):
    bl_label = "Display Settings"
    bl_idname = "VIEW3D_PT_softviz_display"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SoftViz'
    bl_parent_id = "VIEW3D_PT_softviz"
    bl_order = 0

    def draw(self, context):
        s = context.scene.softviz_settings
        l = self.layout
        l.prop(s, "use_xray")
        l.prop(s, "use_screen_space")
        l.prop(s, "dot_size", slider=True)
        l.prop(s, "alpha_fade", slider=True)


class VIEW3D_PT_softviz_colors(bpy.types.Panel):
    bl_label = "Heatmap Colors"
    bl_idname = "VIEW3D_PT_softviz_colors"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SoftViz'
    bl_parent_id = "VIEW3D_PT_softviz"
    bl_order = 1

    def draw(self, context):
        ramp_node = get_ramp_node()
        l = self.layout
        if ramp_node:
            l.template_color_ramp(ramp_node, "color_ramp", expand=True)
            l.operator("view3d.softviz_reset_ramp", icon='FILE_REFRESH')


class VIEW3D_PT_softviz_mode(bpy.types.Panel):
    bl_label = "Mode"
    bl_idname = "VIEW3D_PT_softviz_mode"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SoftViz'
    bl_parent_id = "VIEW3D_PT_softviz"
    bl_order = 2

    def draw(self, context):
        l = self.layout
        s = context.scene.softviz_settings
        obj = context.active_object

        l.row().prop(s, "viz_mode", expand=True)

        if s.viz_mode == 'VERTEX_GROUP':
            if obj and obj.type == 'MESH' and obj.vertex_groups:
                l.prop_search(s, "vgroup_name", obj, "vertex_groups", text="Group")
            else:
                l.label(text="No vertex groups found", icon='INFO')

        elif s.viz_mode == 'SHAPE_KEY':
            if obj and obj.type == 'MESH' and obj.data.shape_keys:
                l.prop_search(s, "shape_key_name", obj.data.shape_keys, "key_blocks", text="Key")
            else:
                l.label(text="No shape keys found", icon='INFO')

# -------------------------------------------------
# REGISTER
# -------------------------------------------------
classes = (
    SoftVizSettings,
    VIEW3D_OT_softviz_transform_spy,
    VIEW3D_OT_softviz_toggle,
    VIEW3D_OT_softviz_reset_ramp,
    VIEW3D_PT_softviz,
    VIEW3D_PT_softviz_display,
    VIEW3D_PT_softviz_colors,
    VIEW3D_PT_softviz_mode,
)

def register_spy_keymaps():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc is None:
        return
    km = kc.keymaps.new(name='Mesh', space_type='EMPTY')
    for key, ttype in (('G', 'TRANSLATE'), ('R', 'ROTATE'), ('S', 'RESIZE')):
        kmi = km.keymap_items.new(
            'view3d.softviz_transform_spy', key, 'PRESS',
            any=False, shift=False, ctrl=False, alt=False,
        )
        kmi.properties.transform_type = ttype
        _SV_KEYMAPS.append((km, kmi))

def unregister_spy_keymaps():
    for km, kmi in _SV_KEYMAPS:
        km.keymap_items.remove(kmi)
    _SV_KEYMAPS.clear()

def register():
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.softviz_settings = bpy.props.PointerProperty(type=SoftVizSettings)
    bpy.types.Scene.softviz_running = bpy.props.BoolProperty(default=False)

    bpy.app.handlers.load_post.append(softviz_load_post)
    
    if not bpy.app.timers.is_registered(init_nodegroup_timer):
        bpy.app.timers.register(init_nodegroup_timer, first_interval=0.1)
        
    if not bpy.app.timers.is_registered(softviz_cache_timer):
        bpy.app.timers.register(softviz_cache_timer)

    sync_softviz_draw_handler()
    register_spy_keymaps()

def unregister():
    unregister_spy_keymaps()
    remove_draw_handler()

    scenes = _bpy_scenes()
    if scenes is not None:
        for scene in scenes:
            scene.softviz_running = False

    if softviz_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(softviz_load_post)
        
    if bpy.app.timers.is_registered(init_nodegroup_timer):
        bpy.app.timers.unregister(init_nodegroup_timer)
        
    if bpy.app.timers.is_registered(softviz_cache_timer):
        bpy.app.timers.unregister(softviz_cache_timer)

    remove_softviz_ramp_nodegroup()

    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    del bpy.types.Scene.softviz_settings
    del bpy.types.Scene.softviz_running