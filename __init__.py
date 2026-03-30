import bpy, bmesh, gpu, heapq, time, traceback
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, kdtree

bl_info = {
    "name": "SoftViz Pro",
    "author": "Niels Couvreur",
    "version": (4, 5, 0),
    "blender": (4, 5, 0), 
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
# Total push-constant budget: mat4(64) + vec4(16)*3 + float(4) + float(4) + int(4) = 124 bytes.
_SOFTVIZ_VERT_NEW = """
void main() {
    float factor;
    if (u_screen_mode == 0) {
        factor = u_dot_size * 0.005;
    } else if (u_screen_mode == 2) {
        factor = u_ortho_half;
    } else {
        float depth = max(0.01, -dot(u_view_z_row, vec4(pos, 1.0)));
        factor = depth * u_dot_size * 0.0005;
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
    info.push_constant("FLOAT", "u_dot_size")
    info.push_constant("FLOAT", "u_ortho_half")
    info.push_constant("INT", "u_screen_mode")
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
        self.batch = None
        self.batch_hash = None
        self.ramp_lut = None
        self.ramp_lut_key = None
        self.draw_error_logged = False

VIZ_CACHE = SoftVizCache()

# -------------------------------------------------
# DRAW HANDLER SAFETY
# -------------------------------------------------
def remove_draw_handler():
    global DRAW_HANDLE, SHADER, SOFTVIZ_SHADER_FAILED
    if DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(DRAW_HANDLE, 'WINDOW')
        except:
            pass
        DRAW_HANDLE = None
    VIZ_CACHE.batch = None
    VIZ_CACHE.batch_hash = None
    VIZ_CACHE.draw_error_logged = False
    SHADER = None
    SOFTVIZ_SHADER_FAILED = False

@bpy.app.handlers.persistent
def softviz_load_post(dummy):
    remove_draw_handler()

    for scene in bpy.data.scenes:
        scene.softviz_running = False
        # Force native Connected Only to default to OFF on load
        scene.tool_settings.use_proportional_connected = False
        
    if not bpy.app.timers.is_registered(init_nodegroup_timer):
        bpy.app.timers.register(init_nodegroup_timer, first_interval=0.1)

# -------------------------------------------------
# DEBOUNCE TIMER
# -------------------------------------------------
def softviz_cache_timer():
    scene = bpy.context.scene
    if getattr(scene, 'softviz_running', False):
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
        return ng

    ng = bpy.data.node_groups.new(NG_NAME, 'ShaderNodeTree')
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

def object_needs_evaluated_deform_cage(obj):
    if obj.type == 'MESH' and obj.data.shape_keys:
        if getattr(obj, "use_shape_key_edit_mode", False):
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

def transform_modal_active(ctx):
    ao = getattr(ctx, "active_operator", None)
    if ao is None:
        return False
    bid = getattr(ao, "bl_idname", "") or ""
    return bid.startswith("transform.")

# -------------------------------------------------
# DRAW
# -------------------------------------------------
def draw_callback():
    ctx = bpy.context
    ts = ctx.tool_settings
    s = ctx.scene.softviz_settings

    if s.viz_mode == 'PROPORTIONAL' and not ts.use_proportional_edit:
        return

    edit_objs = [o for o in ctx.objects_in_mode if o.type == 'MESH']
    if not edit_objs: return

    ramp_node = get_ramp_node()
    if ramp_node:
        ramp_node.id_data.update_tag()
        ctx.view_layer.update()

    ramp = ramp_node.color_ramp if ramp_node else None

    bms = {}
    for obj in edit_objs:
        bms[obj] = bmesh.from_edit_mesh(obj.data)

    live_tf = transform_modal_active(ctx)
    cage_coords_by_obj = {}
    for obj in edit_objs:
        if object_needs_evaluated_deform_cage(obj):
            c = eval_vert_world_coords_for_draw_cage(
                obj, ctx, len(obj.data.vertices))
            if c is not None:
                cage_coords_by_obj[obj] = c

    # ------- VERTEX GROUP mode -------
    if s.viz_mode == 'VERTEX_GROUP':
        vert_weights = []
        for obj in edit_objs:
            bm = bms[obj]
            mat = obj.matrix_world
            cage = cage_coords_by_obj.get(obj)

            vg = obj.vertex_groups.get(s.vgroup_name) if s.vgroup_name else None
            if not vg:
                vg = obj.vertex_groups.active
            if not vg:
                continue

            dvert_layer = bm.verts.layers.deform.verify()
            vg_idx = vg.index
            for v in bm.verts:
                w = v[dvert_layer].get(vg_idx, 0.0)
                if w > 0.0:
                    wp = vert_world_pos(mat, v, cage)
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
            displacements = [
                (sk.data[i].co - basis.data[i].co).length
                for i in range(len(mesh.vertices))
            ]
            max_d = max(displacements) if displacements else 0.0
            if max_d < 1e-6:
                continue

            bm = bms[obj]
            mat = obj.matrix_world
            cage = cage_coords_by_obj.get(obj)
            bm.verts.ensure_lookup_table()
            for v in bm.verts:
                d = displacements[v.index]
                if d > 0.0:
                    wp = vert_world_pos(mat, v, cage)
                    vert_weights.append((wp, d / max_d))

        if not vert_weights: return

    # ------- PROPORTIONAL mode -------
    else:
        rad = ts.proportional_size

        cache_key_elements = [
            ts.proportional_size,
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
                getattr(obj, "use_shape_key_edit_mode", False),
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

        if live_tf:
            # Modal G/R/S: viewport uses evaluated geometry; recalc every frame
            # (debounce would leave weights frozen during the drag).
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

            global_centers = []
            for obj in edit_objs:
                bm = bms[obj]
                mat = obj.matrix_world
                cage = cage_coords_by_obj.get(obj)
                sel = [v for v in bm.verts if v.select]
                global_centers.extend([vert_world_pos(mat, v, cage) for v in sel])
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
                                p_v = vert_world_pos(mat, v, cage)
                                p_n = vert_world_pos(mat, neighbor, cage)
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
                            wp = vert_world_pos(mat, v, cage)
                            _, _, dist = kd.find(wp)
                            if dist <= rad:
                                w = falloff(dist / rad, ts.proportional_edit_falloff)
                                if w > 0.0:
                                    obj_weights.append((v.index, w))

                        VIZ_CACHE.weights[obj.name] = obj_weights

        vert_weights = []
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
                    vert_weights.append((wp, w))
                except IndexError:
                    pass

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
    elif s.viz_mode == 'VERTEX_GROUP':
        data_hash = ('VG', s.vgroup_name, len(vert_weights))
    else:
        data_hash = ('SK', s.shape_key_name, len(vert_weights))

    pos_fp = hash(tuple(
        (round(wp.x, 3), round(wp.y, 3), round(wp.z, 3))
        for wp, _ in vert_weights
    ))
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
        SHADER.uniform_float("u_dot_size", s.dot_size)
        SHADER.uniform_float("u_ortho_half", ortho_half)
        SHADER.uniform_int("u_screen_mode", (screen_mode,))
        VIZ_CACHE.batch.draw(SHADER)
    except Exception as ex:
        if not VIZ_CACHE.draw_error_logged:
            print("SoftViz: GPU draw/uniforms failed:", ex)
            traceback.print_exc()
            VIZ_CACHE.draw_error_logged = True

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
    bl_label = "SoftViz 4.5"
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
    VIEW3D_OT_softviz_toggle,
    VIEW3D_OT_softviz_reset_ramp,
    VIEW3D_PT_softviz,
    VIEW3D_PT_softviz_display,
    VIEW3D_PT_softviz_colors,
    VIEW3D_PT_softviz_mode,
)

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

def unregister():
    remove_draw_handler()

    if softviz_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(softviz_load_post)
        
    if bpy.app.timers.is_registered(init_nodegroup_timer):
        bpy.app.timers.unregister(init_nodegroup_timer)
        
    if bpy.app.timers.is_registered(softviz_cache_timer):
        bpy.app.timers.unregister(softviz_cache_timer)

    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    del bpy.types.Scene.softviz_settings
    del bpy.types.Scene.softviz_running