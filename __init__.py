import bpy, bmesh, gpu, heapq, time
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

VIZ_CACHE = SoftVizCache()

# -------------------------------------------------
# DRAW HANDLER SAFETY
# -------------------------------------------------
def remove_draw_handler():
    global DRAW_HANDLE
    if DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(DRAW_HANDLE, 'WINDOW')
        except:
            pass
        DRAW_HANDLE = None

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
def modifier_edit_display_signature(obj):
    return tuple(
        (m.name, m.type, m.show_viewport, m.show_in_editmode)
        for m in obj.modifiers
    )

def object_uses_modifier_edit_display(obj):
    for mod in obj.modifiers:
        if mod.show_viewport and mod.show_in_editmode:
            return True
    return False

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

def vert_world_pos(mat, v, cage_coords):
    if cage_coords is not None and v.index < len(cage_coords):
        return cage_coords[v.index]
    return mat @ v.co

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

    depsgraph = ctx.evaluated_depsgraph_get()
    cage_coords_by_obj = {}
    for obj in edit_objs:
        if object_uses_modifier_edit_display(obj):
            c = eval_vert_world_coords(obj, depsgraph, len(obj.data.vertices))
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

        if current_hash != VIZ_CACHE.hash:
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

    coords, colors, indices = [], [], []
    vc = 0

    rv3d = ctx.region_data
    if not rv3d: return
    
    view_mat = rv3d.view_matrix
    view_inv = view_mat.inverted().to_3x3()
    right_base = view_inv @ Vector((1, 0, 0))
    up_base = view_inv @ Vector((0, 1, 0))

    for wp, w in vert_weights:
        if s.use_screen_space:
            if rv3d.is_perspective:
                depth = max(0.01, -(view_mat @ wp).z) 
                factor = depth * s.dot_size * 0.0005 
            else:
                factor = rv3d.view_distance * s.dot_size * 0.0005 
        else:
            factor = s.dot_size * 0.005 

        right = right_base * factor
        up = up_base * factor
        
        if ramp:
            r, g, b, a = ramp.evaluate(w)
        else:
            r, g, b, a = (1, 0, 0, 1)

        a = (a * (1 - s.alpha_fade)) + (w * s.alpha_fade)
        col = (r, g, b, a)

        coords += [wp - right - up, wp + right - up, wp + right + up, wp - right + up]
        colors += [col] * 4
        indices += [(vc, vc + 1, vc + 2), (vc, vc + 2, vc + 3)]
        vc += 4

    if not coords: return

    sh = gpu.shader.from_builtin('SMOOTH_COLOR')
    batch = batch_for_shader(sh, 'TRIS', {"pos": coords, "color": colors}, indices=indices)
    gpu.state.blend_set('ALPHA')
    
    if s.use_xray:
        gpu.state.depth_test_set('ALWAYS')
    else:
        gpu.state.depth_test_set('LESS_EQUAL')
        
    sh.bind()
    batch.draw(sh)

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
        self.layout.operator(
            "view3d.softviz_toggle",
            depress=context.scene.softviz_running,
            icon='PARTICLES',
        )


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