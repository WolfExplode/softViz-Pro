import bpy, bmesh, gpu, json, heapq, time
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
# PRESETS
# -------------------------------------------------
class SoftVizAddonPrefs(bpy.types.AddonPreferences):
    bl_idname = __name__
    presets: bpy.props.StringProperty(default="{}")
    def draw(self, context):
        self.layout.label(text="SoftViz Presets")

def prefs():
    return bpy.context.preferences.addons[__name__].preferences

def serialize_ramp(r):
    return [(e.position, list(e.color)) for e in r.elements]

def deserialize_ramp(r, data):
    while len(r.elements) > 1:
        r.elements.remove(r.elements[0])
    for i, (p, c) in enumerate(data):
        if i == 0:
            r.elements[0].position = p
            r.elements[0].color = c
        else:
            e = r.elements.new(p)
            e.color = c

def is_same_ramp(data1, data2):
    if len(data1) != len(data2): return False
    for (p1, c1), (p2, c2) in zip(data1, data2):
        if abs(p1 - p2) > 0.001: return False
        for v1, v2 in zip(c1, c2):
            if abs(v1 - v2) > 0.001: return False
    return True

def get_active_preset_name():
    ramp_node = get_ramp_node()
    if not ramp_node:
        return None

    current = serialize_ramp(ramp_node.color_ramp)
    store = json.loads(prefs().presets)

    for name, data in store.items():
        if is_same_ramp(data, current):
            return name
    return None

# -------------------------------------------------
# PRESET OPERATORS
# -------------------------------------------------
class VIEW3D_OT_softviz_save_preset(bpy.types.Operator):
    bl_idname = "view3d.softviz_save_preset"
    bl_label = "Save Preset"
    name: bpy.props.StringProperty(name="Name")

    def execute(self, context):
        ramp = get_ramp_node().color_ramp
        store = json.loads(prefs().presets)
        store[self.name] = serialize_ramp(ramp)
        prefs().presets = json.dumps(store)
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

class VIEW3D_OT_softviz_load_preset(bpy.types.Operator):
    bl_idname = "view3d.softviz_load_preset"
    bl_label = "Load Preset"
    name: bpy.props.StringProperty()

    def execute(self, context):
        ramp = get_ramp_node().color_ramp
        store = json.loads(prefs().presets)
        if self.name in store:
            deserialize_ramp(ramp, store[self.name])
        return {'FINISHED'}

class VIEW3D_OT_softviz_update_preset(bpy.types.Operator):
    bl_idname = "view3d.softviz_update_preset"
    bl_label = "Update Preset"
    name: bpy.props.StringProperty()

    def execute(self, context):
        ramp = get_ramp_node().color_ramp
        store = json.loads(prefs().presets)
        store[self.name] = serialize_ramp(ramp)
        prefs().presets = json.dumps(store)
        return {'FINISHED'}

class VIEW3D_OT_softviz_delete_preset(bpy.types.Operator):
    bl_idname = "view3d.softviz_delete_preset"
    bl_label = "Delete Preset"
    name: bpy.props.StringProperty()

    def execute(self, context):
        store = json.loads(prefs().presets)
        if self.name in store:
            del store[self.name]
            prefs().presets = json.dumps(store)
        return {'FINISHED'}

class VIEW3D_OT_softviz_rename_preset(bpy.types.Operator):
    bl_idname = "view3d.softviz_rename_preset"
    bl_label = "Rename Preset"

    name: bpy.props.StringProperty()
    new_name: bpy.props.StringProperty(name="New Name")

    def execute(self, context):
        store = json.loads(prefs().presets)
        if self.name in store and self.new_name:
            store[self.new_name] = store.pop(self.name)
            prefs().presets = json.dumps(store)
        return {'FINISHED'}

    def invoke(self, context, event):
        self.new_name = self.name
        return context.window_manager.invoke_props_dialog(self)

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
    if not ts.use_proportional_edit: return
    
    edit_objs = [o for o in ctx.objects_in_mode if o.type == 'MESH']
    if not edit_objs: return

    ramp_node = get_ramp_node()
    if ramp_node:
        ramp_node.id_data.update_tag()
        ctx.view_layer.update()

    ramp = ramp_node.color_ramp if ramp_node else None
    s = ctx.scene.softviz_settings
    rad = ts.proportional_size

    bms = {}
    cache_key_elements = [
        ts.proportional_size,
        ts.proportional_edit_falloff,
        ts.use_proportional_connected
    ]
    
    for obj in edit_objs:
        bm = bmesh.from_edit_mesh(obj.data)
        bms[obj] = bm
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

    depsgraph = ctx.evaluated_depsgraph_get()
    cage_coords_by_obj = {}
    for obj in edit_objs:
        if object_uses_modifier_edit_display(obj):
            c = eval_vert_world_coords(obj, depsgraph, len(obj.data.vertices))
            if c is not None:
                cage_coords_by_obj[obj] = c

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

# -------------------------------------------------
# UI
# -------------------------------------------------
class VIEW3D_PT_softviz(bpy.types.Panel):
    bl_label = "SoftViz 4.5"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SoftViz'

    def draw(self, context):
        l = self.layout
        ts = context.tool_settings
        s = context.scene.softviz_settings
        store = json.loads(prefs().presets)

        ramp_node = get_ramp_node()
        active = get_active_preset_name()

        box = l.box()
        box.label(text="Proportional")
        box.prop(ts, "use_proportional_edit")
        box.prop(ts, "proportional_size")
        box.prop(ts, "proportional_edit_falloff")
        box.prop(ts, "use_proportional_connected")

        l.operator("view3d.softviz_toggle",
                   depress=context.scene.softviz_running,
                   icon='PARTICLES')

        box = l.box()
        box.label(text="Display Settings")
        box.prop(s, "use_xray")
        box.prop(s, "use_screen_space")
        box.prop(s, "dot_size", slider=True)
        box.prop(s, "alpha_fade", slider=True)

        box = l.box()
        box.label(text="Heatmap Colors")
        if ramp_node:
            box.template_color_ramp(ramp_node, "color_ramp", expand=True)
            box.operator("view3d.softviz_reset_ramp", icon='FILE_REFRESH')

        box = l.box()
        box.label(text="Presets")
        box.label(text=f"Active Preset: {active if active else 'None'}")
        box.operator("view3d.softviz_save_preset", icon='ADD')

        for name in store.keys():
            r = box.row(align=True)

            op = r.operator("view3d.softviz_load_preset", text=name)
            op.name = name

            up = r.operator("view3d.softviz_update_preset", text="", icon='FILE_TICK')
            up.name = name

            rn = r.operator("view3d.softviz_rename_preset", text="", icon='GREASEPENCIL')
            rn.name = name

            de = r.operator("view3d.softviz_delete_preset", text="", icon='TRASH')
            de.name = name

# -------------------------------------------------
# REGISTER
# -------------------------------------------------
classes = (
    SoftVizAddonPrefs,
    SoftVizSettings,
    VIEW3D_OT_softviz_toggle,
    VIEW3D_OT_softviz_reset_ramp,
    VIEW3D_OT_softviz_save_preset,
    VIEW3D_OT_softviz_load_preset,
    VIEW3D_OT_softviz_update_preset,
    VIEW3D_OT_softviz_rename_preset,
    VIEW3D_OT_softviz_delete_preset,
    VIEW3D_PT_softviz,
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