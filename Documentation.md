# Developer documentation

## Proportional radius during transform (API)

We cannot know the current proportional radius during transform from RNA alone. Blender doesn't expose it until the modal operator updates its internal state (often on mouse move or confirm, not scroll). Any approach that only reads `ts.proportional_size` during the modal is unreliable.

**Mitigation in this add-on:** a keymap-triggered operator with `MODAL_PRIORITY` runs alongside G/R/S, sees wheel events (with `PASS_THROUGH` to the transform), tracks an estimated radius, and snapshots mesh world positions at transform start so fall distances match Blender’s proportional field while verts move.

```python
# Starting radius is accurate before the transform modal owns the event loop
ts = bpy.context.scene.tool_settings
last_known_radius = ts.proportional_size
bpy.ops.transform.translate('INVOKE_DEFAULT')
```

Older idea: infer scroll from final vs initial radius after confirm — still valid as a cross-check.

### Event order and `MODAL_PRIORITY`

It was once assumed Python could not see scroll during transform because modals consume events in stack order. **Blender 4.2+** adds `MODAL_PRIORITY` so an operator can handle events before other modals if it returns `PASS_THROUGH` for events the transform should still receive.

Reference: [Modal operators (4.2 Python API)](https://developer.blender.org/docs/release_notes/4.2/python_api/#modal-operators)

```python
class MyEventSpyOperator(bpy.types.Operator):
    bl_idname = "my.event_spy"
    bl_label = "Event Spy"
    bl_options = {'MODAL_PRIORITY'}

    def modal(self, context, event):
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            ...
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
```

### Spy / operator return values

When the spy does not attach its own modal (`_softviz_spy_should_run` is false), `invoke` must not return `{'RUNNING_MODAL'}` without `modal_handler_add(self)` — it runs the transform and returns `{'FINISHED'}` (or `{'CANCELLED'}`) so only the transform owns the modal stack.

### `bpy.data` during register

During add-on `register()`, `bpy.data` may be `_RestrictData` with no `.scenes`; use `getattr(bpy.data, "scenes", None)` where needed (see `_bpy_scenes()`).

---

## Technical overview (implementation)

### Distance modes

- **Connected Only off:** Euclidean distance in 3D via a KD-tree from seed points (selected verts at transform start when the spy snapshot is active; otherwise live positions).
- **Connected Only on:** Dijkstra-style propagation along edges, using edge lengths from snapshot or live world positions as appropriate.

### GPU drawing

Influence points are drawn with `gpu` / `batch_for_shader` so the overlay stays usable on denser meshes.

### Depth (X-Ray)

Matches `draw_callback`: `gpu.state.depth_test_set('LESS_EQUAL')` when **X-Ray (Show Through)** is off (dots respect surface depth); when on, **`ALWAYS`** so depth tests always pass and dots draw through the mesh like the UI label suggests.

### Dot sizing

- **World space:** dot scale follows scene units / zoom.
- **Screen space:** size tied to depth / ortho distance for more stable pixel width.

### Caching

Weights are rebuilt when inputs change; during an active spy session, proportional weights rebuild every draw frame so motion and scroll stay current.

---

## Transform + scroll sync (current behavior)

- **Scroll radius:** `MODAL_PRIORITY` spy adjusts an internal `_SV_MODAL_RADIUS` and uses it in `draw_callback` while RNA may be stale.
- **Falloff vs large drags:** `_SV_TRANSFORM_SNAPSHOT` stores per-vertex world positions at G/R/S start; weight math uses the snapshot during the spy session so the field does not “travel” incorrectly with dragged verts.
- **Spy gating:** runs only when the heatmap is on, panel mode is **Proportional**, and Blender **proportional editing** is enabled (`_softviz_spy_should_run`).
- **Keymap:** the spy is registered on addon **Mesh** keymap items for **G**, **R**, and **S** (unmodified). Starting move/rotate/scale only from the toolbar or menus does **not** run the spy — scroll/snapshot sync applies to keyboard shortcuts handled by that keymap.

---

## Modifier “Display in Edit Mode” (limitations)

- SoftViz aligns with **evaluated** vertex positions when a supported **deform** modifier has viewport visibility and **Display in Edit Mode** enabled (e.g. Armature), matching the edit cage for those cases.
- **Subdivision**, **Multires**, and other topology-changing modifiers are not supported for cage alignment: the Python API does not give a stable mapping from cage verts back to original mesh indices. If only those show in edit mode, the overlay uses base mesh positions.
- If both deform and topology-changing modifiers show in edit mode, the add-on temporarily clears **Display in Edit Mode** on known topology-changing types only while sampling evaluation, then restores settings so deform-only evaluation still matches where possible.
- The viewport may still show subdiv handles; dots follow the evaluated mesh used after that suppression, not necessarily the subdiv cage wire.

