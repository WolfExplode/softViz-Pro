# Documentation for developer's thoughts

We cannot know the current proportional radius during transform. Blender doesn't expose it until the modal operator updates its internal state, which happens on mouse move or confirm, not scroll. So any fix that depends on reading ts.proportional_size during transform will be broken by design.

In blender, we can set Proportional editing radius to a specific value after we invoke a transform operator G/R/S. This means we can solve our Scroll Wheel Sync limitation by reading the current 

```python
# Get the tool settings BEFORE the transform starts
ts = bpy.context.scene.tool_settings
last_known_radius = ts.proportional_size  # This is accurate here

# Now invoke the transform (this enters modal state)
bpy.ops.transform.translate('INVOKE_DEFAULT')
```

What you can do:
Know the starting radius at transform begin
Know the final radius after transform confirm (via ts.proportional_size again, or the Redo panel)
Infer that if the user scrolled, the final != initial

if we scroll wheel or pageup/down we can record and set `bpy.context.scene.tool_settings.proportional_size = x.xx` depending on if we scroll up or down. 
then after we commit the transform, we can read from the real radius and update the heatmap to match.





Is it possible to detect that scroll events occurred while Blender's modal operator is running? Blender's event handling passes events through modal operators in LIFO order, so the transform modal consumes scroll events before my operator would even see them. 

do some research to figure out if what we are trying to do is even possible. 
**No, you cannot reliably detect scroll events while Blender's transform modal is running without either:**
- **Consuming the event** (blocking the transform operator), or
- **Using C-level access** (not available in Python API)
- The Python API only exposes the **RNA layer**, not the **internal modal state**
- The event system is **LIFO (Last-In-First-Out)**, once the transform modal is active, it consumes events before any Python handler could see them
- The **live radius value** exists only in C memory (`t->prop_size`), not in RNA (`ts->proportional_size`)



There's a blender addon, screencast keys. when I do a transform operator, it can read keyboard inputs. how is this possible?
does transform operator only consume specifically the scroll wheel and not other keyboard keys?

in Blender 4.2+
```python
class MyEventSpyOperator(bpy.types.Operator):
    bl_idname = "my.event_spy"
    bl_label = "Event Spy"
    bl_options = {'MODAL_PRIORITY'}  # Handle events BEFORE transform!
    
    def modal(self, context, event):
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            print(f"Scroll detected! {event.type}")
            # You can read it, but if you return 'RUNNING_MODAL', 
            # you BLOCK the transform operator
            
        return {'PASS_THROUGH'}  # Let transform have it too
    
    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
```
With [MODAL_PRIORITY](https://developer.blender.org/docs/release_notes/4.2/python_api/#modal-operators), your operator gets events before the transform operator . You can detect the scroll event, but you must return PASS_THROUGH if you want the transform operator to also receive it. 

so this will allow us to read that a scroll up or down input has been sent and update our internal assumption of what the radius might be. 


# Documentation for end user thoughts
nothing right now...