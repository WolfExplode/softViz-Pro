# softViz-Pro
SoftViz Pro: Implementation & Technical Overview

SoftViz Pro is a viewport overlay for Blender 4.2+ designed to bridge the feedback gap in the Proportional Editing workflow. While Blender’s native tool uses a 2D circle to represent a 3D influence, SoftViz calculates and draws a per-vertex heatmap directly onto the geometry (similar to 3dsMax, maya,...).

---
Usage:
	Access the settings via the SoftViz tab in the 3D Viewport N-Panel.

	Make sure you're in edit mode.

	Turn on proportional editing.

	Turn on the SoftViz Heatmap.

---
Technical Features:

Geodesic vs. Euclidean Distance:
	When Connected Only is disabled, the tool uses a KDTree to calculate Euclidean (straight-line) distance through 3D space.

	When Connected Only is enabled, it switches to a Dijkstra-based pathfinding algorithm. This calculates the topological distance by "walking" along connected edges, ensuring the 	heatmap accurately stops at physical mesh gaps or separate manifold parts.



GPU-Accelerated Drawing: 
	Points are rendered using Blender's gpu module. This allows the overlay to remain performant even on high-poly meshes by utilizing shaders instead of standard viewport primitives.



Depth Testing (X-Ray):
	Users can toggle the depth state. When X-Ray is disabled, the script uses a LESS_EQUAL depth test against the current buffer, culling any influence points hidden behind the mesh's 	front-facing polygons.


---
Adaptive Scaling:

	World Space: Dots scale with the scene units (becoming larger as you zoom in).

	Screen Space: Dots are calculated based on camera depth and viewport distance to maintain a consistent pixel-width on the monitor.

	Caching & Performance: To maintain high framerates, the mathematical weights are cached. Calculations only re-trigger when the mesh selection, geometry count, or proportional 	settings are modified.
---

Operational Note:

	Due to the architecture of Blender’s Python API, certain data is inaccessible during an active state (while the G, R, or S transform operators are running).

	Scroll Wheel Sync: If you adjust the proportional radius using the scroll wheel while moving a vertex, the Python API cannot read the new radius value in real-time.


---
Update Behavior: 
	The heatmap remains static during the scroll and will instantly recalculate and snap to the correct size as soon as the transform is confirmed (left-click/Enter) or the mouse 	pause-timer (0.2s) expires.

---
Modifier “Display in Edit Mode” (limitations):

	SoftViz aligns the overlay with evaluated vertex positions when a deform modifier (e.g. Armature) has both viewport visibility (eye icon) and “Display in Edit Mode” enabled. That matches the cage you edit for those modifiers.

	Subdivision Surface, Multires, and other modifiers that change vertex count are not supported for cage alignment: Blender’s Python API does not expose the on-cage positions mapped back to original mesh indices. If only those modifiers use “Display in Edit Mode”, SoftViz keeps using base mesh positions (the heatmap ignores that display).

	If Armature (or another supported deform modifier) and Subdivision both show in edit mode, SoftViz temporarily clears “Display in Edit Mode” on known topology-changing modifiers only while it samples the evaluated mesh each frame, then restores your settings. That way Armature deformation still applies to the overlay without a vertex-count mismatch. Subdivision cage positions remain a Blender limitation for the overlay.

	On Cage (Subdivision): the viewport may still show handles on the subdivided surface; SoftViz dots follow the mesh used for evaluation after that brief suppression (deform stack), not the subdivided cage wire.
