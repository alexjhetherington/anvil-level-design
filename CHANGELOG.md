# Anvil Level Design Changelog

## 1.3.1
- Change material matching strategy to name based (fixes crash related to material disambiguation)
- Stop repeated checks for hotspot file if it doesn't exist

## 1.3.0
- Handle MULTI-UV MAPS
- Integration tests
- Fix increasing size of 0 area face failing to apply UVs
- Significant refactor of new face UVing (moved to a general rather than tool specific approach)
- Cube cut initial UVs are more sensible
- Fix alt click not working on adjacent faces the first time after entering edit mode
- Use Blender's walk navigation for WASD view

## 1.2.11
- Fix duplicating objects resets UVs
- When box is built in edit mode, box is selected

## 1.2.10
- Instance selection also works on collection instances
- Fix hotspotting overzealously rejecting combining faces
- Fix switching to a new object with selected face fails to update UV settings panel / applies old scale

## 1.2.9
- Anvil grid scales with scene length unit
- Fix cube cut and box builder snap/overlay not aligning with Blender grid for imperial units
- Don't show cube cut and box builder grid overlay in ortho views (grid is already there)
- Fix initial texture application setting incorrect UV scale
- Improve reliability of image texture preview
- Debug logging is now a scene property
- Texture preview shows previous image selected (disabled)
- New boxes from box builder are textured with the previously selected texture
- Selecting a texture in the file browser will apply it to an entire object while in object mode
- Fix pick texture (alt right mouse) crash

## 1.2.8
- Swapped Level Design and Hotspot Mapping workspaces
- Fix selecting instances (I hope)
- Make it clearer that scaling objects is not the appropriate way to use Anvil
- Very minor UV lock styling fix
- Improve extrusion UV handling (maintain non default scales)

## 1.2.7
- Add debug logging

## 1.2.6
- Add Anvil's own SELECT CONNECTED
- Hold Ctrl at the start of a box build or cube cut to lock the axis
- Prevent cube cut and box builder being active at the same time
- Cube cut and box builder grids should appear immediately (before mouse move)
- Add keybindings (default none) for UV shortcuts
- (Hidden) select invalid UVs
- Gated some tools to Level Design / Hotspot Mapping workspace that were missed
- Organise the keybindings section in the addon preferences into sections

## 1.2.5
- Selecting, texture application, cube cut and box builder all now ignore culled backfaces
- New select mode: PAINT SELECT. Hold ctrl, left mouse, and drag to add all vertices, edges, or faces crossed to the selection (by default overrides 'shortest path to'). 
- Alt left mouse to apply textures now works hold left mouse and drag

## 1.2.4
- Fix hotspot mapping orientation type icon sizing (probably)

## 1.2.3
- Reorganise N panels
- Introduce default material settings
- Improve clarity of UI when multiple faces with conflicting textures and UV settings
- Alt right click and file browser selection can now only apply images to selected faces (removed the concept of 'active image' when faces aren't selected)
- Alt left click to apply texture to face only works when a single face is selected
- Vertex and Edge select mode are always treated as 'no faces selected' for the purposes of the Anvil material workflow

## 1.2.2
- Add BOX BUILDER
- Cube Cut and Box Builder can make rotated construction boxes by holding shift
- Cube Cut from orthogonal view has infinite depth
- Improve Cube Cut and Box Builder snapping (fix bug; improve mouse feel)
- Even more Cube Cut and Box Builder stability
- Fix active image stale reference on undo
- Fix an issue with undo and UVs

## 1.2.1

- Cube Cut cube is now oriented based on start face normal
- More Cube Cut stability (including defensive error messaging when done on invalid faces)
- Fix version in init!

## 1.2.0

- Add HOTSPOT MAPPING
- Various undo / redo improvements
- Slight colour change on 3d view snap preview
- Refactor workspace creation to use a template .blend file (much more stable)
- Add workspace creation buttons into Anvil addon preferences
- Addon disabled outside of specialised workspaces
- Cube cut stability improved massively

## 1.1.1
- Cube cut opposite face now cuts when exactly aligned with a mesh face
- New mesh faces will be appropriately projected (extrusion was already working, bridging edges etc did not)
- Fix regression causing geometry edits back to the original state in one modal operation distorts UVs (see comments in code)
- Speculative mouse look reliability improvements

## 1.1.0

- Introduce CUBE CUT. Press C in edit mode to enter cube cut mode. See Readme for more details.
- Fix grids not displaying in orthographic view in some cases
- On Level Design workspace creation, set appropriate blender file settings
- Set paths to relative on save
- UV lock is now on a per object basis
- Allow camera navigation keys to be remapped

## 1.0.3

- Fix applied textures sometimes being rotated 180 degrees extra
- Speculative fix for applying UVs to faces with opposite windings
- Applying textures to opposite faces mirrors the rotation over the reference axis
- Applying UVs to non-connected faces applies offset more intelligently
- If file is loaded into edit mode with face selected, that face is used to set active image
- Default material is always scale 1 (initial texturing makes more sense)
- Fix grid hotkeys not working in object mode when file first loaded (they worked once toggled in and out of edit mode)
- Face UV Mode (Press T to snap texture to edge based on mouse position) (removed UV Rotation panel)
- Fix UV shortcut buttons
- Manually adjusting scale, rotation, and offset only affect the value adjusted
- Allow face aligned project to be scaled

## 1.0.2

- Fix vertex and edge sliding garbling UVs by disabling 'correct uv' by default (HACK - see etc in Readme for details)
- Improve face local axis calculations to handle 0 length edges
- Fix UVs no longer working when vertex count on object changes (e.g. merged vertices)

## 1.0.1

- Appropriately scoped keybindings so enabling / disabling the addon doesn't permanently break you keybindings (sorry for any inconvenience caused)
- Keybindings menu in the addon preference (allows you to set all keybindings and for example turn off features by disabling the keybinds)
- Added some general information about exporting to the readme (non addon specific)
- Changed file browser interactions from modal to just a timed function (open models prevent you from reloading scripts)
- Picking material from a different object updates the texture preview in a timely fashion
- Fixed mesh modification reseting scale in some cases
- Fixed loop cuts (actually just force disabled blenders own UV correction which was breaking everything)
- Addon free cam is now blocked in orthographic mode