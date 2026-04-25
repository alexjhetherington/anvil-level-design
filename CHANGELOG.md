# Anvil Level Design Changelog

# 1.6.4
- Fix Alt Left Mouse on another object crash
- Alt Left Mouse on another object handles differently scaled objects correctly
- Alt Left Mouse correctly ignores hidden faces
- Remove WELD undo transient states

# 1.6.3 
- Supports flipped textures (excluding hotspots)
- UV Transform texture preview respects closest filtering
- Correct UV Transform preview on scaled objects
- Fix switching to the Level Design workspace incorrectly applying textures in some cases
- Default show edge length on
- Add show edge length, subdivisions, and Unit system to addon preferences
- Fix automerge vertices on spin continuing to trigger after the spin

# 1.6.2
- Improve back face selection culling edge / vert selection (fix bugs and more heuristics in ambiguous situations)
- Added hotspot tutorial (see examples/hotspot_tutorial.blend)
- Fix regression: cancelling a transform modal causes UV warping
- Fix for corrupted faces after cube-cutting non-convex geometry
- Move toggle grid snapping mode to ctrl+g, freeing shift+g for the blender default select hotkey
- Fix spin UV issues (Blender's in built spin triggers undo/redo every tick?!)
- Add automerge selected vertices after spin (in some cases Blender's in built spin vertex merge does not work)
- Improve UV transform mode behaviour when edges and vertices are very close
- Add UV transform edge-edge snapping
- Add UV transform vertex snapping during move
- Fix UV transform disabling itself forever in vertex / edge select modes
- UV transform gizmo visible through faces
- UV transform works with multiple selected faces
- Add axis aligned transforms to UV transform

# 1.6.1
- Move gltf export items into the default blender gltf export; remove Anvil's separate scaled export
- Add default default material settings to addon preferences (allows you to define the default material settings for new .blend files)
- UV Transform mode can no longer be entered multiple times simultaneously
- UV Transform mode exists on leaving edit mode
- Fix error in specific UV Transform mode case on new objects

# 1.6.0
- Add UV TRANSFORM mode.
- Shift+L adds connected island to selection
- Default rotation snapping on 

# 1.5.3
- Geometry grid is now triplanar
- Improve z fighting on grid overlay
- Improve edge selection highlighted when grid overlay is on
- Cube cut / box builder snap to world aligned grid

# 1.5.2
- Allow users to customise combine face angle limit (hotspotting)
- Default seam angle is 33d

## 1.5.1
- Add vignette to texture snapping modes
- Add quad grid texture snapping (T with a grid of quads selected)
- Alt-left click applies texture via affine mapping in some cases (should support applying skewed and distorted textures)
- Fix (I think) Face snapping (T on an individual face)  on irregular faces causing sheer 
- Improve mapping of shader to texture file (helps with custom shaders)
- Large code organisation changes (bug risk!)
- Fix hotspot seam issues (introduced in 1.5.0)

## 1.5.0
- BREAKING CHANGE (hotspot definitions)
- Significantly(!) improve hotspot performance
- Manually select hotspots
- Fixed hotspots won't be changed by automatic hotspotting
- Adjust Anvil panels for clarity
- Hotspots can now be used without an external file
- The hotspots external file path can be chosen
- Fix edge selection edge case (2 overlapping front faces) 
- Set front face orientation to transparent on addon start

## 1.4.7
- Shift-G toggles between snap modes
- Alt-G toggles a grid over geometry faces
- Ctrl-Alt Left mouse applies a UV without changing the material
- Ctrl-Alt Right mouse picks a UV without changing the material
- Fix apply stretched texture not applying UVs

## 1.4.6
- WELD folded plane
- Add debug button to display overlapping faces
- Box build with 0 height and abs(depth) > 0 creates a plane (not a thin box)
- Undoing does not clear previously selected image
- Entering vertex paint enables face orientation (and leaving vertex paint mode returns to the previous value)
- Fix cube cut triangulation UV issue
- Backface respecting selection can select non mesh objects

## 1.4.5
- Fix various bevel UV issues

## 1.4.4
- Align actual and default material panel styling
- Add emission and specular to material settings

## 1.4.3
- Vertex colour node setup changed so Godot picks it up

## 1.4.2
- Added option to enable vertex colours on materials (automatically mixes in the colour attribute with the image texture) (colour attribute always added by default)
- Walk Navigation hold works in vertex paint mode (quick colour is rebound to 'C' by default)
- Add metallic to materials
- significant changes to Weld's undo / redo logic (likely fixes edge case bugs I'm not aware of)
- Box builder adds to selection if there is a current selection, otherwise ends with no selection (reduces keys required for subsequent cuts)
- Weld Invert keeps selection (reduces keys required for subsequent cuts)
- Weld Corridor and Weld Bridge Edge loops end with no selection (reduces keys required for subsequent cuts)
- Fix regression caused by blender update stopping initial texture apply on default cube and new boxes in object mode

## 1.4.1
- MINIMUM BLENDER VERSION 5.1
- Add pick and apply STRETCH texture
- Add very obvious error message when user's blender version is too low

## 1.4.0
- Significant refactor to face change identification
- WELD mode (see README)
- WELD corridor
- WELD bridge edge loops
- WELD invert box
- Alt right click now picks scale
- Cube cut now cuts faces that are coplanar with the cube if they point outward
- Fix some incorrect uvs after Blender triangulate
- Even more sensible texturing on currently blank textured faces
- Fix a class of bugs relating to invalid cache on undo / redo
- Texture application from file browser handles undo properly
- Addon initialising works when creating new files from inside blender, ignoring specialised templates

## 1.3.3
- Fix incorrect UV on extruded face when using the extrusion tool
- Fix incorrect UV on some loop cuts
- Improve UV results for bevels

## 1.3.2
- Alt left click works across objects
- Finishing a cube cut switches to edge select and leaves boundary edges selected
- Vertex loop select works through culled backfaces
- Front face near edge selection improved

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