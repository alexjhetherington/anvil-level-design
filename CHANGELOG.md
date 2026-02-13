# Anvil Level Design Changelog

## 1.2.2
- Add Box Builder!

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