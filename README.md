# Anvil Level Design

## Intro

Anvil Level Design (Anvil LD) is an addon for blender that combines various tools to speed-up video game level design. It has a particular focus on material application but the overall scope is varied.

Anvil LD is particularly inspired by Trenchbroom.

Anvil LD is a hobby project. It is likely buggy.

## Features

### Material and UV Management

Select a face and Alt Left Mouse button to apply the same material to another face.

Select an image file in the file browser and Alt Left Mouse button to apply the image as a material to a face.

Anvil LD manages materials to prevent duplicates. Use the Cleanup Unused Materials buttons if unused materials become bothersome.

The currently selected image is previewed in the side panel; here the texture transparency channel can be linked to the shader, and roughness can be adjusted. 

Fix Alpha Bleed is a utility tool that edits the source image to set transparent pixels to a specified colour, to fix some cases of visible edges on transparent cutout materials (the blender 'premultiply alpha' setting can fix this without editing the source image, but .GLB exports don't support this flag).

Alt Left Mouse tries to seamlessly tile textures across different faces and around corners.

Manual UV adjustments are possible via the Anvil LD panel:
* Scale, Rotation, and Offset can be manually set. At 1 scale, the pixels per meter setting controls how large materials appear in the level
* UV shapes that do not match the 3d face are not strongly support but:
* When UV lock is turned on, adjusting a face will cause the applied material to warp along with the face
* When UV lock is turned off, adjust a face will not affect the material in world space e.g. when extending a wall, the applied wall material will remain natural looking (bricks won't stretch)
* Shortcuts are provided to reset scale, rotation, and offset (Face-Aligned project); centre and fit materials to faces; and snap the edge of a material to face edges

### Camera and Viewport Tools

Hotkeys are intended to be blender default.

Holding down Right Mouse Button in a 3d view will enable game engine style WASD flying. This is a custom implemention, replacing Blender's walk mode due to its unsatisfying interaction with holding buttons.

Middle mouse button orbits the camera in perspective grid views.

Middle mouse button in orthographic grid views will pan the camera. 

Num keys set orthographic view angles even when camera rotation is locked.

A useful hotkey to remember is numpad . to focus objects.

Use '[' and ']' to double and half the grid size.

### Export

A new import mode is provided; scaled GLB. This allows you to globally adjust the scale when exporting as GLB.

The quick export button exports the scene with previously used settings, skipping popups.

## Additional Settings

For a more convenient experience I strongly recommend you adjust the following (indeed, some features will not work correctly without these adjustments):
* Enable Grid snapping
* Set unit system to None; set grid subdivisions to 1 (grid size is constant regardless of zoom level)
* Lock camera rotation for orthographic grid views
* Enable X-Ray mode
* Change Focal Length (it helps with interiors)

## Limitations / Known Issues

* Changing grid size does not work during an operation
* Grid snapping does not work when zoomed out far in orthographic mode (blender bug: https://projects.blender.org/blender/blender/issues/137957)
* Due to file browser API restrictions, you must hold alt-click to pick an image if it is already selected

