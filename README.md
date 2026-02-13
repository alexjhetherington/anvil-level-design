# Anvil Level Design Readme

## Intro

Anvil Level Design (Anvil LD) is an addon for blender that combines various tools to speed-up video game level design. It has a particular focus on material application but the overall scope is varied.

Anvil LD is particularly inspired by Trenchbroom.

Anvil LD is a hobby project. It is likely buggy.

## Installation & Use

Download the zip from github: on the main github screen click code -> Download zip.

Edit -> Preferences -> v arrow -> Install From Disk -> Select the zip file

(For the avoidance of doubt, the zip should contain \_\_init.py\_\_ at the root.)

Anvil adds new workspaces 'Hotspot Mapping' and 'Level Design.' You must be in those workspaces to use Anvil features. If you lose them, go to addon preferences -> Anvil -> Create X Workspace.

You can see and remap addon hotkeys in addon preferences.

## Features

### Material and UV Management

Select a face and Alt Left Mouse button to apply the same material to another face.

Select an image file in the file browser and Alt Left Mouse button to apply the image as a material to a face.

Pick a material from a face without selecting it (including from different objects) using Alt Right Mouse

Anvil LD manages materials to prevent duplicates. Use the Cleanup Unused Materials buttons if unused materials become bothersome.

The currently selected image is previewed in the side panel; here the texture transparency channel can be linked to the shader, and roughness can be adjusted. 

Fix Alpha Bleed is a utility tool that edits the source image to set transparent pixels to a specified colour, to fix some cases of visible edges on transparent cutout materials (the blender 'premultiply alpha' setting can fix this without editing the source image, but .GLB exports don't support this flag).

Alt Left Mouse tries to seamlessly tile textures across different faces and around corners.

Manual UV adjustments are possible via the Anvil LD panel:
* Scale, Rotation, and Offset can be manually set. At 1 scale, the pixels per meter setting controls how large materials appear in the level
* UV shapes that do not match the 3d face are not strongly support but:
* When UV lock is turned on, adjusting a face will cause the applied material to warp along with the face
* When UV lock is turned off, adjust a face will not affect the material in world space e.g. when extending a wall, the applied wall material will remain natural looking (bricks won't stretch)
* Shortcuts are provided to reset scale, rotation, and offset (Face-Aligned project); centre and fit materials to faces

Select a face and press T to enter Face UV Mode. In this mode by default the bottom of the texture will snap to the edge *closest to the mouse cusor*. Use WASD to select different texture edges. Use Q and E to set FIT modes.

### Hotspot Mapping

Hotspot mapping automatically assigns UV coordinates by matching face shapes to predefined regions on a texture atlas. It allows you to quickly add details to models.

#### Setting Up Hotspot Maps

Hotspot maps are defined in the Image Editor:

1. Open your texture atlas in Hotspot Mapping workspace
2. Click the Hotspot Edit tool in the left sidebar
2. In the Anvil panel (N key), click "Assign Hotspottable" to mark the texture as a hotspot source
3. Click "Add Hotspot" to create a new hotspot region
4. Click and drag on the image to define rectangular hotspot areas
5. Hotspots can be resized by dragging their edges or corners, and moved by dragging the centre

Hotspot data is stored in a `hotspots.json` file next to your .blend file. This file can be shared between projects or checked into version control.

As in 3d views using this addon, '[' and ']' control pixel snapping.

#### Orientation Types

Each hotspot has an orientation type that controls which faces it can be applied to. Click the orientation button next to a hotspot, or on the orientation icon on the hotspot itself, to cycle through types:

* **Any** - Can be applied to any face. Rotation is randomised
* **Upwards** - Only applies to wall faces (vertical surfaces). The texture's top edge will point upward in world space. Useful for textures with a clear "up" direction like bricks or siding
* **Floor** - Only applies to floor faces (surfaces facing up). Rotation is randomised
* **Ceiling** - Only applies to ceiling faces (surfaces facing down). Rotation is randomised

#### Applying Hotspots

In the 3D viewport, the Anvil panel provides hotspot controls.

You can manually apply hotspots to selected faces (or all faces if none are selected or the object is in edit mode) using the Apply Hotspot button.

You can toggle auto applying hotspots when making geometry edits; this only adjusts hotspots on moved geometry.

##### Allow Combined Faces & Seam Mode

The hotspotting algorithm is able to treat connected faces as one face for the purpose of hotspot application. This works on series of faces that curve or bend. The algorithm attempts to find groups of faces that when taken together can be roughly transformed into a rectangle. The algorithm splits 'islands' based on normals and user added seams.

This functionality can be turned off by unchecking allow combined faces.

Whether or not this functionality is enabled, the hotspotting algorithm adds seams automatically to help UV unwrapping. The seam mode controls what happens to these seams.
* Maintain User Seams clears the automatically added seams so from your perspective seams are unchanged
* Display All Seams (unsurprisingly) displays all seams; it is primarily useful to see what the algorithm is treating as an island if your hotspot textures don't make it clear
* Clear All Seams is there just in case

For now the hotspotting algorithm does not attempt to split non-rectangular islands. As an example: if the algorithm finds an L shaped island of 3 faces, it won't split it into 2 rectangular islands; it will treat each face separately.

##### Size Weight & Texture Thoughts

Once the hotspotting algorithm has chosen islands, it must choose the closest matching hotspot for each island. 

By default the hotspot with the closest matching aspect ratio the island will be chosen, to preserve the texture / pixel aspect ratio.

Depending on the texture atlas used, the closest matching aspect ratio hotspot (or one of the multiple closest matching aspect ratio hotspots) may be small, leading to a blurry texture.

Drag the Size Weight to introduce weighting towards hotspots that match the same texel density.

It is important to have a wide variety of aspect ratios and sizes in your texture atlas to ensure that hotspotting will look good on any given object you create.

### Geometry Tools

#### Cube Cut

Press C in edit mode to enter cube cut mode.

Click on a face in the 3d view to start drawing the cube; move your mouse and click again to define the rectangular face that will be cut. Move your mouse a third time to define the cube; click a third time to make the cut.

Cube cut avoids N-gons and T-Junctions.

Cube cut only affects selected faces (if no faces are selected, it will affect all faces).

#### Box Builder

Press B in edit mode to enter box builder mode.

Click on a face in the 3d view to start drawing the cube; move your mouse and click again to define the rectangular face that will be cut. Move your mouse a third time to define the cube; click a third time to make the box.

A 0 depth box will become a plane.

### Camera and Viewport Tools

Hotkeys are intended to be blender default.

Holding down Right Mouse Button in a 3d view will enable game engine style WASD flying. This is a custom implemention, replacing Blender's walk mode due to its unsatisfying interaction with holding buttons.

Middle mouse button orbits the camera in perspective grid views.

Middle mouse button in orthographic grid views will pan the camera. 

Num keys set orthographic view angles even when camera rotation is locked.

A useful hotkey to remember is numpad . to focus objects.

Use '[' and ']' to double and half the grid size.

### Exporting

Right now there are no bespoke export behaviours. This means you can work with Blender and your game engine as you normally would.

.blend files are typically not directly imported by game engines (though it visually seems like that)

Godot and Unity can both automatically translate the .blend to a hidden intermediary format (.gltf and .fbx respectively).

.fbx image file paths are defined on export.

.gltf image file paths are defined by the file paths blender uses. Be aware of the 'set paths to relative' and 'set paths to absolute' blender options. Be aware that for Godot if your image files are in the same folder as your .blend file and paths are set to absolute, you may not notice until a collaborator opens the project and has missing textures on the imported level (because they do not have the same folder structure as you).

To help you avoid unexpected behaviour, **Anvil sets paths to relative on save.**

.glb files include the images in the exported file. Good for mods but not recommended if you're making a game with multiple levels that will share the same materials.

Image files getting copied into your project unexpectedly is a sign that you are referencing image files that are outside of the project folder or the exported level contains image files; your game engine is making sure the images are available in your project and their import settings can be edited.

In short, the simplest workflow is to keep images used for level design in your project folder:
* For Unity - work with a .blend file in your project
* For Godot - ensure the reference to blender is configured in settings, then work with a .blend file in your project; ensure image paths are relative in the .blend file
* For Unreal - export a .fbx to your project (untested)

If you need non-PBR custom materials use in-engine approaches
* For Unity - Set the import settings to search the project folder for materials with the appropriate name
* For Godot - Write an import script to create/find and set external materials for each model material
* For Unreal - tbd (I'm not an unreal dev)

#### Scaled GLB

A new import mode is provided; scaled GLB. This allows you to globally adjust the scale when exporting as GLB.

The quick export button exports the scene with previously used settings, skipping popups.

## Keybindings and Feature Toggling

All keybinds can be adjusted in the normal Blender keymap settings.

Keybinds added or modified by this addon are exposed in the addon preferences so they can be found easily.

By disabling keybinds you can disable features (like the free camera).

## Additional Settings

When the addon is loaded it sets the following:
* Enable Grid snapping
* Set unit system to None; set grid subdivisions to 1 (grid size is constant regardless of zoom level)
* Lock camera rotation for orthographic grid views (hotkeys still work)

For a more convenient experience I recommend you consider adjusting the following:
* Change Focal Length (view panel) (it helps with interiors)
* Auto Merge Vertices (options in edit mode)

## Limitations / Known Issues / Notes on Blender / Hacks / etc

* Changing grid size does not work during an operation
* Grid snapping does not work when zoomed out far in orthographic mode (blender bug: https://projects.blender.org/blender/blender/issues/137957)
* Due to file browser API restrictions, you must hold alt-click to pick an image if it is already selected
* Blender has built in UV correct: correct face attributes. It defaults to off. It doesn't work well when extruding faces orthoganally
* Blender has built in UV correct: correct UV (vertex and edge slide). It defaults to on. It mostly works well but has a few annoyances (sometimes just doesn't work; base UV is affected by initial move when using the G hotkey). It also (guess) causes face data to move in memory, causing crashes when Anvil UV is working. UV correct is not exposed via Python, so we disable it by directly accessing blender memory. Brittle!