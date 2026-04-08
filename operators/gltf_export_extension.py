import bpy
import bmesh

from ..core.logging import debug_log


class ANVIL_PT_gltf_export_panel(bpy.types.Panel):
    """Anvil settings panel in the glTF export dialog.

    Uses an inner layout.panel() to match the styling of the built-in glTF
    exporter sections (Include, Transform, Data, Animation) which are all
    drawn with layout.panel() rather than as Panel subclasses.
    """
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_label = ""
    bl_parent_id = "FILE_PT_operator"
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, context):
        operator = context.space_data.active_operator
        return operator is not None and operator.bl_idname == "EXPORT_SCENE_OT_gltf"

    def draw(self, context):
        props = context.scene.level_design_props

        # Enable "Remember Export Settings" so quick re-export works correctly
        operator = context.space_data.active_operator
        if operator and hasattr(operator, 'will_save_settings'):
            operator.will_save_settings = True

        header, body = self.layout.panel("GLTF_export_anvil", default_closed=True)
        header.use_property_split = False
        header.prop(props, "gltf_anvil_enabled", text="")
        header.label(text="Anvil")
        if body:
            body.active = props.gltf_anvil_enabled
            body.use_property_split = True
            body.use_property_decorate = False

            col = body.column()
            col.prop(props, "gltf_anvil_scale")
            col.prop(props, "gltf_anvil_apply_modifiers")
            col.prop(props, "gltf_anvil_separate_loose")
            col.prop(props, "gltf_anvil_debug")


class glTF2ExportUserExtension:
    """Hook into Blender's glTF exporter to apply Anvil preprocessing.

    Uses pre_export_hook to clone the scene and apply destructive operations
    before the exporter collects any data, then post_export_hook to clean up.
    """

    def __init__(self):
        # Required by the glTF exporter extension system
        from io_scene_gltf2.io.com.gltf2_io_extensions import Extension
        self.Extension = Extension
        self._export_scene = None
        self._original_window_scene = None

    def pre_export_hook(self, export_settings):
        props = bpy.context.scene.level_design_props
        if not props.gltf_anvil_enabled:
            return

        debug_log("[glTF Anvil] Starting Anvil export preprocessing")

        self._original_window_scene = bpy.context.window.scene

        # Clone the scene (FULL_COPY duplicates objects + data)
        bpy.ops.scene.new(type='FULL_COPY')
        export_scene = bpy.context.scene
        export_scene.name = "Anvil_Export_Temp"
        self._export_scene = export_scene

        debug_log(f"[glTF Anvil] Created temp scene: {export_scene.name}")

        try:
            _apply_modifiers(export_scene, props.gltf_anvil_apply_modifiers)
            _apply_scale(export_scene, props.gltf_anvil_scale)
            _separate_loose(export_scene, props.gltf_anvil_separate_loose)
        except Exception as e:
            print(f"Level Design Tools: Error during Anvil export preprocessing: {e}")
            _cleanup(self._export_scene, self._original_window_scene,
                     props.gltf_anvil_debug)
            self._export_scene = None
            self._original_window_scene = None
            raise

        # Force the exporter to only export the active (cloned) scene.
        # Without this, if "Active Scene" is unchecked the exporter would
        # iterate bpy.data.scenes and export the original unmodified scene too.
        export_settings['gltf_active_scene'] = True

        debug_log("[glTF Anvil] Preprocessing complete, handing off to glTF exporter")

    def post_export_hook(self, export_settings):
        # Store the filepath for quick re-export (regardless of Anvil enabled state)
        filepath = export_settings.get('gltf_filepath', '')
        if filepath:
            # Use original scene props if we cloned, otherwise current scene
            target_scene = self._original_window_scene or bpy.context.scene
            target_scene.level_design_props.last_export_filepath = filepath

        if self._export_scene is None:
            return

        props = self._original_window_scene.level_design_props
        _cleanup(self._export_scene, self._original_window_scene,
                 props.gltf_anvil_debug)
        self._export_scene = None
        self._original_window_scene = None


def _apply_modifiers(scene, enabled):
    if not enabled:
        return

    debug_log("[glTF Anvil] Applying modifiers")

    for obj in scene.objects:
        if obj.type != 'MESH' or not obj.modifiers:
            continue

        # Make this object active so modifier_apply works
        for o in scene.objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        while obj.modifiers:
            mod_name = obj.modifiers[0].name
            try:
                bpy.ops.object.modifier_apply(modifier=mod_name)
                debug_log(f"[glTF Anvil]   Applied modifier '{mod_name}' on {obj.name}")
            except RuntimeError:
                debug_log(f"[glTF Anvil]   Could not apply modifier '{mod_name}' on {obj.name}, removing")
                obj.modifiers.remove(obj.modifiers[0])


def _apply_scale(scene, scale):
    import math

    if math.isclose(scale, 1.0, rel_tol=1e-6):
        return

    debug_log(f"[glTF Anvil] Applying scale: {scale}")

    for obj in scene.objects:
        # Scale location relative to world origin
        obj.location = obj.location * scale

        # Combined scale factor
        scale_vec = obj.scale * scale

        if obj.type == 'MESH':
            mesh = obj.data
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.scale(bm, vec=scale_vec, verts=bm.verts)
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()

        elif obj.type == 'CURVE':
            curve = obj.data
            for spline in curve.splines:
                if spline.type == 'BEZIER':
                    for point in spline.bezier_points:
                        point.co.x *= scale_vec.x
                        point.co.y *= scale_vec.y
                        point.co.z *= scale_vec.z
                        point.handle_left.x *= scale_vec.x
                        point.handle_left.y *= scale_vec.y
                        point.handle_left.z *= scale_vec.z
                        point.handle_right.x *= scale_vec.x
                        point.handle_right.y *= scale_vec.y
                        point.handle_right.z *= scale_vec.z
                else:  # NURBS or POLY
                    for point in spline.points:
                        point.co.x *= scale_vec.x
                        point.co.y *= scale_vec.y
                        point.co.z *= scale_vec.z
        else:
            continue

        # Reset scale to 1 since we baked it into the data
        obj.scale = (1.0, 1.0, 1.0)


def _separate_loose(scene, enabled):
    if not enabled:
        return

    debug_log("[glTF Anvil] Separating loose meshes")

    # Collect mesh objects (list() since we'll be adding new objects)
    mesh_objects = [obj for obj in scene.objects if obj.type == 'MESH']

    for obj in mesh_objects:
        for o in scene.objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Need to enter edit mode for mesh.separate
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            bpy.ops.mesh.separate(type='LOOSE')
        except RuntimeError:
            debug_log(f"[glTF Anvil]   Could not separate {obj.name}")
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')


def _cleanup(export_scene, original_window_scene, debug):
    if debug:
        debug_log("[glTF Anvil] Debug mode: keeping export scene")
        # Switch back to original but leave the export scene intact
        bpy.context.window.scene = original_window_scene
        return

    debug_log("[glTF Anvil] Cleaning up temp scene")

    # Switch back to original scene first
    bpy.context.window.scene = original_window_scene

    # Delete the export scene and its objects
    bpy.data.scenes.remove(export_scene, do_unlink=True)

    # Clean up orphaned data
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for curve in list(bpy.data.curves):
        if curve.users == 0:
            bpy.data.curves.remove(curve)


def register():
    bpy.utils.register_class(ANVIL_PT_gltf_export_panel)


def unregister():
    bpy.utils.unregister_class(ANVIL_PT_gltf_export_panel)
