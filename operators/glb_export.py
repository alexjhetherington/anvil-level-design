import bpy


class LEVELDESIGN_OT_export_gltf_quick(bpy.types.Operator):
    """Re-export using the last glTF export settings and filepath"""
    bl_idname = "leveldesign.export_gltf_quick"
    bl_label = "Quick Export glTF"

    @classmethod
    def poll(cls, context):
        props = context.scene.level_design_props
        return props.last_export_filepath != ""

    def execute(self, context):
        props = context.scene.level_design_props
        filepath = props.last_export_filepath

        if not filepath:
            self.report({'ERROR'}, "No previous export. Use File > Export > glTF 2.0 first.")
            return {'CANCELLED'}

        # Retrieve saved glTF export settings
        saved = context.scene.get("glTF2ExportSettings")
        if saved:
            kwargs = {k: v for k, v in saved.items()}
        else:
            kwargs = {}

        # Always override filepath and disable the file browser
        kwargs['filepath'] = filepath

        bpy.ops.export_scene.gltf(**kwargs)

        self.report({'INFO'}, f"Exported to {filepath}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(LEVELDESIGN_OT_export_gltf_quick)


def unregister():
    bpy.utils.unregister_class(LEVELDESIGN_OT_export_gltf_quick)
