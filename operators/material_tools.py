import bpy
from bpy.types import Operator

from ..utils import (
    find_material_with_image,
    get_texture_node_from_material,
    get_principled_bsdf_from_material,
    is_texture_alpha_connected,
)
from ..handlers import get_active_image


def get_used_material_indices(obj):
    """Return set of material indices actually used by faces."""
    if obj.type != 'MESH' or not obj.data:
        return set()

    used_indices = set()
    for poly in obj.data.polygons:
        used_indices.add(poly.material_index)
    return used_indices


def cleanup_unused_material_slots(obj):
    """Remove material slots not used by any face. Returns count removed."""
    if obj.type != 'MESH' or not obj.data:
        return 0

    removed = 0
    used_indices = get_used_material_indices(obj)

    # Work backwards to avoid index shifting issues
    for i in range(len(obj.material_slots) - 1, -1, -1):
        if i not in used_indices:
            obj.active_material_index = i
            with bpy.context.temp_override(object=obj):
                bpy.ops.object.material_slot_remove()
            # Recalculate used indices after removal (they shift down)
            used_indices = get_used_material_indices(obj)
            removed += 1

    return removed


class LEVELDESIGN_OT_set_interpolation_closest(Operator):
    """Setting interpolation of image texture to closest"""

    bl_idname = "leveldesign.set_interpolation_closest"
    bl_label = "Set interpolation to closest"

    def execute(self, context):
        image = get_active_image()
        mat = find_material_with_image(image)
        tex = get_texture_node_from_material(mat)
        if tex:
            tex.interpolation = 'Closest'
        return {'FINISHED'}


class LEVELDESIGN_OT_set_interpolation_linear(Operator):
    """Setting interpolation of image texture to linear"""

    bl_idname = "leveldesign.set_interpolation_linear"
    bl_label = "Set interpolation to linear"

    def execute(self, context):
        image = get_active_image()
        mat = find_material_with_image(image)
        tex = get_texture_node_from_material(mat)
        if tex:
            tex.interpolation = 'Linear'
        return {'FINISHED'}


class LEVELDESIGN_OT_toggle_texture_alpha(Operator):
    """Toggle connecting texture alpha to material alpha"""

    bl_idname = "leveldesign.toggle_texture_alpha"
    bl_label = "Toggle Texture Alpha"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        image = get_active_image()
        if not image:
            return False
        mat = find_material_with_image(image)
        return mat is not None

    def execute(self, context):
        image = get_active_image()
        mat = find_material_with_image(image)
        tex = get_texture_node_from_material(mat)
        bsdf = get_principled_bsdf_from_material(mat)

        if not tex or not bsdf:
            self.report({'WARNING'}, "Material missing texture or BSDF node")
            return {'CANCELLED'}

        nt = mat.node_tree

        if is_texture_alpha_connected(mat):
            # Disconnect alpha
            for link in list(nt.links):
                if (
                    link.from_node == tex
                    and link.from_socket.name == "Alpha"
                    and link.to_node == bsdf
                    and link.to_socket.name == "Alpha"
                ):
                    nt.links.remove(link)
            mat.blend_method = 'OPAQUE'
        else:
            # Connect alpha
            nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            mat.blend_method = 'CLIP'

        return {'FINISHED'}


class LEVELDESIGN_OT_fix_alpha_bleed(Operator):
    """Set RGB of transparent pixels to a color to fix edge bleeding"""

    bl_idname = "leveldesign.fix_alpha_bleed"
    bl_label = "Fix Alpha Bleed"
    bl_options = {'REGISTER', 'UNDO'}

    color: bpy.props.FloatVectorProperty(
        name="Fill Color",
        subtype='COLOR',
        default=(0.0, 0.0, 0.0),
        min=0.0,
        max=1.0,
        description="Color to set transparent pixels to",
    )

    alpha_threshold: bpy.props.FloatProperty(
        name="Alpha Threshold",
        default=0.01,
        min=0.0,
        max=1.0,
        description="Pixels with alpha below this value will be modified",
    )

    @classmethod
    def poll(cls, context):
        image = get_active_image()
        return image is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        image = get_active_image()

        if image.packed_file:
            image.unpack(method='USE_ORIGINAL')

        width, height = image.size
        pixels = list(image.pixels[:])

        modified_count = 0
        for i in range(0, len(pixels), 4):
            alpha = pixels[i + 3]
            if alpha < self.alpha_threshold:
                pixels[i] = self.color[0]  # R
                pixels[i + 1] = self.color[1]  # G
                pixels[i + 2] = self.color[2]  # B
                modified_count += 1

        image.pixels[:] = pixels
        image.update()

        if image.filepath:
            image.save()
            self.report(
                {'INFO'},
                f"Fixed {modified_count} pixels and saved to {image.filepath}",
            )
        else:
            self.report(
                {'WARNING'},
                f"Fixed {modified_count} pixels but image has no filepath - pack or save manually",
            )

        return {'FINISHED'}


class LEVELDESIGN_OT_cleanup_unused_materials(Operator):
    """Remove unused materials created by the addon (IMG_ prefix)"""

    bl_idname = "leveldesign.cleanup_unused_materials"
    bl_label = "Cleanup Unused Materials"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        slots_removed = 0
        materials_removed = 0

        # First, clean up unused material slots from all mesh objects
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                slots_removed += cleanup_unused_material_slots(obj)

        # Then remove materials with no users
        for mat in list(bpy.data.materials):
            if mat.name.startswith("IMG_") and mat.users == 0:
                bpy.data.materials.remove(mat)
                materials_removed += 1

        if slots_removed > 0 or materials_removed > 0:
            self.report(
                {'INFO'},
                f"Removed {slots_removed} slot(s), {materials_removed} material(s)",
            )
        else:
            self.report({'INFO'}, "No unused materials to remove")

        return {'FINISHED'}


classes = (
    LEVELDESIGN_OT_set_interpolation_closest,
    LEVELDESIGN_OT_set_interpolation_linear,
    LEVELDESIGN_OT_toggle_texture_alpha,
    LEVELDESIGN_OT_fix_alpha_bleed,
    LEVELDESIGN_OT_cleanup_unused_materials,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
