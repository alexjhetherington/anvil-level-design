import bpy
import bmesh
import math
from bpy.types import Operator
from mathutils import Vector

from ..utils import align_2d_shape_to_square, derive_transform_from_uvs, get_texture_dimensions_from_material, get_selected_faces_or_report
from ..handlers import cache_face_data, cache_single_face
from ..properties import set_updating_from_selection, sync_scale_tracking


class LEVELDESIGN_OT_face_aligned_project(Operator):
    """Project UVs aligned to each face independently"""
    bl_idname = "leveldesign.face_aligned_project"
    bl_label = "Face-Aligned Projection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH' and context.mode == 'EDIT_MESH'

    def execute(self, context):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        uv_layer = bm.loops.layers.uv.verify()

        selected_faces = get_selected_faces_or_report(self, bm)
        if not selected_faces:
            return {'CANCELLED'}

        props = context.scene.level_design_props
        ppm = props.pixels_per_meter

        for face in selected_faces:
            # Get texture dimensions for this face's material
            mat = me.materials[face.material_index] if face.material_index < len(me.materials) else None
            tex_meters_u, tex_meters_v = get_texture_dimensions_from_material(mat, ppm)
            uv_per_meter_u = 1.0 / tex_meters_u
            uv_per_meter_v = 1.0 / tex_meters_v

            normal = face.normal

            # Determine best projection axis based on face normal
            abs_normal = Vector((abs(normal.x), abs(normal.y), abs(normal.z)))

            if abs_normal.z > abs_normal.x and abs_normal.z > abs_normal.y:
                # Project from Z (top/bottom view)
                for loop in face.loops:
                    loop[uv_layer].uv = (
                        loop.vert.co.x * uv_per_meter_u,
                        loop.vert.co.y * uv_per_meter_v
                    )
            elif abs_normal.y > abs_normal.x:
                # Project from Y (front/back view)
                for loop in face.loops:
                    loop[uv_layer].uv = (
                        loop.vert.co.x * uv_per_meter_u,
                        loop.vert.co.z * uv_per_meter_v
                    )
            else:
                # Project from X (side view)
                for loop in face.loops:
                    loop[uv_layer].uv = (
                        loop.vert.co.y * uv_per_meter_u,
                        loop.vert.co.z * uv_per_meter_v
                    )

        bmesh.update_edit_mesh(me)
        self.report({'INFO'}, f"Projected {len(selected_faces)} faces")

        # Update properties to reflect projection - scale defaults to 1,1
        set_updating_from_selection(True)
        try:
            props.texture_scale_u = 1.0
            props.texture_scale_v = 1.0
            props.texture_rotation = 0.0
            # Offset will be derived from the resulting UVs
        finally:
            set_updating_from_selection(False)
            sync_scale_tracking(context)

        # Store data for UV lock if enabled
        if props.uv_lock:
            cache_face_data(context)

        return {'FINISHED'}


class LEVELDESIGN_OT_align_uv(Operator):
    """Align texture to face edges"""
    bl_idname = "leveldesign.align_uv"
    bl_label = "Align UV"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        items=[
            ('LEFT', "Left", ""),
            ('RIGHT', "Right", ""),
            ('TOP', "Top", ""),
            ('BOTTOM', "Bottom", ""),
            ('CENTER', "Center", ""),
        ]
    )

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH' and context.mode == 'EDIT_MESH'

    def execute(self, context):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        uv_layer = bm.loops.layers.uv.verify()
        selected_faces = get_selected_faces_or_report(self, bm)
        if not selected_faces:
            return {'CANCELLED'}

        for face in selected_faces:
            uvs = [loop[uv_layer].uv.copy() for loop in face.loops]

            if not uvs:
                continue

            # Get UV bounds
            min_u = min(uv.x for uv in uvs)
            max_u = max(uv.x for uv in uvs)
            min_v = min(uv.y for uv in uvs)
            max_v = max(uv.y for uv in uvs)

            center_u = (min_u + max_u) / 2
            center_v = (min_v + max_v) / 2

            # Calculate offset based on direction
            offset_u = 0
            offset_v = 0

            if self.direction == 'LEFT':
                offset_u = -min_u
            elif self.direction == 'RIGHT':
                offset_u = 1.0 - max_u
            elif self.direction == 'TOP':
                offset_v = 1.0 - max_v
            elif self.direction == 'BOTTOM':
                offset_v = -min_v
            elif self.direction == 'CENTER':
                offset_u = 0.5 - center_u
                offset_v = 0.5 - center_v

            # Apply offset
            for loop in face.loops:
                loop[uv_layer].uv.x += offset_u
                loop[uv_layer].uv.y += offset_v

        bmesh.update_edit_mesh(me)
        return {'FINISHED'}


class LEVELDESIGN_OT_fit_to_face(Operator):
    """Fit texture to fill the entire face (0-1 UV space)"""
    bl_idname = "leveldesign.fit_to_face"
    bl_label = "Fit to Face"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH' and context.mode == 'EDIT_MESH'

    def execute(self, context):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        uv_layer = bm.loops.layers.uv.verify()
        selected_faces = get_selected_faces_or_report(self, bm)
        if not selected_faces:
            return {'CANCELLED'}

        for face in selected_faces:
            uvs = [loop[uv_layer].uv.copy() for loop in face.loops]

            if not uvs:
                continue

            # Get UV bounds
            min_u = min(uv.x for uv in uvs)
            max_u = max(uv.x for uv in uvs)
            min_v = min(uv.y for uv in uvs)
            max_v = max(uv.y for uv in uvs)

            width = max_u - min_u
            height = max_v - min_v

            if width < 0.0001 or height < 0.0001:
                continue

            # Scale and offset to fit 0-1
            for loop in face.loops:
                u = (loop[uv_layer].uv.x - min_u) / width
                v = (loop[uv_layer].uv.y - min_v) / height
                loop[uv_layer].uv = (u, v)

        bmesh.update_edit_mesh(me)
        return {'FINISHED'}


class LEVELDESIGN_OT_rotate_uv(Operator):
    """Rotate UVs by specified angle"""
    bl_idname = "leveldesign.rotate_uv"
    bl_label = "Rotate UV"
    bl_options = {'REGISTER', 'UNDO'}

    angle: bpy.props.FloatProperty(default=90.0)

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH' and context.mode == 'EDIT_MESH'

    def execute(self, context):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        uv_layer = bm.loops.layers.uv.verify()
        selected_faces = get_selected_faces_or_report(self, bm)
        if not selected_faces:
            return {'CANCELLED'}

        props = context.scene.level_design_props
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        for face in selected_faces:
            # Get center of UV coordinates for this face
            uvs = [loop[uv_layer].uv.copy() for loop in face.loops]
            center_u = sum(uv.x for uv in uvs) / len(uvs)
            center_v = sum(uv.y for uv in uvs) / len(uvs)

            # Rotate around center
            for loop in face.loops:
                uv = loop[uv_layer].uv
                # Translate to origin
                u = uv.x - center_u
                v = uv.y - center_v
                # Rotate
                new_u = u * cos_a - v * sin_a
                new_v = u * sin_a + v * cos_a
                # Translate back
                loop[uv_layer].uv.x = new_u + center_u
                loop[uv_layer].uv.y = new_v + center_v

        bmesh.update_edit_mesh(me)

        # Update rotation property to reflect new rotation
        if selected_faces:
            uvs = [loop[uv_layer].uv.copy() for loop in selected_faces[0].loops]
            if len(uvs) >= 2:
                edge_u = uvs[1].x - uvs[0].x
                edge_v = uvs[1].y - uvs[0].y
                current_angle = math.degrees(math.atan2(edge_v, edge_u))
                props.texture_rotation = current_angle

        return {'FINISHED'}


class LEVELDESIGN_OT_snap_rotation_to_edge(Operator):
    """Snap texture edge to face edge"""
    bl_idname = "leveldesign.snap_rotation_to_edge"
    bl_label = "Snap Rotation to Edge"
    bl_options = {'REGISTER', 'UNDO'}

    texture_edge: bpy.props.EnumProperty(
        items=[
            ('TOP', "Top", "Align top of texture"),
            ('BOTTOM', "Bottom", "Align bottom of texture"),
            ('LEFT', "Left", "Align left of texture"),
            ('RIGHT', "Right", "Align right of texture"),
        ]
    )

    direction: bpy.props.EnumProperty(
        items=[
            ('NEXT', "Next Edge", "Snap to next edge"),
            ('PREV', "Previous Edge", "Snap to previous edge"),
        ]
    )

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH' and context.mode == 'EDIT_MESH'

    def execute(self, context):
        obj = context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        uv_layer = bm.loops.layers.uv.verify()
        selected_faces = get_selected_faces_or_report(self, bm)
        if not selected_faces:
            return {'CANCELLED'}

        props = context.scene.level_design_props

        # Map texture_edge to squareEdgeIndex: 0=bottom, 1=right, 2=top, 3=left
        square_edge_map = {'BOTTOM': 0, 'RIGHT': 1, 'TOP': 2, 'LEFT': 3}
        square_edge_index = square_edge_map[self.texture_edge]

        for face in selected_faces:
            num_edges = len(face.loops)
            if num_edges < 2:
                continue

            # Increment or decrement edge index with wrapping
            if self.direction == 'NEXT':
                props.edge_index = (props.edge_index + 1) % num_edges
            else:  # PREV
                props.edge_index = (props.edge_index - 1) % num_edges

            # Get current UV shape
            shape = [(loop[uv_layer].uv.x, loop[uv_layer].uv.y) for loop in face.loops]

            # Align the shape edge to the square edge
            aligned = align_2d_shape_to_square(shape, props.edge_index, square_edge_index)

            # Apply the transformed UVs back
            for i, loop in enumerate(face.loops):
                loop[uv_layer].uv.x = aligned[i][0]
                loop[uv_layer].uv.y = aligned[i][1]

            # Derive and update rotation and offset from the resulting UVs
            transform = derive_transform_from_uvs(face, uv_layer, props.pixels_per_meter, me)
            if transform:
                set_updating_from_selection(True)
                try:
                    props.texture_rotation = transform['rotation']
                    props.texture_offset_x = transform['offset_x']
                    props.texture_offset_y = transform['offset_y']
                finally:
                    set_updating_from_selection(False)

            # Update cache so world-scale UV system uses the new rotation/offset
            cache_single_face(face, uv_layer, props.pixels_per_meter, me)

        bmesh.update_edit_mesh(me)
        return {'FINISHED'}


classes = (
    LEVELDESIGN_OT_face_aligned_project,
    LEVELDESIGN_OT_align_uv,
    LEVELDESIGN_OT_fit_to_face,
    LEVELDESIGN_OT_rotate_uv,
    LEVELDESIGN_OT_snap_rotation_to_edge,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
