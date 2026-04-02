# Re-export all public symbols for convenient imports.
# Usage: from ..core import debug_log, is_level_design_workspace, ...
# Or import from submodules directly: from ..core.logging import debug_log

from .logging import debug_log, DEBUG_KEEP_HOTSPOT_SEAMS
from .workspace_check import (
    LEVEL_DESIGN_WORKSPACE_NAME,
    HOTSPOT_MAPPING_WORKSPACE_NAME,
    is_level_design_workspace,
    is_hotspot_mapping_workspace,
)
from .face_id import (
    FACE_ID_LAYER_NAME,
    FIXED_HOTSPOT_LAYER_NAME,
    get_face_id_layer,
    get_fixed_hotspot_layer,
    ensure_face_ids,
    assign_face_id,
    find_face_by_id,
    reindex_face_ids,
    save_face_selection,
    restore_face_selection,
    get_selected_faces_or_report,
    get_selected_face_count,
)
from .geometry import (
    get_local_x_from_verts_3d,
    compute_normal_from_verts,
    are_verts_coplanar,
    get_local_x_from_verts_2d,
    normalize_offset,
    align_2d_shape_to_square,
)
from .materials import (
    get_image_from_material,
    find_material_with_image,
    create_material_with_image,
    get_principled_bsdf_from_material,
    get_texture_node_from_material,
    get_texture_dimensions_from_material,
    is_texture_alpha_connected,
    is_vertex_colors_enabled,
    remove_unused_nodes,
    get_default_material_settings,
    get_selected_image_path,
)
from .uv_projection import (
    get_face_local_axes,
    derive_transform_from_uvs,
    face_aligned_project,
    compute_uv_projection_from_face,
    apply_uv_projection_to_face,
    transfer_uvs_from_projection,
    needs_affine_transfer,
    extract_affine_from_face,
    apply_uv_to_face,
    apply_affine_to_face,
)
from .uv_layers import (
    sync_uv_map_settings,
    get_render_active_uv_layer,
    get_unlocked_uv_layers,
    get_all_uv_layers,
    get_locked_uv_layers,
)
from .hotspot_queries import (
    face_has_hotspot_material,
    object_has_hotspot_material,
    get_connected_faces,
    any_connected_face_has_hotspot,
    get_all_hotspot_faces,
)
