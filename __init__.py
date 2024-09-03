import os
import folder_paths

rembg_path = os.path.join(folder_paths.models_dir, 'rembg')
if not os.path.exists(rembg_path):
    os.makedirs(rembg_path)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']