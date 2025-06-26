"""
Utility modules for the Referee Detection System

This package contains various utility functions and classes used throughout
the application for common operations like file handling, image processing,
hash management, and validation.
"""

# Explicit imports instead of star imports for better code clarity
from .file_utils import (
    ensure_directory_exists,
    get_file_extension,
    is_valid_image_file,
    is_valid_video_file,
    allowed_file,
    save_uploaded_file
)

from .image_utils import (
    resize_image_maintaining_aspect,
    add_padding_to_bbox,
    convert_bbox_format
)

from .hash_utils import (
    calculate_image_hash,
    is_hash_registered,
    register_hash,
    load_image_hashes,
    is_duplicate_image,
    register_image_hash,
    clear_hash_cache
)

from .validation_utils import (
    validate_bbox,
    validate_class_id,
    validate_file_size,
    validate_image_upload
)

__all__ = [
    # File utilities
    'ensure_directory_exists',
    'get_file_extension',
    'is_valid_image_file',
    'is_valid_video_file',
    'allowed_file',
    'save_uploaded_file',
    
    # Image utilities
    'resize_image_maintaining_aspect',
    'add_padding_to_bbox',
    'convert_bbox_format',
    
    # Hash utilities
    'calculate_image_hash',
    'is_hash_registered',
    'register_hash',
    'load_image_hashes',
    'is_duplicate_image',
    'register_image_hash',
    'clear_hash_cache',
    
    # Validation utilities
    'validate_bbox',
    'validate_class_id',
    'validate_file_size',
    'validate_image_upload'
] 