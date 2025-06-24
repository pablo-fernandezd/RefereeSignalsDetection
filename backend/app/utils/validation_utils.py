"""
Validation Utility Functions

This module provides utility functions for validating inputs,
including bounding boxes, class IDs, file formats, and other data validation.
"""

from typing import List, Union, Optional, Any, Dict, Tuple
from pathlib import Path
import re
from flask import Request

from config.settings import ModelConfig, FileConfig


def validate_bbox(bbox: List[Union[int, float]], 
                 image_shape: Optional[tuple] = None,
                 format_type: str = 'xyxy') -> bool:
    """
    Validate a bounding box.
    
    Args:
        bbox: Bounding box coordinates
        image_shape: Image shape as (height, width) for bounds checking
        format_type: Format of the bbox ('xyxy', 'xywh', 'xywhn')
        
    Returns:
        True if bounding box is valid, False otherwise
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    
    try:
        # Convert to float for validation
        coords = [float(x) for x in bbox]
    except (ValueError, TypeError):
        return False
    
    # Check for NaN or infinite values
    if any(not (-float('inf') < x < float('inf')) for x in coords):
        return False
    
    if format_type == 'xyxy':
        x1, y1, x2, y2 = coords
        # Check that x2 > x1 and y2 > y1
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Check bounds if image shape is provided
        if image_shape:
            h, w = image_shape
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return False
                
    elif format_type == 'xywh':
        x, y, w, h = coords
        # Width and height must be positive
        if w <= 0 or h <= 0:
            return False
        
        # Check bounds if image shape is provided
        if image_shape:
            img_h, img_w = image_shape
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                return False
                
    elif format_type == 'xywhn':
        # Normalized coordinates should be between 0 and 1
        if any(x < 0 or x > 1 for x in coords):
            return False
        
        x_center, y_center, w, h = coords
        # Width and height must be positive
        if w <= 0 or h <= 0:
            return False
    else:
        return False
    
    return True


def validate_class_id(class_id: Union[int, str], 
                     valid_classes: List[Union[int, str]] = None) -> bool:
    """
    Validate a class ID.
    
    Args:
        class_id: Class identifier to validate
        valid_classes: List of valid class identifiers
        
    Returns:
        True if class ID is valid, False otherwise
    """
    if valid_classes is None:
        # Use default referee classes
        valid_classes = [cls['id'] for cls in ModelConfig.REFEREE_CLASSES]
    
    return class_id in valid_classes


def validate_signal_class(signal_class: str) -> bool:
    """
    Validate a signal class name.
    
    Args:
        signal_class: Signal class name to validate
        
    Returns:
        True if signal class is valid, False otherwise
    """
    return signal_class in ModelConfig.SIGNAL_CLASSES or signal_class == 'none'


def validate_file_upload(file_data: Any, 
                        allowed_extensions: List[str] = None,
                        max_size: int = None) -> tuple[bool, str]:
    """
    Validate an uploaded file.
    
    Args:
        file_data: File data object (Flask request.files)
        allowed_extensions: List of allowed file extensions
        max_size: Maximum file size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_data:
        return False, "No file provided"
    
    if not hasattr(file_data, 'filename') or not file_data.filename:
        return False, "No filename provided"
    
    # Check file extension
    if allowed_extensions is None:
        allowed_extensions = list(FileConfig.ALLOWED_IMAGE_EXTENSIONS)
    
    file_ext = Path(file_data.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return False, f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
    
    # Check file size if provided
    if max_size is None:
        max_size = FileConfig.MAX_FILE_SIZE
    
    if hasattr(file_data, 'content_length') and file_data.content_length:
        if file_data.content_length > max_size:
            return False, f"File too large. Maximum size: {max_size // (1024*1024)}MB"
    
    return True, ""


def validate_youtube_url(url: str) -> bool:
    """
    Validate a YouTube URL.
    
    Args:
        url: YouTube URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    if not isinstance(url, str) or not url.strip():
        return False
    
    # YouTube URL patterns
    patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(?:www\.)?youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/v/[\w-]+',
    ]
    
    for pattern in patterns:
        if re.match(pattern, url.strip()):
            return True
    
    return False


def validate_confidence_threshold(threshold: Union[float, str]) -> bool:
    """
    Validate a confidence threshold value.
    
    Args:
        threshold: Confidence threshold to validate
        
    Returns:
        True if threshold is valid, False otherwise
    """
    try:
        threshold_float = float(threshold)
        return 0.0 <= threshold_float <= 1.0
    except (ValueError, TypeError):
        return False


def validate_model_size(size: Union[int, str]) -> bool:
    """
    Validate a model input size.
    
    Args:
        size: Model size to validate
        
    Returns:
        True if size is valid, False otherwise
    """
    try:
        size_int = int(size)
        # Common model sizes (multiples of 32 for most YOLO models)
        valid_sizes = [320, 416, 512, 608, 640, 832, 1024]
        return size_int in valid_sizes
    except (ValueError, TypeError):
        return False


def validate_json_data(data: dict, required_fields: List[str]) -> tuple[bool, str]:
    """
    Validate JSON data contains required fields.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Data must be a dictionary"
    
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, ""


def validate_pagination_params(page: Union[int, str], 
                             per_page: Union[int, str],
                             max_per_page: int = 100) -> tuple[bool, str, int, int]:
    """
    Validate pagination parameters.
    
    Args:
        page: Page number
        per_page: Items per page
        max_per_page: Maximum allowed items per page
        
    Returns:
        Tuple of (is_valid, error_message, validated_page, validated_per_page)
    """
    try:
        page_int = int(page)
        per_page_int = int(per_page)
    except (ValueError, TypeError):
        return False, "Page and per_page must be integers", 0, 0
    
    if page_int < 1:
        return False, "Page must be >= 1", 0, 0
    
    if per_page_int < 1:
        return False, "Per_page must be >= 1", 0, 0
    
    if per_page_int > max_per_page:
        return False, f"Per_page cannot exceed {max_per_page}", 0, 0
    
    return True, "", page_int, per_page_int


def validate_coordinate_range(value: Union[int, float], 
                            min_val: Union[int, float] = 0,
                            max_val: Union[int, float] = None) -> bool:
    """
    Validate that a coordinate value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value (None for no limit)
        
    Returns:
        True if value is valid, False otherwise
    """
    try:
        val = float(value)
        if val < min_val:
            return False
        if max_val is not None and val > max_val:
            return False
        return True
    except (ValueError, TypeError):
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    if not isinstance(filename, str):
        return "unnamed_file"
    
    # Remove path separators and other problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f]', '', sanitized)
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        max_name_length = 255 - len(ext)
        sanitized = name[:max_name_length] + ext
    
    # Ensure it's not empty
    if not sanitized or sanitized.isspace():
        return "unnamed_file"
    
    return sanitized.strip()


def validate_image_upload(request: Request) -> Dict[str, Any]:
    """
    Validate an image upload request.
    
    Args:
        request: Flask request object
        
    Returns:
        Dictionary with validation result:
        - valid: bool - Whether validation passed
        - error: str - Error message if validation failed
    """
    # Check if image is in request
    if 'image' not in request.files:
        return {'valid': False, 'error': 'No image file provided'}
    
    file = request.files['image']
    
    # Check if file was selected
    if not file or not file.filename:
        return {'valid': False, 'error': 'No file selected'}
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in FileConfig.ALLOWED_IMAGE_EXTENSIONS:
        return {
            'valid': False, 
            'error': f'File type not allowed. Allowed types: {", ".join(FileConfig.ALLOWED_IMAGE_EXTENSIONS)}'
        }
    
    # Check file size (read current position)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > FileConfig.MAX_FILE_SIZE:
        return {
            'valid': False,
            'error': f'File too large. Maximum size: {FileConfig.MAX_FILE_SIZE / (1024*1024):.1f}MB'
        }
    
    if file_size == 0:
        return {'valid': False, 'error': 'File is empty'}
    
    return {'valid': True, 'error': None}


def validate_bbox(bbox: List[int], image_shape: Tuple[int, int]) -> bool:
    """
    Validate that a bounding box is within image bounds and has valid dimensions.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        image_shape: Image shape (height, width)
        
    Returns:
        True if bounding box is valid, False otherwise
    """
    if not bbox or len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    height, width = image_shape
    
    # Check if coordinates are valid numbers
    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return False
    
    # Check if coordinates are within image bounds
    if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
        return False
    
    # Check if bounding box has positive area
    if x2 <= x1 or y2 <= y1:
        return False
    
    return True


def validate_confidence_threshold(threshold: float) -> Dict[str, Any]:
    """
    Validate a confidence threshold value.
    
    Args:
        threshold: Confidence threshold to validate
        
    Returns:
        Dictionary with validation result:
        - valid: bool - Whether validation passed
        - error: str - Error message if validation failed
    """
    if not isinstance(threshold, (int, float)):
        return {'valid': False, 'error': 'Threshold must be a number'}
    
    if threshold < 0.0 or threshold > 1.0:
        return {'valid': False, 'error': 'Threshold must be between 0.0 and 1.0'}
    
    return {'valid': True, 'error': None}


def validate_coordinates(x: int, y: int, width: int, height: int) -> bool:
    """
    Validate that coordinates are within specified bounds.
    
    Args:
        x: X coordinate
        y: Y coordinate
        width: Maximum width
        height: Maximum height
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    return (isinstance(x, int) and isinstance(y, int) and 
            0 <= x < width and 0 <= y < height)


def validate_crop_parameters(crop_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate crop parameters from a request.
    
    Args:
        crop_data: Dictionary containing crop parameters
        
    Returns:
        Dictionary with validation result:
        - valid: bool - Whether validation passed
        - error: str - Error message if validation failed
    """
    required_fields = ['original_filename', 'crop_filename', 'bbox']
    
    for field in required_fields:
        if field not in crop_data:
            return {'valid': False, 'error': f'Missing required field: {field}'}
    
    # Validate bbox
    bbox = crop_data.get('bbox')
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {'valid': False, 'error': 'Invalid bounding box format'}
    
    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return {'valid': False, 'error': 'Bounding box coordinates must be numbers'}
    
    return {'valid': True, 'error': None}


def validate_filename(filename: str) -> Dict[str, Any]:
    """
    Validate a filename for security and format.
    
    Args:
        filename: Filename to validate
        
    Returns:
        Dictionary with validation result:
        - valid: bool - Whether validation passed
        - error: str - Error message if validation failed
    """
    if not filename:
        return {'valid': False, 'error': 'Filename cannot be empty'}
    
    # Check for dangerous characters
    dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in filename for char in dangerous_chars):
        return {'valid': False, 'error': 'Filename contains invalid characters'}
    
    # Check filename length
    if len(filename) > 255:
        return {'valid': False, 'error': 'Filename too long'}
    
    return {'valid': True, 'error': None}


def sanitize_input(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitize input string by removing potentially dangerous characters.
    
    Args:
        input_string: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(input_string, str):
        return ""
    
    # Remove potentially dangerous characters
    sanitized = input_string.strip()
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized 