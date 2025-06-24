"""
Image Processing Utilities for Referee Detection System

This module provides utilities for image loading, processing, and manipulation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
from PIL import Image

from config.settings import FileConfig


def load_image_safely(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Safely load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array, or None if loading failed
    """
    try:
        path = Path(image_path)
        if not path.exists():
            return None
        
        # Try loading with OpenCV first
        image = cv2.imread(str(path))
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Fall back to PIL
        pil_image = Image.open(path)
        return np.array(pil_image.convert('RGB'))
        
    except Exception:
        return None


def normalize_image_for_model(image: np.ndarray) -> np.ndarray:
    """
    Normalize image for model input.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Normalized image
    """
    # Convert to float32 and normalize to [0, 1]
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    return image


def add_padding_to_bbox(bbox: List[int], image_shape: Tuple[int, int], padding: int = None) -> List[int]:
    """
    Add padding to a bounding box while keeping it within image bounds.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        image_shape: Image shape (height, width)
        padding: Padding amount in pixels (defaults to config value)
        
    Returns:
        Padded bounding box coordinates
    """
    if padding is None:
        padding = FileConfig.CROP_PADDING
    
    x1, y1, x2, y2 = bbox
    height, width = image_shape
    
    # Add padding
    x1_padded = max(0, x1 - padding)
    y1_padded = max(0, y1 - padding)
    x2_padded = min(width - 1, x2 + padding)
    y2_padded = min(height - 1, y2 + padding)
    
    return [x1_padded, y1_padded, x2_padded, y2_padded]


def convert_bbox_format(bbox: List[float], from_format: str, to_format: str, 
                       image_shape: Tuple[int, int]) -> List[float]:
    """
    Convert bounding box between different formats.
    
    Args:
        bbox: Bounding box coordinates
        from_format: Source format ('xyxy', 'xywh', 'xywhn')
        to_format: Target format ('xyxy', 'xywh', 'xywhn')
        image_shape: Image shape (height, width) for normalized formats
        
    Returns:
        Converted bounding box coordinates
    """
    height, width = image_shape
    
    # Convert to xyxy first
    if from_format == 'xyxy':
        x1, y1, x2, y2 = bbox
    elif from_format == 'xywh':
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif from_format == 'xywhn':
        x, y, w, h = bbox
        x1 = x * width
        y1 = y * height
        x2 = (x + w) * width
        y2 = (y + h) * height
    else:
        raise ValueError(f"Unsupported from_format: {from_format}")
    
    # Convert from xyxy to target format
    if to_format == 'xyxy':
        return [x1, y1, x2, y2]
    elif to_format == 'xywh':
        return [x1, y1, x2 - x1, y2 - y1]
    elif to_format == 'xywhn':
        return [(x1 / width), (y1 / height), ((x2 - x1) / width), ((y2 - y1) / height)]
    else:
        raise ValueError(f"Unsupported to_format: {to_format}")


def crop_image(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Crop an image using bounding box coordinates.
    
    Args:
        image: Input image as numpy array
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    
    return image[y1:y2, x1:x2]


def save_image_safely(image: np.ndarray, save_path: Union[str, Path]) -> bool:
    """
    Safely save an image to file.
    
    Args:
        image: Image to save as numpy array
        save_path: Path where to save the image
        
    Returns:
        True if save was successful, False otherwise
    """
    try:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        success = cv2.imwrite(str(path), image_bgr)
        return success and path.exists()
        
    except Exception:
        return False


def resize_image(image: np.ndarray, target_size: Union[int, Tuple[int, int]], 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize an image to target size.
    
    Args:
        image: Input image as numpy array
        target_size: Target size as (width, height) or single dimension
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    target_width, target_height = target_size
    
    if maintain_aspect:
        # Calculate scale to fit within target size
        height, width = image.shape[:2]
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Pad to exact target size if needed
        if new_width != target_width or new_height != target_height:
            # Create padded image
            padded = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
            
            # Center the resized image
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
            
            return padded
        
        return resized
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def calculate_image_hash(image_path: Union[str, Path]) -> str:
    """
    Calculate MD5 hash of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MD5 hash string
    """
    import hashlib
    
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def get_image_info(image_path: Union[str, Path]) -> Optional[dict]:
    """
    Get basic information about an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image info or None if failed
    """
    try:
        path = Path(image_path)
        if not path.exists():
            return None
        
        image = load_image_safely(path)
        if image is None:
            return None
        
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        file_size = path.stat().st_size
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'file_size': file_size,
            'format': path.suffix.lower()
        }
        
    except Exception:
        return None 