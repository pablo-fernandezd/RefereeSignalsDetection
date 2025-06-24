"""
Referee Detector Wrapper

This module provides a simplified interface for referee detection,
maintaining compatibility with existing code while using the new inference engine.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class RefereeDetector:
    """
    Simplified interface for referee detection operations.
    Wraps the InferenceEngine to maintain compatibility with existing code.
    """
    
    def __init__(self):
        self.inference_engine = InferenceEngine()
        logger.info("Referee detector initialized")
    
    def detect_referee(self, 
                      image_path: str, 
                      crop_save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect referee in an image and optionally save a crop.
        
        This method maintains compatibility with the original detect_referee function
        from the legacy inference.py file.
        
        Args:
            image_path: Path to the image file
            crop_save_path: Optional path where to save the detected referee crop
            
        Returns:
            Dictionary containing detection results:
            - detected: bool - Whether a referee was detected
            - bbox: list - Bounding box coordinates [x1, y1, x2, y2] if detected
            - crop_path: str - Path to saved crop if crop_save_path provided
        """
        try:
            # Determine if we should save crop
            save_crop = crop_save_path is not None
            
            # Run detection
            result = self.inference_engine.detect_referee(
                image_path=image_path,
                save_crop=save_crop,
                crop_save_path=crop_save_path
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in referee detection: {e}")
            return {'detected': False, 'bbox': None, 'crop_path': None}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the referee detection model.
        
        Returns:
            Dictionary containing model information
        """
        info = self.inference_engine.get_model_info()
        return info.get('models', {}).get('referee', {})
    
    def update_confidence_threshold(self, threshold: float) -> bool:
        """
        Update the confidence threshold for referee detection.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
            
        Returns:
            True if threshold was updated successfully, False otherwise
        """
        return self.inference_engine.update_confidence_threshold(threshold)


# Global singleton instance for efficient reuse
_global_detector = None

def _get_detector():
    """Get the global detector instance (lazy initialization)."""
    global _global_detector
    if _global_detector is None:
        _global_detector = RefereeDetector()
    return _global_detector

# Legacy function for backward compatibility
def detect_referee(image_path: str, crop_save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Legacy function for referee detection - maintains backward compatibility.
    
    Args:
        image_path: Path to the image file
        crop_save_path: Optional path where to save the detected referee crop
        
    Returns:
        Dictionary containing detection results
    """
    detector = _get_detector()
    return detector.detect_referee(image_path, crop_save_path) 