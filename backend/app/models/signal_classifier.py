"""
Signal Classifier Wrapper

This module provides a simplified interface for hand signal classification,
maintaining compatibility with existing code while using the new inference engine.
"""

import logging
from typing import Dict, Any
from pathlib import Path

from .inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class SignalClassifier:
    """
    Simplified interface for hand signal classification operations.
    Wraps the InferenceEngine to maintain compatibility with existing code.
    """
    
    def __init__(self):
        self.inference_engine = InferenceEngine()
        logger.info("Signal classifier initialized")
    
    def detect_signal(self, crop_path: str) -> Dict[str, Any]:
        """
        Detect and classify hand signal in a referee crop image.
        
        This method maintains compatibility with the original detect_signal function
        from the legacy inference.py file.
        
        Args:
            crop_path: Path to the referee crop image
            
        Returns:
            Dictionary containing classification results:
            - predicted_class: str - Predicted signal class name or None
            - confidence: float - Classification confidence
            - bbox_xywhn: list - Normalized bounding box coordinates or None
        """
        try:
            # Run signal classification
            result = self.inference_engine.classify_signal(crop_path)
            
            # Return in expected format for backward compatibility
            return {
                'predicted_class': result.get('predicted_class'),
                'confidence': result.get('confidence', 0.0),
                'bbox_xywhn': result.get('bbox_xywhn')
            }
            
        except Exception as e:
            logger.error(f"Error in signal classification: {e}")
            return {
                'predicted_class': None,
                'confidence': 0.0,
                'bbox_xywhn': None
            }
    
    def get_all_predictions(self, crop_path: str) -> Dict[str, Any]:
        """
        Get all signal predictions with their confidences.
        
        Args:
            crop_path: Path to the referee crop image
            
        Returns:
            Dictionary containing all classification results
        """
        try:
            result = self.inference_engine.classify_signal(crop_path)
            return result
            
        except Exception as e:
            logger.error(f"Error getting all predictions: {e}")
            return {
                'predicted_class': None,
                'confidence': 0.0,
                'bbox_xywhn': None,
                'all_predictions': []
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the signal classification model.
        
        Returns:
            Dictionary containing model information
        """
        info = self.inference_engine.get_model_info()
        return info.get('models', {}).get('signal', {})
    
    def update_confidence_threshold(self, threshold: float) -> bool:
        """
        Update the confidence threshold for signal classification.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
            
        Returns:
            True if threshold was updated successfully, False otherwise
        """
        return self.inference_engine.update_confidence_threshold(threshold)


# Global singleton instance for efficient reuse
_global_classifier = None

def _get_classifier():
    """Get the global classifier instance (lazy initialization)."""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = SignalClassifier()
    return _global_classifier

# Legacy function for backward compatibility
def detect_signal(crop_path: str) -> Dict[str, Any]:
    """
    Legacy function for signal detection - maintains backward compatibility.
    
    Args:
        crop_path: Path to the referee crop image
        
    Returns:
        Dictionary containing classification results
    """
    classifier = _get_classifier()
    return classifier.detect_signal(crop_path) 