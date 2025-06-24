"""
Inference Engine for Referee Detection System

This module provides a unified interface for AI model inference operations,
including referee detection and signal classification. It handles model loading,
caching, and provides high-level inference methods with proper error handling.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import threading

import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Import with proper path handling
import sys
from pathlib import Path

# Add necessary paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config.settings import ModelConfig, DirectoryConfig
    from utils.image_utils import (
        load_image_safely, normalize_image_for_model, 
        add_padding_to_bbox, convert_bbox_format
    )
    from utils.validation_utils import validate_bbox, validate_signal_class
except ImportError:
    from app.config.settings import ModelConfig, DirectoryConfig
    from app.utils.image_utils import (
        load_image_safely, normalize_image_for_model, 
        add_padding_to_bbox, convert_bbox_format
    )
    from app.utils.validation_utils import validate_bbox, validate_signal_class

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model loading and caching with thread safety.
    Implements singleton pattern to ensure models are loaded only once.
    """
    
    _instance = None
    _lock = threading.Lock()
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_type: str) -> Optional[YOLO]:
        """
        Get a loaded model instance, loading it if necessary.
        
        Args:
            model_type: Type of model ('referee' or 'signal')
            
        Returns:
            Loaded YOLO model instance or None if loading failed
        """
        if model_type not in self._models:
            self._load_model(model_type)
        
        return self._models.get(model_type)
    
    def _load_model(self, model_type: str) -> bool:
        """
        Load a model from disk.
        
        Args:
            model_type: Type of model to load ('referee' or 'signal')
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            if model_type == 'referee':
                model_path = ModelConfig.REFEREE_MODEL_PATH
            elif model_type == 'signal':
                model_path = ModelConfig.SIGNAL_MODEL_PATH
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
            
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            logger.info(f"Loading {model_type} model from {model_path}")
            
            # Load model and move to appropriate device
            model = YOLO(str(model_path))
            model.to(ModelConfig.DEVICE)
            model.fuse()  # Optimize model for inference
            
            self._models[model_type] = model
            logger.info(f"Successfully loaded {model_type} model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            return False
    
    def unload_models(self):
        """Unload all models to free memory."""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models unloaded")


class InferenceEngine:
    """
    High-level inference engine for referee detection and signal classification.
    Provides unified interface for both detection tasks with proper error handling.
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.device = ModelConfig.DEVICE
        self.confidence_threshold = ModelConfig.CONFIDENCE_THRESHOLD
        self.model_size = ModelConfig.MODEL_SIZE
        
        logger.info(f"Inference engine initialized with device: {self.device}")
        logger.info("Models will be loaded on first use (lazy loading)")
    
    def detect_referee(self, 
                      image_path: str, 
                      save_crop: bool = False,
                      crop_save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect referee in an image and optionally save a crop.
        
        Args:
            image_path: Path to the image file
            save_crop: Whether to save the detected referee crop
            crop_save_path: Path where to save the crop (if save_crop is True)
            
        Returns:
            Dictionary containing detection results:
            - detected: bool - Whether a referee was detected
            - bbox: list - Bounding box coordinates [x1, y1, x2, y2] if detected
            - confidence: float - Detection confidence if detected
            - crop_path: str - Path to saved crop if save_crop is True
        """
        result = {
            'detected': False,
            'bbox': None,
            'confidence': 0.0,
            'crop_path': None
        }
        
        try:
            # Load and validate image
            image = load_image_safely(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return result
            
            # Get referee model
            model = self.model_manager.get_model('referee')
            if model is None:
                logger.error("Referee model not available")
                return result
            
            # Run inference
            results = model(image, conf=self.confidence_threshold, verbose=False)
            
            if not results or len(results) == 0:
                logger.debug("No detection results returned")
                return result
            
            detection_result = results[0]
            
            # Check if any referees were detected
            if (detection_result.boxes is not None and 
                len(detection_result.boxes) > 0):
                
                # Get the detection with highest confidence
                best_idx = 0
                if len(detection_result.boxes) > 1:
                    confidences = detection_result.boxes.conf.cpu().numpy()
                    best_idx = np.argmax(confidences)
                
                # Extract bounding box and confidence
                bbox_tensor = detection_result.boxes.xyxy[best_idx]
                confidence_tensor = detection_result.boxes.conf[best_idx]
                
                bbox = bbox_tensor.cpu().numpy().astype(int).tolist()
                confidence = float(confidence_tensor.cpu().numpy())
                
                # Validate bounding box
                if not validate_bbox(bbox, image.shape[:2]):
                    logger.warning(f"Invalid bounding box detected: {bbox}")
                    return result
                
                # Add padding to bounding box
                padded_bbox = add_padding_to_bbox(bbox, image.shape[:2])
                
                result.update({
                    'detected': True,
                    'bbox': padded_bbox,
                    'confidence': confidence
                })
                
                # Save crop if requested
                if save_crop and crop_save_path:
                    crop_success = self._save_referee_crop(
                        image, padded_bbox, crop_save_path
                    )
                    if crop_success:
                        result['crop_path'] = crop_save_path
                    else:
                        logger.warning(f"Failed to save crop to {crop_save_path}")
                
                logger.debug(f"Referee detected with confidence {confidence:.3f}")
            else:
                logger.debug("No referee detected in image")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during referee detection: {e}")
            return result
    
    def classify_signal(self, image_path: str) -> Dict[str, Any]:
        """
        Classify hand signal in a referee crop image.
        
        Args:
            image_path: Path to the referee crop image
            
        Returns:
            Dictionary containing classification results:
            - predicted_class: str - Predicted signal class name or None
            - confidence: float - Classification confidence
            - bbox_xywhn: list - Normalized bounding box coordinates or None
            - all_predictions: list - All predictions with confidences
        """
        result = {
            'predicted_class': None,
            'confidence': 0.0,
            'bbox_xywhn': None,
            'all_predictions': []
        }
        
        try:
            # Load and validate image
            image = load_image_safely(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return result
            
            # Get signal model
            model = self.model_manager.get_model('signal')
            if model is None:
                logger.error("Signal model not available")
                return result
            
            # Normalize image for model input
            normalized_image = normalize_image_for_model(image)
            
            # Run inference
            results = model(normalized_image, conf=self.confidence_threshold, verbose=False)
            
            if not results or len(results) == 0:
                logger.debug("No signal detection results returned")
                return result
            
            detection_result = results[0]
            
            # Check if any signals were detected
            if (detection_result.boxes is not None and 
                len(detection_result.boxes) > 0):
                
                # Get all predictions
                all_predictions = self._extract_all_predictions(detection_result)
                result['all_predictions'] = all_predictions
                
                if all_predictions:
                    # Use the prediction with highest confidence
                    best_prediction = all_predictions[0]
                    
                    result.update({
                        'predicted_class': best_prediction['class_name'],
                        'confidence': best_prediction['confidence'],
                        'bbox_xywhn': best_prediction['bbox_xywhn']
                    })
                    
                    logger.debug(f"Signal classified as {best_prediction['class_name']} "
                               f"with confidence {best_prediction['confidence']:.3f}")
            else:
                logger.debug("No hand signal detected in crop")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during signal classification: {e}")
            return result
    
    def _save_referee_crop(self, 
                          image: np.ndarray, 
                          bbox: List[int], 
                          save_path: str) -> bool:
        """
        Save a cropped referee image to disk.
        
        Args:
            image: Original image as numpy array
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            save_path: Path where to save the crop
            
        Returns:
            True if crop was saved successfully, False otherwise
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Crop the image
            cropped_image = image[y1:y2, x1:x2]
            
            # Validate crop dimensions
            if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0:
                logger.error("Invalid crop dimensions")
                return False
            
            # Resize to model size for consistency
            resized_crop = cv2.resize(cropped_image, 
                                    (self.model_size, self.model_size),
                                    interpolation=cv2.INTER_AREA)
            
            # Ensure save directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the crop
            success = cv2.imwrite(save_path, resized_crop)
            
            if success and Path(save_path).exists():
                logger.debug(f"Crop saved successfully to {save_path}")
                return True
            else:
                logger.error(f"Failed to save crop to {save_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving referee crop: {e}")
            return False
    
    def _extract_all_predictions(self, detection_result) -> List[Dict[str, Any]]:
        """
        Extract all predictions from detection results, sorted by confidence.
        
        Args:
            detection_result: YOLO detection result object
            
        Returns:
            List of prediction dictionaries sorted by confidence (highest first)
        """
        predictions = []
        
        try:
            boxes = detection_result.boxes
            class_indices = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            bbox_xywhn = boxes.xywhn.cpu().numpy()
            
            for i, (class_idx, confidence, bbox) in enumerate(
                zip(class_indices, confidences, bbox_xywhn)
            ):
                class_idx = int(class_idx)
                
                # Get class name
                if 0 <= class_idx < len(ModelConfig.SIGNAL_CLASSES):
                    class_name = ModelConfig.SIGNAL_CLASSES[class_idx]
                else:
                    class_name = f"unknown_{class_idx}"
                
                # Validate signal class
                if not validate_signal_class(class_name):
                    logger.warning(f"Invalid signal class detected: {class_name}")
                    continue
                
                predictions.append({
                    'class_name': class_name,
                    'class_id': class_idx,
                    'confidence': float(confidence),
                    'bbox_xywhn': bbox.tolist()
                })
            
            # Sort by confidence (highest first)
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error extracting predictions: {e}")
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'model_size': self.model_size,
            'models': {}
        }
        
        for model_type in ['referee', 'signal']:
            model = self.model_manager.get_model(model_type)
            if model is not None:
                info['models'][model_type] = {
                    'loaded': True,
                    'classes': list(model.names.values()) if hasattr(model, 'names') else [],
                    'num_classes': len(model.names) if hasattr(model, 'names') else 0
                }
            else:
                info['models'][model_type] = {
                    'loaded': False,
                    'classes': [],
                    'num_classes': 0
                }
        
        return info
    
    def update_confidence_threshold(self, threshold: float) -> bool:
        """
        Update the confidence threshold for detections.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
            
        Returns:
            True if threshold was updated successfully, False otherwise
        """
        if not (0.0 <= threshold <= 1.0):
            logger.error(f"Invalid confidence threshold: {threshold}")
            return False
        
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to {threshold}")
        return True 