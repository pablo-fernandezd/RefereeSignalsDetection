"""
Image Processing Service

This service handles all business logic related to image processing,
referee detection, and crop management. It separates business logic
from HTTP handling in the routes.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np

from ..models.referee_detector import detect_referee
from ..utils import (
    save_uploaded_file, is_duplicate_image, register_image_hash,
    clear_hash_cache, validate_image_upload
)
from ..config.settings import DirectoryConfig, ModelConfig

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass


class ImageService:
    """Service for handling image processing operations"""
    
    @staticmethod
    def process_uploaded_image(file) -> Dict[str, Any]:
        """
        Process an uploaded image for referee detection.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Dict containing processing results
            
        Raises:
            ImageProcessingError: If processing fails
        """
        try:
            # Save uploaded file
            filename = save_uploaded_file(file, DirectoryConfig.UPLOAD_FOLDER)
            upload_path = DirectoryConfig.UPLOAD_FOLDER / filename
            
            # Process for referee detection
            crop_filename = f"temp_crop_{filename}"
            crop_path = DirectoryConfig.CROPS_FOLDER / crop_filename
            
            result = detect_referee(str(upload_path), crop_save_path=str(crop_path))
            
            if result['detected']:
                return {
                    'success': True,
                    'filename': filename,
                    'crop_filename': crop_filename,
                    'crop_url': f'/api/referee_crop_image/{crop_filename}',
                    'bbox': result['bbox']
                }
            else:
                return {
                    'success': False,
                    'error': 'No referee detected',
                    'filename': filename
                }
                
        except Exception as e:
            logger.error(f"Error processing uploaded image: {e}")
            raise ImageProcessingError(f"Failed to process image: {str(e)}")
    
    @staticmethod
    def get_pending_images() -> Dict[str, Any]:
        """
        Get list of pending images for labeling.
        
        Returns:
            Dict containing list of pending images and count
        """
        try:
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg']:
                image_files.extend(DirectoryConfig.UPLOAD_FOLDER.glob(f'*{ext}'))
            
            # Filter out temporary and processed files
            filtered_files = []
            for img_path in image_files:
                name = img_path.name
                if not (name.startswith('temp_crop_') or 
                       name.startswith('referee_') or 
                       name.startswith('yt_signal_crop_')):
                    filtered_files.append(name)
            
            filtered_files.sort()
            return {
                'images': filtered_files,
                'count': len(filtered_files)
            }
            
        except Exception as e:
            logger.error(f"Error getting pending images: {e}")
            raise ImageProcessingError(f"Failed to get pending images: {str(e)}")
    
    @staticmethod
    def confirm_crop(original_filename: str, crop_filename: str, bbox: List[int]) -> Dict[str, Any]:
        """
        Confirm and save a referee crop for training.
        
        Args:
            original_filename: Name of the original image file
            crop_filename: Name of the crop file
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Dict containing confirmation results
        """
        try:
            # Check if the original image has been processed before
            original_path = DirectoryConfig.UPLOAD_FOLDER / original_filename
            if not original_path.exists():
                raise ImageProcessingError(f'Original image not found: {original_filename}')
            
            # Clear cache to ensure fresh data
            clear_hash_cache()
            
            # Check for duplicate
            if is_duplicate_image(original_path):
                logger.info(f"Duplicate image detected: {original_filename}")
                return {
                    'status': 'warning',
                    'action': 'duplicate_detected',
                    'message': 'This image has been processed before. It will not be saved to training data to avoid duplicates.',
                    'crop_filename_for_signal': crop_filename,
                    'duplicate': True
                }
            
            # Register the image hash to prevent future duplicates
            if register_image_hash(original_path):
                logger.info(f"Registered hash for image: {original_filename}")
            else:
                logger.warning(f"Failed to register hash for image: {original_filename}")
            
            # Save to training data
            ImageService._save_positive_referee_sample(original_path, bbox)
            
            logger.info(f"Saving referee crop for training: {crop_filename}")
            return {
                'status': 'ok',
                'crop_filename_for_signal': crop_filename,
                'message': 'Crop confirmed and saved for training',
                'duplicate': False
            }
            
        except Exception as e:
            logger.error(f"Error in confirm_crop: {e}")
            raise ImageProcessingError(f"Failed to confirm crop: {str(e)}")
    
    @staticmethod
    def create_manual_crop(original_filename: str, bbox: Optional[List[int]], 
                          class_id: int = 0, proceed_to_signal: bool = True) -> Dict[str, Any]:
        """
        Process a manual crop of an image.
        
        Args:
            original_filename: Name of the original image file
            bbox: Bounding box coordinates [x1, y1, x2, y2] or None for negative samples
            class_id: Class ID (0 for referee, -1 for none)
            proceed_to_signal: Whether to proceed to signal labeling
            
        Returns:
            Dict containing manual crop results
        """
        try:
            logger.info(f"Manual crop parameters: filename={original_filename}, bbox={bbox}, class_id={class_id}")
            
            original_path = DirectoryConfig.UPLOAD_FOLDER / original_filename
            if not original_path.exists():
                raise ImageProcessingError(f'Original image not found: {original_filename}')
            
            # Handle negative samples (no referee)
            if class_id == -1 or not bbox or len(bbox) == 0:
                return ImageService._save_negative_referee_sample(original_path)
            
            if len(bbox) != 4:
                raise ImageProcessingError('Invalid bbox parameter - must be [x1, y1, x2, y2]')
            
            # Check for duplicates
            clear_hash_cache()
            if is_duplicate_image(original_path):
                logger.info(f"Duplicate image detected: {original_filename}")
                return {
                    'status': 'warning',
                    'action': 'duplicate_detected',
                    'message': 'This image has been processed before. It will not be saved to training data to avoid duplicates.',
                    'duplicate': True
                }
            
            # Create crop
            crop_filename = ImageService._create_crop_from_bbox(original_path, bbox)
            
            # Register hash and save to training data
            register_image_hash(original_path)
            ImageService._save_positive_referee_sample(original_path, bbox)
            
            # Determine response based on workflow
            if proceed_to_signal:
                return {
                    'status': 'ok',
                    'crop_filename_for_signal': crop_filename,
                    'message': 'Manual crop created successfully - ready for signal detection'
                }
            else:
                return {
                    'status': 'ok',
                    'crop_filename': crop_filename,
                    'message': 'Manual crop saved to training data'
                }
                
        except Exception as e:
            logger.error(f"Error in create_manual_crop: {e}")
            raise ImageProcessingError(f"Failed to create manual crop: {str(e)}")
    
    @staticmethod
    def _save_positive_referee_sample(original_path: Path, bbox: List[int]) -> None:
        """Save original image as positive referee sample with YOLO annotation"""
        try:
            referee_training_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
            referee_training_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = original_path.stem
            extension = original_path.suffix
            timestamp = int(datetime.now().timestamp() * 1000)
            
            positive_filename = f"referee_positive_{timestamp}{extension}"
            positive_txt_filename = f"referee_positive_{timestamp}.txt"
            
            positive_path = referee_training_dir / positive_filename
            positive_label_path = referee_training_dir / positive_txt_filename
            
            # Ensure unique filename
            counter = 1
            while positive_path.exists():
                positive_filename = f"referee_positive_{timestamp}_{counter}{extension}"
                positive_txt_filename = f"referee_positive_{timestamp}_{counter}.txt"
                positive_path = referee_training_dir / positive_filename
                positive_label_path = referee_training_dir / positive_txt_filename
                counter += 1
            
            # Move original image to referee training
            shutil.move(str(original_path), str(positive_path))
            
            # Create YOLO annotation file
            ImageService._create_yolo_annotation(positive_path, positive_label_path, bbox)
            
            logger.info(f"Saved positive referee sample: {positive_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save positive referee sample: {e}")
            raise
    
    @staticmethod
    def _save_negative_referee_sample(original_path: Path) -> Dict[str, Any]:
        """Save original image as negative referee sample"""
        try:
            referee_training_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
            referee_training_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = original_path.stem
            extension = original_path.suffix
            timestamp = int(datetime.now().timestamp() * 1000)
            
            negative_filename = f"referee_negative_{timestamp}{extension}"
            negative_txt_filename = f"referee_negative_{timestamp}.txt"
            
            negative_path = referee_training_dir / negative_filename
            negative_label_path = referee_training_dir / negative_txt_filename
            
            # Ensure unique filename
            counter = 1
            while negative_path.exists():
                negative_filename = f"referee_negative_{timestamp}_{counter}{extension}"
                negative_txt_filename = f"referee_negative_{timestamp}_{counter}.txt"
                negative_path = referee_training_dir / negative_filename
                negative_label_path = referee_training_dir / negative_txt_filename
                counter += 1
            
            # Move image to referee training
            shutil.move(str(original_path), str(negative_path))
            
            # Create empty txt file for negative sample
            negative_label_path.touch()
            
            logger.info(f"Saved negative referee sample: {negative_filename}")
            
            return {
                'status': 'ok',
                'action': 'saved_as_negative',
                'message': f'Image saved as negative sample: {negative_filename}'
            }
            
        except Exception as e:
            logger.error(f"Failed to save negative referee sample: {e}")
            raise
    
    @staticmethod
    def _create_crop_from_bbox(original_path: Path, bbox: List[int]) -> str:
        """Create a crop from the original image using bounding box"""
        # Read the original image
        image = cv2.imread(str(original_path))
        if image is None:
            raise ImageProcessingError('Failed to load original image')
        
        # Extract and validate coordinates
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        height, width = image.shape[:2]
        
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            raise ImageProcessingError('Invalid bounding box coordinates')
        
        # Crop and resize
        cropped_image = image[y1:y2, x1:x2]
        resized_crop = cv2.resize(cropped_image, 
                                (ModelConfig.MODEL_SIZE, ModelConfig.MODEL_SIZE),
                                interpolation=cv2.INTER_AREA)
        
        # Generate crop filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = original_path.stem
        crop_filename = f"manual_crop_{base_name}_{timestamp}.jpg"
        crop_path = DirectoryConfig.CROPS_FOLDER / crop_filename
        
        # Ensure directory exists and save
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(crop_path), resized_crop)
        
        if not success or not crop_path.exists():
            raise ImageProcessingError('Failed to save cropped image')
        
        logger.info(f"Successfully created manual crop: {crop_filename}")
        return crop_filename
    
    @staticmethod
    def _create_yolo_annotation(image_path: Path, label_path: Path, bbox: List[int]) -> None:
        """Create YOLO format annotation file"""
        # Read image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            # Fallback annotation if image can't be read
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 1.0 1.0\n")
            return
        
        img_height, img_width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Convert to normalized YOLO format (center_x, center_y, width, height)
        center_x = ((x1 + x2) / 2) / img_width
        center_y = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Write annotation (class 0 for referee)
        with open(label_path, 'w') as f:
            f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")