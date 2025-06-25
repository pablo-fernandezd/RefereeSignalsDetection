"""
Real Data Augmentation Engine for YOLO Training

This module implements the hybrid approach using:
- Albumentations for complex transformations
- YOLO built-in augmentations for basic transformations
- Real image processing during training pipeline
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
import yaml

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    A = None
    ToTensorV2 = None
    ALBUMENTATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlbumentationsAugmenter:
    """Handles complex augmentations using Albumentations library."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transform = self._build_transform()
    
    def _build_transform(self) -> Optional[A.Compose]:
        """Build Albumentations transform pipeline."""
        if not ALBUMENTATIONS_AVAILABLE:
            logger.warning("Albumentations not available, skipping complex augmentations")
            return None
        
        transforms = []
        
        # Geometric transformations
        if self.config.get('horizontal_flip', False):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if self.config.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=0.3))
        
        if self.config.get('rotation', False):
            transforms.append(A.Rotate(
                limit=15, 
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ))
        
        if self.config.get('shift_scale_rotate', False):
            transforms.append(A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ))
        
        # Color augmentations
        if self.config.get('brightness', False):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ))
        
        if self.config.get('hue_saturation', False):
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ))
        
        if self.config.get('channel_shuffle', False):
            transforms.append(A.ChannelShuffle(p=0.3))
        
        # Quality degradation
        if self.config.get('gaussian_blur', False):
            transforms.append(A.GaussianBlur(
                blur_limit=(3, 7),
                p=0.3
            ))
        
        if self.config.get('motion_blur', False):
            transforms.append(A.MotionBlur(
                blur_limit=7,
                p=0.3
            ))
        
        if self.config.get('gaussian_noise', False):
            transforms.append(A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=0.3
            ))
        
        if self.config.get('jpeg_compression', False):
            transforms.append(A.ImageCompression(
                quality_lower=85,
                quality_upper=100,
                p=0.3
            ))
        
        # Weather effects
        if self.config.get('rain', False):
            transforms.append(A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=7,
                brightness_coefficient=0.7,
                rain_type=None,
                p=0.2
            ))
        
        if self.config.get('fog', False):
            transforms.append(A.RandomFog(
                fog_coef_lower=0.3,
                fog_coef_upper=1,
                alpha_coef=0.08,
                p=0.2
            ))
        
        # Advanced augmentations
        if self.config.get('elastic_transform', False):
            transforms.append(A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03,
                p=0.3
            ))
        
        if self.config.get('grid_distortion', False):
            transforms.append(A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.3
            ))
        
        if self.config.get('coarse_dropout', False):
            transforms.append(A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ))
        
        if not transforms:
            return None
        
        # Always include bbox parameters for object detection
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            )
        )
    
    def augment(self, image: np.ndarray, bboxes: List[List[float]], class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """Apply augmentations to image and bounding boxes."""
        if self.transform is None:
            return image, bboxes, class_labels
        
        try:
            augmented = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            return (
                augmented['image'],
                augmented['bboxes'],
                augmented['class_labels']
            )
        
        except Exception as e:
            logger.warning(f"Albumentations augmentation failed: {e}")
            return image, bboxes, class_labels


class YOLOAugmenter:
    """Handles YOLO built-in augmentations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def get_yolo_augmentation_params(self) -> Dict[str, Any]:
        """Generate YOLO augmentation parameters for training."""
        params = {}
        
        # Basic augmentations
        if self.config.get('horizontal_flip', True):
            params['fliplr'] = 0.5
        
        if self.config.get('vertical_flip', False):
            params['flipud'] = 0.1
        
        if self.config.get('mosaic', True):
            params['mosaic'] = 1.0
        
        if self.config.get('mixup', False):
            params['mixup'] = 0.3
        
        if self.config.get('copy_paste', False):
            params['copy_paste'] = 0.3
        
        # Color augmentations
        if self.config.get('hsv_h', True):
            params['hsv_h'] = 0.015  # Hue augmentation
        
        if self.config.get('hsv_s', True):
            params['hsv_s'] = 0.7    # Saturation augmentation
        
        if self.config.get('hsv_v', True):
            params['hsv_v'] = 0.4    # Value augmentation
        
        # Geometric augmentations
        if self.config.get('degrees', True):
            params['degrees'] = 10.0  # Rotation degrees
        
        if self.config.get('translate', True):
            params['translate'] = 0.1  # Translation
        
        if self.config.get('scale', True):
            params['scale'] = 0.5      # Scale augmentation
        
        if self.config.get('shear', True):
            params['shear'] = 2.0      # Shear augmentation
        
        if self.config.get('perspective', True):
            params['perspective'] = 0.0002  # Perspective transformation
        
        return params


class DataAugmentationEngine:
    """Main engine that coordinates both Albumentations and YOLO augmentations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_albumentations = config.get('use_albumentations', True)
        self.use_yolo_augmentations = config.get('use_yolo_augmentations', True)
        
        # Initialize augmenters
        self.albumentations_augmenter = AlbumentationsAugmenter(config) if self.use_albumentations else None
        self.yolo_augmenter = YOLOAugmenter(config) if self.use_yolo_augmentations else None
        
        logger.info(f"Data augmentation engine initialized - "
                   f"Albumentations: {self.use_albumentations}, "
                   f"YOLO: {self.use_yolo_augmentations}")
    
    def process_training_data(self, data_dir: Path, output_dir: Path, augmentation_factor: int = 2) -> Dict[str, Any]:
        """Process training data with augmentations."""
        if not self.use_albumentations:
            logger.info("Skipping Albumentations preprocessing - will use YOLO built-in only")
            return self._get_yolo_config()
        
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(data_dir.glob(ext))
            
            if not image_files:
                logger.warning(f"No images found in {data_dir}")
                return self._get_yolo_config()
            
            processed_count = 0
            augmented_count = 0
            
            for img_file in image_files:
                # Copy original image
                original_output = output_dir / img_file.name
                if not original_output.exists():
                    import shutil
                    shutil.copy2(img_file, original_output)
                    processed_count += 1
                
                # Copy original label if exists
                label_file = img_file.with_suffix('.txt')
                if label_file.exists():
                    label_output = output_dir / label_file.name
                    if not label_output.exists():
                        shutil.copy2(label_file, label_output)
                
                # Generate augmented versions
                if self.albumentations_augmenter and self.albumentations_augmenter.transform:
                    for aug_idx in range(augmentation_factor - 1):  # -1 because we already have original
                        try:
                            augmented_img, augmented_labels = self._augment_image_with_labels(
                                img_file, label_file, aug_idx
                            )
                            
                            if augmented_img is not None:
                                # Save augmented image
                                aug_img_name = f"{img_file.stem}_aug{aug_idx}{img_file.suffix}"
                                aug_img_path = output_dir / aug_img_name
                                cv2.imwrite(str(aug_img_path), augmented_img)
                                
                                # Save augmented labels
                                if augmented_labels:
                                    aug_label_path = output_dir / f"{img_file.stem}_aug{aug_idx}.txt"
                                    with open(aug_label_path, 'w') as f:
                                        for label in augmented_labels:
                                            f.write(' '.join(map(str, label)) + '\n')
                                
                                augmented_count += 1
                        
                        except Exception as e:
                            logger.warning(f"Failed to augment {img_file.name}: {e}")
            
            result = {
                'processed_images': processed_count,
                'augmented_images': augmented_count,
                'total_images': processed_count + augmented_count,
                'output_dir': str(output_dir)
            }
            
            # Add YOLO configuration
            result.update(self._get_yolo_config())
            
            logger.info(f"Data augmentation complete: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Data augmentation failed: {e}")
            return self._get_yolo_config()
    
    def _augment_image_with_labels(self, img_file: Path, label_file: Path, aug_idx: int) -> Tuple[Optional[np.ndarray], Optional[List[List]]]:
        """Augment a single image with its labels."""
        try:
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                return None, None
            
            # Load labels if they exist
            bboxes = []
            class_labels = []
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
            
            # Apply augmentations
            if self.albumentations_augmenter:
                aug_image, aug_bboxes, aug_class_labels = self.albumentations_augmenter.augment(
                    image, bboxes, class_labels
                )
                
                # Convert back to YOLO format
                aug_labels = []
                for bbox, class_id in zip(aug_bboxes, aug_class_labels):
                    aug_labels.append([class_id] + list(bbox))
                
                return aug_image, aug_labels
            
            return image, [[class_labels[i]] + bboxes[i] for i in range(len(bboxes))]
        
        except Exception as e:
            logger.error(f"Failed to augment image {img_file}: {e}")
            return None, None
    
    def _get_yolo_config(self) -> Dict[str, Any]:
        """Get YOLO augmentation configuration."""
        yolo_config = {
            'yolo_augmentations': {},
            'use_yolo_augmentations': self.use_yolo_augmentations
        }
        
        if self.yolo_augmenter:
            yolo_config['yolo_augmentations'] = self.yolo_augmenter.get_yolo_augmentation_params()
        
        return yolo_config
    
    def get_augmentation_summary(self) -> Dict[str, Any]:
        """Get a summary of enabled augmentations."""
        summary = {
            'albumentations_enabled': self.use_albumentations and ALBUMENTATIONS_AVAILABLE,
            'yolo_enabled': self.use_yolo_augmentations,
            'enabled_augmentations': []
        }
        
        for key, value in self.config.items():
            if isinstance(value, bool) and value and key != 'use_albumentations' and key != 'use_yolo_augmentations':
                summary['enabled_augmentations'].append(key)
        
        return summary


def create_augmentation_engine(config: Dict[str, Any]) -> DataAugmentationEngine:
    """Factory function to create a data augmentation engine."""
    return DataAugmentationEngine(config)


def get_default_augmentation_config() -> Dict[str, Any]:
    """Get default augmentation configuration for the hybrid approach."""
    return {
        # Control flags
        'use_albumentations': True,
        'use_yolo_augmentations': True,
        
        # Basic augmentations (used by both)
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation': True,
        'brightness': True,
        'contrast': True,
        
        # Albumentations-specific
        'hue_saturation': True,
        'gaussian_blur': False,
        'motion_blur': False,
        'gaussian_noise': False,
        'jpeg_compression': False,
        'rain': False,
        'fog': False,
        'elastic_transform': False,
        'grid_distortion': False,
        'coarse_dropout': False,
        'shift_scale_rotate': False,
        'channel_shuffle': False,
        
        # YOLO-specific
        'mosaic': True,
        'mixup': False,
        'copy_paste': False,
        'hsv_h': True,
        'hsv_s': True,
        'hsv_v': True,
        'degrees': True,
        'translate': True,
        'scale': True,
        'shear': True,
        'perspective': True
    }


def validate_augmentation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize augmentation configuration."""
    default_config = get_default_augmentation_config()
    validated_config = {}
    
    for key, default_value in default_config.items():
        if key in config:
            # Ensure boolean values
            if isinstance(default_value, bool):
                validated_config[key] = bool(config[key])
            else:
                validated_config[key] = config[key]
        else:
            validated_config[key] = default_value
    
    return validated_config 