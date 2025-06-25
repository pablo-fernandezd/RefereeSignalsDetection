"""
Dynamic Model Configuration Utility

This module handles dynamic model configurations, YAML file management,
and provides utilities for scalable model management.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ModelConfigManager:
    """Manages dynamic model configurations and YAML files."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / 'data'
        self.models_config_file = self.data_dir / 'models_config.json'
        self.models_config = self._load_models_config()
    
    def _load_models_config(self) -> Dict[str, Any]:
        """Load models configuration from file."""
        if self.models_config_file.exists():
            try:
                with open(self.models_config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load models config: {e}")
        
        # Default configuration
        return {
            'referee': {
                'type': 'detection',
                'data_dir': 'referee_training_data',
                'classes': ['referee'],
                'task': 'object_detection'
            },
            'signal': {
                'type': 'detection',
                'data_dir': 'signal_training_data', 
                'classes': ['armLeft', 'armRight', 'hits', 'leftServe', 'net', 'outside', 'rightServe', 'touched'],
                'task': 'signal_detection'
            }
        }
    
    def _save_models_config(self):
        """Save models configuration to file."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.models_config_file, 'w') as f:
                json.dump(self.models_config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save models config: {e}")
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get model information."""
        return self.models_config.get(model_type, {})
    
    def get_model_classes(self, model_type: str) -> List[str]:
        """Get classes for a model type."""
        model_info = self.get_model_info(model_type)
        
        # Try to load from YAML file first
        yaml_path = self.data_dir / model_info.get('data_dir', f'{model_type}_training_data') / 'data.yaml'
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'names' in yaml_data:
                        return yaml_data['names']
            except Exception as e:
                logger.warning(f"Failed to load classes from YAML: {e}")
        
        # Fallback to config
        return model_info.get('classes', [])
    
    def update_model_classes(self, model_type: str, classes: List[str]) -> bool:
        """Update classes for a model type."""
        try:
            # Update in memory config
            if model_type not in self.models_config:
                self.models_config[model_type] = {
                    'type': 'detection',
                    'data_dir': f'{model_type}_training_data',
                    'task': f'{model_type}_detection'
                }
            
            self.models_config[model_type]['classes'] = classes
            
            # Update YAML file if it exists
            model_info = self.get_model_info(model_type)
            yaml_path = self.data_dir / model_info.get('data_dir', f'{model_type}_training_data') / 'data.yaml'
            
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f) or {}
                
                yaml_data['names'] = classes
                yaml_data['nc'] = len(classes)
                
                with open(yaml_path, 'w') as f:
                    yaml.dump(yaml_data, f, default_flow_style=False)
                
                logger.info(f"Updated YAML file for {model_type} with {len(classes)} classes")
            
            # Save config
            self._save_models_config()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model classes: {e}")
            return False
    
    def get_data_directory(self, model_type: str) -> Path:
        """Get data directory for a model type."""
        model_info = self.get_model_info(model_type)
        data_dir_name = model_info.get('data_dir', f'{model_type}_training_data')
        return self.data_dir / data_dir_name
    
    def create_yaml_config(self, model_type: str, dataset_dir: Path, splits_info: Dict) -> Path:
        """Create YOLO dataset configuration file."""
        classes = self.get_model_classes(model_type)
        
        config = {
            'path': str(dataset_dir),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': len(classes),
            'names': classes
        }
        
        yaml_path = dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created YOLO config for {model_type} model with {len(classes)} classes")
        return yaml_path
    
    def add_new_model_type(self, model_type: str, config: Dict[str, Any]) -> bool:
        """Add a new model type configuration."""
        try:
            self.models_config[model_type] = {
                'type': config.get('type', 'detection'),
                'data_dir': config.get('data_dir', f'{model_type}_training_data'),
                'classes': config.get('classes', []),
                'task': config.get('task', f'{model_type}_detection')
            }
            
            self._save_models_config()
            logger.info(f"Added new model type: {model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add new model type: {e}")
            return False
    
    def get_all_model_types(self) -> List[str]:
        """Get all available model types."""
        return list(self.models_config.keys())

class DataAugmentationManager:
    """Manages data augmentation options and processing with real implementation."""
    
    AVAILABLE_AUGMENTATIONS = {
        # Basic augmentations (supported by both Albumentations and YOLO)
        'horizontal_flip': {
            'name': 'Horizontal Flip',
            'description': 'Randomly flip images horizontally',
            'default': True,
            'category': 'geometric',
            'supported_by': ['albumentations', 'yolo']
        },
        'vertical_flip': {
            'name': 'Vertical Flip', 
            'description': 'Randomly flip images vertically',
            'default': False,
            'category': 'geometric',
            'supported_by': ['albumentations', 'yolo']
        },
        'rotation': {
            'name': 'Rotation',
            'description': 'Randomly rotate images (±15 degrees)',
            'default': True,
            'category': 'geometric',
            'supported_by': ['albumentations', 'yolo']
        },
        'brightness': {
            'name': 'Brightness',
            'description': 'Randomly adjust brightness',
            'default': True,
            'category': 'color',
            'supported_by': ['albumentations', 'yolo']
        },
        'contrast': {
            'name': 'Contrast',
            'description': 'Randomly adjust contrast',
            'default': True,
            'category': 'color',
            'supported_by': ['albumentations', 'yolo']
        },
        
        # Advanced Albumentations-only augmentations
        'hue_saturation': {
            'name': 'Hue & Saturation',
            'description': 'Randomly adjust hue and saturation',
            'default': True,
            'category': 'color',
            'supported_by': ['albumentations']
        },
        'gaussian_blur': {
            'name': 'Gaussian Blur',
            'description': 'Apply random gaussian blur',
            'default': False,
            'category': 'quality',
            'supported_by': ['albumentations']
        },
        'motion_blur': {
            'name': 'Motion Blur',
            'description': 'Apply random motion blur',
            'default': False,
            'category': 'quality',
            'supported_by': ['albumentations']
        },
        'gaussian_noise': {
            'name': 'Gaussian Noise',
            'description': 'Add random gaussian noise',
            'default': False,
            'category': 'quality',
            'supported_by': ['albumentations']
        },
        'jpeg_compression': {
            'name': 'JPEG Compression',
            'description': 'Simulate JPEG compression artifacts',
            'default': False,
            'category': 'quality',
            'supported_by': ['albumentations']
        },
        'rain': {
            'name': 'Rain Effect',
            'description': 'Add realistic rain effect',
            'default': False,
            'category': 'weather',
            'supported_by': ['albumentations']
        },
        'fog': {
            'name': 'Fog Effect',
            'description': 'Add realistic fog effect',
            'default': False,
            'category': 'weather',
            'supported_by': ['albumentations']
        },
        'elastic_transform': {
            'name': 'Elastic Transform',
            'description': 'Apply elastic deformation',
            'default': False,
            'category': 'geometric',
            'supported_by': ['albumentations']
        },
        'grid_distortion': {
            'name': 'Grid Distortion',
            'description': 'Apply grid-based distortion',
            'default': False,
            'category': 'geometric',
            'supported_by': ['albumentations']
        },
        'coarse_dropout': {
            'name': 'Coarse Dropout',
            'description': 'Randomly mask rectangular regions',
            'default': False,
            'category': 'occlusion',
            'supported_by': ['albumentations']
        },
        'shift_scale_rotate': {
            'name': 'Shift Scale Rotate',
            'description': 'Combined shift, scale and rotate',
            'default': False,
            'category': 'geometric',
            'supported_by': ['albumentations']
        },
        'channel_shuffle': {
            'name': 'Channel Shuffle',
            'description': 'Randomly shuffle color channels',
            'default': False,
            'category': 'color',
            'supported_by': ['albumentations']
        },
        
        # YOLO-specific augmentations
        'mosaic': {
            'name': 'Mosaic',
            'description': 'Combine 4 images into one (YOLO-style)',
            'default': True,
            'category': 'composition',
            'supported_by': ['yolo']
        },
        'mixup': {
            'name': 'MixUp',
            'description': 'Blend two images together',
            'default': False,
            'category': 'composition',
            'supported_by': ['yolo']
        },
        'copy_paste': {
            'name': 'Copy Paste',
            'description': 'Copy objects between images',
            'default': False,
            'category': 'composition',
            'supported_by': ['yolo']
        },
        'hsv_h': {
            'name': 'HSV Hue',
            'description': 'YOLO HSV hue augmentation',
            'default': True,
            'category': 'color',
            'supported_by': ['yolo']
        },
        'hsv_s': {
            'name': 'HSV Saturation',
            'description': 'YOLO HSV saturation augmentation',
            'default': True,
            'category': 'color',
            'supported_by': ['yolo']
        },
        'hsv_v': {
            'name': 'HSV Value',
            'description': 'YOLO HSV value augmentation',
            'default': True,
            'category': 'color',
            'supported_by': ['yolo']
        },
        'degrees': {
            'name': 'Rotation Degrees',
            'description': 'YOLO rotation augmentation',
            'default': True,
            'category': 'geometric',
            'supported_by': ['yolo']
        },
        'translate': {
            'name': 'Translation',
            'description': 'YOLO translation augmentation',
            'default': True,
            'category': 'geometric',
            'supported_by': ['yolo']
        },
        'scale': {
            'name': 'Scale',
            'description': 'YOLO scale augmentation',
            'default': True,
            'category': 'geometric',
            'supported_by': ['yolo']
        },
        'shear': {
            'name': 'Shear',
            'description': 'YOLO shear augmentation',
            'default': True,
            'category': 'geometric',
            'supported_by': ['yolo']
        },
        'perspective': {
            'name': 'Perspective',
            'description': 'YOLO perspective transformation',
            'default': True,
            'category': 'geometric',
            'supported_by': ['yolo']
        }
    }
    
    @classmethod
    def get_available_augmentations(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available augmentation options."""
        return cls.AVAILABLE_AUGMENTATIONS
    
    @classmethod
    def get_default_augmentations(cls) -> Dict[str, bool]:
        """Get default augmentation settings."""
        return {
            key: info['default'] 
            for key, info in cls.AVAILABLE_AUGMENTATIONS.items()
        }
    
    @classmethod
    def validate_augmentation_config(cls, config: Dict[str, bool]) -> Dict[str, bool]:
        """Validate and filter augmentation configuration."""
        return {
            key: value 
            for key, value in config.items() 
            if key in cls.AVAILABLE_AUGMENTATIONS
        }
    
    @classmethod
    def get_augmentations_by_category(cls) -> Dict[str, List[str]]:
        """Get augmentations grouped by category."""
        categories = {}
        for key, info in cls.AVAILABLE_AUGMENTATIONS.items():
            category = info.get('category', 'other')
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        return categories
    
    @classmethod
    def get_augmentations_by_engine(cls, engine: str) -> List[str]:
        """Get augmentations supported by specific engine (albumentations/yolo)."""
        return [
            key for key, info in cls.AVAILABLE_AUGMENTATIONS.items()
            if engine in info.get('supported_by', [])
        ]
    
    @classmethod
    def create_real_augmentation_engine(cls, config: Dict[str, Any]):
        """Create a real data augmentation engine with the given configuration."""
        try:
            from app.utils.data_augmentation import create_augmentation_engine, validate_augmentation_config
            
            # Validate and prepare configuration
            validated_config = validate_augmentation_config(config)
            
            # Create and return the engine
            return create_augmentation_engine(validated_config)
            
        except ImportError as e:
            logger.error(f"Failed to import data augmentation engine: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create augmentation engine: {e}")
            return None
    
    @classmethod
    def get_hybrid_config_template(cls) -> Dict[str, Any]:
        """Get a template configuration for hybrid augmentation approach."""
        return {
            # Engine control
            'use_albumentations': True,
            'use_yolo_augmentations': True,
            
            # Basic augmentations (both engines)
            'horizontal_flip': True,
            'rotation': True,
            'brightness': True,
            'contrast': True,
            
            # Albumentations advanced features
            'hue_saturation': True,
            'gaussian_blur': False,
            'motion_blur': False,
            'gaussian_noise': False,
            'rain': False,
            'fog': False,
            'elastic_transform': False,
            'grid_distortion': False,
            'coarse_dropout': False,
            
            # YOLO-specific features
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

# Global instance
_config_manager = None

def get_config_manager(base_dir: Optional[Path] = None) -> ModelConfigManager:
    """Get or create the global config manager instance."""
    global _config_manager
    
    if _config_manager is None:
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        _config_manager = ModelConfigManager(base_dir)
    
    return _config_manager 