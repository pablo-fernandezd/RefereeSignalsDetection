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
    """Manages data augmentation options and processing."""
    
    AVAILABLE_AUGMENTATIONS = {
        'horizontal_flip': {
            'name': 'Horizontal Flip',
            'description': 'Randomly flip images horizontally',
            'default': True
        },
        'vertical_flip': {
            'name': 'Vertical Flip', 
            'description': 'Randomly flip images vertically',
            'default': False
        },
        'rotation': {
            'name': 'Rotation',
            'description': 'Randomly rotate images (±15 degrees)',
            'default': True
        },
        'brightness': {
            'name': 'Brightness',
            'description': 'Randomly adjust brightness',
            'default': True
        },
        'contrast': {
            'name': 'Contrast',
            'description': 'Randomly adjust contrast',
            'default': True
        },
        'saturation': {
            'name': 'Saturation',
            'description': 'Randomly adjust color saturation',
            'default': False
        },
        'hue': {
            'name': 'Hue',
            'description': 'Randomly adjust hue',
            'default': False
        },
        'gaussian_blur': {
            'name': 'Gaussian Blur',
            'description': 'Apply random gaussian blur',
            'default': False
        },
        'noise': {
            'name': 'Gaussian Noise',
            'description': 'Add random gaussian noise',
            'default': False
        },
        'mosaic': {
            'name': 'Mosaic',
            'description': 'Combine 4 images into one (YOLO-style)',
            'default': True
        },
        'mixup': {
            'name': 'MixUp',
            'description': 'Blend two images together',
            'default': False
        },
        'cutmix': {
            'name': 'CutMix',
            'description': 'Cut and paste patches between images',
            'default': False
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