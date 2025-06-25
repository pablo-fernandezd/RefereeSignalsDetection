"""
Model Registry System for YOLO Models

This module provides comprehensive model management including:
- Model version tracking and metadata
- Downloads from Ultralytics Hub
- Custom model uploads
- Active model deployment for inference
- Model performance tracking
- Scalable model storage and organization
"""

import os
import json
import logging
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    torch = None
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Represents metadata for a model version."""
    
    def __init__(self, model_id: str, model_type: str, version: str):
        self.model_id = model_id
        self.model_type = model_type
        self.version = version
        self.created_at = datetime.now().isoformat()
        self.is_active = False
        self.source = 'unknown'  # 'ultralytics', 'upload', 'training'
        self.file_path = None
        self.file_size_mb = 0
        self.model_hash = None
        self.performance_metrics = {}
        self.training_config = {}
        self.ultralytics_info = {}
        self.tags = []
        self.description = ""
        self.download_count = 0
        self.last_used = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'version': self.version,
            'created_at': self.created_at,
            'is_active': self.is_active,
            'source': self.source,
            'file_path': str(self.file_path) if self.file_path else None,
            'file_size_mb': self.file_size_mb,
            'model_hash': self.model_hash,
            'performance_metrics': self.performance_metrics,
            'training_config': self.training_config,
            'ultralytics_info': self.ultralytics_info,
            'tags': self.tags,
            'description': self.description,
            'download_count': self.download_count,
            'last_used': self.last_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        metadata = cls(data['model_id'], data['model_type'], data['version'])
        metadata.created_at = data.get('created_at', metadata.created_at)
        metadata.is_active = data.get('is_active', False)
        metadata.source = data.get('source', 'unknown')
        metadata.file_path = Path(data['file_path']) if data.get('file_path') else None
        metadata.file_size_mb = data.get('file_size_mb', 0)
        metadata.model_hash = data.get('model_hash')
        metadata.performance_metrics = data.get('performance_metrics', {})
        metadata.training_config = data.get('training_config', {})
        metadata.ultralytics_info = data.get('ultralytics_info', {})
        metadata.tags = data.get('tags', [])
        metadata.description = data.get('description', "")
        metadata.download_count = data.get('download_count', 0)
        metadata.last_used = data.get('last_used')
        return metadata


class UltralyticsDownloader:
    """Handles downloading models from Ultralytics."""
    
    AVAILABLE_MODELS = {
        'yolov8n': {'size': 'nano', 'params': '3.2M', 'description': 'Fastest, lowest accuracy'},
        'yolov8s': {'size': 'small', 'params': '11.2M', 'description': 'Good speed/accuracy balance'},
        'yolov8m': {'size': 'medium', 'params': '25.9M', 'description': 'Higher accuracy'},
        'yolov8l': {'size': 'large', 'params': '43.7M', 'description': 'High accuracy'},
        'yolov8x': {'size': 'extra_large', 'params': '68.2M', 'description': 'Highest accuracy'},
        'yolov9c': {'size': 'compact', 'params': '25.5M', 'description': 'YOLOv9 compact'},
        'yolov9e': {'size': 'extended', 'params': '58.1M', 'description': 'YOLOv9 extended'},
        'yolov10n': {'size': 'nano', 'params': '2.3M', 'description': 'YOLOv10 nano'},
        'yolov10s': {'size': 'small', 'params': '7.2M', 'description': 'YOLOv10 small'},
        'yolov10m': {'size': 'medium', 'params': '15.4M', 'description': 'YOLOv10 medium'},
        'yolov10l': {'size': 'large', 'params': '24.4M', 'description': 'YOLOv10 large'},
        'yolov10x': {'size': 'extra_large', 'params': '29.5M', 'description': 'YOLOv10 extra large'},
        'yolo11n': {'size': 'nano', 'params': '2.6M', 'description': 'YOLO11 nano'},
        'yolo11s': {'size': 'small', 'params': '9.4M', 'description': 'YOLO11 small'},
        'yolo11m': {'size': 'medium', 'params': '20.1M', 'description': 'YOLO11 medium'},
        'yolo11l': {'size': 'large', 'params': '25.3M', 'description': 'YOLO11 large'},
        'yolo11x': {'size': 'extra_large', 'params': '56.9M', 'description': 'YOLO11 extra large'}
    }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get list of available Ultralytics models."""
        return cls.AVAILABLE_MODELS
    
    @classmethod
    def download_model(cls, model_name: str, save_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        """Download a model from Ultralytics."""
        if not ULTRALYTICS_AVAILABLE:
            return False, "Ultralytics not available", {}
        
        if model_name not in cls.AVAILABLE_MODELS:
            return False, f"Model {model_name} not available", {}
        
        try:
            logger.info(f"Downloading {model_name} from Ultralytics...")
            
            # Create YOLO model (this will download if not cached)
            model = YOLO(f"{model_name}.pt")
            
            # Get the actual model file path
            model_file = None
            if hasattr(model, 'ckpt_path'):
                model_file = Path(model.ckpt_path)
            elif hasattr(model, 'model_path'):
                model_file = Path(model.model_path)
            
            # Try to find the downloaded file in common locations
            if not model_file or not model_file.exists():
                import ultralytics
                possible_paths = [
                    Path.home() / '.cache' / 'ultralytics' / f"{model_name}.pt",
                    Path(ultralytics.__file__).parent / 'weights' / f"{model_name}.pt",
                    Path.cwd() / f"{model_name}.pt"
                ]
                
                for path in possible_paths:
                    if path.exists():
                        model_file = path
                        break
            
            if not model_file or not model_file.exists():
                return False, f"Could not locate downloaded model file for {model_name}", {}
            
            # Copy to target location
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_file, save_path)
            
            # Get model info
            model_info = {
                'architecture': model_name,
                'task': 'detect',
                'nc': getattr(model.model, 'nc', 80) if hasattr(model, 'model') else 80,
                'names': getattr(model.model, 'names', {}) if hasattr(model, 'model') else {},
                'ultralytics_version': getattr(ultralytics, '__version__', 'unknown') if 'ultralytics' in locals() else 'unknown',
                'original_size': cls.AVAILABLE_MODELS[model_name]
            }
            
            logger.info(f"Successfully downloaded {model_name} to {save_path}")
            return True, "Download successful", model_info
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False, str(e), {}


class ModelRegistry:
    """Central registry for managing all model versions."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.registry_dir = base_dir / 'model_registry'
        self.models_dir = self.registry_dir / 'models'
        self.active_dir = self.registry_dir / 'active'
        self.metadata_file = self.registry_dir / 'registry.json'
        
        # Create directories
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self.models: Dict[str, List[ModelMetadata]] = self._load_registry()
        
        # Initialize with default models if empty
        if not self.models:
            self._initialize_default_models()
    
    def _load_registry(self) -> Dict[str, List[ModelMetadata]]:
        """Load model registry from file."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            registry = {}
            for model_type, models_data in data.items():
                registry[model_type] = [
                    ModelMetadata.from_dict(model_data) 
                    for model_data in models_data
                ]
            
            logger.info(f"Loaded model registry with {sum(len(models) for models in registry.values())} models")
            return registry
            
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            return {}
    
    def _save_registry(self):
        """Save model registry to file."""
        try:
            data = {}
            for model_type, models in self.models.items():
                data[model_type] = [model.to_dict() for model in models]
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Model registry saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def _initialize_default_models(self):
        """Initialize registry with some default Ultralytics models."""
        default_models = [
            ('referee', 'yolov8n', 'Default nano model for referee detection'),
            ('referee', 'yolov8s', 'Default small model for referee detection'),
            ('signal', 'yolov8n', 'Default nano model for signal detection'),
            ('signal', 'yolov8s', 'Default small model for signal detection')
        ]
        
        for model_type, model_name, description in default_models:
            try:
                self.add_ultralytics_model(model_type, model_name, description, auto_download=False)
            except Exception as e:
                logger.warning(f"Failed to add default model {model_name}: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        return file_path.stat().st_size / (1024 * 1024)
    
    def add_ultralytics_model(self, model_type: str, model_name: str, description: str = "", auto_download: bool = True) -> Tuple[bool, str, Optional[str]]:
        """Add a model from Ultralytics."""
        try:
            # Generate model ID
            model_id = f"{model_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            version = f"v1.0_{model_name}"
            
            # Create metadata
            metadata = ModelMetadata(model_id, model_type, version)
            metadata.source = 'ultralytics'
            metadata.description = description or f"Ultralytics {model_name} model for {model_type} detection"
            metadata.tags = ['ultralytics', model_name, model_type]
            
            # Set file path
            model_file = self.models_dir / f"{model_id}.pt"
            metadata.file_path = model_file
            
            if auto_download:
                # Download the model
                success, message, model_info = UltralyticsDownloader.download_model(model_name, model_file)
                
                if not success:
                    return False, f"Download failed: {message}", None
                
                # Update metadata with model info
                metadata.ultralytics_info = model_info
                metadata.file_size_mb = self._get_file_size_mb(model_file)
                metadata.model_hash = self._calculate_file_hash(model_file)
            
            # Add to registry
            if model_type not in self.models:
                self.models[model_type] = []
            
            self.models[model_type].append(metadata)
            self._save_registry()
            
            logger.info(f"Added Ultralytics model {model_name} for {model_type}")
            return True, "Model added successfully", model_id
            
        except Exception as e:
            logger.error(f"Failed to add Ultralytics model: {e}")
            return False, str(e), None
    
    def upload_model(self, model_type: str, file_path: Path, description: str = "", tags: List[str] = None) -> Tuple[bool, str, Optional[str]]:
        """Upload a custom model file."""
        try:
            if not file_path.exists():
                return False, "Model file does not exist", None
            
            if file_path.suffix.lower() != '.pt':
                return False, "Only .pt files are supported", None
            
            # Generate model ID
            model_id = f"{model_type}_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            version = f"v1.0_custom"
            
            # Create metadata
            metadata = ModelMetadata(model_id, model_type, version)
            metadata.source = 'upload'
            metadata.description = description or f"Custom uploaded model for {model_type} detection"
            metadata.tags = tags or ['custom', 'upload', model_type]
            
            # Copy file to registry
            model_file = self.models_dir / f"{model_id}.pt"
            shutil.copy2(file_path, model_file)
            
            metadata.file_path = model_file
            metadata.file_size_mb = self._get_file_size_mb(model_file)
            metadata.model_hash = self._calculate_file_hash(model_file)
            
            # Try to load model and get info
            try:
                if ULTRALYTICS_AVAILABLE:
                    model = YOLO(str(model_file))
                    metadata.ultralytics_info = {
                        'nc': getattr(model.model, 'nc', 0) if hasattr(model, 'model') else 0,
                        'names': getattr(model.model, 'names', {}) if hasattr(model, 'model') else {},
                        'task': 'detect'
                    }
            except Exception as e:
                logger.warning(f"Could not load model info: {e}")
            
            # Add to registry
            if model_type not in self.models:
                self.models[model_type] = []
            
            self.models[model_type].append(metadata)
            self._save_registry()
            
            logger.info(f"Uploaded custom model for {model_type}")
            return True, "Model uploaded successfully", model_id
            
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return False, str(e), None
    
    def deploy_model(self, model_id: str) -> Tuple[bool, str]:
        """Deploy a model as active for inference."""
        try:
            # Find the model
            model_metadata = None
            model_type = None
            
            for mtype, models in self.models.items():
                for model in models:
                    if model.model_id == model_id:
                        model_metadata = model
                        model_type = mtype
                        break
                if model_metadata:
                    break
            
            if not model_metadata:
                return False, f"Model {model_id} not found"
            
            if not model_metadata.file_path or not model_metadata.file_path.exists():
                return False, f"Model file not found: {model_metadata.file_path}"
            
            # Deactivate all models of this type
            for model in self.models[model_type]:
                model.is_active = False
            
            # Activate the selected model
            model_metadata.is_active = True
            model_metadata.last_used = datetime.now().isoformat()
            
            # Copy to active directory with standard name
            active_model_path = self.active_dir / f"active_{model_type}_detection.pt"
            shutil.copy2(model_metadata.file_path, active_model_path)
            
            # Update legacy model paths for backward compatibility
            legacy_paths = [
                self.base_dir / 'models' / f"best{model_type.title()}Detection.pt",
                self.base_dir / f"best{model_type.title()}Detection.pt"
            ]
            
            for legacy_path in legacy_paths:
                legacy_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(model_metadata.file_path, legacy_path)
            
            self._save_registry()
            
            logger.info(f"Deployed model {model_id} as active {model_type} detector")
            return True, f"Model {model_id} deployed successfully"
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False, str(e)
    
    def get_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all models, optionally filtered by type."""
        models = []
        
        for mtype, model_list in self.models.items():
            if model_type and mtype != model_type:
                continue
            
            for model in model_list:
                models.append(model.to_dict())
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        return models
    
    def get_active_model(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get the active model for a type."""
        if model_type not in self.models:
            return None
        
        for model in self.models[model_type]:
            if model.is_active:
                return model.to_dict()
        
        return None
    
    def delete_model(self, model_id: str) -> Tuple[bool, str]:
        """Delete a model from the registry."""
        try:
            # Find and remove the model
            for model_type, models in self.models.items():
                for i, model in enumerate(models):
                    if model.model_id == model_id:
                        # Don't delete if it's the active model
                        if model.is_active:
                            return False, "Cannot delete active model. Deploy another model first."
                        
                        # Delete the file
                        if model.file_path and model.file_path.exists():
                            model.file_path.unlink()
                        
                        # Remove from registry
                        del models[i]
                        self._save_registry()
                        
                        logger.info(f"Deleted model {model_id}")
                        return True, "Model deleted successfully"
            
            return False, f"Model {model_id} not found"
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False, str(e)
    
    def get_model_file_path(self, model_type: str) -> Optional[Path]:
        """Get the file path of the active model for a type."""
        active_model_path = self.active_dir / f"active_{model_type}_detection.pt"
        
        if active_model_path.exists():
            return active_model_path
        
        # Fallback to legacy paths
        legacy_paths = [
            self.base_dir / 'models' / f"best{model_type.title()}Detection.pt",
            self.base_dir / f"best{model_type.title()}Detection.pt"
        ]
        
        for path in legacy_paths:
            if path.exists():
                return path
        
        return None
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, Any]) -> bool:
        """Update performance metrics for a model."""
        try:
            for models in self.models.values():
                for model in models:
                    if model.model_id == model_id:
                        model.performance_metrics.update(metrics)
                        self._save_registry()
                        return True
            return False
        except Exception as e:
            logger.error(f"Failed to update model metrics: {e}")
            return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the model registry."""
        stats = {
            'total_models': 0,
            'models_by_type': {},
            'models_by_source': {},
            'active_models': {},
            'total_size_mb': 0
        }
        
        for model_type, models in self.models.items():
            stats['models_by_type'][model_type] = len(models)
            stats['total_models'] += len(models)
            
            for model in models:
                # Count by source
                source = model.source
                stats['models_by_source'][source] = stats['models_by_source'].get(source, 0) + 1
                
                # Track active models
                if model.is_active:
                    stats['active_models'][model_type] = model.model_id
                
                # Sum file sizes
                stats['total_size_mb'] += model.file_size_mb
        
        return stats


# Global registry instance
_model_registry = None

def get_model_registry(base_dir: Optional[Path] = None) -> ModelRegistry:
    """Get or create the global model registry instance."""
    global _model_registry
    
    if _model_registry is None:
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        _model_registry = ModelRegistry(base_dir)
    
    return _model_registry


def initialize_model_registry(base_dir: Optional[Path] = None):
    """Initialize the model registry."""
    global _model_registry
    _model_registry = ModelRegistry(base_dir or Path(__file__).parent.parent.parent)
    logger.info("Model registry initialized") 