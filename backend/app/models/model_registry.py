"""
Model Registry System for YOLO Models

This module provides comprehensive model management including:
- Model version tracking and metadata
- Downloads from Ultralytics Hub
- Custom model uploads
- Active model deployment for inference
- Model performance tracking
- Scalable model storage and organization
- Automatic YOLO version detection and validation
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

# Import YOLO detection utilities
try:
    from app.utils.model_detection import (
        YOLOVersionDetector, 
        ModelCompatibilityValidator, 
        detect_and_validate_model
    )
    MODEL_DETECTION_AVAILABLE = True
except ImportError:
    YOLOVersionDetector = None
    ModelCompatibilityValidator = None
    detect_and_validate_model = None
    MODEL_DETECTION_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Represents metadata for a model version."""
    
    def __init__(self, model_id: str, model_type: str, version: str):
        self.model_id = model_id
        self.model_type = model_type
        self.version = version
        self.created_at = datetime.now().isoformat()
        self.is_active = False
        self.source = 'unknown'  # 'ultralytics', 'upload', 'training', 'legacy'
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
        
        # New fields for YOLO version detection
        self.yolo_version = None
        self.yolo_architecture = None
        self.compatibility_status = 'unknown'  # 'compatible', 'incompatible', 'unknown', 'warning'
        self.validation_results = {}
        self.base_model_version = None  # For training: what base model this was trained from
    
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
            'last_used': self.last_used,
            'yolo_version': self.yolo_version,
            'yolo_architecture': self.yolo_architecture,
            'compatibility_status': self.compatibility_status,
            'validation_results': self.validation_results,
            'base_model_version': self.base_model_version
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
        
        # New fields
        metadata.yolo_version = data.get('yolo_version')
        metadata.yolo_architecture = data.get('yolo_architecture')
        metadata.compatibility_status = data.get('compatibility_status', 'unknown')
        metadata.validation_results = data.get('validation_results', {})
        metadata.base_model_version = data.get('base_model_version')
        
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
        # First, try to import existing models from legacy directory
        self._import_existing_models()
        
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
    
    def _import_existing_models(self):
        """Import existing models from legacy directories into the registry."""
        try:
            legacy_paths = [
                (self.base_dir / 'models' / 'bestRefereeDetection.pt', 'referee'),
                (self.base_dir / 'models' / 'bestSignalsDetection.pt', 'signal'),
                (self.base_dir / 'bestRefereeDetection.pt', 'referee'),
                (self.base_dir / 'bestSignalsDetection.pt', 'signal'),
                (self.base_dir.parent / 'models' / 'bestRefereeDetection.pt', 'referee'),
                (self.base_dir.parent / 'models' / 'bestSignalsDetection.pt', 'signal'),
                (self.base_dir.parent / 'bestRefereeDetection.pt', 'referee'),
                (self.base_dir.parent / 'bestSignalsDetection.pt', 'signal'),
            ]
            
            for model_path, model_type in legacy_paths:
                if model_path.exists():
                    try:
                        # Generate model ID for legacy model
                        model_id = f"{model_type}_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        version = "v1.0_legacy"
                        
                        # Create metadata
                        metadata = ModelMetadata(model_id, model_type, version)
                        metadata.source = 'legacy'
                        metadata.description = f"Imported legacy {model_type} detection model"
                        metadata.tags = ['legacy', 'imported', model_type]
                        
                        # Copy to registry
                        model_file = self.models_dir / f"{model_id}.pt"
                        shutil.copy2(model_path, model_file)
                        
                        metadata.file_path = model_file
                        metadata.file_size_mb = self._get_file_size_mb(model_file)
                        metadata.model_hash = self._calculate_file_hash(model_file)
                        
                        # Try to get model info and detect YOLO version
                        try:
                            if ULTRALYTICS_AVAILABLE:
                                model = YOLO(str(model_file))
                                metadata.ultralytics_info = {
                                    'nc': getattr(model.model, 'nc', 0) if hasattr(model, 'model') else 0,
                                    'names': getattr(model.model, 'names', {}) if hasattr(model, 'model') else {},
                                    'task': 'detect'
                                }
                                
                                # Detect YOLO version automatically
                                detection_results = self._detect_and_validate_model(model_file, model_type)
                                metadata.yolo_version = detection_results.get('yolo_version', 'unknown')
                                metadata.yolo_architecture = detection_results.get('yolo_architecture')
                                metadata.compatibility_status = detection_results.get('compatibility_status', 'unknown')
                                metadata.validation_results = detection_results.get('validation_results', {})
                                
                                # Add version to tags
                                if metadata.yolo_version and metadata.yolo_version != 'unknown':
                                    metadata.tags.append(metadata.yolo_version)
                                if metadata.yolo_architecture:
                                    metadata.tags.append(metadata.yolo_architecture)
                                
                                logger.info(f"Detected YOLO version {metadata.yolo_version} for legacy model {model_path}")
                                
                        except Exception as e:
                            logger.warning(f"Could not load model info for {model_path}: {e}")
                            # Fallback: try to detect from filename
                            if MODEL_DETECTION_AVAILABLE:
                                try:
                                    detection_results = self._detect_and_validate_model(model_file, model_type)
                                    metadata.yolo_version = detection_results.get('yolo_version', 'unknown')
                                    metadata.yolo_architecture = detection_results.get('yolo_architecture')
                                    metadata.compatibility_status = detection_results.get('compatibility_status', 'unknown')
                                    metadata.validation_results = detection_results.get('validation_results', {})
                                except Exception as e2:
                                    logger.warning(f"Version detection also failed: {e2}")
                                    metadata.yolo_version = 'unknown'
                                    metadata.compatibility_status = 'unknown'
                        
                        # Add to registry
                        if model_type not in self.models:
                            self.models[model_type] = []
                        
                        # Check if this specific legacy model already exists by checking file path
                        legacy_exists = any(
                            model.source == 'legacy' and 
                            model.file_path and 
                            model.file_path.exists() and
                            model.file_path.stat().st_size > 0
                            for model in self.models[model_type]
                        )
                        
                        if not legacy_exists:
                            self.models[model_type].append(metadata)
                            
                            # Set as active if no other active model
                            if not any(model.is_active for model in self.models[model_type]):
                                metadata.is_active = True
                                # Copy to active directory
                                active_model_path = self.active_dir / f"active_{model_type}_detection.pt"
                                shutil.copy2(model_file, active_model_path)
                            
                            logger.info(f"Imported legacy {model_type} model: {model_path}")
                    
                    except Exception as e:
                        logger.warning(f"Failed to import legacy model {model_path}: {e}")
            
            # Save registry after importing
            self._save_registry()
            
        except Exception as e:
            logger.error(f"Failed to import existing models: {e}")
    
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
            
            # Try to load model and get info
            try:
                if ULTRALYTICS_AVAILABLE:
                    model = YOLO(str(model_file))
                    metadata.ultralytics_info = {
                        'nc': getattr(model.model, 'nc', 0) if hasattr(model, 'model') else 0,
                        'names': getattr(model.model, 'names', {}) if hasattr(model, 'model') else {},
                        'task': 'detect'
                    }
                    
                    # Detect YOLO version automatically
                    detection_results = self._detect_and_validate_model(model_file, model_type)
                    metadata.yolo_version = detection_results.get('yolo_version', 'unknown')
                    metadata.yolo_architecture = detection_results.get('yolo_architecture')
                    metadata.compatibility_status = detection_results.get('compatibility_status', 'unknown')
                    metadata.validation_results = detection_results.get('validation_results', {})
                    
                    # Add version to tags
                    if metadata.yolo_version and metadata.yolo_version != 'unknown':
                        if metadata.yolo_version not in metadata.tags:
                            metadata.tags.append(metadata.yolo_version)
                    if metadata.yolo_architecture:
                        if metadata.yolo_architecture not in metadata.tags:
                            metadata.tags.append(metadata.yolo_architecture)
                    
                    logger.info(f"Detected YOLO version {metadata.yolo_version} for uploaded model")
                    
            except Exception as e:
                logger.warning(f"Could not load model info: {e}")
                # Fallback: try to detect from filename only
                if MODEL_DETECTION_AVAILABLE:
                    try:
                        detection_results = self._detect_and_validate_model(model_file, model_type)
                        metadata.yolo_version = detection_results.get('yolo_version', 'unknown')
                        metadata.yolo_architecture = detection_results.get('yolo_architecture')
                        metadata.compatibility_status = detection_results.get('compatibility_status', 'unknown')
                        metadata.validation_results = detection_results.get('validation_results', {})
                    except Exception as e2:
                        logger.warning(f"Version detection also failed: {e2}")
                        metadata.yolo_version = 'unknown'
                        metadata.compatibility_status = 'unknown'
            
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
                    
                    # Detect YOLO version automatically
                    detection_results = self._detect_and_validate_model(model_file, model_type)
                    metadata.yolo_version = detection_results.get('yolo_version', 'unknown')
                    metadata.yolo_architecture = detection_results.get('yolo_architecture')
                    metadata.compatibility_status = detection_results.get('compatibility_status', 'unknown')
                    metadata.validation_results = detection_results.get('validation_results', {})
                    
                    # Add version to tags
                    if metadata.yolo_version and metadata.yolo_version != 'unknown':
                        if metadata.yolo_version not in metadata.tags:
                            metadata.tags.append(metadata.yolo_version)
                    if metadata.yolo_architecture:
                        if metadata.yolo_architecture not in metadata.tags:
                            metadata.tags.append(metadata.yolo_architecture)
                    
                    logger.info(f"Detected YOLO version {metadata.yolo_version} for uploaded model")
                    
            except Exception as e:
                logger.warning(f"Could not load model info: {e}")
                # Fallback: try to detect from filename only
                if MODEL_DETECTION_AVAILABLE:
                    try:
                        detection_results = self._detect_and_validate_model(model_file, model_type)
                        metadata.yolo_version = detection_results.get('yolo_version', 'unknown')
                        metadata.yolo_architecture = detection_results.get('yolo_architecture')
                        metadata.compatibility_status = detection_results.get('compatibility_status', 'unknown')
                        metadata.validation_results = detection_results.get('validation_results', {})
                    except Exception as e2:
                        logger.warning(f"Version detection also failed: {e2}")
                        metadata.yolo_version = 'unknown'
                        metadata.compatibility_status = 'unknown'
            
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
    
    def update_model_metadata(self, model_id: str, description: Optional[str] = None, 
                            tags: Optional[List[str]] = None, 
                            performance_metrics: Optional[Dict[str, Any]] = None,
                            training_config: Optional[Dict[str, Any]] = None,
                            version: Optional[str] = None) -> Tuple[bool, str]:
        """Update model metadata (description, tags, metrics, version, etc.)."""
        try:
            for models in self.models.values():
                for model in models:
                    if model.model_id == model_id:
                        # Update fields if provided
                        if description is not None:
                            model.description = description
                        if tags is not None:
                            model.tags = tags
                        if performance_metrics is not None:
                            model.performance_metrics.update(performance_metrics)
                        if training_config is not None:
                            model.training_config.update(training_config)
                        if version is not None:
                            model.version = version
                        
                        # Update modified timestamp
                        model.last_used = datetime.now().isoformat()
                        
                        self._save_registry()
                        logger.info(f"Updated metadata for model {model_id}")
                        return True, "Model metadata updated successfully"
            
            return False, f"Model {model_id} not found"
        except Exception as e:
            logger.error(f"Failed to update model metadata: {e}")
            return False, str(e)
    
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
    
    def _detect_and_validate_model(self, model_path: Path, model_type: str) -> Dict[str, Any]:
        """
        Detect YOLO version and validate model compatibility.
        
        Args:
            model_path: Path to model file
            model_type: Expected model type
            
        Returns:
            Detection and validation results
        """
        if not MODEL_DETECTION_AVAILABLE:
            logger.warning("Model detection utilities not available")
            return {
                'yolo_version': 'unknown',
                'yolo_architecture': None,
                'compatibility_status': 'unknown',
                'validation_results': {'error': 'Detection utilities not available'}
            }
        
        try:
            # Run detection and validation
            results = detect_and_validate_model(model_path, model_type)
            
            # Extract key information
            version_info = results.get('version_detection', {})
            validation_info = results.get('compatibility_validation', {})
            summary = results.get('summary', {})
            
            # Determine compatibility status
            compatibility_status = 'unknown'
            if summary.get('is_valid', False):
                compatibility_status = 'compatible'
            elif summary.get('is_compatible', False):
                compatibility_status = 'warning'  # Compatible but with issues
            elif summary.get('main_issues'):
                compatibility_status = 'incompatible'
            
            return {
                'yolo_version': summary.get('detected_version', 'unknown'),
                'yolo_architecture': version_info.get('architecture'),
                'compatibility_status': compatibility_status,
                'validation_results': {
                    'confidence': summary.get('confidence', 0.0),
                    'issues': summary.get('main_issues', []),
                    'recommendations': summary.get('recommendations', []),
                    'model_info': validation_info.get('model_info', {}),
                    'performance_test': validation_info.get('performance_test', {}),
                    'detection_details': version_info
                }
            }
            
        except Exception as e:
            logger.error(f"Model detection failed: {e}")
            return {
                'yolo_version': 'unknown',
                'yolo_architecture': None,
                'compatibility_status': 'unknown',
                'validation_results': {'error': str(e)}
            }
    
    def validate_model_for_training(self, model_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate if a model can be used for training.
        
        Args:
            model_id: ID of the model to validate
            
        Returns:
            Tuple of (can_train, message, validation_details)
        """
        # Find the model
        model_metadata = None
        for models in self.models.values():
            for model in models:
                if model.model_id == model_id:
                    model_metadata = model
                    break
            if model_metadata:
                break
        
        if not model_metadata:
            return False, f"Model {model_id} not found", {}
        
        if not model_metadata.file_path or not model_metadata.file_path.exists():
            return False, f"Model file not found: {model_metadata.file_path}", {}
        
        # Check compatibility status
        if model_metadata.compatibility_status == 'incompatible':
            issues = model_metadata.validation_results.get('issues', [])
            return False, f"Model is incompatible: {'; '.join(issues)}", model_metadata.validation_results
        
        # Check if it's a supported YOLO version
        if MODEL_DETECTION_AVAILABLE:
            supported_versions = YOLOVersionDetector.get_supported_versions()
            if model_metadata.yolo_version and model_metadata.yolo_version not in supported_versions:
                return False, f"YOLO version '{model_metadata.yolo_version}' is not supported for training", {}
        
        # Additional validation for training
        validation_results = model_metadata.validation_results
        performance_test = validation_results.get('performance_test', {})
        
        if not performance_test.get('success', False):
            return False, f"Model failed performance test: {performance_test.get('error', 'unknown error')}", validation_results
        
        # Check if model can be loaded for training
        try:
            if ULTRALYTICS_AVAILABLE:
                model = YOLO(str(model_metadata.file_path))
                # Test that we can access model for training
                if not hasattr(model, 'model') or model.model is None:
                    return False, "Model structure is not accessible for training", {}
        except Exception as e:
            return False, f"Cannot load model for training: {str(e)}", {}
        
        return True, "Model is ready for training", validation_results
    
    def update_model_yolo_version(self, model_id: str, yolo_version: str, yolo_architecture: str = None) -> Tuple[bool, str]:
        """
        Manually update the YOLO version information for a model.
        
        Args:
            model_id: ID of the model to update
            yolo_version: YOLO version (e.g., 'yolov8', 'yolo11', 'yolov12')
            yolo_architecture: Specific architecture (e.g., 'yolov8n', 'yolo11s')
            
        Returns:
            Tuple of (success, message)
        """
        # Find the model
        model_metadata = None
        for models in self.models.values():
            for model in models:
                if model.model_id == model_id:
                    model_metadata = model
                    break
            if model_metadata:
                break
        
        if not model_metadata:
            return False, f"Model {model_id} not found"
        
        # Validate the version
        if MODEL_DETECTION_AVAILABLE:
            version_info = YOLOVersionDetector.get_version_info(yolo_version)
            if not version_info:
                return False, f"Unknown YOLO version: {yolo_version}"
            
            # Check if architecture matches version
            if yolo_architecture and yolo_architecture not in version_info.get('architecture_names', []):
                return False, f"Architecture {yolo_architecture} is not valid for {yolo_version}"
        
        # Update metadata
        old_version = model_metadata.yolo_version
        model_metadata.yolo_version = yolo_version
        model_metadata.yolo_architecture = yolo_architecture
        
        # Update compatibility status based on new version
        if MODEL_DETECTION_AVAILABLE:
            version_info = YOLOVersionDetector.get_version_info(yolo_version)
            if version_info and version_info.get('supported', False):
                model_metadata.compatibility_status = 'compatible'
            else:
                model_metadata.compatibility_status = 'warning'
        
        # Add to tags if not already there
        if yolo_version not in model_metadata.tags:
            model_metadata.tags.append(yolo_version)
        if yolo_architecture and yolo_architecture not in model_metadata.tags:
            model_metadata.tags.append(yolo_architecture)
        
        # Save changes
        self._save_registry()
        
        logger.info(f"Updated model {model_id} YOLO version from {old_version} to {yolo_version}")
        return True, f"YOLO version updated to {yolo_version}"


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