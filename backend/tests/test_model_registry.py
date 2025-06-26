"""
Tests for Model Registry functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from app.models.model_registry import ModelRegistry, ModelMetadata, UltralyticsDownloader


class TestModelMetadata:
    """Test ModelMetadata class."""
    
    def test_model_metadata_creation(self):
        """Test creating ModelMetadata instance."""
        metadata = ModelMetadata("test_id", "referee", "v1.0")
        
        assert metadata.model_id == "test_id"
        assert metadata.model_type == "referee"
        assert metadata.version == "v1.0"
        assert metadata.is_active == False
        assert metadata.source == 'unknown'
        assert metadata.yolo_version is None
        assert metadata.compatibility_status == 'unknown'
    
    def test_model_metadata_to_dict(self):
        """Test converting ModelMetadata to dictionary."""
        metadata = ModelMetadata("test_id", "referee", "v1.0")
        metadata.yolo_version = "yolov8"
        metadata.compatibility_status = "compatible"
        
        data = metadata.to_dict()
        
        assert isinstance(data, dict)
        assert data['model_id'] == "test_id"
        assert data['model_type'] == "referee"
        assert data['version'] == "v1.0"
        assert data['yolo_version'] == "yolov8"
        assert data['compatibility_status'] == "compatible"
    
    def test_model_metadata_from_dict(self):
        """Test creating ModelMetadata from dictionary."""
        data = {
            'model_id': 'test_id',
            'model_type': 'signal',
            'version': 'v2.0',
            'yolo_version': 'yolo11',
            'compatibility_status': 'compatible'
        }
        
        metadata = ModelMetadata.from_dict(data)
        
        assert metadata.model_id == "test_id"
        assert metadata.model_type == "signal"
        assert metadata.version == "v2.0"
        assert metadata.yolo_version == "yolo11"
        assert metadata.compatibility_status == "compatible"


class TestUltralyticsDownloader:
    """Test UltralyticsDownloader functionality."""
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = UltralyticsDownloader.get_available_models()
        
        assert isinstance(models, dict)
        assert 'yolov8n' in models
        assert 'yolov8s' in models
        assert 'yolo11n' in models
        
        # Check model info structure
        yolov8n_info = models['yolov8n']
        assert 'size' in yolov8n_info
        assert 'params' in yolov8n_info
        assert 'description' in yolov8n_info
    
    @patch('app.models.model_registry.YOLO')
    def test_download_model_success(self, mock_yolo, temp_dir):
        """Test successful model download."""
        # Mock YOLO model
        mock_model = Mock()
        mock_model.ckpt_path = temp_dir / "yolov8n.pt"
        mock_model.ckpt_path.touch()  # Create the file
        mock_yolo.return_value = mock_model
        
        save_path = temp_dir / "downloaded_model.pt"
        success, message, info = UltralyticsDownloader.download_model("yolov8n", save_path)
        
        assert success == True
        assert "successful" in message.lower()
        assert isinstance(info, dict)
        assert save_path.exists()
    
    def test_download_model_invalid(self, temp_dir):
        """Test downloading invalid model."""
        save_path = temp_dir / "invalid_model.pt"
        success, message, info = UltralyticsDownloader.download_model("invalid_model", save_path)
        
        assert success == False
        assert "not available" in message
        assert isinstance(info, dict)


class TestModelRegistry:
    """Test ModelRegistry functionality."""
    
    @pytest.fixture
    def registry(self, temp_dir):
        """Create a test model registry."""
        return ModelRegistry(temp_dir)
    
    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert registry.base_dir.exists()
        assert registry.registry_dir.exists()
        assert registry.models_dir.exists()
        assert registry.active_dir.exists()
        assert isinstance(registry.models, dict)
    
    def test_add_model_metadata(self, registry, sample_model_file):
        """Test adding model metadata."""
        # Get initial count
        initial_count = len(registry.models.get("referee", []))
        
        # Create metadata
        metadata = ModelMetadata("test_model", "referee", "v1.0")
        metadata.file_path = sample_model_file
        metadata.file_size_mb = 5.0
        metadata.yolo_version = "yolov8"
        metadata.compatibility_status = "compatible"
        
        # Add to registry
        if "referee" not in registry.models:
            registry.models["referee"] = []
        registry.models["referee"].append(metadata)
        
        # Verify
        assert len(registry.models["referee"]) == initial_count + 1
        assert registry.models["referee"][-1].model_id == "test_model"
    
    def test_get_models(self, registry):
        """Test getting all models."""
        models = registry.get_models()
        assert isinstance(models, list)
        
        # Add a test model
        metadata = ModelMetadata("test_model", "referee", "v1.0")
        if "referee" not in registry.models:
            registry.models["referee"] = []
        registry.models["referee"].append(metadata)
        
        models = registry.get_models()
        assert len(models) >= 1
        
        # Check model structure
        model_dict = models[0]
        assert 'model_id' in model_dict
        assert 'model_type' in model_dict
        assert 'version' in model_dict
    
    def test_get_models_by_type(self, registry):
        """Test getting models filtered by type."""
        # Add test models
        referee_metadata = ModelMetadata("referee_model", "referee", "v1.0")
        signal_metadata = ModelMetadata("signal_model", "signal", "v1.0")
        
        registry.models["referee"] = [referee_metadata]
        registry.models["signal"] = [signal_metadata]
        
        # Get referee models
        referee_models = registry.get_models("referee")
        assert len(referee_models) == 1
        assert referee_models[0]['model_type'] == "referee"
        
        # Get signal models
        signal_models = registry.get_models("signal")
        assert len(signal_models) == 1
        assert signal_models[0]['model_type'] == "signal"
    
    def test_deploy_model(self, registry, sample_model_file):
        """Test deploying a model as active."""
        # Add a test model
        metadata = ModelMetadata("test_model", "referee", "v1.0")
        metadata.file_path = sample_model_file
        
        registry.models["referee"] = [metadata]
        
        # Deploy the model
        success, message = registry.deploy_model("test_model")
        
        assert success == True
        assert "deployed" in message.lower()
        assert metadata.is_active == True
        
        # Check active model file was created
        active_file = registry.active_dir / "active_referee_detection.pt"
        assert active_file.exists()
    
    def test_delete_model(self, registry, sample_model_file):
        """Test deleting a model."""
        # Add a test model
        metadata = ModelMetadata("test_model", "referee", "v1.0")
        metadata.file_path = sample_model_file
        
        registry.models["referee"] = [metadata]
        
        # Delete the model
        success, message = registry.delete_model("test_model")
        
        assert success == True
        assert "deleted" in message.lower()
        assert len(registry.models["referee"]) == 0
    
    def test_update_model_metadata(self, registry):
        """Test updating model metadata."""
        # Add a test model
        metadata = ModelMetadata("test_model", "referee", "v1.0")
        registry.models["referee"] = [metadata]
        
        # Update metadata
        success, message = registry.update_model_metadata(
            "test_model",
            description="Updated description",
            tags=["new", "tags"],
            performance_metrics={"accuracy": 0.95}
        )
        
        assert success == True
        assert metadata.description == "Updated description"
        assert metadata.tags == ["new", "tags"]
        assert metadata.performance_metrics["accuracy"] == 0.95
    
    @patch('app.models.model_registry.MODEL_DETECTION_AVAILABLE', True)
    @patch('app.models.model_registry.detect_and_validate_model')
    def test_detect_and_validate_model(self, mock_detect, registry, sample_model_file):
        """Test model detection and validation."""
        # Mock detection results
        mock_detect.return_value = {
            'version_detection': {
                'detected_version': 'yolov8',
                'architecture': 'yolov8n',
                'confidence': 0.9
            },
            'compatibility_validation': {
                'is_valid': True,
                'is_compatible': True,
                'compatibility_issues': [],
                'model_info': {'task': 'detect'}
            },
            'summary': {
                'detected_version': 'yolov8',
                'is_supported': True,
                'is_compatible': True,
                'is_valid': True,
                'confidence': 0.9,
                'main_issues': [],
                'recommendations': []
            }
        }
        
        result = registry._detect_and_validate_model(sample_model_file, "referee")
        
        assert result['yolo_version'] == 'yolov8'
        assert result['yolo_architecture'] == 'yolov8n'
        assert result['compatibility_status'] == 'compatible'
        assert result['validation_results']['confidence'] == 0.9
    
    def test_validate_model_for_training(self, registry, sample_model_file):
        """Test validating model for training."""
        # Add a compatible test model
        metadata = ModelMetadata("test_model", "referee", "v1.0")
        metadata.file_path = sample_model_file
        metadata.compatibility_status = "compatible"
        metadata.yolo_version = "yolov8"
        metadata.validation_results = {
            'performance_test': {'success': True}
        }
        
        registry.models["referee"] = [metadata]
        
        # Mock YOLO loading
        with patch('app.models.model_registry.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_yolo.return_value = mock_model
            
            can_train, message, details = registry.validate_model_for_training("test_model")
            
            assert can_train == True
            assert "ready for training" in message.lower()
    
    def test_get_registry_stats(self, registry):
        """Test getting registry statistics."""
        # Add test models
        referee_metadata = ModelMetadata("referee_model", "referee", "v1.0")
        referee_metadata.is_active = True
        signal_metadata = ModelMetadata("signal_model", "signal", "v1.0")
        
        registry.models["referee"] = [referee_metadata]
        registry.models["signal"] = [signal_metadata]
        
        stats = registry.get_registry_stats()
        
        assert isinstance(stats, dict)
        assert stats['total_models'] == 2
        assert stats['models_by_type']['referee'] == 1
        assert stats['models_by_type']['signal'] == 1
        assert stats['active_models']['referee'] == "referee_model" 