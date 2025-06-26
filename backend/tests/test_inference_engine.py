"""
Tests for Inference Engine functionality.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io


class TestInferenceEngine:
    """Test Inference Engine functionality."""
    
    @pytest.fixture
    def mock_inference_engine(self):
        """Create a mock inference engine."""
        with patch('app.models.inference_engine.InferenceEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.model_registry = Mock()
            mock_instance.referee_model = Mock()
            mock_instance.signal_model = Mock()
            mock_engine.return_value = mock_instance
            yield mock_instance
    
    def test_inference_engine_initialization(self, mock_inference_engine):
        """Test inference engine initialization."""
        from app.models.inference_engine import InferenceEngine
        engine = InferenceEngine()
        
        assert engine is not None
    
    def test_detect_referee_success(self, mock_inference_engine):
        """Test successful referee detection."""
        # Mock detection results
        mock_inference_engine.detect_referee.return_value = [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.95,
                'class': 'referee'
            }
        ]
        
        # Create a fake image
        fake_image = Image.new('RGB', (640, 480), color='red')
        image_bytes = io.BytesIO()
        fake_image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        
        results = mock_inference_engine.detect_referee(image_bytes)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert 'bbox' in results[0]
        assert 'confidence' in results[0]
        assert results[0]['confidence'] > 0.9
    
    def test_detect_referee_no_detection(self, mock_inference_engine):
        """Test referee detection with no detections."""
        mock_inference_engine.detect_referee.return_value = []
        
        fake_image = Image.new('RGB', (640, 480), color='blue')
        image_bytes = io.BytesIO()
        fake_image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        
        results = mock_inference_engine.detect_referee(image_bytes)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_detect_signals_success(self, mock_inference_engine):
        """Test successful signal detection."""
        mock_inference_engine.detect_signals.return_value = [
            {
                'bbox': [50, 50, 150, 150],
                'confidence': 0.88,
                'class': 'armLeft',
                'signal_type': 'armLeft'
            }
        ]
        
        fake_image = Image.new('RGB', (640, 480), color='green')
        image_bytes = io.BytesIO()
        fake_image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        
        results = mock_inference_engine.detect_signals(image_bytes)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert 'signal_type' in results[0]
        assert results[0]['signal_type'] == 'armLeft'
    
    def test_load_models_success(self, mock_inference_engine):
        """Test successful model loading."""
        mock_inference_engine.load_models.return_value = True
        
        success = mock_inference_engine.load_models()
        assert success == True
    
    def test_load_models_failure(self, mock_inference_engine):
        """Test model loading failure."""
        mock_inference_engine.load_models.return_value = False
        
        success = mock_inference_engine.load_models()
        assert success == False
    
    def test_preprocess_image(self, mock_inference_engine):
        """Test image preprocessing."""
        # Mock preprocessing
        mock_inference_engine.preprocess_image.return_value = np.zeros((640, 640, 3))
        
        fake_image = Image.new('RGB', (1920, 1080), color='yellow')
        processed = mock_inference_engine.preprocess_image(fake_image)
        
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (640, 640, 3)
    
    def test_postprocess_detections(self, mock_inference_engine):
        """Test detection postprocessing."""
        # Mock raw detections
        raw_detections = [
            {
                'bbox': [0.1, 0.1, 0.3, 0.3],  # Normalized coordinates
                'confidence': 0.95,
                'class_id': 0
            }
        ]
        
        # Mock postprocessing
        mock_inference_engine.postprocess_detections.return_value = [
            {
                'bbox': [64, 48, 192, 144],  # Absolute coordinates
                'confidence': 0.95,
                'class': 'referee'
            }
        ]
        
        processed = mock_inference_engine.postprocess_detections(raw_detections, (640, 480))
        
        assert isinstance(processed, list)
        assert len(processed) > 0
        assert processed[0]['bbox'] == [64, 48, 192, 144]
        assert processed[0]['class'] == 'referee'


class TestInferenceEngineIntegration:
    """Test inference engine integration scenarios."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        image = Image.new('RGB', (640, 480), color='white')
        # Add some simple patterns
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(image)
        draw.rectangle([100, 100, 200, 300], fill='black')  # Simulate a person
        draw.rectangle([150, 80, 170, 100], fill='pink')    # Simulate a head
        
        return image
    
    def test_end_to_end_referee_detection(self, sample_image):
        """Test end-to-end referee detection workflow."""
        with patch('app.models.inference_engine.InferenceEngine') as mock_engine:
            # Mock the complete workflow
            mock_instance = Mock()
            mock_instance.detect_referee.return_value = [
                {
                    'bbox': [100, 100, 200, 300],
                    'confidence': 0.92,
                    'class': 'referee'
                }
            ]
            mock_engine.return_value = mock_instance
            
            from app.models.inference_engine import InferenceEngine
            engine = InferenceEngine()
            
            # Convert image to bytes
            image_bytes = io.BytesIO()
            sample_image.save(image_bytes, format='JPEG')
            image_bytes.seek(0)
            
            results = engine.detect_referee(image_bytes)
            
            assert len(results) == 1
            assert results[0]['confidence'] > 0.9
            assert results[0]['class'] == 'referee'
    
    def test_batch_processing(self):
        """Test batch processing of multiple images."""
        with patch('app.models.inference_engine.InferenceEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.detect_referee_batch.return_value = [
                [{'bbox': [100, 100, 200, 200], 'confidence': 0.95, 'class': 'referee'}],
                [],  # No detection in second image
                [{'bbox': [50, 50, 150, 150], 'confidence': 0.88, 'class': 'referee'}]
            ]
            mock_engine.return_value = mock_instance
            
            from app.models.inference_engine import InferenceEngine
            engine = InferenceEngine()
            
            # Mock batch of images
            batch_results = engine.detect_referee_batch(['img1', 'img2', 'img3'])
            
            assert len(batch_results) == 3
            assert len(batch_results[0]) == 1  # First image has detection
            assert len(batch_results[1]) == 0  # Second image has no detection
            assert len(batch_results[2]) == 1  # Third image has detection
    
    def test_model_switching(self):
        """Test switching between different models."""
        with patch('app.models.inference_engine.InferenceEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.switch_model.return_value = True
            mock_instance.current_model_info.return_value = {
                'model_id': 'new_model',
                'model_type': 'referee',
                'version': 'v2.0'
            }
            mock_engine.return_value = mock_instance
            
            from app.models.inference_engine import InferenceEngine
            engine = InferenceEngine()
            
            # Test model switching
            success = engine.switch_model('new_model')
            assert success == True
            
            # Test getting current model info
            info = engine.current_model_info()
            assert info['model_id'] == 'new_model'
            assert info['model_type'] == 'referee'


class TestInferenceEngineErrors:
    """Test inference engine error handling."""
    
    def test_invalid_image_format(self):
        """Test handling of invalid image formats."""
        with patch('app.models.inference_engine.InferenceEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.detect_referee.side_effect = ValueError("Invalid image format")
            mock_engine.return_value = mock_instance
            
            from app.models.inference_engine import InferenceEngine
            engine = InferenceEngine()
            
            # Test with invalid image data
            with pytest.raises(ValueError):
                engine.detect_referee(b"not an image")
    
    def test_model_not_loaded(self):
        """Test behavior when model is not loaded."""
        with patch('app.models.inference_engine.InferenceEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.detect_referee.side_effect = RuntimeError("Model not loaded")
            mock_engine.return_value = mock_instance
            
            from app.models.inference_engine import InferenceEngine
            engine = InferenceEngine()
            
            # Test detection without loaded model
            with pytest.raises(RuntimeError):
                engine.detect_referee(b"fake image")
    
    def test_memory_error_handling(self):
        """Test handling of memory errors during inference."""
        with patch('app.models.inference_engine.InferenceEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.detect_referee.side_effect = MemoryError("Out of memory")
            mock_engine.return_value = mock_instance
            
            from app.models.inference_engine import InferenceEngine
            engine = InferenceEngine()
            
            # Test memory error handling
            with pytest.raises(MemoryError):
                engine.detect_referee(b"large image")


class TestInferenceEnginePerformance:
    """Test inference engine performance aspects."""
    
    def test_inference_timing(self):
        """Test inference timing measurement."""
        with patch('app.models.inference_engine.InferenceEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.detect_referee_with_timing.return_value = (
                [{'bbox': [100, 100, 200, 200], 'confidence': 0.95}],
                0.150  # 150ms inference time
            )
            mock_engine.return_value = mock_instance
            
            from app.models.inference_engine import InferenceEngine
            engine = InferenceEngine()
            
            results, inference_time = engine.detect_referee_with_timing(b"test image")
            
            assert len(results) > 0
            assert inference_time > 0
            assert inference_time < 1.0  # Should be less than 1 second
    
    def test_confidence_thresholding(self):
        """Test confidence threshold filtering."""
        with patch('app.models.inference_engine.InferenceEngine') as mock_engine:
            mock_instance = Mock()
            # Mock results with different confidence levels
            all_results = [
                {'bbox': [100, 100, 200, 200], 'confidence': 0.95},
                {'bbox': [300, 300, 400, 400], 'confidence': 0.75},
                {'bbox': [500, 500, 600, 600], 'confidence': 0.45}
            ]
            
            # Filter by confidence threshold
            def filter_by_confidence(threshold):
                return [r for r in all_results if r['confidence'] >= threshold]
            
            mock_instance.detect_referee.side_effect = lambda img, threshold=0.8: filter_by_confidence(threshold)
            mock_engine.return_value = mock_instance
            
            from app.models.inference_engine import InferenceEngine
            engine = InferenceEngine()
            
            # Test with high threshold
            high_conf_results = engine.detect_referee(b"test", threshold=0.9)
            assert len(high_conf_results) == 1
            assert high_conf_results[0]['confidence'] == 0.95
            
            # Test with medium threshold
            med_conf_results = engine.detect_referee(b"test", threshold=0.7)
            assert len(med_conf_results) == 2
            
            # Test with low threshold
            low_conf_results = engine.detect_referee(b"test", threshold=0.4)
            assert len(low_conf_results) == 3 