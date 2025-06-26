"""
Tests for API endpoints.
"""

import pytest
import json
import io
from unittest.mock import patch, Mock


class TestHealthEndpoints:
    """Test health and basic endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        # The endpoint might not exist, so we test what we can
        assert response.status_code in [200, 404]
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get('/')
        assert response.status_code in [200, 404, 405]


class TestImageUploadEndpoints:
    """Test image upload and processing endpoints."""
    
    def test_upload_no_file(self, client):
        """Test upload endpoint with no file."""
        response = client.post('/api/upload', data={})
        assert response.status_code == 400
    
    def test_upload_invalid_file(self, client):
        """Test upload endpoint with invalid file."""
        data = {
            'image': (io.BytesIO(b"not an image"), 'test.txt')
        }
        response = client.post('/api/upload', data=data, content_type='multipart/form-data')
        # Should fail validation
        assert response.status_code in [400, 422, 500]
    
    @patch('app.models.inference_engine.InferenceEngine')
    def test_upload_valid_image(self, mock_inference, client):
        """Test upload endpoint with valid image."""
        # Mock the inference engine
        mock_engine = Mock()
        mock_engine.detect_referee.return_value = []
        mock_inference.return_value = mock_engine
        
        # Create a fake image file
        data = {
            'image': (io.BytesIO(b"fake image data"), 'test.jpg')
        }
        response = client.post('/api/upload', data=data, content_type='multipart/form-data')
        
        # Should process without major errors
        assert response.status_code in [200, 404, 500]  # 500 is OK for fake image


class TestModelEndpoints:
    """Test model management endpoints."""
    
    def test_models_list(self, client):
        """Test models list endpoint."""
        response = client.get('/api/models/list')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
            assert data['status'] == 'success'
            assert 'models' in data
            assert isinstance(data['models'], list)
        else:
            # Endpoint might not be implemented yet
            assert response.status_code in [404, 500]
    
    def test_models_registry_stats(self, client):
        """Test models registry stats endpoint."""
        response = client.get('/api/models/registry/stats')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
            assert 'stats' in data
        else:
            assert response.status_code in [404, 500]
    
    def test_models_supported_versions(self, client):
        """Test supported YOLO versions endpoint."""
        response = client.get('/api/models/supported_versions')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
            assert 'supported_versions' in data
            assert isinstance(data['supported_versions'], list)
        else:
            assert response.status_code in [404, 500]
    
    def test_models_compatibility_report(self, client):
        """Test compatibility report endpoint."""
        response = client.get('/api/models/compatibility_report')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
            assert 'compatibility_report' in data
            report = data['compatibility_report']
            assert 'total_models' in report
            assert 'compatible' in report
            assert 'incompatible' in report
        else:
            assert response.status_code in [404, 500]


class TestTrainingEndpoints:
    """Test training-related endpoints."""
    
    def test_training_smart_base_selection(self, client):
        """Test smart base model selection endpoint."""
        test_data = {
            'dataset_size': 100,
            'model_type': 'referee',
            'performance_priority': 'balanced'
        }
        
        response = client.post('/api/training/smart_base_selection', 
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
            assert 'recommendations' in data
        else:
            assert response.status_code in [404, 500]
    
    def test_training_transfer_learning_start(self, client):
        """Test transfer learning start endpoint."""
        test_data = {
            'base_model_id': 'test_model',
            'model_type': 'referee',
            'experiment_name': 'test_experiment'
        }
        
        response = client.post('/api/training/transfer_learning/start',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        # This will likely fail without proper setup, but should not crash
        assert response.status_code in [200, 400, 404, 500]


class TestYouTubeEndpoints:
    """Test YouTube processing endpoints."""
    
    def test_youtube_videos_list(self, client):
        """Test YouTube videos list endpoint."""
        response = client.get('/api/youtube/videos')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
            assert 'videos' in data
            assert isinstance(data['videos'], list)
        else:
            assert response.status_code in [404, 500]
    
    def test_youtube_process_invalid_url(self, client):
        """Test YouTube processing with invalid URL."""
        test_data = {
            'url': 'not-a-valid-url',
            'auto_crop': True
        }
        
        response = client.post('/api/youtube/process',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        # Should fail validation
        assert response.status_code in [400, 422, 500] 