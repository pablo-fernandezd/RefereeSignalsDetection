"""
Pytest configuration and shared fixtures for the Referee Detection System tests.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set test environment variables
os.environ['TESTING'] = 'True'
os.environ['FLASK_ENV'] = 'testing'


@pytest.fixture(scope="session")
def app():
    """Create and configure a test Flask app."""
    try:
        from app.main import create_app
        app = create_app()
        app.config.update({
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret-key"
        })
        return app
    except ImportError:
        # Fallback for simple Flask app structure
        from app import app
        app.config['TESTING'] = True
        return app


@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create a test CLI runner for the Flask app."""
    return app.test_cli_runner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model():
    """Create a mock YOLO model for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = [Mock()]
    mock_model.predict.return_value[0].boxes = Mock()
    mock_model.predict.return_value[0].boxes.data = []
    return mock_model


@pytest.fixture
def sample_model_file(temp_dir):
    """Create a sample model file for testing."""
    model_file = temp_dir / "test_model.pt"
    model_file.write_bytes(b"fake model data")
    return model_file


@pytest.fixture
def mock_ultralytics():
    """Mock ultralytics YOLO for testing."""
    with patch('app.utils.model_detection.YOLO') as mock_yolo:
        mock_instance = Mock()
        mock_instance.model = Mock()
        mock_instance.model.nc = 1
        mock_instance.model.names = {0: 'referee'}
        mock_instance.task = 'detect'
        mock_instance.predict.return_value = [Mock()]
        mock_yolo.return_value = mock_instance
        yield mock_yolo


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Ensure test directories exist
    test_dirs = [
        Path("temp_uploads"),
        Path("test_data"),
        Path("test_models")
    ]
    
    for test_dir in test_dirs:
        test_dir.mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after test
    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True) 