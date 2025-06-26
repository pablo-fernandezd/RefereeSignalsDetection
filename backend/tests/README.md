# Referee Detection System - Test Suite

This directory contains comprehensive unit tests for the Referee Detection System backend.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── test_api.py                # API endpoint tests
├── test_model_registry.py     # Model registry functionality tests
├── test_model_detection.py    # YOLO version detection tests
├── test_inference_engine.py   # Inference engine tests
├── test_youtube_processor.py  # YouTube processing tests
└── test_utils.py              # Utility function tests
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-mock
```

### Running All Tests

```bash
# From backend directory
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=app --cov-report=html

# Using the test runner script
python run_tests.py
```

### Running Specific Tests

```bash
# Run specific test file
python -m pytest tests/test_api.py -v

# Run specific test class
python -m pytest tests/test_model_registry.py::TestModelRegistry -v

# Run specific test method
python -m pytest tests/test_api.py::TestHealthEndpoints::test_health_check -v
```

### Test Configuration

Tests are configured using `pytest.ini`:
- Test discovery: `test_*.py` files
- Verbose output with short tracebacks
- Warning suppression for cleaner output
- Color output enabled

## Test Categories

### 🌐 API Tests (`test_api.py`)
- **Health Endpoints**: Basic connectivity tests
- **Image Upload**: File upload and validation
- **Model Management**: Model registry API endpoints
- **Training Endpoints**: Training-related API tests
- **YouTube Processing**: YouTube API endpoint tests

### 🤖 Model Registry Tests (`test_model_registry.py`)
- **ModelMetadata**: Model metadata creation and serialization
- **UltralyticsDownloader**: Model downloading functionality
- **ModelRegistry**: Core registry operations (add, delete, deploy)
- **Model Validation**: Training compatibility checks

### 🔍 Model Detection Tests (`test_model_detection.py`)
- **YOLOVersionDetector**: YOLO version detection from files
- **ModelCompatibilityValidator**: Model compatibility validation
- **Integration Tests**: End-to-end detection workflows

### 🧠 Inference Engine Tests (`test_inference_engine.py`)
- **Basic Operations**: Model loading and inference
- **Error Handling**: Invalid inputs and failure scenarios
- **Performance**: Timing and threshold validation
- **Integration**: End-to-end inference workflows

### 📹 YouTube Processor Tests (`test_youtube_processor.py`)
- **Processor Initialization**: YouTube processor setup
- **URL Validation**: YouTube URL pattern validation
- **API Structure**: Expected request/response formats
- **Utility Functions**: Video ID extraction and filename sanitization

### 🛠️ Utility Tests (`test_utils.py`)
- **File Operations**: File handling and validation
- **Validation Functions**: Input validation utilities
- **Data Processing**: Basic data manipulation functions

## Test Environment

### Environment Variables
- `TESTING=True`: Enables test mode
- `FLASK_ENV=testing`: Sets Flask to testing environment

### Fixtures
Tests use pytest fixtures for:
- **Temporary directories**: Clean test environments
- **Mock objects**: Isolated component testing
- **Sample data**: Consistent test inputs

### Mocking Strategy
- **External Dependencies**: YOLO models, file systems
- **Network Calls**: API requests and downloads
- **Heavy Operations**: Model loading and inference

## Continuous Integration

### GitHub Actions
The test suite runs automatically on:
- Push to main/master/develop branches
- Pull requests
- Multiple Python versions (3.8, 3.9, 3.10, 3.11)

### Test Workflow
1. **Environment Setup**: Install dependencies and system packages
2. **Directory Creation**: Set up required test directories
3. **Test Execution**: Run all test suites
4. **Import Validation**: Verify app can be imported
5. **Code Structure Check**: Validate file organization

## Writing New Tests

### Test Structure
```python
"""
Tests for [component name].
"""

import pytest
from unittest.mock import Mock, patch


class TestComponentName:
    """Test [component] functionality."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        expected = "expected_result"
        
        # Act
        result = function_to_test()
        
        # Assert
        assert result == expected
```

### Best Practices
1. **Descriptive Names**: Use clear, descriptive test names
2. **Arrange-Act-Assert**: Structure tests clearly
3. **Mock External Dependencies**: Isolate units under test
4. **Test Edge Cases**: Include error conditions and boundary cases
5. **Use Fixtures**: Reuse common test setup
6. **Document Intent**: Add docstrings explaining test purpose

### Adding New Test Files
1. Create file with `test_` prefix
2. Import required modules and fixtures
3. Organize tests into logical classes
4. Add to CI workflow if needed

## Test Coverage

Current test coverage includes:
- ✅ API endpoints (basic functionality)
- ✅ Model registry operations
- ✅ YOLO version detection
- ✅ Inference engine core functions
- ✅ YouTube processing utilities
- ✅ File and validation utilities

### Coverage Goals
- Maintain >80% code coverage
- Cover all critical paths
- Include error handling scenarios
- Test integration points

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the backend directory
cd backend
# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:.
```

**Missing Dependencies**
```bash
# Install all test dependencies
pip install -r requirements.txt
```

**Permission Errors**
```bash
# Ensure test directories are writable
chmod 755 temp_uploads test_data test_models
```

### Debug Mode
```bash
# Run with verbose output and no capture
python -m pytest tests/ -v -s --tb=long
```

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add new tests for new functionality
4. Update this documentation if needed
5. Verify CI pipeline passes

For questions or issues with tests, please check the logs and ensure all dependencies are properly installed. 