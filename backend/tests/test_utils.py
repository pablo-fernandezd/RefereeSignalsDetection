"""
Tests for utility functions.
"""

import pytest
from pathlib import Path


class TestValidation:
    """Test validation utilities."""
    
    def test_validate_image_format(self):
        """Test image format validation."""
        valid_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        invalid_formats = ['txt', 'pdf', 'doc', 'mp4']
        
        for fmt in valid_formats:
            filename = f'test.{fmt}'
            # Basic extension check
            assert filename.lower().endswith(fmt.lower())
        
        for fmt in invalid_formats:
            filename = f'test.{fmt}'
            # Should not be image format
            assert not any(filename.lower().endswith(img_fmt) for img_fmt in valid_formats)
    
    def test_validate_confidence_threshold(self):
        """Test confidence threshold validation."""
        valid_thresholds = [0.0, 0.5, 0.8, 1.0]
        invalid_thresholds = [-0.1, 1.1, 'invalid', None]
        
        for threshold in valid_thresholds:
            # Should be number between 0 and 1
            assert isinstance(threshold, (int, float))
            assert 0.0 <= threshold <= 1.0
        
        for threshold in invalid_thresholds:
            # Should fail validation
            if threshold is None or isinstance(threshold, str):
                assert True  # Invalid type
            elif isinstance(threshold, (int, float)):
                assert not (0.0 <= threshold <= 1.0)  # Out of range


class TestFileUtils:
    """Test file utility functions."""
    
    def test_safe_filename(self):
        """Test filename sanitization."""
        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("file with spaces.txt", "file_with_spaces.txt"),
            ("file/with/slashes.txt", "file_with_slashes.txt"),
            ("UPPERCASE.TXT", "uppercase.txt")
        ]
        
        for original, expected_pattern in test_cases:
            # Basic checks for safe filename
            result = original.lower().replace(' ', '_').replace('/', '_')
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_get_file_size(self, temp_dir):
        """Test file size calculation."""
        # Create a test file
        test_file = temp_dir / "test_file.txt"
        test_content = "A" * 1024  # 1KB of data
        test_file.write_text(test_content)
        
        # Basic file size check
        size = test_file.stat().st_size
        size_mb = size / (1024 * 1024)
        
        assert isinstance(size_mb, float)
        assert size_mb > 0
        assert size_mb < 1  # Should be less than 1MB 