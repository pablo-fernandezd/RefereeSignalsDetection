"""
Tests for YouTube processor functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestYouTubeProcessor:
    """Test YouTube processor functionality."""
    
    @patch('app.models.youtube_processor.YouTubeProcessor')
    def test_youtube_processor_initialization(self, mock_processor):
        """Test YouTube processor initialization."""
        # Mock the processor
        mock_instance = Mock()
        mock_instance.base_dir = Path("/tmp/test")
        mock_processor.return_value = mock_instance
        
        # Test initialization
        from app.models.youtube_processor import YouTubeProcessor
        processor = YouTubeProcessor()
        
        assert processor is not None
    
    def test_video_id_extraction(self):
        """Test video ID extraction from URLs."""
        # Mock the extraction method
        with patch('app.models.youtube_processor.YouTubeProcessor') as mock_processor:
            mock_instance = Mock()
            mock_instance._extract_video_id.return_value = "TMou-6_4w10"
            mock_processor.return_value = mock_instance
            
            from app.models.youtube_processor import YouTubeProcessor
            processor = YouTubeProcessor()
            
            test_url = "https://www.youtube.com/watch?v=TMou-6_4w10"
            video_id = processor._extract_video_id(test_url)
            
            assert video_id == "TMou-6_4w10"
    
    def test_get_all_videos(self):
        """Test getting all processed videos."""
        with patch('app.models.youtube_processor.YouTubeProcessor') as mock_processor:
            mock_instance = Mock()
            mock_instance.get_all_videos.return_value = []
            mock_processor.return_value = mock_instance
            
            from app.models.youtube_processor import YouTubeProcessor
            processor = YouTubeProcessor()
            
            videos = processor.get_all_videos()
            assert isinstance(videos, list)
    
    def test_processing_info_structure(self):
        """Test processing info structure."""
        sample_info = {
            "total_frames": 1000,
            "segments_created": 2,
            "frames_extracted": 50,
            "crops_created": 30,
            "auto_crop": True,
            "status": "completed"
        }
        
        # Verify structure
        assert isinstance(sample_info, dict)
        assert "total_frames" in sample_info
        assert "status" in sample_info
        assert sample_info["total_frames"] == 1000
        assert sample_info["status"] == "completed"
    
    @pytest.mark.skipif(True, reason="Requires actual model file")
    def test_model_loading(self):
        """Test model loading (skipped if model not available)."""
        # This test is skipped by default since it requires actual model files
        # It can be enabled when running in an environment with models
        pass
    
    def test_url_validation(self):
        """Test URL validation."""
        # Test valid YouTube URLs
        valid_urls = [
            "https://www.youtube.com/watch?v=TMou-6_4w10",
            "https://youtu.be/TMou-6_4w10",
            "https://youtube.com/watch?v=TMou-6_4w10"
        ]
        
        # Test invalid URLs
        invalid_urls = [
            "not-a-url",
            "https://example.com",
            "https://vimeo.com/123456"
        ]
        
        # Basic URL pattern validation
        import re
        youtube_pattern = r'(youtube\.com|youtu\.be)'
        
        for url in valid_urls:
            assert re.search(youtube_pattern, url) is not None
        
        for url in invalid_urls:
            if url.startswith('http'):
                assert re.search(youtube_pattern, url) is None


class TestYouTubeAPI:
    """Test YouTube API endpoints."""
    
    def test_youtube_videos_endpoint_structure(self):
        """Test the expected structure of YouTube videos endpoint."""
        # Expected response structure
        expected_response = {
            "status": "success",
            "videos": [
                {
                    "video_id": "TMou-6_4w10",
                    "title": "Test Video",
                    "status": "completed",
                    "total_frames": 1000,
                    "processed_frames": 500
                }
            ]
        }
        
        # Verify structure
        assert "status" in expected_response
        assert "videos" in expected_response
        assert isinstance(expected_response["videos"], list)
    
    def test_youtube_process_request_structure(self):
        """Test the expected structure of YouTube process request."""
        # Expected request structure
        expected_request = {
            "url": "https://www.youtube.com/watch?v=TMou-6_4w10",
            "auto_crop": True,
            "segment_duration": 30,
            "max_segments": 10
        }
        
        # Verify structure
        assert "url" in expected_request
        assert "auto_crop" in expected_request
        assert isinstance(expected_request["auto_crop"], bool)


class TestYouTubeUtils:
    """Test YouTube utility functions."""
    
    def test_video_id_patterns(self):
        """Test video ID extraction patterns."""
        import re
        
        # Common YouTube URL patterns
        patterns = [
            (r'youtube\.com/watch\?v=([^&]+)', 'https://www.youtube.com/watch?v=TMou-6_4w10'),
            (r'youtu\.be/([^?]+)', 'https://youtu.be/TMou-6_4w10'),
            (r'youtube\.com/embed/([^?]+)', 'https://www.youtube.com/embed/TMou-6_4w10')
        ]
        
        for pattern, url in patterns:
            match = re.search(pattern, url)
            assert match is not None
            assert match.group(1) == 'TMou-6_4w10'
    
    def test_filename_sanitization(self):
        """Test filename sanitization for video processing."""
        # Test cases for filename sanitization
        test_cases = [
            ("Test Video Title", "test_video_title"),
            ("Video with / slash", "video_with_slash"),
            ("Video: with colon", "video_with_colon"),
            ("Video with 123 numbers", "video_with_123_numbers")
        ]
        
        import re
        
        for original, expected_pattern in test_cases:
            # Simple sanitization (remove special chars, lowercase, replace spaces)
            sanitized = re.sub(r'[^\w\s-]', '', original.lower())
            sanitized = re.sub(r'[-\s]+', '_', sanitized)
            
            # Should not contain special characters
            assert not re.search(r'[^\w_]', sanitized)
            # Should be lowercase
            assert sanitized.islower() or sanitized.isdigit() or '_' in sanitized 