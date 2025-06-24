"""
Application Configuration Settings

This module contains all configuration settings for the Referee Detection System.
It provides a centralized location for managing application constants,
file paths, model settings, and other configuration parameters.
"""

import os
from pathlib import Path

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Directory structure configuration
class DirectoryConfig:
    """Configuration for directory paths used throughout the application"""
    
    # Static file directories
    UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
    CROPS_FOLDER = BASE_DIR / 'static' / 'referee_crops'
    SIGNALS_FOLDER = BASE_DIR / 'static' / 'signals'
    
    # Training data directories  
    REFEREE_TRAINING_DATA_FOLDER = BASE_DIR / 'data' / 'referee_training_data'
    SIGNAL_TRAINING_DATA_FOLDER = BASE_DIR / 'data' / 'signal_training_data'
    
    # YouTube processing directories
    YOUTUBE_VIDEOS_FOLDER = BASE_DIR / 'data' / 'youtube_videos'
    
    # Model directories - using MODELS_DIR for consistency
    MODELS_DIR = PROJECT_ROOT / 'models'
    MODELS_FOLDER = PROJECT_ROOT / 'models'  # Keep both for compatibility
    
    # Registry files
    IMAGE_HASH_REGISTRY = BASE_DIR / 'data' / 'image_hashes.txt'
    
    # Base directory for external access
    BASE_DIR = BASE_DIR

# Flask application configuration
class FlaskConfig:
    """Flask application configuration settings"""
    
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_very_secret_key_here')
    # Disable debug mode to prevent auto-restart during video processing
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', 5000))

# CORS configuration
class CORSConfig:
    """Cross-Origin Resource Sharing configuration"""
    
    METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    ORIGINS = os.getenv('ALLOWED_ORIGINS', '*')

# Model configuration
class ModelConfig:
    """AI model configuration settings"""
    
    # Device configuration
    DEVICE = 'cuda' if os.getenv('USE_CUDA', 'auto') == 'true' else 'cpu'
    
    # Model file paths
    REFEREE_MODEL_PATH = DirectoryConfig.MODELS_FOLDER / 'bestRefereeDetection.pt'
    SIGNAL_MODEL_PATH = DirectoryConfig.MODELS_FOLDER / 'bestSignalsDetection.pt'
    
    # Model parameters
    MODEL_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.4
    
    # Signal classes (adjust according to your model)
    SIGNAL_CLASSES = [
        'armLeft', 'armRight', 'hits', 'leftServe', 
        'net', 'outside', 'rightServe', 'touched'
    ]
    
    # Referee classes
    REFEREE_CLASSES = [
        {'name': 'referee', 'id': 0},
        {'name': 'none', 'id': -1}  # Special value for no detection
    ]

# YouTube processing configuration
class YouTubeConfig:
    """YouTube video processing configuration"""
    
    SEGMENT_DURATION = 600  # 10 minutes in seconds
    FRAME_EXTRACTION_FPS = 1  # Extract one frame per second
    THUMBNAIL_WIDTH = 200
    
    # Download options
    DOWNLOAD_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    USE_FIREFOX_COOKIES = True

# File processing configuration
class FileConfig:
    """File processing and validation configuration"""
    
    ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov'}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Crop parameters
    CROP_PADDING = 20  # Pixels to add around detected objects

# Database configuration (for future use)
class DatabaseConfig:
    """Database configuration (placeholder for future implementation)"""
    
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///referee_detection.db')

# Logging configuration
class LoggingConfig:
    """Logging configuration settings"""
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = BASE_DIR / 'logs' / 'app.log'

# Ensure all necessary directories exist
def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        DirectoryConfig.UPLOAD_FOLDER,
        DirectoryConfig.CROPS_FOLDER,
        DirectoryConfig.SIGNALS_FOLDER,
        DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER,
        DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER,
        DirectoryConfig.YOUTUBE_VIDEOS_FOLDER,
        LoggingConfig.LOG_FILE.parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Ensure the hash registry file exists
    if not DirectoryConfig.IMAGE_HASH_REGISTRY.exists():
        DirectoryConfig.IMAGE_HASH_REGISTRY.touch() 