"""
Training Data Service

This service handles all operations related to training data management,
statistics, and file operations for both referee and signal training data.
"""

import logging
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

from ..config.settings import DirectoryConfig, ModelConfig

logger = logging.getLogger(__name__)


class TrainingDataError(Exception):
    """Custom exception for training data operations"""
    pass


class TrainingDataService:
    """Service for managing training data operations"""
    
    @staticmethod
    def get_referee_training_count() -> Dict[str, int]:
        """
        Get count of positive and negative referee training samples.
        
        Returns:
            Dict with positive_count and negative_count
        """
        try:
            training_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
            if not training_dir.exists():
                return {'positive_count': 0, 'negative_count': 0}
            
            positive_count = 0
            negative_count = 0
            
            for file_path in training_dir.glob('*.txt'):
                if file_path.stat().st_size == 0:
                    negative_count += 1  # Empty files are negative samples
                else:
                    positive_count += 1  # Non-empty files are positive samples
            
            return {
                'positive_count': positive_count,
                'negative_count': negative_count
            }
            
        except Exception as e:
            logger.error(f"Error getting referee training count: {e}")
            raise TrainingDataError(f"Failed to get referee training count: {str(e)}")
    
    @staticmethod
    def get_signal_classes() -> Dict[str, List[str]]:
        """
        Get available signal classes from configuration and training data.
        
        Returns:
            Dict containing list of signal classes
        """
        try:
            # Get classes from configuration
            classes = ModelConfig.SIGNAL_CLASSES.copy()
            
            # Add 'none' class for negative samples if not present
            if 'none' not in classes:
                classes.append('none')
            
            return {'classes': classes}
            
        except Exception as e:
            logger.error(f"Error getting signal classes: {e}")
            raise TrainingDataError(f"Failed to get signal classes: {str(e)}")
    
    @staticmethod
    def get_signal_class_counts() -> Dict[str, int]:
        """
        Get count of samples for each signal class.
        
        Returns:
            Dict with counts for each signal class
        """
        try:
            training_dir = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER
            if not training_dir.exists():
                return {}
            
            class_counts = {}
            
            # Count image files by their prefix
            for image_file in training_dir.glob('*.jpg'):
                filename = image_file.name
                
                # Extract class name from filename (assuming format: signal_<class>_<timestamp>.jpg)
                if filename.startswith('signal_'):
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        class_name = parts[1]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            return class_counts
            
        except Exception as e:
            logger.error(f"Error getting signal class counts: {e}")
            raise TrainingDataError(f"Failed to get signal class counts: {str(e)}")
    
    @staticmethod
    def move_referee_training_data(destination_dir: str = None) -> Dict[str, Any]:
        """
        Move referee training data to a destination directory.
        
        Args:
            destination_dir: Optional custom destination directory
            
        Returns:
            Dict with operation results
        """
        try:
            source_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
            if not source_dir.exists():
                return {'moved': [], 'dst': 'N/A - source directory not found'}
            
            # Default destination is global referee training data
            if destination_dir is None:
                global_dir = DirectoryConfig.BASE_DIR.parent / 'data' / 'referee_training_data'
            else:
                global_dir = Path(destination_dir)
            
            global_dir.mkdir(parents=True, exist_ok=True)
            
            moved_files = []
            for file_path in source_dir.glob('*'):
                if file_path.is_file():
                    dest_path = global_dir / file_path.name
                    
                    # Ensure unique filename
                    counter = 1
                    original_dest = dest_path
                    while dest_path.exists():
                        stem = original_dest.stem
                        suffix = original_dest.suffix
                        dest_path = global_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    shutil.move(str(file_path), str(dest_path))
                    moved_files.append(file_path.name)
            
            logger.info(f"Moved {len(moved_files)} referee training files to {global_dir}")
            return {
                'moved': moved_files,
                'dst': str(global_dir)
            }
            
        except Exception as e:
            logger.error(f"Error moving referee training data: {e}")
            raise TrainingDataError(f"Failed to move referee training data: {str(e)}")
    
    @staticmethod
    def move_signal_training_data(destination_dir: str = None) -> Dict[str, Any]:
        """
        Move signal training data to a destination directory.
        
        Args:
            destination_dir: Optional custom destination directory
            
        Returns:
            Dict with operation results
        """
        try:
            source_dir = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER
            if not source_dir.exists():
                return {'moved': [], 'dst': 'N/A - source directory not found'}
            
            # Default destination is global signal training data
            if destination_dir is None:
                global_dir = DirectoryConfig.BASE_DIR.parent / 'data' / 'signal_training_data'
            else:
                global_dir = Path(destination_dir)
            
            global_dir.mkdir(parents=True, exist_ok=True)
            
            moved_files = []
            for file_path in source_dir.glob('*'):
                if file_path.is_file() and file_path.name != 'data.yaml':
                    dest_path = global_dir / file_path.name
                    
                    # Ensure unique filename
                    counter = 1
                    original_dest = dest_path
                    while dest_path.exists():
                        stem = original_dest.stem
                        suffix = original_dest.suffix
                        dest_path = global_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    shutil.move(str(file_path), str(dest_path))
                    moved_files.append(file_path.name)
            
            logger.info(f"Moved {len(moved_files)} signal training files to {global_dir}")
            return {
                'moved': moved_files,
                'dst': str(global_dir)
            }
            
        except Exception as e:
            logger.error(f"Error moving signal training data: {e}")
            raise TrainingDataError(f"Failed to move signal training data: {str(e)}")
    
    @staticmethod
    def delete_referee_training_data() -> Dict[str, Any]:
        """
        Delete all referee training data.
        
        Returns:
            Dict with operation results
        """
        try:
            training_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
            if not training_dir.exists():
                return {'status': 'success', 'deleted_count': 0}
            
            deleted_count = 0
            for file_path in training_dir.glob('*'):
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} referee training files")
            return {
                'status': 'success',
                'deleted_count': deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting referee training data: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @staticmethod
    def delete_signal_training_data() -> Dict[str, Any]:
        """
        Delete all signal training data.
        
        Returns:
            Dict with operation results
        """
        try:
            training_dir = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER
            if not training_dir.exists():
                return {'status': 'success', 'deleted_count': 0}
            
            deleted_count = 0
            for file_path in training_dir.glob('*'):
                if file_path.is_file() and file_path.name != 'data.yaml':
                    file_path.unlink()
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} signal training files")
            return {
                'status': 'success',
                'deleted_count': deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting signal training data: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }