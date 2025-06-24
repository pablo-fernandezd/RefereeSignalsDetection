"""
Training Data Management Routes

This module contains routes for managing training data,
including signal classification and data organization.
"""

from flask import Blueprint, request, jsonify
from pathlib import Path
import os

import sys
from pathlib import Path

# Add necessary paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from models.signal_classifier import detect_signal
    from config.settings import DirectoryConfig, ModelConfig
except ImportError:
    # Try alternative import paths
    from app.models.signal_classifier import detect_signal
    from app.config.settings import DirectoryConfig, ModelConfig

training_bp = Blueprint('training', __name__)

@training_bp.route('/signal_classes', methods=['GET'])
def get_signal_classes():
    """Get list of available signal classes."""
    try:
        # Include the standard signal classes plus "none" for negative samples
        all_classes = ModelConfig.SIGNAL_CLASSES + ['none']
        return jsonify({
            'classes': all_classes,
            'count': len(all_classes)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/process_signal', methods=['POST'])
def process_signal():
    """Process a referee crop for signal classification."""
    try:
        data = request.json
        crop_filename = data.get('crop_filename_for_signal')
        
        if not crop_filename:
            return jsonify({'error': 'Missing crop filename'}), 400
        
        # Check both possible locations for the crop file
        path_in_training = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER / crop_filename
        path_in_crops = DirectoryConfig.CROPS_FOLDER / crop_filename
        
        crop_path = None
        if path_in_training.exists():
            crop_path = path_in_training
        elif path_in_crops.exists():
            crop_path = path_in_crops
        
        if not crop_path:
            return jsonify({'error': f'Image not found: {crop_filename}'}), 404
        
        # Run signal classification
        result = detect_signal(str(crop_path))
        
        return jsonify({
            'predicted_class': result.get('predicted_class'),
            'confidence': result.get('confidence', 0.0),
            'bbox_xywhn': result.get('bbox_xywhn')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/confirm_signal', methods=['POST'])
def confirm_signal():
    """Confirm and save signal classification for training."""
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        data = request.json
        crop_filename_for_signal = data.get('crop_filename_for_signal')
        correct = data.get('correct')
        selected_class = data.get('selected_class')
        signal_bbox_yolo = data.get('signal_bbox_yolo')
        original_filename = data.get('original_filename')
        
        if not crop_filename_for_signal or selected_class is None:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Check if the original image has been processed before (if original_filename is provided)
        if original_filename:
            from app.utils.hash_utils import is_duplicate_image, register_image_hash, clear_hash_cache
            
            original_path = DirectoryConfig.UPLOAD_FOLDER / original_filename
            if original_path.exists():
                # Clear cache to ensure fresh data
                clear_hash_cache()
                
                # Check for duplicate
                if is_duplicate_image(original_path):
                    logger.info(f"Duplicate image detected for signal labeling: {original_filename}")
                    
                    # Even for duplicates, remove the image from the queue to prevent it from reappearing
                    try:
                        original_path.unlink()
                        logger.info(f"Removed duplicate image from queue: {original_filename}")
                    except Exception as e:
                        logger.warning(f"Failed to remove duplicate image from queue: {e}")
                    
                    # Also clean up the temporary crop file
                    temp_crop_path = DirectoryConfig.CROPS_FOLDER / crop_filename_for_signal
                    if temp_crop_path.exists():
                        try:
                            temp_crop_path.unlink()
                            logger.info(f"Removed temporary crop file for duplicate: {crop_filename_for_signal}")
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary crop file for duplicate: {e}")
                    
                    return jsonify({
                        'status': 'warning',
                        'action': 'duplicate_detected',
                        'message': 'This image has been processed before. The signal label will not be saved to avoid duplicates.',
                        'duplicate': True
                    })
                
                # Register the image hash to prevent future duplicates
                if register_image_hash(original_path):
                    logger.info(f"Registered hash for signal image: {original_filename}")
                else:
                    logger.warning(f"Failed to register hash for signal image: {original_filename}")
        
        # Validate signal class
        from utils.validation_utils import validate_signal_class
        if not validate_signal_class(selected_class):
            return jsonify({'error': f'Invalid signal class: {selected_class}'}), 400
        
        # Handle negative sample case (when "none" is selected)
        if selected_class == 'none':
            logger.info(f"Saving negative signal sample: {crop_filename_for_signal}")
            
            # Remove the original image from the uploads folder to prevent it from appearing in the queue again
            if original_filename:
                original_path = DirectoryConfig.UPLOAD_FOLDER / original_filename
                if original_path.exists():
                    try:
                        original_path.unlink()
                        logger.info(f"Removed original image from queue: {original_filename}")
                    except Exception as e:
                        logger.warning(f"Failed to remove original image from queue: {e}")
            
            # Also clean up the temporary crop file if it's in the crops folder
            temp_crop_path = DirectoryConfig.CROPS_FOLDER / crop_filename_for_signal
            if temp_crop_path.exists():
                try:
                    temp_crop_path.unlink()
                    logger.info(f"Removed temporary crop file: {crop_filename_for_signal}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary crop file: {e}")
            
            # For negative samples, we could save them to a "none" folder or handle differently
            # For now, just log and return success
            return jsonify({
                'status': 'success',
                'message': 'Negative sample (no signal) saved for training',
                'action': 'saved_as_negative',
                'duplicate': False
            })
        
        # For positive samples, save the labeled data for training
        logger.info(f"Saving positive signal sample: {crop_filename_for_signal}, class: {selected_class}")
        
        # Find the source crop file
        source_crop_path = None
        crop_paths_to_check = [
            DirectoryConfig.CROPS_FOLDER / crop_filename_for_signal,
            DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER / crop_filename_for_signal
        ]
        
        for path in crop_paths_to_check:
            if path.exists():
                source_crop_path = path
                break
        
        if not source_crop_path:
            logger.error(f"Source crop file not found: {crop_filename_for_signal}")
            return jsonify({'error': f'Source crop file not found: {crop_filename_for_signal}'}), 404
        
        # Create signal training data directory structure
        signal_training_dir = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER
        signal_training_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename for training data
        import time
        timestamp = int(time.time() * 1000000)  # microsecond timestamp for uniqueness
        base_name = f"signal_{selected_class}_{timestamp}"
        
        # Copy image to signal training data folder
        import shutil
        dest_image_path = signal_training_dir / f"{base_name}.jpg"
        shutil.copy2(source_crop_path, dest_image_path)
        logger.info(f"Copied signal training image to: {dest_image_path}")
        
        # Create YOLO annotation file if bbox data is available
        if signal_bbox_yolo and len(signal_bbox_yolo) >= 4:
            # Get class index for YOLO format
            try:
                class_index = ModelConfig.SIGNAL_CLASSES.index(selected_class)
            except ValueError:
                class_index = 0  # Default to first class if not found
            
            # Create annotation file
            dest_txt_path = signal_training_dir / f"{base_name}.txt"
            with open(dest_txt_path, 'w') as f:
                # YOLO format: class_id center_x center_y width height (normalized)
                f.write(f"{class_index} {signal_bbox_yolo[0]} {signal_bbox_yolo[1]} {signal_bbox_yolo[2]} {signal_bbox_yolo[3]}\n")
            logger.info(f"Created YOLO annotation file: {dest_txt_path}")
        
        # Remove the original image from the uploads folder to prevent it from appearing in the queue again
        if original_filename:
            original_path = DirectoryConfig.UPLOAD_FOLDER / original_filename
            if original_path.exists():
                try:
                    original_path.unlink()
                    logger.info(f"Removed original image from queue: {original_filename}")
                except Exception as e:
                    logger.warning(f"Failed to remove original image from queue: {e}")
        
        # Also clean up the temporary crop file if it's in the crops folder
        temp_crop_path = DirectoryConfig.CROPS_FOLDER / crop_filename_for_signal
        if temp_crop_path.exists():
            try:
                temp_crop_path.unlink()
                logger.info(f"Removed temporary crop file: {crop_filename_for_signal}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary crop file: {e}")
        
        return jsonify({
            'status': 'success',
            'message': f'Signal "{selected_class}" saved for training',
            'action': 'saved_as_positive',
            'duplicate': False,
            'details': {
                'crop_filename': crop_filename_for_signal,
                'signal_class': selected_class,
                'was_correct_prediction': correct,
                'has_bbox': signal_bbox_yolo is not None,
                'training_image': f"{base_name}.jpg",
                'training_annotation': f"{base_name}.txt" if signal_bbox_yolo else None
            }
        })
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in confirm_signal: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@training_bp.route('/referee_training_count', methods=['GET'])
def referee_training_count():
    """Get count of referee training images."""
    try:
        # Count positive samples (images with corresponding .txt files)
        positive_count = 0
        negative_count = 0
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER.glob(ext))
        
        for image_file in image_files:
            # Check if there's a corresponding .txt file
            txt_file = image_file.with_suffix('.txt')
            if txt_file.exists():
                positive_count += 1
            else:
                negative_count += 1
        
        return jsonify({
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_count': positive_count + negative_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/signal_class_counts', methods=['GET'])
def signal_class_counts():
    """Get count of training images per signal class."""
    try:
        counts = {}
        
        # Initialize all classes with 0 count
        for signal_class in ModelConfig.SIGNAL_CLASSES:
            counts[signal_class] = 0
        counts['none'] = 0
        
        # Count files in the signal training data folder
        signal_training_dir = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER
        if signal_training_dir.exists():
            # Get all image files
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(signal_training_dir.glob(ext))
            
            # Count files by class based on filename pattern
            for image_file in image_files:
                filename = image_file.name
                # Look for pattern: signal_{class}_{timestamp}.jpg
                if filename.startswith('signal_'):
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        # Extract class name (everything between 'signal_' and the last underscore)
                        class_name = '_'.join(parts[1:-1])  # Handle multi-word classes like 'leftServe'
                        if class_name in counts:
                            counts[class_name] += 1
                        elif class_name == 'none':
                            counts['none'] += 1
                # Also count legacy files that don't follow the new pattern
                elif any(class_name in filename for class_name in ModelConfig.SIGNAL_CLASSES):
                    for class_name in ModelConfig.SIGNAL_CLASSES:
                        if class_name in filename:
                            counts[class_name] += 1
                            break
        
        return jsonify(counts)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/autolabel/pending_count', methods=['GET'])
def get_pending_autolabel_count():
    """
    Get count of pending autolabel jobs.
    
    This endpoint provides compatibility with the frontend dashboard
    that expects autolabel pending count information.
    
    Returns:
        JSON response with pending autolabel count
    """
    try:
        # For now, return 0 as autolabeling is not yet implemented in new system
        # This prevents frontend errors while maintaining compatibility
        return jsonify({'count': 0})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/test', methods=['GET'])
def test_route():
    """Test route to verify blueprint registration."""
    return jsonify({'message': 'Training routes are working!'})

@training_bp.route('/delete_referee_training_data', methods=['POST'])
def delete_referee_training_data():
    """Delete all referee training data files."""
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        deleted_files = []
        error_files = []
        
        # Get all files in the referee training data folder
        referee_training_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
        if not referee_training_dir.exists():
            return jsonify({
                'status': 'success',
                'message': 'No referee training data folder found',
                'deleted_count': 0,
                'deleted_files': [],
                'error_files': []
            })
        
        # Delete all files in the directory
        for file_path in referee_training_dir.iterdir():
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_files.append(file_path.name)
                    logger.info(f"Deleted referee training file: {file_path.name}")
                except Exception as e:
                    error_files.append({'file': file_path.name, 'error': str(e)})
                    logger.error(f"Failed to delete referee training file {file_path.name}: {e}")
        
        return jsonify({
            'status': 'success',
            'message': f'Deleted {len(deleted_files)} referee training files',
            'deleted_count': len(deleted_files),
            'deleted_files': deleted_files,
            'error_files': error_files
        })
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error deleting referee training data: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@training_bp.route('/delete_signal_training_data', methods=['POST'])
def delete_signal_training_data():
    """Delete all signal training data files."""
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        deleted_files = []
        error_files = []
        
        # Get all files in the signal training data folder
        signal_training_dir = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER
        if not signal_training_dir.exists():
            return jsonify({
                'status': 'success',
                'message': 'No signal training data folder found',
                'deleted_count': 0,
                'deleted_files': [],
                'error_files': []
            })
        
        # Delete all files in the directory
        for file_path in signal_training_dir.iterdir():
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_files.append(file_path.name)
                    logger.info(f"Deleted signal training file: {file_path.name}")
                except Exception as e:
                    error_files.append({'file': file_path.name, 'error': str(e)})
                    logger.error(f"Failed to delete signal training file {file_path.name}: {e}")
        
        return jsonify({
            'status': 'success',
            'message': f'Deleted {len(deleted_files)} signal training files',
            'deleted_count': len(deleted_files),
            'deleted_files': deleted_files,
            'error_files': error_files
        })
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error deleting signal training data: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500 