"""
Image Processing Routes for Referee Detection System

This module contains all routes related to image upload, processing,
and referee detection functionality.
"""

from flask import Blueprint, request, jsonify, send_from_directory
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

import sys
from pathlib import Path

# Add necessary paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from models.referee_detector import detect_referee
    from utils.file_utils import allowed_file, save_uploaded_file
    from utils.validation_utils import validate_image_upload
    from config.settings import DirectoryConfig
except ImportError:
    # Try alternative import paths
    from app.models.referee_detector import detect_referee
    from app.utils.file_utils import allowed_file, save_uploaded_file
    from app.utils.validation_utils import validate_image_upload
    from app.config.settings import DirectoryConfig

image_bp = Blueprint('image', __name__)

@image_bp.route('/upload', methods=['POST'])
def upload_image():
    """Upload and process an image for referee detection."""
    try:
        # Validate upload
        validation_result = validate_image_upload(request)
        if not validation_result['valid']:
            return jsonify({'error': validation_result['error']}), 400
        
        # Save uploaded file
        file = request.files['image']
        filename = save_uploaded_file(file, DirectoryConfig.UPLOAD_FOLDER)
        
        # Process for referee detection
        upload_path = DirectoryConfig.UPLOAD_FOLDER / filename
        crop_filename = f"temp_crop_{filename}"
        crop_path = DirectoryConfig.CROPS_FOLDER / crop_filename
        
        result = detect_referee(str(upload_path), crop_save_path=str(crop_path))
        
        if result['detected']:
            return jsonify({
                'filename': filename,
                'crop_filename': crop_filename,
                'crop_url': f'/api/referee_crop_image/{crop_filename}',
                'bbox': result['bbox']
            })
        else:
            return jsonify({
                'error': 'No referee detected',
                'filename': filename
            }), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@image_bp.route('/pending_images', methods=['GET'])
def get_pending_images():
    """Get list of pending images for labeling."""
    try:
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_files.extend(DirectoryConfig.UPLOAD_FOLDER.glob(f'*{ext}'))
        
        # Filter out temporary and processed files
        filtered_files = []
        for img_path in image_files:
            name = img_path.name
            if not (name.startswith('temp_crop_') or 
                   name.startswith('referee_') or 
                   name.startswith('yt_signal_crop_')):
                filtered_files.append(name)
        
        filtered_files.sort()
        return jsonify({
            'images': filtered_files,
            'count': len(filtered_files)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@image_bp.route('/referee_crop_image/<filename>', methods=['GET'])
def get_referee_crop_image(filename):
    """Serve referee crop images."""
    try:
        crop_path = DirectoryConfig.CROPS_FOLDER / filename
        if crop_path.exists():
            return send_from_directory(DirectoryConfig.CROPS_FOLDER, filename)
        else:
            return jsonify({'error': 'Referee crop image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@image_bp.route('/uploads/<filename>')
def get_uploaded_image(filename):
    """Serve uploaded images."""
    try:
        return send_from_directory(DirectoryConfig.UPLOAD_FOLDER, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@image_bp.route('/confirm_crop', methods=['POST'])
def confirm_crop():
    """Confirm and save a referee crop for training."""
    try:
        data = request.json
        original_filename = data.get('original_filename')
        crop_filename = data.get('crop_filename')
        bbox = data.get('bbox')
        
        if not all([original_filename, crop_filename, bbox]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Check if the original image has been processed before
        from app.utils.hash_utils import is_duplicate_image, register_image_hash, clear_hash_cache
        
        original_path = DirectoryConfig.UPLOAD_FOLDER / original_filename
        if not original_path.exists():
            return jsonify({'error': f'Original image not found: {original_filename}'}), 404
        
        # Clear cache to ensure fresh data
        clear_hash_cache()
        
        # Check for duplicate
        if is_duplicate_image(original_path):
            logger.info(f"Duplicate image detected: {original_filename}")
            return jsonify({
                'status': 'warning',
                'action': 'duplicate_detected',
                'message': 'This image has been processed before. It will not be saved to training data to avoid duplicates.',
                'crop_filename_for_signal': crop_filename,
                'duplicate': True
            })
        
        # Register the image hash to prevent future duplicates
        if register_image_hash(original_path):
            logger.info(f"Registered hash for image: {original_filename}")
        else:
            logger.warning(f"Failed to register hash for image: {original_filename}")
        
        # Save the crop and create training data
        logger.info(f"Saving referee crop for training: {crop_filename}")
        return jsonify({
            'status': 'ok',
            'crop_filename_for_signal': crop_filename,
            'message': 'Crop confirmed and saved for training',
            'duplicate': False
        })
        
    except Exception as e:
        logger.error(f"Error in confirm_crop: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@image_bp.route('/manual_crop', methods=['POST'])
def manual_crop():
    """Process a manual crop of an image."""
    try:
        logger.info("Manual crop request received")
        data = request.json
        original_filename = data.get('original_filename')
        bbox = data.get('bbox')
        class_id = data.get('class_id', 0)
        proceed_to_signal = data.get('proceedToSignal', True)
        
        logger.info(f"Manual crop parameters: filename={original_filename}, bbox={bbox}, class_id={class_id}")
        
        if not original_filename:
            return jsonify({'error': 'Missing original_filename parameter'}), 400
        
        # Handle negative samples (no referee)
        if class_id == -1 or not bbox or len(bbox) == 0:
            # Save as negative sample - no crop needed
            logger.info("Saving as negative sample")
            return jsonify({
                'status': 'ok',
                'action': 'saved_as_negative',
                'message': 'Image saved as negative sample (no referee)'
            })
        
        if not bbox or len(bbox) != 4:
            return jsonify({'error': 'Invalid bbox parameter - must be [x1, y1, x2, y2]'}), 400
        
        # Load the original image
        original_path = DirectoryConfig.UPLOAD_FOLDER / original_filename
        logger.info(f"Looking for original image at: {original_path}")
        if not original_path.exists():
            return jsonify({'error': f'Original image not found: {original_filename}'}), 404
        
        import cv2
        import numpy as np
        from datetime import datetime
        
        # Read the original image
        logger.info("Reading original image")
        image = cv2.imread(str(original_path))
        if image is None:
            logger.error("Failed to load original image with OpenCV")
            return jsonify({'error': 'Failed to load original image'}), 500
        
        logger.info(f"Original image shape: {image.shape}")
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        logger.info(f"Bounding box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # Validate coordinates
        height, width = image.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            logger.error(f"Invalid bounding box coordinates for image size {width}x{height}")
            return jsonify({'error': 'Invalid bounding box coordinates'}), 400
        
        # Crop the image
        logger.info("Cropping image")
        cropped_image = image[y1:y2, x1:x2]
        logger.info(f"Cropped image shape: {cropped_image.shape}")
        
        # Resize to model size for consistency
        from config.settings import ModelConfig
        logger.info(f"Resizing to model size: {ModelConfig.MODEL_SIZE}")
        resized_crop = cv2.resize(cropped_image, 
                                (ModelConfig.MODEL_SIZE, ModelConfig.MODEL_SIZE),
                                interpolation=cv2.INTER_AREA)
        
        # Generate crop filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = original_filename.rsplit('.', 1)[0]  # Remove extension
        crop_filename = f"manual_crop_cropped_{base_name}_{timestamp}.jpg"
        crop_path = DirectoryConfig.CROPS_FOLDER / crop_filename
        
        logger.info(f"Saving crop to: {crop_path}")
        
        # Ensure directory exists
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the crop
        success = cv2.imwrite(str(crop_path), resized_crop)
        logger.info(f"cv2.imwrite returned: {success}")
        
        if not success:
            logger.error("cv2.imwrite failed")
            return jsonify({'error': 'Failed to save cropped image - cv2.imwrite failed'}), 500
        
        if not crop_path.exists():
            logger.error(f"File was not created at {crop_path}")
            return jsonify({'error': 'Failed to save cropped image - file not found after save'}), 500
        
        logger.info(f"Successfully created manual crop: {crop_filename}")
        
        # Determine response based on workflow
        if proceed_to_signal:
            return jsonify({
                'status': 'ok',
                'crop_filename_for_signal': crop_filename,
                'message': 'Manual crop created successfully - ready for signal detection'
            })
        else:
            return jsonify({
                'status': 'ok',
                'crop_filename': crop_filename,
                'message': 'Manual crop saved to training data'
            })
        
    except Exception as e:
        logger.error(f"Error in manual_crop: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@image_bp.route('/process_queued_image_referee', methods=['POST'])
def process_queued_image_referee():
    """Process a queued image for referee detection."""
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
        
        upload_path = DirectoryConfig.UPLOAD_FOLDER / filename
        if not upload_path.exists():
            return jsonify({'error': 'Image not found in queue'}), 404
        
        # Run referee detection
        crop_filename = f"temp_crop_{filename}"
        crop_path = DirectoryConfig.CROPS_FOLDER / crop_filename
        result = detect_referee(str(upload_path), crop_save_path=str(crop_path))

        if result['detected']:
            return jsonify({
                'filename': filename,
                'crop_filename': crop_filename,
                'crop_url': f'/api/referee_crop_image/{crop_filename}',
                'bbox': result['bbox']
            })
        else:
            return jsonify({
                'error': 'No referee detected',
                'filename': filename,
                'action': 'saved_as_negative'
            }), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@image_bp.route('/process_crop_for_signal', methods=['POST'])
def process_crop_for_signal():
    """Process a confirmed referee crop for signal detection."""
    try:
        data = request.json
        crop_filename_for_signal = data.get('crop_filename_for_signal')
        
        if not crop_filename_for_signal:
            return jsonify({'error': 'crop_filename_for_signal is required'}), 400
        
        # Check both possible locations for the crop file
        path_in_training = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER / crop_filename_for_signal
        path_in_crops = DirectoryConfig.CROPS_FOLDER / crop_filename_for_signal
        
        crop_path = None
        if path_in_training.exists():
            crop_path = path_in_training
        elif path_in_crops.exists():
            crop_path = path_in_crops
        
        if not crop_path:
            return jsonify({'error': f'Crop image not found: {crop_filename_for_signal}'}), 404
        
        # Import and use signal detection
        from models.signal_classifier import detect_signal
        result = detect_signal(str(crop_path))
        
        return jsonify({
            'predicted_class': result.get('predicted_class'),
            'confidence': result.get('confidence', 0.0),
            'bbox_xywhn': result.get('bbox_xywhn'),
            'crop_filename_for_signal': crop_filename_for_signal
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@image_bp.route('/referee_training_count', methods=['GET'])
def get_referee_training_count():
    """Get count of referee training data (positive and negative samples)."""
    try:
        # Count files in referee training data directory
        referee_data_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
        
        positive_count = 0
        negative_count = 0
        
        if referee_data_dir.exists():
            # Count image files
            for ext in ['.png', '.jpg', '.jpeg']:
                image_files = list(referee_data_dir.glob(f'*{ext}'))
                
                # Classify as positive or negative based on filename patterns
                for img_file in image_files:
                    name = img_file.name.lower()
                    if 'negative' in name or 'none' in name or 'no_referee' in name:
                        negative_count += 1
                    else:
                        positive_count += 1
        
        return jsonify({
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_count': positive_count + negative_count
        })
        
    except Exception as e:
        logger.error(f"Error getting referee training count: {e}")
        return jsonify({
            'positive_count': 0,
            'negative_count': 0,
            'total_count': 0
        })

@image_bp.route('/signal_classes', methods=['GET'])
def get_signal_classes():
    """Get list of signal classes."""
    try:
        # Try to load from YAML file first
        yaml_path = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER / 'data.yaml'
        
        if yaml_path.exists():
            import yaml
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'names' in yaml_data:
                        classes = yaml_data['names']
                        # Add 'none' if not present for negative samples
                        if 'none' not in classes:
                            classes.append('none')
                        return jsonify({'classes': classes})
            except Exception as e:
                logger.warning(f"Failed to load classes from YAML: {e}")
        
        # Fallback to default classes
        from config.settings import ModelConfig
        default_classes = ['armLeft', 'armRight', 'hits', 'leftServe', 'net', 'outside', 'rightServe', 'touched', 'none']
        
        return jsonify({'classes': default_classes})
        
    except Exception as e:
        logger.error(f"Error getting signal classes: {e}")
        return jsonify({'classes': ['none']})

@image_bp.route('/signal_class_counts', methods=['GET'])
def get_signal_class_counts():
    """Get count of images per signal class."""
    try:
        signal_data_dir = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER
        class_counts = {}
        
        if signal_data_dir.exists():
            # Get signal classes
            signal_classes_response = get_signal_classes()
            signal_classes = signal_classes_response.get_json().get('classes', [])
            
            # Initialize counts
            for class_name in signal_classes:
                class_counts[class_name] = 0
            
            # Count files by examining label files
            label_files = list(signal_data_dir.glob('*.txt'))
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(signal_classes):
                                    class_name = signal_classes[class_id]
                                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                except Exception:
                    continue
            
            # Also count files with class names in filename
            for ext in ['.png', '.jpg', '.jpeg']:
                image_files = list(signal_data_dir.glob(f'*{ext}'))
                for img_file in image_files:
                    name = img_file.name.lower()
                    for class_name in signal_classes:
                        if class_name.lower() in name:
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            break
        
        return jsonify(class_counts)
        
    except Exception as e:
        logger.error(f"Error getting signal class counts: {e}")
        return jsonify({})

@image_bp.route('/move_referee_training', methods=['POST'])
def move_referee_training():
    """Move referee training data to global training folder."""
    try:
        import shutil
        from pathlib import Path
        
        # Source and destination directories
        src_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
        dst_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'referee_training_data'
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        moved_files = []
        
        if src_dir.exists():
            for file_path in src_dir.iterdir():
                if file_path.is_file():
                    dst_path = dst_dir / file_path.name
                    if not dst_path.exists():  # Avoid overwriting
                        shutil.move(str(file_path), str(dst_path))
                        moved_files.append(file_path.name)
        
        return jsonify({
            'status': 'success',
            'moved': moved_files,
            'count': len(moved_files),
            'dst': str(dst_dir)
        })
        
    except Exception as e:
        logger.error(f"Error moving referee training data: {e}")
        return jsonify({'error': str(e)}), 500

@image_bp.route('/move_signal_training', methods=['POST'])
def move_signal_training():
    """Move signal training data to global training folder."""
    try:
        import shutil
        from pathlib import Path
        
        # Source and destination directories
        src_dir = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER
        dst_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'signal_training_data'
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        moved_files = []
        
        if src_dir.exists():
            for file_path in src_dir.iterdir():
                if file_path.is_file():
                    dst_path = dst_dir / file_path.name
                    if not dst_path.exists():  # Avoid overwriting
                        shutil.move(str(file_path), str(dst_path))
                        moved_files.append(file_path.name)
        
        return jsonify({
            'status': 'success',
            'moved': moved_files,
            'count': len(moved_files),
            'dst': str(dst_dir)
        })
        
    except Exception as e:
        logger.error(f"Error moving signal training data: {e}")
        return jsonify({'error': str(e)}), 500

@image_bp.route('/delete_referee_training_data', methods=['POST'])
def delete_referee_training_data():
    """Delete all referee training data."""
    try:
        import shutil
        
        referee_data_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
        deleted_count = 0
        
        if referee_data_dir.exists():
            # Count files before deletion
            all_files = list(referee_data_dir.iterdir())
            deleted_count = len([f for f in all_files if f.is_file()])
            
            # Delete all files in the directory
            for file_path in all_files:
                if file_path.is_file():
                    file_path.unlink()
        
        return jsonify({
            'status': 'success',
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} referee training files'
        })
        
    except Exception as e:
        logger.error(f"Error deleting referee training data: {e}")
        return jsonify({'error': str(e)}), 500

@image_bp.route('/delete_signal_training_data', methods=['POST'])
def delete_signal_training_data():
    """Delete all signal training data."""
    try:
        import shutil
        
        signal_data_dir = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER
        deleted_count = 0
        
        if signal_data_dir.exists():
            # Count files before deletion
            all_files = list(signal_data_dir.iterdir())
            deleted_count = len([f for f in all_files if f.is_file() and not f.name == 'data.yaml'])
            
            # Delete all files except data.yaml
            for file_path in all_files:
                if file_path.is_file() and file_path.name != 'data.yaml':
                    file_path.unlink()
        
        return jsonify({
            'status': 'success',
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} signal training files'
        })
        
    except Exception as e:
        logger.error(f"Error deleting signal training data: {e}")
        return jsonify({'error': str(e)}), 500

@image_bp.route('/process_signal', methods=['POST'])
def process_signal():
    """Process a signal detection request from labeling queue."""
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
        
        # Check if file exists in uploads directory
        upload_path = DirectoryConfig.UPLOAD_FOLDER / filename
        if not upload_path.exists():
            return jsonify({'error': 'Image not found in queue'}), 404
        
        # Import and use signal detection
        from app.models.signal_classifier import detect_signal
        result = detect_signal(str(upload_path))
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'predicted_class': result.get('predicted_class'),
            'confidence': result.get('confidence', 0.0),
            'bbox_xywhn': result.get('bbox_xywhn'),
            'message': 'Signal detection completed'
        })
        
    except Exception as e:
        logger.error(f"Error in process_signal: {e}")
        return jsonify({'error': str(e)}), 500 