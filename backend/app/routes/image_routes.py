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