"""
Image Processing Routes for Referee Detection System

This module contains all routes related to image upload, processing,
and referee detection functionality. Business logic has been moved
to the service layer for better separation of concerns.
"""

from flask import Blueprint, request, jsonify, send_from_directory
import logging

from ..services import ImageService, ImageProcessingError, TrainingDataService, TrainingDataError
from ..utils import validate_image_upload
from ..config.settings import DirectoryConfig

logger = logging.getLogger(__name__)

image_bp = Blueprint('image', __name__)


@image_bp.route('/upload', methods=['POST'])
def upload_image():
    """Upload and process an image for referee detection."""
    try:
        # Validate upload
        validation_result = validate_image_upload(request)
        if not validation_result['valid']:
            return jsonify({'error': validation_result['error']}), 400
        
        # Process image using service
        file = request.files['image']
        result = ImageService.process_uploaded_image(file)
        
        if result['success']:
            return jsonify({
                'filename': result['filename'],
                'crop_filename': result['crop_filename'],
                'crop_url': result['crop_url'],
                'bbox': result['bbox']
            })
        else:
            return jsonify({
                'error': result['error'],
                'filename': result['filename']
            }), 404
            
    except ImageProcessingError as e:
        logger.error(f"Image processing error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in upload_image: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@image_bp.route('/pending_images', methods=['GET'])
def get_pending_images():
    """Get list of pending images for labeling."""
    try:
        result = ImageService.get_pending_images()
        return jsonify(result)
        
    except ImageProcessingError as e:
        logger.error(f"Error getting pending images: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in get_pending_images: {e}")
        return jsonify({'error': 'Internal server error'}), 500


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
        logger.error(f"Error serving crop image: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@image_bp.route('/uploads/<filename>')
def get_uploaded_image(filename):
    """Serve uploaded images."""
    try:
        return send_from_directory(DirectoryConfig.UPLOAD_FOLDER, filename)
    except Exception as e:
        logger.error(f"Error serving uploaded image: {e}")
        return jsonify({'error': 'Internal server error'}), 500


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
        
        # Use service to confirm crop
        result = ImageService.confirm_crop(original_filename, crop_filename, bbox)
        
        if result['status'] == 'warning':
            return jsonify(result)
        else:
            return jsonify(result)
            
    except ImageProcessingError as e:
        logger.error(f"Image processing error in confirm_crop: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in confirm_crop: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@image_bp.route('/manual_crop', methods=['POST'])
def manual_crop():
    """Process a manual crop of an image."""
    try:
        data = request.json
        original_filename = data.get('original_filename')
        bbox = data.get('bbox')
        class_id = data.get('class_id', 0)
        proceed_to_signal = data.get('proceedToSignal', True)
        
        if not original_filename:
            return jsonify({'error': 'Missing original_filename parameter'}), 400
        
        # Use service to create manual crop
        result = ImageService.create_manual_crop(
            original_filename, bbox, class_id, proceed_to_signal
        )
        
        return jsonify(result)
        
    except ImageProcessingError as e:
        logger.error(f"Image processing error in manual_crop: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in manual_crop: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# Training data related routes
@image_bp.route('/referee_training_count', methods=['GET'])
def get_referee_training_count():
    """Get count of referee training samples."""
    try:
        result = TrainingDataService.get_referee_training_count()
        return jsonify(result)
        
    except TrainingDataError as e:
        logger.error(f"Training data error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in get_referee_training_count: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@image_bp.route('/signal_classes', methods=['GET'])
def get_signal_classes():
    """Get available signal classes."""
    try:
        result = TrainingDataService.get_signal_classes()
        return jsonify(result)
        
    except TrainingDataError as e:
        logger.error(f"Training data error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in get_signal_classes: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@image_bp.route('/signal_class_counts', methods=['GET'])
def get_signal_class_counts():
    """Get count of samples for each signal class."""
    try:
        result = TrainingDataService.get_signal_class_counts()
        return jsonify(result)
        
    except TrainingDataError as e:
        logger.error(f"Training data error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in get_signal_class_counts: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@image_bp.route('/move_referee_training', methods=['POST'])
def move_referee_training():
    """Move referee training data to global directory."""
    try:
        result = TrainingDataService.move_referee_training_data()
        return jsonify(result)
        
    except TrainingDataError as e:
        logger.error(f"Training data error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in move_referee_training: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@image_bp.route('/move_signal_training', methods=['POST'])
def move_signal_training():
    """Move signal training data to global directory."""
    try:
        result = TrainingDataService.move_signal_training_data()
        return jsonify(result)
        
    except TrainingDataError as e:
        logger.error(f"Training data error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in move_signal_training: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@image_bp.route('/delete_referee_training_data', methods=['POST'])
def delete_referee_training_data():
    """Delete all referee training data."""
    try:
        result = TrainingDataService.delete_referee_training_data()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in delete_referee_training_data: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@image_bp.route('/delete_signal_training_data', methods=['POST'])
def delete_signal_training_data():
    """Delete all signal training data."""
    try:
        result = TrainingDataService.delete_signal_training_data()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in delete_signal_training_data: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# Signal processing related routes (kept minimal for now)
@image_bp.route('/process_crop_for_signal', methods=['POST'])
def process_crop_for_signal():
    """Process a crop for signal detection."""
    try:
        data = request.json
        crop_filename_for_signal = data.get('crop_filename_for_signal')
        
        if not crop_filename_for_signal:
            return jsonify({'error': 'Missing crop_filename_for_signal parameter'}), 400
        
        # Import signal classifier
        from ..models.signal_classifier import classify_signal
        
        crop_path = DirectoryConfig.CROPS_FOLDER / crop_filename_for_signal
        if not crop_path.exists():
            return jsonify({'error': 'Crop file not found'}), 404
        
        # Process signal
        signal_result = classify_signal(str(crop_path))
        
        return jsonify({
            'signal_image_url': f'/api/referee_crop_image/{crop_filename_for_signal}',
            'prediction': signal_result.get('prediction', 'unknown'),
            'confidence': signal_result.get('confidence', 0.0),
            'all_predictions': signal_result.get('all_predictions', [])
        })
        
    except Exception as e:
        logger.error(f"Error in process_crop_for_signal: {e}")
        return jsonify({'error': str(e)}), 500


@image_bp.route('/confirm_signal', methods=['POST'])
def confirm_signal():
    """Confirm signal classification."""
    try:
        data = request.json
        crop_filename_for_signal = data.get('crop_filename_for_signal')
        correct = data.get('correct', False)
        selected_class = data.get('selected_class')
        signal_bbox_yolo = data.get('signal_bbox_yolo')
        original_filename = data.get('original_filename')
        
        if not crop_filename_for_signal:
            return jsonify({'error': 'Missing crop_filename_for_signal parameter'}), 400
        
        # Import signal confirmation logic
        from ..models.signal_classifier import confirm_signal_classification
        
        result = confirm_signal_classification(
            crop_filename_for_signal=crop_filename_for_signal,
            correct=correct,
            selected_class=selected_class,
            signal_bbox_yolo=signal_bbox_yolo,
            original_filename=original_filename
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in confirm_signal: {e}")
        return jsonify({'error': str(e)}), 500 