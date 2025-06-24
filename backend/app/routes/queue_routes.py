"""
Queue Management Routes

This module contains routes for managing image processing queues
and batch operations.
"""

from flask import Blueprint, request, jsonify
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

# Import configuration
import sys
from pathlib import Path

# Add necessary paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config.settings import DirectoryConfig
except ImportError:
    from app.config.settings import DirectoryConfig

queue_bp = Blueprint('queue', __name__)

@queue_bp.route('/status', methods=['GET'])
def get_queue_status():
    """Get queue processing status."""
    try:
        return jsonify({
            'queue_length': 0,
            'status': 'Queue processing not yet implemented in new system',
            'message': 'Please use the old system for queue functionality'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@queue_bp.route('/image/<filename>', methods=['DELETE'])
def delete_queue_image(filename):
    """Delete an image from the labeling queue."""
    try:
        # Check if the image exists in the uploads folder
        image_path = Path(DirectoryConfig.UPLOAD_FOLDER) / filename
        
        if not image_path.exists():
            return jsonify({'error': 'Image not found in queue'}), 404
        
        # Delete the image file
        image_path.unlink()
        logger.info(f"Deleted image from queue: {filename}")
        
        # Also clean up any associated temporary crop files
        crops_folder = Path(DirectoryConfig.CROPS_FOLDER)
        temp_crop_name = f"temp_crop_{filename}"
        temp_crop_path = crops_folder / temp_crop_name
        
        if temp_crop_path.exists():
            temp_crop_path.unlink()
            logger.info(f"Deleted associated temp crop: {temp_crop_name}")
        
        return jsonify({
            'success': True,
            'message': f'Image "{filename}" has been permanently deleted from the queue'
        })
        
    except Exception as e:
        logger.error(f"Error deleting image {filename}: {e}")
        return jsonify({'error': str(e)}), 500 