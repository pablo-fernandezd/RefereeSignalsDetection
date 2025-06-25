"""
Image Processing Routes for Referee Detection System

This module contains all routes related to image upload, processing,
and referee detection functionality.
"""

from flask import Blueprint, request, jsonify, send_from_directory
from pathlib import Path
import os
import logging
import time

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
        
        # Save the original image to referee training data
        try:
            referee_training_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
            referee_training_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename for positive referee sample
            base_name = original_path.stem  # filename without extension
            extension = original_path.suffix  # .jpg, .png, etc.
            positive_filename = f"referee_positive_{base_name}{extension}"
            positive_txt_filename = f"referee_positive_{base_name}.txt"
            
            positive_path = referee_training_dir / positive_filename
            positive_label_path = referee_training_dir / positive_txt_filename
            
            # Ensure unique filename if it already exists
            counter = 1
            while positive_path.exists():
                positive_filename = f"referee_positive_{base_name}_{counter}{extension}"
                positive_txt_filename = f"referee_positive_{base_name}_{counter}.txt"
                positive_path = referee_training_dir / positive_filename
                positive_label_path = referee_training_dir / positive_txt_filename
                counter += 1
            
            # Move original image to referee training
            import shutil
            shutil.move(str(original_path), str(positive_path))
            
            # Create non-empty txt file for positive sample (referee detected)
            # Use normalized bbox coordinates for YOLO format
            with open(positive_label_path, 'w') as f:
                # Convert bbox to normalized coordinates
                # bbox is [x1, y1, x2, y2] in pixels
                import cv2
                temp_img = cv2.imread(str(positive_path))
                if temp_img is not None:
                    img_height, img_width = temp_img.shape[:2]
                    x1, y1, x2, y2 = bbox
                    
                    # Convert to center_x, center_y, width, height (normalized)
                    center_x = ((x1 + x2) / 2) / img_width
                    center_y = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Class 0 for referee (assuming single class)
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                else:
                    # Fallback if image can't be read
                    f.write("0 0.5 0.5 1.0 1.0\n")
            
            logger.info(f"Saved positive referee sample: {positive_filename} with bbox in {positive_txt_filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save positive referee sample: {e}")
        
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
            # Save as negative sample - move original image to referee training
            logger.info("Saving as negative sample")
            
            try:
                # Get the original image path
                original_path = DirectoryConfig.UPLOAD_FOLDER / original_filename
                if not original_path.exists():
                    return jsonify({'error': f'Original image not found: {original_filename}'}), 404
                
                # Generate filename for negative referee sample
                referee_training_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
                referee_training_dir.mkdir(parents=True, exist_ok=True)
                
                # Use original filename but with negative prefix
                base_name = original_path.stem  # filename without extension
                extension = original_path.suffix  # .jpg, .png, etc.
                negative_filename = f"referee_negative_{base_name}{extension}"
                negative_txt_filename = f"referee_negative_{base_name}.txt"
                
                negative_path = referee_training_dir / negative_filename
                negative_label_path = referee_training_dir / negative_txt_filename
                
                # Ensure unique filename if it already exists
                counter = 1
                while negative_path.exists():
                    negative_filename = f"referee_negative_{base_name}_{counter}{extension}"
                    negative_txt_filename = f"referee_negative_{base_name}_{counter}.txt"
                    negative_path = referee_training_dir / negative_filename
                    negative_label_path = referee_training_dir / negative_txt_filename
                    counter += 1
                
                # Move image to referee training
                import shutil
                shutil.move(str(original_path), str(negative_path))
                
                # Create empty txt file for negative sample (same name as image but .txt)
                with open(negative_label_path, 'w') as f:
                    pass  # Empty file
                
                logger.info(f"Saved negative referee sample: {negative_filename} with empty {negative_txt_filename}")
                
                return jsonify({
                    'status': 'ok',
                    'action': 'saved_as_negative',
                    'message': f'Image saved as negative sample: {negative_filename}'
                })
                
            except Exception as e:
                logger.error(f"Failed to save negative referee sample: {e}")
                return jsonify({'error': f'Failed to save negative sample: {str(e)}'}), 500
        
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
        
        # Save the original image to referee training data with YOLO annotation
        try:
            # Check if the original image has been processed before
            from app.utils.hash_utils import is_duplicate_image, register_image_hash, clear_hash_cache
            
            # Clear cache to ensure fresh data
            clear_hash_cache()
            
            # Check for duplicate
            if is_duplicate_image(original_path):
                logger.info(f"Duplicate image detected: {original_filename}")
                # Still return success but note it's a duplicate
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
            
            # Generate filename for positive referee sample
            referee_training_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
            referee_training_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = original_path.stem  # filename without extension
            extension = original_path.suffix  # .jpg, .png, etc.
            positive_filename = f"referee_confirmed_{int(datetime.now().timestamp() * 1000)}{extension}"
            positive_txt_filename = f"referee_confirmed_{int(datetime.now().timestamp() * 1000)}.txt"
            
            positive_path = referee_training_dir / positive_filename
            positive_label_path = referee_training_dir / positive_txt_filename
            
            # Ensure unique filename if it already exists
            counter = 1
            while positive_path.exists():
                timestamp_ms = int(datetime.now().timestamp() * 1000)
                positive_filename = f"referee_confirmed_{timestamp_ms}_{counter}{extension}"
                positive_txt_filename = f"referee_confirmed_{timestamp_ms}_{counter}.txt"
                positive_path = referee_training_dir / positive_filename
                positive_label_path = referee_training_dir / positive_txt_filename
                counter += 1
            
            # Move original image to referee training
            import shutil
            shutil.move(str(original_path), str(positive_path))
            
            # Create YOLO annotation file for positive sample (referee detected)
            with open(positive_label_path, 'w') as f:
                # Convert bbox to normalized coordinates for YOLO format
                # bbox is [x1, y1, x2, y2] in pixels
                img_height, img_width = image.shape[:2]
                
                # Convert to center_x, center_y, width, height (normalized)
                center_x = ((x1 + x2) / 2) / img_width
                center_y = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Class 0 for referee (assuming single class)
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            logger.info(f"Moved original image to referee training: {positive_filename}")
            logger.info(f"Created YOLO annotation file: {positive_txt_filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save positive referee sample: {e}")
        
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
            # No referee detected - save as negative sample to referee training
            try:
                # Generate filename for negative referee sample using original filename
                referee_training_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
                referee_training_dir.mkdir(parents=True, exist_ok=True)
                
                # Use original filename but with negative prefix
                base_name = upload_path.stem  # filename without extension
                extension = upload_path.suffix  # .jpg, .png, etc.
                negative_filename = f"referee_negative_{base_name}{extension}"
                negative_txt_filename = f"referee_negative_{base_name}.txt"
                
                negative_path = referee_training_dir / negative_filename
                negative_label_path = referee_training_dir / negative_txt_filename
                
                # Ensure unique filename if it already exists
                counter = 1
                while negative_path.exists():
                    negative_filename = f"referee_negative_{base_name}_{counter}{extension}"
                    negative_txt_filename = f"referee_negative_{base_name}_{counter}.txt"
                    negative_path = referee_training_dir / negative_filename
                    negative_label_path = referee_training_dir / negative_txt_filename
                    counter += 1
                
                # Move image to referee training
                import shutil
                shutil.move(str(upload_path), str(negative_path))
                
                # Create empty txt file for negative sample (same name as image but .txt)
                with open(negative_label_path, 'w') as f:
                    pass  # Empty file
                
                logger.info(f"Saved negative referee sample: {negative_filename} with empty {negative_txt_filename}")
                
            except Exception as e:
                logger.warning(f"Failed to save negative referee sample: {e}")
            
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
    """Get count of referee training data by counting txt files only."""
    try:
        # Count txt files in referee training data directory
        referee_data_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
        
        positive_count = 0
        negative_count = 0
        
        if referee_data_dir.exists():
            # Count txt files only
            txt_files = list(referee_data_dir.glob('*.txt'))
            
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r') as f:
                        content = f.read().strip()
                    
                    if not content:
                        # Empty txt file = negative sample (no referee)
                        negative_count += 1
                    else:
                        # Non-empty txt file = positive sample (referee detected)
                        positive_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error reading {txt_file}: {e}")
                    continue
        
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
    """Get count of images per signal class by counting txt files only."""
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
            
            # Count ONLY txt files
            label_files = list(signal_data_dir.glob('*.txt'))
            
            for label_file in label_files:
                # Skip data.yaml related files
                if label_file.name == 'data.yaml':
                    continue
                    
                try:
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                        
                    if not content:
                        # Empty txt file = negative sample (none/no signal)
                        class_counts['none'] = class_counts.get('none', 0) + 1
                    else:
                        # Parse the content for class ID
                        lines = content.split('\n')
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(signal_classes) - 1:  # Exclude 'none' from regular classes
                                    class_name = signal_classes[class_id]
                                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                except Exception as e:
                    logger.warning(f"Error reading {label_file}: {e}")
                    continue
        
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
    """Delete all referee training data except yaml files."""
    try:
        import shutil
        
        referee_data_dir = DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER
        deleted_count = 0
        
        if referee_data_dir.exists():
            # Count files before deletion (excluding yaml files)
            all_files = list(referee_data_dir.iterdir())
            deleted_count = len([f for f in all_files if f.is_file() and not f.name.endswith('.yaml')])
            
            # Delete all files except yaml files
            for file_path in all_files:
                if file_path.is_file() and not file_path.name.endswith('.yaml'):
                    file_path.unlink()
        
        return jsonify({
            'status': 'success',
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} referee training files (kept yaml files)'
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
        
        # Check if file exists in crops directory (where manual crops are saved)
        crop_path = DirectoryConfig.CROPS_FOLDER / filename
        if not crop_path.exists():
            # Also check uploads directory as fallback
            upload_path = DirectoryConfig.UPLOAD_FOLDER / filename
            if upload_path.exists():
                crop_path = upload_path
            else:
                return jsonify({'error': f'Image not found: {filename}'}), 404
        
        # Import and use signal detection
        from app.models.signal_classifier import detect_signal
        result = detect_signal(str(crop_path))
        
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

@image_bp.route('/confirm_signal', methods=['POST'])
def confirm_signal():
    """Confirm signal classification and save to training data."""
    try:
        data = request.json
        original_filename = data.get('original_filename')
        crop_filename_for_signal = data.get('crop_filename_for_signal')
        selected_class = data.get('selected_class')
        confidence = data.get('confidence', 0.0)
        
        # Validate required parameters
        if not all([original_filename, crop_filename_for_signal, selected_class]):
            return jsonify({'error': 'Missing required parameters: original_filename, crop_filename_for_signal, and selected_class are required'}), 400
        
        # Check if crop file exists
        crop_path = DirectoryConfig.CROPS_FOLDER / crop_filename_for_signal
        if not crop_path.exists():
            # Also check uploads directory as fallback
            upload_path = DirectoryConfig.UPLOAD_FOLDER / crop_filename_for_signal
            if upload_path.exists():
                crop_path = upload_path
            else:
                return jsonify({'error': f'Crop file not found: {crop_filename_for_signal}'}), 404
        
        # Always save to signal training data with the selected class
        try:
            # Import hash utilities
            from app.utils.hash_utils import calculate_image_hash, is_hash_registered
            import time
            
            # Calculate hash to check for duplicates
            image_hash = calculate_image_hash(str(crop_path))
            
            if is_hash_registered(image_hash):
                return jsonify({
                    'status': 'warning',
                    'action': 'duplicate_detected',
                    'message': f'This image appears to be a duplicate and has not been saved for training.'
                })
            
            # Generate unique filename for training data
            timestamp = int(time.time() * 1000000)
            training_filename = f"signal_{selected_class}_{timestamp}"
            
            # Copy image to training folder
            signal_training_dir = DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER
            signal_training_dir.mkdir(parents=True, exist_ok=True)
            
            training_image_path = signal_training_dir / f"{training_filename}.jpg"
            training_label_path = signal_training_dir / f"{training_filename}.txt"
            
            # Copy the image
            import shutil
            shutil.copy2(str(crop_path), str(training_image_path))
            
            # Create YOLO format label
            with open(training_label_path, 'w') as f:
                if selected_class == 'none':
                    # Empty file for negative samples (no signal)
                    pass  # Write nothing - empty file
                else:
                    # Get class ID from signal classes (excluding 'none')
                    signal_classes_response = get_signal_classes()
                    signal_classes = signal_classes_response.get_json().get('classes', [])
                    signal_classes_without_none = [cls for cls in signal_classes if cls != 'none']
                    
                    try:
                        class_id = signal_classes_without_none.index(selected_class)
                        # Full image bounding box (normalized coordinates)
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                    except ValueError:
                        # If class not found, leave empty (negative sample)
                        pass
            
            # Add hash to hash file  
            hash_file = DirectoryConfig.IMAGE_HASH_REGISTRY
            hash_file.parent.mkdir(parents=True, exist_ok=True)
            with open(hash_file, 'a') as f:
                f.write(f"{image_hash}\n")
            
            # Clean up the original crop file
            crop_path.unlink()
            
            # Note: Original image should already be moved to referee training by manual_crop endpoint
            # No need to move it again here
            logger.info(f"Signal training data saved. Original image should already be in referee training data.")
            
            return jsonify({
                'status': 'success',
                'message': f'Signal "{selected_class}" saved to training data. Original image moved to referee training.',
                'training_file': training_filename,
                'confidence': confidence
            })
            
        except Exception as e:
            logger.error(f"Error saving signal training data: {e}")
            return jsonify({'error': f'Failed to save training data: {str(e)}'}), 500

        
    except Exception as e:
        logger.error(f"Error in confirm_signal: {e}")
        return jsonify({'error': str(e)}), 500 