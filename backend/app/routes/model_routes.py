"""
Model Registry API Routes

This module provides REST API endpoints for the model registry system:
- Model listing and filtering
- Ultralytics model downloads
- Custom model uploads
- Model deployment and activation
- Model deletion and management
- Registry statistics
"""

import os
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Import the model registry
from app.models.model_registry import get_model_registry, UltralyticsDownloader

logger = logging.getLogger(__name__)

# Create blueprint
model_bp = Blueprint('model', __name__)


@model_bp.route('/registry/stats', methods=['GET'])
def get_registry_stats():
    """Get model registry statistics."""
    try:
        registry = get_model_registry()
        stats = registry.get_registry_stats()
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get registry stats: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/list', methods=['GET'])
def list_models():
    """List all models, optionally filtered by type."""
    try:
        model_type = request.args.get('type')
        registry = get_model_registry()
        models = registry.get_models(model_type)
        
        return jsonify({
            'status': 'success',
            'models': models,
            'total': len(models),
            'filtered_by': model_type
        })
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/active/<model_type>', methods=['GET'])
def get_active_model(model_type):
    """Get the active model for a specific type."""
    try:
        registry = get_model_registry()
        active_model = registry.get_active_model(model_type)
        
        if not active_model:
            return jsonify({
                'status': 'success',
                'active_model': None,
                'message': f'No active model for {model_type}'
            })
        
        return jsonify({
            'status': 'success',
            'active_model': active_model
        })
        
    except Exception as e:
        logger.error(f"Failed to get active model: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/ultralytics/available', methods=['GET'])
def get_available_ultralytics_models():
    """Get list of available Ultralytics models for download."""
    try:
        models = UltralyticsDownloader.get_available_models()
        
        return jsonify({
            'status': 'success',
            'available_models': models,
            'total': len(models)
        })
        
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/ultralytics/download', methods=['POST'])
def download_ultralytics_model():
    """Download a model from Ultralytics."""
    try:
        data = request.get_json()
        
        model_type = data.get('model_type')
        model_name = data.get('model_name')
        description = data.get('description', '')
        
        if not model_type or not model_name:
            return jsonify({'error': 'Missing model_type or model_name'}), 400
        
        registry = get_model_registry()
        success, message, model_id = registry.add_ultralytics_model(
            model_type, model_name, description, auto_download=True
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'model_id': model_id
            })
        else:
            return jsonify({'error': message}), 500
        
    except Exception as e:
        logger.error(f"Failed to download Ultralytics model: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/upload', methods=['POST'])
def upload_model():
    """Upload a custom model file."""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get form data
        model_type = request.form.get('model_type')
        description = request.form.get('description', '')
        tags = request.form.get('tags', '').split(',') if request.form.get('tags') else []
        
        if not model_type:
            return jsonify({'error': 'Missing model_type'}), 400
        
        # Validate file type
        if not file.filename.lower().endswith('.pt'):
            return jsonify({'error': 'Only .pt files are supported'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_dir = Path(__file__).parent.parent.parent / 'temp_uploads'
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / filename
        
        file.save(str(temp_path))
        
        try:
            # Add to registry
            registry = get_model_registry()
            success, message, model_id = registry.upload_model(
                model_type, temp_path, description, tags
            )
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': message,
                    'model_id': model_id
                })
            else:
                return jsonify({'error': message}), 500
                
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
        
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/deploy/<model_id>', methods=['POST'])
def deploy_model(model_id):
    """Deploy a model as active for inference."""
    try:
        registry = get_model_registry()
        success, message = registry.deploy_model(model_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message
            })
        else:
            return jsonify({'error': message}), 500
        
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/delete/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model from the registry."""
    try:
        registry = get_model_registry()
        success, message = registry.delete_model(model_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message
            })
        else:
            return jsonify({'error': message}), 500
        
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/edit/<model_id>', methods=['PUT'])
def edit_model(model_id):
    """Edit model metadata (description, tags, version, etc.)."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        registry = get_model_registry()
        success, message = registry.update_model_metadata(
            model_id,
            description=data.get('description'),
            tags=data.get('tags', []),
            performance_metrics=data.get('performance_metrics', {}),
            training_config=data.get('training_config', {}),
            version=data.get('version')
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message
            })
        else:
            return jsonify({'error': message}), 500
        
    except Exception as e:
        logger.error(f"Failed to edit model: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/download/<model_id>', methods=['GET'])
def download_model_file(model_id):
    """Download a model file."""
    try:
        registry = get_model_registry()
        
        # Find the model
        model_data = None
        for model in registry.get_models():
            if model['model_id'] == model_id:
                model_data = model
                break
        
        if not model_data:
            return jsonify({'error': 'Model not found'}), 404
        
        file_path = Path(model_data['file_path'])
        if not file_path.exists():
            return jsonify({'error': 'Model file not found'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"{model_id}.pt",
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Failed to download model file: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/types', methods=['GET'])
def get_model_types():
    """Get all available model types."""
    try:
        registry = get_model_registry()
        stats = registry.get_registry_stats()
        
        model_types = []
        for model_type, count in stats['models_by_type'].items():
            active_model = registry.get_active_model(model_type)
            model_types.append({
                'type': model_type,
                'count': count,
                'active_model': active_model['model_id'] if active_model else None,
                'has_active': active_model is not None
            })
        
        return jsonify({
            'status': 'success',
            'model_types': model_types
        })
        
    except Exception as e:
        logger.error(f"Failed to get model types: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/import/legacy', methods=['POST'])
def import_legacy_models():
    """Import existing models from legacy directories."""
    try:
        registry = get_model_registry()
        registry._import_existing_models()
        
        return jsonify({
            'status': 'success',
            'message': 'Legacy models imported successfully'
        })
        
    except Exception as e:
        logger.error(f"Failed to import legacy models: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/base/download/<model_name>', methods=['POST'])
def download_base_model(model_name):
    """Download a YOLO base model for training new models."""
    try:
        data = request.get_json() or {}
        
        model_type = data.get('model_type', 'custom')
        description = data.get('description', f'Base {model_name} model for training')
        
        if not model_name:
            return jsonify({'error': 'Missing model_name'}), 400
        
        # Check if model is available
        available_models = UltralyticsDownloader.get_available_models()
        if model_name not in available_models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        registry = get_model_registry()
        success, message, model_id = registry.add_ultralytics_model(
            model_type, model_name, description, auto_download=True
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'model_id': model_id,
                'model_info': available_models[model_name]
            })
        else:
            return jsonify({'error': message}), 500
        
    except Exception as e:
        logger.error(f"Failed to download base model: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/workflow/types', methods=['GET'])
def get_workflow_types():
    """Get available workflow types for model creation."""
    try:
        workflow_types = [
            {
                'type': 'referee',
                'name': 'Referee Detection',
                'description': 'Detect referees in volleyball images',
                'input': 'Full volleyball court images',
                'output': 'Referee bounding boxes',
                'base_models': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
            },
            {
                'type': 'signal',
                'name': 'Signal Classification',
                'description': 'Classify referee hand signals',
                'input': 'Referee crop images',
                'output': 'Signal class predictions',
                'base_models': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
            },
            {
                'type': 'player',
                'name': 'Player Detection',
                'description': 'Detect players in volleyball images',
                'input': 'Full volleyball court images',
                'output': 'Player bounding boxes',
                'base_models': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
            },
            {
                'type': 'ball',
                'name': 'Ball Detection',
                'description': 'Detect volleyball in images',
                'input': 'Volleyball court images',
                'output': 'Ball bounding boxes',
                'base_models': ['yolov8n', 'yolov8s', 'yolov8m']
            },
            {
                'type': 'court',
                'name': 'Court Segmentation',
                'description': 'Segment volleyball court areas',
                'input': 'Full court images',
                'output': 'Court segmentation masks',
                'base_models': ['yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg']
            },
            {
                'type': 'custom',
                'name': 'Custom Detection',
                'description': 'Custom object detection model',
                'input': 'Custom images',
                'output': 'Custom object detections',
                'base_models': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'yolo11n', 'yolo11s', 'yolo11m']
            }
        ]
        
        return jsonify({
            'status': 'success',
            'workflow_types': workflow_types
        })
        
    except Exception as e:
        logger.error(f"Failed to get workflow types: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/validate_for_training/<model_id>', methods=['GET'])
def validate_model_for_training(model_id):
    """Validate if a model can be used for training."""
    try:
        registry = get_model_registry()
        can_train, message, validation_details = registry.validate_model_for_training(model_id)
        
        return jsonify({
            'status': 'success',
            'can_train': can_train,
            'message': message,
            'validation_details': validation_details
        })
        
    except Exception as e:
        logger.error(f"Failed to validate model for training: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/detect_version/<model_id>', methods=['POST'])
def detect_model_version(model_id):
    """Re-detect YOLO version for a model."""
    try:
        registry = get_model_registry()
        
        # Find the model
        model_metadata = None
        for models in registry.models.values():
            for model in models:
                if model.model_id == model_id:
                    model_metadata = model
                    break
            if model_metadata:
                break
        
        if not model_metadata:
            return jsonify({'error': f'Model {model_id} not found'}), 404
        
        if not model_metadata.file_path or not model_metadata.file_path.exists():
            return jsonify({'error': f'Model file not found: {model_metadata.file_path}'}), 404
        
        # Re-detect version
        detection_results = registry._detect_and_validate_model(
            model_metadata.file_path, 
            model_metadata.model_type
        )
        
        # Update metadata
        model_metadata.yolo_version = detection_results.get('yolo_version', 'unknown')
        model_metadata.yolo_architecture = detection_results.get('yolo_architecture')
        model_metadata.compatibility_status = detection_results.get('compatibility_status', 'unknown')
        model_metadata.validation_results = detection_results.get('validation_results', {})
        
        # Add version to tags
        if model_metadata.yolo_version and model_metadata.yolo_version != 'unknown':
            if model_metadata.yolo_version not in model_metadata.tags:
                model_metadata.tags.append(model_metadata.yolo_version)
        if model_metadata.yolo_architecture:
            if model_metadata.yolo_architecture not in model_metadata.tags:
                model_metadata.tags.append(model_metadata.yolo_architecture)
        
        # Save changes
        registry._save_registry()
        
        return jsonify({
            'status': 'success',
            'detection_results': detection_results,
            'updated_model': model_metadata.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Failed to detect model version: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/update_version/<model_id>', methods=['POST'])
def update_model_version(model_id):
    """Manually update YOLO version information for a model."""
    try:
        data = request.get_json()
        yolo_version = data.get('yolo_version')
        yolo_architecture = data.get('yolo_architecture')
        
        if not yolo_version:
            return jsonify({'error': 'yolo_version is required'}), 400
        
        registry = get_model_registry()
        success, message = registry.update_model_yolo_version(
            model_id, yolo_version, yolo_architecture
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message
            })
        else:
            return jsonify({'error': message}), 400
        
    except Exception as e:
        logger.error(f"Failed to update model version: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/supported_versions', methods=['GET'])
def get_supported_yolo_versions():
    """Get list of supported YOLO versions."""
    try:
        # Import detection utilities
        try:
            from app.utils.model_detection import YOLOVersionDetector
            supported_versions = YOLOVersionDetector.get_supported_versions()
            version_info = {}
            
            for version in supported_versions:
                info = YOLOVersionDetector.get_version_info(version)
                if info:
                    version_info[version] = {
                        'architecture_names': info.get('architecture_names', []),
                        'framework': info.get('framework', 'unknown'),
                        'note': info.get('note', '')
                    }
            
            return jsonify({
                'status': 'success',
                'supported_versions': supported_versions,
                'version_info': version_info
            })
            
        except ImportError:
            # Fallback list if detection utilities not available
            return jsonify({
                'status': 'success',
                'supported_versions': ['yolov8', 'yolov9', 'yolov10', 'yolo11', 'yolov12'],
                'version_info': {
                    'yolov8': {'architecture_names': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']},
                    'yolo11': {'architecture_names': ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']},
                    'yolov12': {'architecture_names': ['yolov12n', 'yolov12s', 'yolov12m', 'yolov12l', 'yolov12x']}
                },
                'note': 'Detection utilities not available, showing basic list'
            })
        
    except Exception as e:
        logger.error(f"Failed to get supported versions: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/compatibility_report', methods=['GET'])
def get_compatibility_report():
    """Get compatibility report for all models."""
    try:
        registry = get_model_registry()
        models = registry.get_models()
        
        report = {
            'total_models': len(models),
            'compatible': 0,
            'incompatible': 0,
            'warning': 0,
            'unknown': 0,
            'by_version': {},
            'issues_summary': [],
            'models_detail': []
        }
        
        for model in models:
            # Ensure compatibility_status is never None
            status = model.get('compatibility_status') or 'unknown'
            if status in report:
                report[status] += 1
            else:
                report['unknown'] += 1
            
            # Count by version - handle None values
            version = model.get('yolo_version') or 'unknown'
            if version not in report['by_version']:
                report['by_version'][version] = {'count': 0, 'compatible': 0}
            report['by_version'][version]['count'] += 1
            if status == 'compatible':
                report['by_version'][version]['compatible'] += 1
            
            # Collect issues - handle None validation_results
            validation_results = model.get('validation_results') or {}
            issues = validation_results.get('issues') or []
            
            # Ensure issues is a list
            if not isinstance(issues, list):
                issues = []
                
            for issue in issues:
                if issue and issue not in report['issues_summary']:
                    report['issues_summary'].append(issue)
            
            # Get recommendations safely
            recommendations = validation_results.get('recommendations') or []
            if not isinstance(recommendations, list):
                recommendations = []
            
            # Add model detail - handle None values
            report['models_detail'].append({
                'model_id': model.get('model_id', 'unknown'),
                'model_type': model.get('model_type', 'unknown'),
                'version': model.get('version', 'unknown'),
                'yolo_version': version,
                'yolo_architecture': model.get('yolo_architecture') or None,
                'compatibility_status': status,
                'source': model.get('source', 'unknown'),
                'issues': issues,
                'recommendations': recommendations
            })
        
        return jsonify({
            'status': 'success',
            'compatibility_report': report
        })
        
    except Exception as e:
        logger.error(f"Failed to generate compatibility report: {e}")
        return jsonify({'error': str(e)}), 500
