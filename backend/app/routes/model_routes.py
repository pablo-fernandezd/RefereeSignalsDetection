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
model_bp = Blueprint('model', __name__, url_prefix='/api/models')


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
