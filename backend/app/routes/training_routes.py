"""
Training Routes for Model Training Workflow

This module provides API endpoints for:
- Dataset statistics and management
- Model training sessions
- Model version management
- Training metrics and monitoring
- Class management for signal models
- Data augmentation configuration
"""

from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import yaml
import os
import sys

# Add current directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

try:
    from app.utils.model_config import get_config_manager, DataAugmentationManager
except ImportError:
    # Fallback import method
    sys.path.insert(0, str(current_dir.parent))
    try:
        from utils.model_config import get_config_manager, DataAugmentationManager
    except ImportError:
        # Create minimal fallback classes
        class ModelConfigManager:
            def __init__(self, base_dir):
                self.base_dir = Path(base_dir)
                self.data_dir = self.base_dir / 'data'
            
            def get_all_model_types(self):
                return ['referee', 'signal']
            
            def get_model_classes(self, model_type):
                if model_type == 'referee':
                    return ['referee']
                elif model_type == 'signal':
                    # Load from YAML if available
                    yaml_path = self.data_dir / 'signal_training_data' / 'data.yaml'
                    if yaml_path.exists():
                        try:
                            with open(yaml_path, 'r') as f:
                                yaml_data = yaml.safe_load(f)
                                return yaml_data.get('names', ['armLeft', 'armRight', 'hits', 'leftServe', 'net', 'outside', 'rightServe', 'touched'])
                        except:
                            pass
                    return ['armLeft', 'armRight', 'hits', 'leftServe', 'net', 'outside', 'rightServe', 'touched']
                return []
            
            def get_data_directory(self, model_type):
                return self.data_dir / f'{model_type}_training_data'
            
            def update_model_classes(self, model_type, classes):
                if model_type == 'signal':
                    yaml_path = self.data_dir / 'signal_training_data' / 'data.yaml'
                    if yaml_path.exists():
                        try:
                            with open(yaml_path, 'r') as f:
                                yaml_data = yaml.safe_load(f) or {}
                            yaml_data['names'] = classes
                            yaml_data['nc'] = len(classes)
                            with open(yaml_path, 'w') as f:
                                yaml.dump(yaml_data, f, default_flow_style=False)
                            return True
                        except Exception as e:
                            logging.error(f"Failed to update YAML: {e}")
                return False
        
        class DataAugmentationManager:
            AVAILABLE_AUGMENTATIONS = {
                'horizontal_flip': {'name': 'Horizontal Flip', 'description': 'Randomly flip images horizontally', 'default': True},
                'rotation': {'name': 'Rotation', 'description': 'Randomly rotate images (±15 degrees)', 'default': True},
                'brightness': {'name': 'Brightness', 'description': 'Randomly adjust brightness', 'default': True},
                'contrast': {'name': 'Contrast', 'description': 'Randomly adjust contrast', 'default': True},
                'mosaic': {'name': 'Mosaic', 'description': 'Combine 4 images into one (YOLO-style)', 'default': True},
            }
            
            @classmethod
            def get_available_augmentations(cls):
                return cls.AVAILABLE_AUGMENTATIONS
            
            @classmethod
            def get_default_augmentations(cls):
                return {key: info['default'] for key, info in cls.AVAILABLE_AUGMENTATIONS.items()}
            
            @classmethod
            def validate_augmentation_config(cls, config):
                return {key: value for key, value in config.items() if key in cls.AVAILABLE_AUGMENTATIONS}
        
        def get_config_manager(base_dir=None):
            if base_dir is None:
                base_dir = Path(__file__).parent.parent.parent
            return ModelConfigManager(base_dir)

logger = logging.getLogger(__name__)

training_bp = Blueprint('training', __name__)

@training_bp.route('/test', methods=['GET'])
def test_route():
    """Test route to verify blueprint registration."""
    return jsonify({'message': 'Training routes are working!'})

@training_bp.route('/dataset_stats/<model_type>', methods=['GET'])
def get_dataset_stats(model_type):
    """Get dataset statistics for a model type."""
    try:
        config_manager = get_config_manager()
        
        if model_type not in config_manager.get_all_model_types():
            return jsonify({'error': 'Invalid model_type'}), 400
        
        # Get source data directory dynamically
        source_dir = config_manager.get_data_directory(model_type)
        
        if not source_dir.exists():
            return jsonify({
                'status': 'success',
                'stats': {
                    'total_images': 0,
                    'total_labels': 0,
                    'classes': {},
                    'message': 'No training data found'
                }
            })
        
        # Count files
        image_files = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
        label_files = list(source_dir.glob('*.txt'))
        
        # Count classes (for detection models)
        class_counts = {}
        classes = config_manager.get_model_classes(model_type)
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(classes):
                                class_name = classes[class_id]
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            except Exception:
                continue
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_images': len(image_files),
                'total_labels': len(label_files),
                'classes': class_counts,
                'model_type': model_type
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get dataset stats: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/model_versions', methods=['GET'])
def get_model_versions():
    """Get all model versions with detailed training metrics."""
    try:
        model_type = request.args.get('model_type', 'referee')
        
        # Mock data with comprehensive metrics for demonstration
        mock_versions = [
            {
                'version_id': f'{model_type}_v3',
                'model_type': model_type,
                'created_at': '2024-06-24T14:00:00Z',
                'is_active': True,
                'model_file': f'best{model_type.title()}Detection.pt',
                'file_size_mb': 14.2,
                'performance_metrics': {
                    'precision': 0.94,
                    'recall': 0.91,
                    'f1_score': 0.925,
                    'mAP50': 0.89,
                    'mAP50_95': 0.82,
                    'training_loss': 0.12,
                    'validation_loss': 0.15,
                    'epochs_trained': 120,
                    'best_epoch': 105,
                    'training_time_minutes': 180
                },
                'training_config': {
                    'epochs': 120,
                    'batch_size': 16,
                    'learning_rate': 0.001,
                    'optimizer': 'AdamW',
                    'data_augmentation': True,
                    'model_size': 'medium'
                }
            },
            {
                'version_id': f'{model_type}_v2',
                'model_type': model_type,
                'created_at': '2024-06-23T10:00:00Z',
                'is_active': False,
                'model_file': f'{model_type}_v2.pt',
                'file_size_mb': 13.8,
                'performance_metrics': {
                    'precision': 0.89,
                    'recall': 0.85,
                    'f1_score': 0.87,
                    'mAP50': 0.82,
                    'mAP50_95': 0.75,
                    'training_loss': 0.18,
                    'validation_loss': 0.21,
                    'epochs_trained': 80,
                    'best_epoch': 72,
                    'training_time_minutes': 95
                },
                'training_config': {
                    'epochs': 80,
                    'batch_size': 8,
                    'learning_rate': 0.0005,
                    'optimizer': 'Adam',
                    'data_augmentation': False,
                    'model_size': 'small'
                }
            },
            {
                'version_id': f'{model_type}_v1',
                'model_type': model_type,
                'created_at': '2024-06-22T16:30:00Z',
                'is_active': False,
                'model_file': f'{model_type}_v1.pt',
                'file_size_mb': 12.5,
                'performance_metrics': {
                    'precision': 0.86,
                    'recall': 0.82,
                    'f1_score': 0.84,
                    'mAP50': 0.78,
                    'mAP50_95': 0.71,
                    'training_loss': 0.22,
                    'validation_loss': 0.25,
                    'epochs_trained': 60,
                    'best_epoch': 55,
                    'training_time_minutes': 75
                },
                'training_config': {
                    'epochs': 60,
                    'batch_size': 4,
                    'learning_rate': 0.0001,
                    'optimizer': 'SGD',
                    'data_augmentation': True,
                    'model_size': 'small'
                }
            }
        ]
        
        return jsonify({
            'status': 'success',
            'versions': mock_versions
        })
        
    except Exception as e:
        logger.error(f"Failed to get model versions: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/download_model/<version_id>', methods=['GET'])
def download_model(version_id):
    """Download a model version file."""
    try:
        # In a real implementation, you would find the actual model file
        base_dir = Path(__file__).parent.parent.parent
        models_dir = base_dir / 'models'
        
        # Mock model file for demonstration
        model_file = models_dir / f"{version_id}.pt"
        
        if not model_file.exists():
            # Create a mock file for demonstration
            models_dir.mkdir(exist_ok=True)
            model_file.write_text("Mock model file content")
        
        return send_file(
            model_file,
            as_attachment=True,
            download_name=f"{version_id}.pt",
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/training_sessions', methods=['GET'])
def get_training_sessions():
    """Get all training sessions with real-time metrics."""
    try:
        # Mock training sessions for demonstration
        mock_sessions = [
            {
                'session_id': 'train_20240624_140000',
                'model_type': 'referee',
                'status': 'training',
                'start_time': '2024-06-24T14:00:00Z',
                'end_time': None,
                'current_epoch': 45,
                'total_epochs': 100,
                'duration_minutes': 67,
                'progress_percentage': 45,
                'current_metrics': {
                    'train_loss': 0.25,
                    'val_loss': 0.28,
                    'precision': 0.87,
                    'recall': 0.84,
                    'f1_score': 0.855,
                    'mAP50': 0.81,
                    'mAP50_95': 0.74
                },
                'best_metrics': {
                    'precision': 0.89,
                    'recall': 0.86,
                    'f1_score': 0.875,
                    'mAP50': 0.83,
                    'mAP50_95': 0.76,
                    'epoch': 42
                },
                'config': {
                    'epochs': 100,
                    'batch_size': 16,
                    'learning_rate': 0.001,
                    'optimizer': 'AdamW'
                }
            },
            {
                'session_id': 'train_20240624_120000',
                'model_type': 'signal',
                'status': 'completed',
                'start_time': '2024-06-24T12:00:00Z',
                'end_time': '2024-06-24T14:30:00Z',
                'current_epoch': 120,
                'total_epochs': 120,
                'duration_minutes': 150,
                'progress_percentage': 100,
                'current_metrics': {
                    'train_loss': 0.12,
                    'val_loss': 0.15,
                    'precision': 0.94,
                    'recall': 0.91,
                    'f1_score': 0.925,
                    'mAP50': 0.89,
                    'mAP50_95': 0.82
                },
                'best_metrics': {
                    'precision': 0.94,
                    'recall': 0.91,
                    'f1_score': 0.925,
                    'mAP50': 0.89,
                    'mAP50_95': 0.82,
                    'epoch': 105
                },
                'config': {
                    'epochs': 120,
                    'batch_size': 16,
                    'learning_rate': 0.001,
                    'optimizer': 'AdamW'
                }
            }
        ]
        
        return jsonify({
            'status': 'success',
            'sessions': mock_sessions
        })
        
    except Exception as e:
        logger.error(f"Failed to get training sessions: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/start_training', methods=['POST'])
def start_training():
    """Start a new model training session with real data augmentation."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['model_type', 'epochs', 'batch_size', 'learning_rate']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate model type
        config_manager = get_config_manager()
        if data['model_type'] not in config_manager.get_all_model_types():
            return jsonify({'error': 'Invalid model_type'}), 400
        
        # Validate and process augmentation config if provided
        augmentation_config = None
        if 'augmentation_config' in data and data['augmentation_config']:
            try:
                # Validate the augmentation configuration
                augmentation_config = DataAugmentationManager.validate_augmentation_config(
                    data['augmentation_config']
                )
                
                # Add augmentation factor if not specified
                if 'augmentation_factor' not in augmentation_config:
                    augmentation_config['augmentation_factor'] = 2
                
                logger.info(f"Augmentation config validated: {len([k for k, v in augmentation_config.items() if v and isinstance(v, bool)])} augmentations enabled")
                
            except Exception as e:
                logger.warning(f"Invalid augmentation config: {e}")
                augmentation_config = DataAugmentationManager.get_hybrid_config_template()
        
        # Prepare dataset with augmentation
        try:
            from app.models.training_engine import get_training_engine
            
            training_engine = get_training_engine()
            
            # Prepare dataset with augmentation
            dataset_metadata = training_engine.dataset_manager.prepare_dataset(
                model_type=data['model_type'],
                train_split=data.get('train_split', 0.7),
                val_split=data.get('val_split', 0.2),
                test_split=data.get('test_split', 0.1),
                augmentation_config=augmentation_config
            )
            
            logger.info(f"Dataset prepared with augmentation: {dataset_metadata.get('augmentation_results', {})}")
            
        except Exception as e:
            logger.warning(f"Dataset preparation with augmentation failed: {e}")
            dataset_metadata = {'error': str(e)}
        
        # Start training session
        try:
            training_config = {
                'model_type': data['model_type'],
                'epochs': data['epochs'],
                'batch_size': data['batch_size'],
                'learning_rate': data['learning_rate'],
                'optimizer': data.get('optimizer', 'AdamW'),
                'augmentation_config': augmentation_config,
                'dataset_metadata': dataset_metadata
            }
            
            session_id = training_engine.start_training(training_config)
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'message': 'Training started successfully with data augmentation',
                'config': training_config,
                'dataset_info': dataset_metadata,
                'augmentation_summary': DataAugmentationManager.create_real_augmentation_engine(augmentation_config).get_augmentation_summary() if augmentation_config else None
            })
            
        except Exception as e:
            logger.error(f"Failed to start training session: {e}")
            return jsonify({'error': f'Training start failed: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/prepare_dataset', methods=['POST'])
def prepare_dataset():
    """Prepare dataset with specified train/val/test splits."""
    try:
        data = request.get_json()
        
        model_type = data.get('model_type')
        train_split = float(data.get('train_split', 0.7))
        val_split = float(data.get('val_split', 0.2))
        test_split = float(data.get('test_split', 0.1))
        
        config_manager = get_config_manager()
        
        if not model_type or model_type not in config_manager.get_all_model_types():
            return jsonify({'error': 'Invalid model_type'}), 400
        
        # Validate splits (allow 0% for any split)
        if abs(train_split + val_split + test_split - 1.0) > 0.001:
            return jsonify({'error': 'Splits must sum to 1.0'}), 400
        
        # Generate dataset ID
        dataset_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get source data directory dynamically
        source_dir = config_manager.get_data_directory(model_type)
        
        if not source_dir.exists():
            return jsonify({'error': 'No training data found'}), 404
        
        # Count available files
        image_files = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
        total_files = len(image_files)
        
        if total_files == 0:
            return jsonify({'error': 'No image files found'}), 404
        
        # Calculate split counts (allow 0 counts)
        train_count = int(total_files * train_split)
        val_count = int(total_files * val_split)
        test_count = total_files - train_count - val_count
        
        # In a real implementation, you would actually split and copy the files
        # For now, return the dataset info
        dataset_info = {
            'dataset_id': dataset_id,
            'model_type': model_type,
            'total_files': total_files,
            'splits': {
                'train': {'count': train_count, 'percentage': train_split},
                'val': {'count': val_count, 'percentage': val_split},
                'test': {'count': test_count, 'percentage': test_split}
            },
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"Prepared dataset {dataset_id} with {total_files} files")
        
        return jsonify({
            'status': 'success',
            'dataset': dataset_info
        })
        
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/model_classes/<model_type>', methods=['GET'])
def get_model_classes(model_type):
    """Get classes for a model type."""
    try:
        config_manager = get_config_manager()
        
        if model_type not in config_manager.get_all_model_types():
            return jsonify({'error': 'Invalid model_type'}), 400
        
        classes = config_manager.get_model_classes(model_type)
        
        return jsonify({
            'status': 'success',
            'classes': classes,
            'model_type': model_type
        })
        
    except Exception as e:
        logger.error(f"Failed to get model classes: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/update_signal_classes', methods=['POST'])
def update_signal_classes():
    """Update signal classes configuration."""
    try:
        data = request.get_json()
        new_classes = data.get('classes', [])
        
        if not isinstance(new_classes, list) or not new_classes:
            return jsonify({'error': 'Invalid classes list'}), 400
        
        # Validate class names
        for class_name in new_classes:
            if not isinstance(class_name, str) or not class_name.strip():
                return jsonify({'error': f'Invalid class name: {class_name}'}), 400
        
        config_manager = get_config_manager()
        
        # Update signal classes
        if config_manager.update_model_classes('signal', new_classes):
            logger.info(f"Updated signal classes: {new_classes}")
            
            return jsonify({
                'status': 'success',
                'classes': new_classes,
                'message': 'Signal classes updated successfully'
            })
        else:
            return jsonify({'error': 'Failed to update signal classes'}), 500
        
    except Exception as e:
        logger.error(f"Failed to update signal classes: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/deploy_model/<version_id>', methods=['POST'])
def deploy_model(version_id):
    """Deploy a model version as active."""
    try:
        logger.info(f"Deploying model version: {version_id}")
        
        # In a real implementation, you would copy the model file and update configuration
        return jsonify({
            'status': 'success',
            'message': f'Model version {version_id} deployed successfully',
            'description': 'This model is now active and will be used for auto-labeling and inference'
        })
        
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/training_metrics/<session_id>', methods=['GET'])
def get_training_metrics(session_id):
    """Get detailed training metrics for visualization."""
    try:
        # Mock detailed metrics for charts
        epochs = list(range(1, 46))  # Current epoch is 45
        
        # Simulate realistic training curves
        train_losses = [1.0 - (i * 0.018) + (0.1 * (i % 5) / 5) for i in epochs]
        val_losses = [1.2 - (i * 0.016) + (0.15 * (i % 7) / 7) for i in epochs]
        precisions = [0.3 + (i * 0.012) + (0.05 * (i % 3) / 3) for i in epochs]
        recalls = [0.25 + (i * 0.013) + (0.04 * (i % 4) / 4) for i in epochs]
        
        chart_data = {
            'epochs': epochs,
            'train_loss': [max(0.1, loss) for loss in train_losses],
            'val_loss': [max(0.15, loss) for loss in val_losses],
            'precision': [min(0.95, prec) for prec in precisions],
            'recall': [min(0.92, rec) for rec in recalls],
            'f1_score': [2 * (p * r) / (p + r) for p, r in zip(precisions, recalls)],
            'mAP50': [min(0.9, 0.2 + (i * 0.015)) for i in epochs],
            'mAP50_95': [min(0.85, 0.15 + (i * 0.013)) for i in epochs]
        }
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'chart_data': chart_data,
            'current_epoch': 45,
            'total_epochs': 100
        })
        
    except Exception as e:
        logger.error(f"Failed to get training metrics: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/available_datasets', methods=['GET'])
def get_available_datasets():
    """Get all available prepared datasets."""
    try:
        # Mock available datasets
        datasets = [
            {
                'dataset_id': 'referee_20240624_140000',
                'model_type': 'referee',
                'total_files': 150,
                'created_at': '2024-06-24T14:00:00Z',
                'splits': {
                    'train': {'count': 105, 'percentage': 0.7},
                    'val': {'count': 30, 'percentage': 0.2},
                    'test': {'count': 15, 'percentage': 0.1}
                }
            },
            {
                'dataset_id': 'signal_20240623_120000',
                'model_type': 'signal',
                'total_files': 200,
                'created_at': '2024-06-23T12:00:00Z',
                'splits': {
                    'train': {'count': 140, 'percentage': 0.7},
                    'val': {'count': 40, 'percentage': 0.2},
                    'test': {'count': 20, 'percentage': 0.1}
                }
            }
        ]
        
        return jsonify({
            'status': 'success',
            'datasets': datasets
        })
        
    except Exception as e:
        logger.error(f"Failed to get available datasets: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/augmentation_options', methods=['GET'])
def get_augmentation_options():
    """Get available data augmentation options with categorization."""
    try:
        return jsonify({
            'status': 'success',
            'augmentations': DataAugmentationManager.get_available_augmentations(),
            'defaults': DataAugmentationManager.get_default_augmentations(),
            'categories': DataAugmentationManager.get_augmentations_by_category(),
            'engines': {
                'albumentations': DataAugmentationManager.get_augmentations_by_engine('albumentations'),
                'yolo': DataAugmentationManager.get_augmentations_by_engine('yolo')
            },
            'hybrid_template': DataAugmentationManager.get_hybrid_config_template()
        })
        
    except Exception as e:
        logger.error(f"Failed to get augmentation options: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/test_augmentation', methods=['POST'])
def test_augmentation():
    """Test data augmentation configuration on a sample image."""
    try:
        data = request.get_json()
        
        model_type = data.get('model_type')
        augmentation_config = data.get('augmentation_config', {})
        
        if not model_type:
            return jsonify({'error': 'Missing model_type'}), 400
        
        # Validate augmentation config
        validated_config = DataAugmentationManager.validate_augmentation_config(augmentation_config)
        
        # Create augmentation engine
        aug_engine = DataAugmentationManager.create_real_augmentation_engine(validated_config)
        
        if not aug_engine:
            return jsonify({'error': 'Failed to create augmentation engine'}), 500
        
        # Get augmentation summary
        summary = aug_engine.get_augmentation_summary()
        
        return jsonify({
            'status': 'success',
            'message': 'Augmentation configuration tested successfully',
            'config': validated_config,
            'summary': summary,
            'enabled_count': len(summary.get('enabled_augmentations', [])),
            'engines_used': {
                'albumentations': summary.get('albumentations_enabled', False),
                'yolo': summary.get('yolo_enabled', False)
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to test augmentation: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/model_types', methods=['GET'])
def get_model_types():
    """Get all available model types."""
    try:
        config_manager = get_config_manager()
        model_types = []
        
        for model_type in config_manager.get_all_model_types():
            model_info = config_manager.get_model_info(model_type)
            model_types.append({
                'type': model_type,
                'display_name': model_type.title(),
                'task': model_info.get('task', f'{model_type}_detection'),
                'data_dir': model_info.get('data_dir'),
                'class_count': len(config_manager.get_model_classes(model_type))
            })
        
        return jsonify({
            'status': 'success',
            'model_types': model_types
        })
        
    except Exception as e:
        logger.error(f"Failed to get model types: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/add_model_type', methods=['POST'])
def add_model_type():
    """Add a new model type."""
    try:
        data = request.get_json()
        
        model_type = data.get('model_type')
        config = data.get('config', {})
        
        if not model_type:
            return jsonify({'error': 'Missing model_type'}), 400
        
        config_manager = get_config_manager()
        
        if config_manager.add_new_model_type(model_type, config):
            return jsonify({
                'status': 'success',
                'message': f'Model type {model_type} added successfully'
            })
        else:
            return jsonify({'error': 'Failed to add model type'}), 500
        
    except Exception as e:
        logger.error(f"Failed to add model type: {e}")
        return jsonify({'error': str(e)}), 500

# Phase 3.1: Enhanced Training Pipeline

@training_bp.route('/smart_base_selection', methods=['POST'])
def smart_base_model_selection():
    """Smart base model selection based on dataset size and requirements."""
    try:
        data = request.get_json()
        
        model_type = data.get('model_type')
        dataset_size = data.get('dataset_size', 0)
        performance_priority = data.get('performance_priority', 'balanced')  # speed, accuracy, balanced
        
        if not model_type:
            return jsonify({'error': 'Missing model_type'}), 400
        
        # Smart selection logic
        recommendations = []
        
        if dataset_size < 100:
            # Small dataset - use smaller models to prevent overfitting
            if performance_priority == 'speed':
                recommendations = ['yolov8n', 'yolo11n']
            elif performance_priority == 'accuracy':
                recommendations = ['yolov8s', 'yolo11s', 'yolov8n']
            else:  # balanced
                recommendations = ['yolov8n', 'yolov8s', 'yolo11n']
        
        elif dataset_size < 500:
            # Medium dataset
            if performance_priority == 'speed':
                recommendations = ['yolov8n', 'yolov8s', 'yolo11n']
            elif performance_priority == 'accuracy':
                recommendations = ['yolov8s', 'yolov8m', 'yolo11s']
            else:  # balanced
                recommendations = ['yolov8s', 'yolov8n', 'yolo11s']
        
        else:
            # Large dataset - can handle larger models
            if performance_priority == 'speed':
                recommendations = ['yolov8s', 'yolov8n', 'yolo11s']
            elif performance_priority == 'accuracy':
                recommendations = ['yolov8m', 'yolov8l', 'yolo11m']
            else:  # balanced
                recommendations = ['yolov8s', 'yolov8m', 'yolo11s']
        
        # Get model details
        from app.models.model_registry import UltralyticsDownloader
        available_models = UltralyticsDownloader.get_available_models()
        
        recommended_models = []
        for model_name in recommendations:
            if model_name in available_models:
                model_info = available_models[model_name].copy()
                model_info['name'] = model_name
                model_info['recommendation_reason'] = _get_recommendation_reason(
                    model_name, dataset_size, performance_priority
                )
                recommended_models.append(model_info)
        
        return jsonify({
            'status': 'success',
            'recommendations': recommended_models,
            'dataset_analysis': {
                'size': dataset_size,
                'size_category': 'small' if dataset_size < 100 else 'medium' if dataset_size < 500 else 'large',
                'performance_priority': performance_priority
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get smart base model selection: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/transfer_learning/start', methods=['POST'])
def start_transfer_learning():
    """Start transfer learning with a base model."""
    try:
        data = request.get_json()
        
        # Required parameters
        model_type = data.get('model_type')
        base_model_id = data.get('base_model_id')
        training_config = data.get('training_config', {})
        
        # Optional parameters
        experiment_name = data.get('experiment_name', f"{model_type}_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if not model_type or not base_model_id:
            return jsonify({'error': 'Missing model_type or base_model_id'}), 400
        
        # Get base model from registry
        from app.models.model_registry import get_model_registry
        registry = get_model_registry()
        
        base_model = None
        for model in registry.get_models():
            if model['model_id'] == base_model_id:
                base_model = model
                break
        
        if not base_model:
            return jsonify({'error': f'Base model {base_model_id} not found'}), 404
        
        # Prepare transfer learning configuration
        transfer_config = {
            'base_model_path': base_model['file_path'],
            'base_model_info': base_model,
            'model_type': model_type,
            'experiment_name': experiment_name,
            'training_config': {
                'epochs': training_config.get('epochs', 100),
                'batch_size': training_config.get('batch_size', 16),
                'learning_rate': training_config.get('learning_rate', 0.001),
                'optimizer': training_config.get('optimizer', 'AdamW'),
                'warmup_epochs': training_config.get('warmup_epochs', 3),
                'patience': training_config.get('patience', 50),
                'save_period': training_config.get('save_period', 10),
                **training_config
            },
            'augmentation_config': data.get('augmentation_config', {}),
            'transfer_learning': True,
            'freeze_backbone': training_config.get('freeze_backbone', False),
            'freeze_epochs': training_config.get('freeze_epochs', 10)
        }
        
        # Create training session
        session_id = f"transfer_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # In a real implementation, this would start the actual training process
        # For now, we'll simulate the training session creation
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'message': 'Transfer learning session started',
            'config': transfer_config,
            'progress_url': f'/api/training/progress/{session_id}',
            'estimated_duration_minutes': _estimate_training_duration(transfer_config)
        })
        
    except Exception as e:
        logger.error(f"Failed to start transfer learning: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/progress/<session_id>', methods=['GET'])
def get_training_progress(session_id):
    """Get real-time training progress for a session."""
    try:
        # In a real implementation, this would read from a training progress file or database
        # For now, we'll simulate progress based on session creation time
        
        # Mock progress data
        progress = {
            'session_id': session_id,
            'status': 'training',  # training, completed, failed, paused
            'current_epoch': 45,
            'total_epochs': 100,
            'progress_percentage': 45.0,
            'elapsed_time_minutes': 67,
            'estimated_remaining_minutes': 82,
            'current_metrics': {
                'train_loss': 0.245,
                'val_loss': 0.278,
                'precision': 0.87,
                'recall': 0.84,
                'f1_score': 0.855,
                'mAP50': 0.81,
                'mAP50_95': 0.74,
                'learning_rate': 0.0008
            },
            'best_metrics': {
                'epoch': 42,
                'val_loss': 0.265,
                'precision': 0.89,
                'recall': 0.86,
                'f1_score': 0.875,
                'mAP50': 0.83,
                'mAP50_95': 0.76
            },
            'training_curve': {
                'epochs': list(range(1, 46)),
                'train_loss': [1.0 - (i * 0.018) + (0.1 * (i % 5) / 5) for i in range(1, 46)],
                'val_loss': [1.2 - (i * 0.016) + (0.15 * (i % 7) / 7) for i in range(1, 46)],
                'mAP50': [0.2 + (i * 0.015) for i in range(1, 46)]
            },
            'gpu_usage': {
                'memory_used_gb': 4.2,
                'memory_total_gb': 8.0,
                'utilization_percent': 85
            },
            'data_loading': {
                'avg_time_ms': 12.5,
                'cache_hit_rate': 0.92
            }
        }
        
        return jsonify({
            'status': 'success',
            'progress': progress
        })
        
    except Exception as e:
        logger.error(f"Failed to get training progress: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/hyperparameter_optimization', methods=['POST'])
def hyperparameter_optimization():
    """Start hyperparameter optimization for a model."""
    try:
        data = request.get_json()
        
        model_type = data.get('model_type')
        base_model_id = data.get('base_model_id')
        optimization_config = data.get('optimization_config', {})
        
        if not model_type or not base_model_id:
            return jsonify({'error': 'Missing model_type or base_model_id'}), 400
        
        # Define hyperparameter search space
        search_space = {
            'learning_rate': optimization_config.get('learning_rate_range', [0.0001, 0.01]),
            'batch_size': optimization_config.get('batch_size_options', [8, 16, 32]),
            'optimizer': optimization_config.get('optimizers', ['AdamW', 'SGD', 'Adam']),
            'weight_decay': optimization_config.get('weight_decay_range', [0.0001, 0.001]),
            'momentum': optimization_config.get('momentum_range', [0.9, 0.99]),
            'warmup_epochs': optimization_config.get('warmup_epochs_range', [1, 5])
        }
        
        optimization_method = optimization_config.get('method', 'grid_search')  # grid_search, random_search, bayesian
        max_trials = optimization_config.get('max_trials', 20)
        
        # Create optimization session
        session_id = f"hpo_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'message': 'Hyperparameter optimization started',
            'search_space': search_space,
            'optimization_method': optimization_method,
            'max_trials': max_trials,
            'progress_url': f'/api/training/hpo_progress/{session_id}'
        })
        
    except Exception as e:
        logger.error(f"Failed to start hyperparameter optimization: {e}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/model_comparison', methods=['POST'])
def model_comparison():
    """Compare performance of multiple models."""
    try:
        data = request.get_json()
        
        model_ids = data.get('model_ids', [])
        comparison_metrics = data.get('metrics', ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score'])
        
        if not model_ids:
            return jsonify({'error': 'No model IDs provided'}), 400
        
        from app.models.model_registry import get_model_registry
        registry = get_model_registry()
        
        comparison_results = []
        
        for model_id in model_ids:
            model = None
            for m in registry.get_models():
                if m['model_id'] == model_id:
                    model = m
                    break
            
            if model:
                comparison_results.append({
                    'model_id': model_id,
                    'model_name': model['version'],
                    'model_type': model['model_type'],
                    'source': model['source'],
                    'file_size_mb': model['file_size_mb'],
                    'performance_metrics': model.get('performance_metrics', {}),
                    'created_at': model['created_at']
                })
        
        # Calculate rankings
        rankings = {}
        for metric in comparison_metrics:
            metric_values = []
            for result in comparison_results:
                value = result['performance_metrics'].get(metric)
                if value is not None:
                    metric_values.append((result['model_id'], value))
            
            # Sort by metric value (higher is better for most metrics)
            metric_values.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = [{'model_id': mid, 'value': val, 'rank': i+1} 
                              for i, (mid, val) in enumerate(metric_values)]
        
        return jsonify({
            'status': 'success',
            'comparison_results': comparison_results,
            'rankings': rankings,
            'summary': {
                'total_models': len(comparison_results),
                'metrics_compared': comparison_metrics,
                'best_overall': _calculate_best_overall_model(comparison_results, comparison_metrics)
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        return jsonify({'error': str(e)}), 500

def _get_recommendation_reason(model_name, dataset_size, performance_priority):
    """Get explanation for model recommendation."""
    reasons = []
    
    if 'n' in model_name:
        reasons.append("Nano model - fast inference and low memory usage")
    elif 's' in model_name:
        reasons.append("Small model - good balance of speed and accuracy")
    elif 'm' in model_name:
        reasons.append("Medium model - higher accuracy with moderate speed")
    elif 'l' in model_name:
        reasons.append("Large model - high accuracy for complex tasks")
    
    if dataset_size < 100:
        reasons.append("Suitable for small datasets (prevents overfitting)")
    elif dataset_size < 500:
        reasons.append("Good for medium-sized datasets")
    else:
        reasons.append("Can leverage large datasets effectively")
    
    if performance_priority == 'speed':
        reasons.append("Optimized for fast inference")
    elif performance_priority == 'accuracy':
        reasons.append("Optimized for high accuracy")
    else:
        reasons.append("Balanced speed and accuracy")
    
    return "; ".join(reasons)

def _estimate_training_duration(config):
    """Estimate training duration in minutes."""
    epochs = config['training_config']['epochs']
    batch_size = config['training_config']['batch_size']
    
    # Simple estimation based on epochs and batch size
    base_time = epochs * 2  # 2 minutes per epoch as baseline
    
    # Adjust for batch size
    if batch_size <= 8:
        base_time *= 1.5
    elif batch_size >= 32:
        base_time *= 0.7
    
    return int(base_time)

def _calculate_best_overall_model(models, metrics):
    """Calculate the best overall model based on multiple metrics."""
    if not models or not metrics:
        return None
    
    scores = {}
    for model in models:
        model_id = model['model_id']
        scores[model_id] = 0
        metric_count = 0
        
        for metric in metrics:
            value = model['performance_metrics'].get(metric)
            if value is not None:
                scores[model_id] += value
                metric_count += 1
        
        if metric_count > 0:
            scores[model_id] /= metric_count  # Average score
    
    if scores:
        best_model_id = max(scores, key=scores.get)
        return {
            'model_id': best_model_id,
            'average_score': scores[best_model_id]
        }
    
    return None 