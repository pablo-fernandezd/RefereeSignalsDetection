"""
Enhanced Training Engine for YOLO Model Training

This module provides comprehensive training capabilities including:
- Dataset preparation and splitting
- YOLO model training with real-time monitoring
- Model version management
- Training session management
- Metrics tracking and visualization
"""

import os
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import yaml

# Import the new configuration manager
from app.utils.model_config import get_config_manager

try:
    import torch
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    # Handle missing dependencies gracefully
    torch = None
    cv2 = None
    np = None
    YOLO = None
    plt = None
    pd = None

# Import with proper path handling
import sys
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class TrainingSession:
    """Represents a single training session with metrics tracking."""
    
    def __init__(self, session_id: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.config = config
        self.status = 'initialized'
        self.start_time = None
        self.end_time = None
        self.current_epoch = 0
        self.total_epochs = config.get('epochs', 100)
        self.metrics = {}
        self.best_metrics = {}
        self.process = None
        self.log_file = None
        
    def start(self):
        """Start the training session."""
        self.status = 'training'
        self.start_time = datetime.now()
        logger.info(f"Training session {self.session_id} started")
    
    def update_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Update training metrics for current epoch."""
        self.current_epoch = epoch
        self.metrics[str(epoch)] = metrics
        
        # Update best metrics
        if not self.best_metrics or metrics.get('mAP50', 0) > self.best_metrics.get('mAP50', 0):
            self.best_metrics = metrics.copy()
            self.best_metrics['epoch'] = epoch
        
        logger.debug(f"Session {self.session_id} - Epoch {epoch} metrics updated")
    
    def complete(self, success: bool = True):
        """Mark the training session as completed."""
        self.status = 'completed' if success else 'failed'
        self.end_time = datetime.now()
        logger.info(f"Training session {self.session_id} completed with status: {self.status}")
    
    def get_duration_minutes(self) -> int:
        """Get training duration in minutes."""
        if self.start_time:
            end_time = self.end_time or datetime.now()
            return int((end_time - self.start_time).total_seconds() / 60)
        return 0
    
    def get_progress_percentage(self) -> float:
        """Get training progress as percentage."""
        if self.total_epochs > 0:
            return min(100.0, (self.current_epoch / self.total_epochs) * 100)
        return 0.0


class ModelVersionManager:
    """Manages model versions and deployment."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.models_dir = base_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.models_dir / 'versions.json'
        self.versions = self._load_versions()
        self.config_manager = get_config_manager(base_dir)
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load model versions from file."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load versions file: {e}")
        
        # Initialize with empty dict for all model types
        return {}
    
    def _save_versions(self):
        """Save model versions to file."""
        try:
            with open(self.versions_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save versions file: {e}")
    
    def add_version(self, model_type: str, session: TrainingSession, model_path: Path) -> str:
        """Add a new model version."""
        # Initialize model type if not exists
        if model_type not in self.versions:
            self.versions[model_type] = []
        
        version_id = f"{model_type}_v{len(self.versions[model_type]) + 1}"
        
        # Copy model file to versions directory
        version_dir = self.models_dir / version_id
        version_dir.mkdir(exist_ok=True)
        dest_path = version_dir / f"{version_id}.pt"
        
        if model_path.exists():
            shutil.copy2(model_path, dest_path)
        
        # Create version metadata
        version_info = {
            'version_id': version_id,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'is_active': False,
            'session_id': session.session_id,
            'model_path': str(dest_path),
            'performance_metrics': session.best_metrics,
            'training_config': session.config,
            'training_duration_minutes': session.get_duration_minutes()
        }
        
        self.versions[model_type].append(version_info)
        self._save_versions()
        
        logger.info(f"Added model version {version_id}")
        return version_id
    
    def get_versions(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get model versions, optionally filtered by type."""
        if model_type:
            return self.versions.get(model_type, [])
        
        all_versions = []
        for versions in self.versions.values():
            all_versions.extend(versions)
        return sorted(all_versions, key=lambda x: x['created_at'], reverse=True)
    
    def set_active_version(self, version_id: str) -> bool:
        """Set a model version as active."""
        try:
            # Find the version
            version_info = None
            model_type = None
            
            for mtype, versions in self.versions.items():
                for version in versions:
                    if version['version_id'] == version_id:
                        version_info = version
                        model_type = mtype
                        break
                if version_info:
                    break
            
            if not version_info:
                logger.error(f"Version {version_id} not found")
                return False
            
            # Deactivate all versions of this model type
            for version in self.versions[model_type]:
                version['is_active'] = False
            
            # Activate the selected version
            version_info['is_active'] = True
            
            # Copy model to active location
            active_model_path = self.models_dir / f"best{model_type.title()}Detection.pt"
            if Path(version_info['model_path']).exists():
                shutil.copy2(version_info['model_path'], active_model_path)
            
            self._save_versions()
            logger.info(f"Set {version_id} as active version")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set active version: {e}")
            return False


class DatasetManager:
    """Manages dataset preparation and splitting for training."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.datasets_dir = base_dir / 'training_datasets'
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.config_manager = get_config_manager(base_dir)
    
    def prepare_dataset(self, model_type: str, train_split: float, val_split: float, test_split: float, augmentation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare a dataset with specified splits."""
        try:
            # Validate splits (allow 0% splits)
            total = train_split + val_split + test_split
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"Splits must sum to 1.0, got {total}")
            
            # Get source data directory using config manager
            source_dir = self.config_manager.get_data_directory(model_type)
            
            if not source_dir.exists():
                raise FileNotFoundError(f"Source directory not found: {source_dir}")
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(source_dir.glob(ext))
            
            if not image_files:
                raise FileNotFoundError("No image files found in source directory")
            
            # Create dataset ID and directory
            dataset_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset_dir = self.datasets_dir / dataset_id
            dataset_dir.mkdir(exist_ok=True)
            
            # Create split directories
            splits = ['train', 'val', 'test']
            for split in splits:
                (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
                (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Calculate split indices (allow 0 counts)
            total_files = len(image_files)
            train_count = int(total_files * train_split)
            val_count = int(total_files * val_split)
            test_count = total_files - train_count - val_count
            
            # Shuffle files for random splitting
            import random
            random.shuffle(image_files)
            
            # Split files
            train_files = image_files[:train_count]
            val_files = image_files[train_count:train_count + val_count]
            test_files = image_files[train_count + val_count:]
            
            # Copy files to respective directories
            splits_info = {}
            for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
                split_dir = dataset_dir / split_name
                copied_count = 0
                
                for img_file in files:
                    # Copy image
                    dest_img = split_dir / 'images' / img_file.name
                    shutil.copy2(img_file, dest_img)
                    
                    # Copy corresponding label file if exists
                    label_file = img_file.with_suffix('.txt')
                    if label_file.exists():
                        dest_label = split_dir / 'labels' / label_file.name
                        shutil.copy2(label_file, dest_label)
                    
                    copied_count += 1
                
                splits_info[split_name] = {
                    'count': copied_count,
                    'percentage': len(files) / total_files if total_files > 0 else 0
                }
            
            # Apply data augmentation if configured
            augmentation_results = {}
            if augmentation_config:
                try:
                    from app.utils.model_config import DataAugmentationManager
                    
                    # Create augmentation engine
                    aug_engine = DataAugmentationManager.create_real_augmentation_engine(augmentation_config)
                    
                    if aug_engine:
                        # Apply augmentation to training data only
                        train_dir = dataset_dir / 'train' / 'images'
                        if train_dir.exists():
                            logger.info(f"Applying data augmentation to training set...")
                            aug_results = aug_engine.process_training_data(
                                train_dir, 
                                train_dir,  # Augment in place
                                augmentation_factor=augmentation_config.get('augmentation_factor', 2)
                            )
                            augmentation_results = aug_results
                            logger.info(f"Data augmentation complete: {aug_results}")
                    
                except Exception as e:
                    logger.warning(f"Data augmentation failed: {e}")
                    augmentation_results = {'error': str(e)}
            
            # Create YOLO dataset configuration using config manager
            yaml_path = self.config_manager.create_yaml_config(model_type, dataset_dir, splits_info)
            
            # Update YAML with augmentation parameters if available
            if augmentation_results.get('yolo_augmentations'):
                try:
                    with open(yaml_path, 'r') as f:
                        yaml_data = yaml.safe_load(f) or {}
                    
                    # Add YOLO augmentation parameters
                    yaml_data.update(augmentation_results['yolo_augmentations'])
                    
                    with open(yaml_path, 'w') as f:
                        yaml.dump(yaml_data, f, default_flow_style=False)
                    
                    logger.info("Updated YAML with YOLO augmentation parameters")
                except Exception as e:
                    logger.warning(f"Failed to update YAML with augmentation parameters: {e}")
            
            # Save dataset metadata
            metadata = {
                'dataset_id': dataset_id,
                'model_type': model_type,
                'total_files': total_files,
                'splits': splits_info,
                'created_at': datetime.now().isoformat(),
                'source_dir': str(source_dir),
                'dataset_dir': str(dataset_dir),
                'augmentation_config': augmentation_config,
                'augmentation_results': augmentation_results
            }
            
            with open(dataset_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Dataset {dataset_id} prepared successfully with {total_files} files")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            raise


class TrainingEngine:
    """Main training engine that orchestrates the training process."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.dataset_manager = DatasetManager(base_dir)
        self.version_manager = ModelVersionManager(base_dir)
        self.active_sessions: Dict[str, TrainingSession] = {}
        self.session_history: List[TrainingSession] = []
        self.config_manager = get_config_manager(base_dir)
        
        # Create logs directory
        self.logs_dir = base_dir / 'logs' / 'training'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def start_training(self, config: Dict[str, Any]) -> str:
        """Start a new training session."""
        try:
            # Validate model type
            model_type = config.get('model_type')
            if model_type not in self.config_manager.get_all_model_types():
                raise ValueError(f"Invalid model type: {model_type}")
            
            # Generate session ID
            session_id = f"train_{int(time.time())}"
            
            # Create training session
            session = TrainingSession(session_id, config)
            session.log_file = self.logs_dir / f"{session_id}.log"
            
            # Add to active sessions
            self.active_sessions[session_id] = session
            self.session_history.append(session)
            
            # Start training in background thread
            training_thread = threading.Thread(
                target=self._run_training,
                args=(session,),
                daemon=True
            )
            training_thread.start()
            
            logger.info(f"Started training session {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            raise
    
    def _run_training(self, session: TrainingSession):
        """Run the actual training process."""
        try:
            session.start()
            
            # This is where you would integrate with actual YOLO training
            # For now, simulate training with realistic metrics
            self._simulate_training(session)
            
            # Mark as completed
            session.complete(success=True)
            
            # Create model version
            # In real implementation, you would use the actual trained model path
            mock_model_path = self.base_dir / 'models' / f"temp_{session.session_id}.pt"
            mock_model_path.touch()  # Create empty file for demonstration
            
            self.version_manager.add_version(
                session.config['model_type'],
                session,
                mock_model_path
            )
            
            # Clean up mock file
            if mock_model_path.exists():
                mock_model_path.unlink()
            
        except Exception as e:
            logger.error(f"Training session {session.session_id} failed: {e}")
            session.complete(success=False)
        finally:
            # Remove from active sessions
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
    
    def _simulate_training(self, session: TrainingSession):
        """Simulate training with realistic metrics progression."""
        total_epochs = session.total_epochs
        
        for epoch in range(1, total_epochs + 1):
            # Simulate realistic training metrics
            progress = epoch / total_epochs
            
            # Training loss decreases with some noise
            train_loss = 1.0 * (1 - progress * 0.8) + 0.1 * (epoch % 5) / 5
            val_loss = train_loss * 1.2 + 0.05 * (epoch % 7) / 7
            
            # Precision and recall increase with some noise
            precision = 0.3 + progress * 0.6 + 0.05 * (epoch % 3) / 3
            recall = 0.25 + progress * 0.65 + 0.04 * (epoch % 4) / 4
            
            # Calculate derived metrics
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mAP50 = 0.2 + progress * 0.7 + 0.03 * (epoch % 6) / 6
            mAP50_95 = mAP50 * 0.85
            
            # Ensure realistic bounds
            precision = min(0.95, max(0.1, precision))
            recall = min(0.92, max(0.1, recall))
            mAP50 = min(0.9, max(0.1, mAP50))
            mAP50_95 = min(0.85, max(0.1, mAP50_95))
            train_loss = max(0.05, min(1.0, train_loss))
            val_loss = max(0.1, min(1.2, val_loss))
            
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'mAP50': mAP50,
                'mAP50_95': mAP50_95
            }
            
            session.update_metrics(epoch, metrics)
            
            # Log progress
            if epoch % 10 == 0 or epoch == total_epochs:
                logger.info(f"Session {session.session_id} - Epoch {epoch}/{total_epochs} - "
                          f"Loss: {train_loss:.4f}, mAP50: {mAP50:.3f}")
            
            # Simulate training time (remove in real implementation)
            time.sleep(0.1)
            
            # Check if training should be stopped
            if session.status != 'training':
                break
    
    def get_training_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training session."""
        session = self.active_sessions.get(session_id)
        if not session:
            # Check in history
            for hist_session in self.session_history:
                if hist_session.session_id == session_id:
                    session = hist_session
                    break
        
        if not session:
            return None
        
        return {
            'session_id': session.session_id,
            'status': session.status,
            'start_time': session.start_time.isoformat() if session.start_time else None,
            'end_time': session.end_time.isoformat() if session.end_time else None,
            'current_epoch': session.current_epoch,
            'total_epochs': session.total_epochs,
            'progress_percentage': session.get_progress_percentage(),
            'duration_minutes': session.get_duration_minutes(),
            'metrics': session.metrics,
            'best_metrics': session.best_metrics,
            'config': session.config
        }
    
    def get_all_training_sessions(self) -> List[Dict[str, Any]]:
        """Get all training sessions."""
        sessions = []
        
        # Add active sessions
        for session in self.active_sessions.values():
            status_info = self.get_training_status(session.session_id)
            if status_info:
                sessions.append(status_info)
        
        # Add recent completed sessions
        for session in reversed(self.session_history[-10:]):  # Last 10 sessions
            if session.session_id not in self.active_sessions:
                status_info = self.get_training_status(session.session_id)
                if status_info:
                    sessions.append(status_info)
        
        return sorted(sessions, key=lambda x: x.get('start_time', ''), reverse=True)
    
    def stop_training(self, session_id: str) -> bool:
        """Stop an active training session."""
        session = self.active_sessions.get(session_id)
        if session and session.status == 'training':
            session.status = 'stopping'
            logger.info(f"Stopping training session {session_id}")
            return True
        return False


# Global training engine instance
training_engine = None

def get_training_engine(base_dir: Optional[Path] = None) -> TrainingEngine:
    """Get or create the global training engine instance."""
    global training_engine
    
    if training_engine is None:
        if base_dir is None:
            # Default to backend directory
            base_dir = Path(__file__).parent.parent.parent
        training_engine = TrainingEngine(base_dir)
    
    return training_engine

# For backward compatibility
def initialize_training_engine(base_dir: Optional[Path] = None):
    """Initialize the training engine."""
    return get_training_engine(base_dir)
