"""
YOLO Model Detection and Validation Utilities

This module provides tools to:
- Detect YOLO version from model files
- Validate model compatibility with Ultralytics
- Extract model architecture information
- Test model loading and functionality
"""

import logging
import torch
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

try:
    from ultralytics import YOLO
    import ultralytics
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ultralytics = None
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class YOLOVersionDetector:
    """Detects YOLO version and architecture from model files."""
    
    # Known YOLO version patterns and their characteristics
    VERSION_PATTERNS = {
        'yolov8': {
            'patterns': [r'yolov8', r'ultralytics.*v8', r'YOLOv8'],
            'architecture_names': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
            'supported': True,
            'framework': 'ultralytics'
        },
        'yolov9': {
            'patterns': [r'yolov9', r'ultralytics.*v9', r'YOLOv9'],
            'architecture_names': ['yolov9c', 'yolov9e'],
            'supported': True,
            'framework': 'ultralytics'
        },
        'yolov10': {
            'patterns': [r'yolov10', r'ultralytics.*v10', r'YOLOv10'],
            'architecture_names': ['yolov10n', 'yolov10s', 'yolov10m', 'yolov10l', 'yolov10x'],
            'supported': True,
            'framework': 'ultralytics'
        },
        'yolo11': {
            'patterns': [r'yolo11', r'ultralytics.*11', r'YOLO11'],
            'architecture_names': ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
            'supported': True,
            'framework': 'ultralytics'
        },
        'yolov11': {
            'patterns': [r'yolov11', r'ultralytics.*v11', r'YOLOv11'],
            'architecture_names': ['yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'],
            'supported': True,
            'framework': 'ultralytics'
        },
        'yolov12': {
            'patterns': [r'yolov12', r'ultralytics.*v12', r'YOLOv12'],
            'architecture_names': ['yolov12n', 'yolov12s', 'yolov12m', 'yolov12l', 'yolov12x'],
            'supported': True,
            'framework': 'ultralytics'
        },
        'yolov5': {
            'patterns': [r'yolov5', r'YOLOv5'],
            'architecture_names': ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
            'supported': False,
            'framework': 'yolov5',
            'note': 'YOLOv5 models may need conversion for Ultralytics compatibility'
        },
        'yolov7': {
            'patterns': [r'yolov7', r'YOLOv7'],
            'architecture_names': ['yolov7', 'yolov7-tiny', 'yolov7x'],
            'supported': False,
            'framework': 'yolov7',
            'note': 'YOLOv7 models may need conversion for Ultralytics compatibility'
        }
    }
    
    @classmethod
    def detect_yolo_version(cls, model_path: Path) -> Dict[str, Any]:
        """
        Detect YOLO version from a model file with improved accuracy.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary with detection results
        """
        result = {
            'detected_version': None,
            'confidence': 0.0,
            'architecture': None,
            'supported': False,
            'framework': 'unknown',
            'model_info': {},
            'validation_result': None,
            'error': None
        }
        
        try:
            if not model_path.exists():
                result['error'] = f"Model file not found: {model_path}"
                return result
            
            # Try multiple detection methods in order of reliability
            detection_methods = [
                cls._detect_with_ultralytics_advanced,
                cls._detect_with_torch_advanced,
                cls._detect_from_filename_advanced
            ]
            
            for method in detection_methods:
                try:
                    method_result = method(model_path)
                    if method_result.get('success') and method_result.get('detected_version'):
                        result.update(method_result)
                        # Remove 'success' key as it's not part of the expected result
                        result.pop('success', None)
                        return result
                except Exception as e:
                    logger.warning(f"Detection method {method.__name__} failed: {e}")
                    continue
            
            # If no method succeeded, return unknown
            result['detected_version'] = 'unknown'
            result['confidence'] = 0.0
            result['error'] = 'Could not detect YOLO version with any method'
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting YOLO version: {e}")
            result['error'] = str(e)
            return result
    
    @classmethod
    def _detect_with_ultralytics_advanced(cls, model_path: Path) -> Dict[str, Any]:
        """Advanced detection using Ultralytics with better version analysis."""
        result = {
            'success': False,
            'detected_version': None,
            'confidence': 0.0,
            'architecture': None,
            'supported': True,
            'framework': 'ultralytics',
            'model_info': {},
            'validation_result': None
        }
        
        if not ULTRALYTICS_AVAILABLE:
            return result
        
        try:
            # Load model with Ultralytics
            model = YOLO(str(model_path))
            
            # Extract comprehensive model information
            model_info = {
                'task': getattr(model, 'task', 'detect'),
                'mode': getattr(model, 'mode', 'predict'),
                'nc': getattr(model.model, 'nc', 0) if hasattr(model, 'model') else 0,
                'names': getattr(model.model, 'names', {}) if hasattr(model, 'model') else {},
            }
            
            # Advanced version detection from model structure
            detected_version, confidence = cls._analyze_model_structure_advanced(model, model_info)
            
            if detected_version:
                # Try to determine architecture size
                architecture = cls._determine_architecture_size(model, detected_version)
                
                result.update({
                    'success': True,
                    'detected_version': detected_version,
                    'confidence': confidence,
                    'architecture': architecture,
                    'model_info': model_info,
                    'validation_result': cls._validate_model_functionality(model)
                })
            
            return result
            
        except Exception as e:
            logger.warning(f"Advanced Ultralytics detection failed: {e}")
            result['error'] = str(e)
            return result
    
    @classmethod
    def _detect_with_torch_advanced(cls, model_path: Path) -> Dict[str, Any]:
        """Advanced detection using PyTorch with detailed checkpoint analysis."""
        result = {
            'success': False,
            'detected_version': None,
            'confidence': 0.5,
            'architecture': None,
            'supported': False,
            'framework': 'pytorch',
            'model_info': {}
        }
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Analyze checkpoint structure for version indicators
            version_indicators = cls._extract_version_indicators(checkpoint)
            
            # Determine version from indicators
            detected_version = cls._determine_version_from_indicators(version_indicators)
            
            if detected_version:
                version_info = cls.VERSION_PATTERNS.get(detected_version, {})
                result.update({
                    'success': True,
                    'detected_version': detected_version,
                    'confidence': 0.7,
                    'supported': version_info.get('supported', False),
                    'framework': version_info.get('framework', 'unknown'),
                    'model_info': version_indicators
                })
            
            return result
            
        except Exception as e:
            logger.warning(f"Advanced PyTorch inspection failed: {e}")
            result['error'] = str(e)
            return result
    
    @classmethod
    def _detect_from_filename_advanced(cls, model_path: Path) -> Dict[str, Any]:
        """Advanced filename-based detection with better pattern matching."""
        result = {
            'success': False,
            'detected_version': None,
            'confidence': 0.3,
            'architecture': None,
            'supported': False,
            'framework': 'unknown'
        }
        
        filename = model_path.name.lower()
        
        # Enhanced filename patterns with priorities
        filename_patterns = [
            # Specific version patterns (higher priority)
            (r'yolo(?:v)?12|v12', 'yolov12', 0.8),
            (r'yolo(?:v)?11|v11', 'yolo11', 0.8),
            (r'yolo(?:v)?10|v10', 'yolov10', 0.7),
            (r'yolo(?:v)?9|v9', 'yolov9', 0.7),
            (r'yolo(?:v)?8|v8', 'yolov8', 0.6),
            (r'yolo(?:v)?7|v7', 'yolov7', 0.5),
            (r'yolo(?:v)?5|v5', 'yolov5', 0.5),
            
            # Architecture-specific patterns
            (r'(?:yolo(?:v)?12|v12)[nslmx]', 'yolov12', 0.9),
            (r'(?:yolo(?:v)?11|v11)[nslmx]', 'yolo11', 0.9),
            (r'(?:yolo(?:v)?8|v8)[nslmx]', 'yolov8', 0.7),
        ]
        
        # Context-based detection (referee/signal models)
        context_hints = []
        if 'referee' in filename:
            context_hints = ['yolov12', 'yolo11', 'yolov8']  # Most likely for referee models
        elif 'signal' in filename:
            context_hints = ['yolo11', 'yolov8', 'yolov12']  # Most likely for signal models
        
        best_match = None
        best_confidence = 0.0
        
        # Check patterns
        import re
        for pattern, version, base_confidence in filename_patterns:
            if re.search(pattern, filename):
                confidence = base_confidence
                # Boost confidence if it matches context hints
                if version in context_hints:
                    confidence += 0.1
                
                if confidence > best_confidence:
                    best_match = version
                    best_confidence = confidence
        
        # If no pattern match but we have context hints, use the most likely
        if not best_match and context_hints:
            best_match = context_hints[0]
            best_confidence = 0.4
        
        if best_match:
            version_info = cls.VERSION_PATTERNS.get(best_match, {})
            # Try to determine architecture
            architecture = None
            for arch_name in version_info.get('architecture_names', []):
                if arch_name.lower() in filename:
                    architecture = arch_name
                    break
            
            result.update({
                'success': True,
                'detected_version': best_match,
                'confidence': best_confidence,
                'architecture': architecture,
                'supported': version_info.get('supported', False),
                'framework': version_info.get('framework', 'unknown')
            })
        
        return result
    
    @classmethod
    def _analyze_model_structure_advanced(cls, model, model_info: Dict) -> Tuple[Optional[str], float]:
        """Advanced analysis of model structure to determine YOLO version."""
        try:
            confidence = 0.0
            detected_version = None
            
            if hasattr(model, 'model'):
                model_obj = model.model
                
                # Check for version-specific attributes and structure
                version_indicators = []
                
                # Check YAML structure for version-specific patterns
                if hasattr(model_obj, 'yaml'):
                    yaml_data = model_obj.yaml
                    if isinstance(yaml_data, dict):
                        # Analyze backbone structure
                        backbone = yaml_data.get('backbone', [])
                        if isinstance(backbone, list) and backbone:
                            # Check for version-specific layer types
                            layer_types = []
                            for layer in backbone:
                                if isinstance(layer, list) and len(layer) >= 3:
                                    layer_types.append(str(layer[2]))
                            
                            # Version-specific layer analysis
                            if 'C2f' in str(layer_types):
                                if 'DFL' in str(yaml_data) or 'v8' in str(yaml_data).lower():
                                    version_indicators.append(('yolov8', 0.8))
                                elif any(v in str(yaml_data).lower() for v in ['v11', '11']):
                                    version_indicators.append(('yolo11', 0.9))
                                elif any(v in str(yaml_data).lower() for v in ['v12', '12']):
                                    version_indicators.append(('yolov12', 0.9))
                                else:
                                    # Default to newer versions for C2f
                                    version_indicators.append(('yolo11', 0.7))
                            
                            elif 'C3' in str(layer_types):
                                version_indicators.append(('yolov5', 0.8))
                            elif 'RepConv' in str(layer_types):
                                version_indicators.append(('yolov7', 0.8))
                
                # Check model attributes for version hints
                if hasattr(model_obj, 'args'):
                    args = model_obj.args
                    if hasattr(args, 'model') and args.model:
                        model_name = str(args.model).lower()
                        if 'v12' in model_name or '12' in model_name:
                            version_indicators.append(('yolov12', 0.9))
                        elif 'v11' in model_name or '11' in model_name:
                            version_indicators.append(('yolo11', 0.9))
                        elif 'v8' in model_name:
                            version_indicators.append(('yolov8', 0.8))
                
                # Check for specific model characteristics
                nc = model_info.get('nc', 0)
                names = model_info.get('names', {})
                
                # Referee models are typically single class, signals are multi-class
                if nc == 1 and 'referee' in str(names).lower():
                    # Recent referee models are likely v12 or v11
                    version_indicators.append(('yolov12', 0.6))
                elif nc > 1 and any(signal in str(names).lower() for signal in ['arm', 'signal', 'serve']):
                    # Signal models are likely v11 or v8
                    version_indicators.append(('yolo11', 0.6))
            
            # Select best version based on indicators
            if version_indicators:
                # Sort by confidence and pick the best
                version_indicators.sort(key=lambda x: x[1], reverse=True)
                detected_version, confidence = version_indicators[0]
            
            return detected_version, confidence
            
        except Exception as e:
            logger.warning(f"Advanced model structure analysis failed: {e}")
            return None, 0.0
    
    @classmethod
    def _extract_version_indicators(cls, checkpoint: Dict) -> Dict[str, Any]:
        """Extract version indicators from model checkpoint."""
        indicators = {}
        
        try:
            # Check for version in metadata
            for key in ['version', 'yolo_version', 'ultralytics_version', 'model_version']:
                if key in checkpoint:
                    indicators[key] = checkpoint[key]
            
            # Check model structure
            if 'model' in checkpoint:
                model_dict = checkpoint['model']
                if hasattr(model_dict, '__dict__'):
                    for attr in ['yaml', 'yaml_file', 'model_name']:
                        if hasattr(model_dict, attr):
                            indicators[attr] = getattr(model_dict, attr)
            
            # Check for training info
            if 'train_args' in checkpoint:
                train_args = checkpoint['train_args']
                if isinstance(train_args, dict):
                    for key in ['model', 'data', 'imgsz']:
                        if key in train_args:
                            indicators[f'train_{key}'] = train_args[key]
            
            # Check optimizer info
            if 'optimizer' in checkpoint:
                indicators['has_optimizer'] = True
            
            # Check for epoch info
            if 'epoch' in checkpoint:
                indicators['epoch'] = checkpoint['epoch']
                
        except Exception as e:
            logger.warning(f"Failed to extract version indicators: {e}")
        
        return indicators
    
    @classmethod
    def _determine_version_from_indicators(cls, indicators: Dict[str, Any]) -> Optional[str]:
        """Determine YOLO version from extracted indicators."""
        try:
            # Direct version indicators
            for key, value in indicators.items():
                if 'version' in key.lower():
                    version_str = str(value).lower()
                    for version in cls.VERSION_PATTERNS.keys():
                        if version.replace('yolo', '').replace('v', '') in version_str:
                            return version
            
            # Model name indicators
            for key, value in indicators.items():
                if 'model' in key.lower():
                    model_str = str(value).lower()
                    if 'v12' in model_str or '12' in model_str:
                        return 'yolov12'
                    elif 'v11' in model_str or '11' in model_str:
                        return 'yolo11'
                    elif 'v8' in model_str:
                        return 'yolov8'
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to determine version from indicators: {e}")
            return None
    
    @classmethod
    def _determine_architecture_size(cls, model, version: str) -> Optional[str]:
        """Determine the architecture size (n, s, m, l, x) from the model."""
        try:
            if not hasattr(model, 'model'):
                return None
            
            # Try to get model size info
            model_obj = model.model
            
            # Check for parameter count or model size hints
            if hasattr(model_obj, 'yaml'):
                yaml_data = model_obj.yaml
                if isinstance(yaml_data, dict):
                    # Look for size indicators in the YAML
                    yaml_str = str(yaml_data).lower()
                    
                    # Check for explicit size in YAML
                    for size in ['n', 's', 'm', 'l', 'x']:
                        if f'{version}{size}' in yaml_str:
                            return f'{version}{size}'
            
            # Fallback: estimate based on parameter count if available
            try:
                total_params = sum(p.numel() for p in model_obj.parameters() if hasattr(p, 'numel'))
                
                # Rough parameter count ranges for different sizes
                if total_params < 5_000_000:  # < 5M
                    return f'{version}n'
                elif total_params < 15_000_000:  # < 15M
                    return f'{version}s'
                elif total_params < 30_000_000:  # < 30M
                    return f'{version}m'
                elif total_params < 50_000_000:  # < 50M
                    return f'{version}l'
                else:  # >= 50M
                    return f'{version}x'
                    
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to determine architecture size: {e}")
            return None
    
    @classmethod
    def _validate_model_functionality(cls, model) -> Dict[str, Any]:
        """Test basic model functionality."""
        validation = {
            'can_load': True,
            'can_predict': False,
            'input_shape': None,
            'output_shape': None,
            'error': None
        }
        
        try:
            # Test prediction with dummy input
            import numpy as np
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            results = model.predict(dummy_image, verbose=False)
            validation['can_predict'] = True
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    validation['output_shape'] = f"boxes: {result.boxes.shape if hasattr(result.boxes, 'shape') else 'unknown'}"
                
        except Exception as e:
            validation['error'] = str(e)
            validation['can_predict'] = False
        
        return validation
    
    @classmethod
    def get_supported_versions(cls) -> List[str]:
        """Get list of supported YOLO versions."""
        return [version for version, info in cls.VERSION_PATTERNS.items() if info['supported']]
    
    @classmethod
    def get_version_info(cls, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific YOLO version."""
        return cls.VERSION_PATTERNS.get(version)


class ModelCompatibilityValidator:
    """Validates model compatibility with the current system."""
    
    @classmethod
    def validate_model(cls, model_path: Path, expected_type: str = None) -> Dict[str, Any]:
        """
        Comprehensive model validation.
        
        Args:
            model_path: Path to model file
            expected_type: Expected model type (referee, signal, etc.)
            
        Returns:
            Validation results
        """
        validation = {
            'is_valid': False,
            'is_compatible': False,
            'version_info': {},
            'compatibility_issues': [],
            'recommendations': [],
            'model_info': {},
            'performance_test': {}
        }
        
        try:
            # Detect YOLO version
            version_info = YOLOVersionDetector.detect_yolo_version(model_path)
            validation['version_info'] = version_info
            
            if version_info.get('error'):
                validation['compatibility_issues'].append(f"Detection error: {version_info['error']}")
                return validation
            
            # Check if version is supported
            if not version_info.get('supported', False):
                validation['compatibility_issues'].append(
                    f"YOLO version '{version_info.get('detected_version', 'unknown')}' is not fully supported"
                )
                validation['recommendations'].append(
                    "Consider converting to a supported YOLO version (v8, v9, v10, v11, v12)"
                )
            
            # Test loading with Ultralytics
            if ULTRALYTICS_AVAILABLE:
                loading_test = cls._test_ultralytics_loading(model_path)
                validation['performance_test'] = loading_test
                
                if loading_test['success']:
                    validation['is_compatible'] = True
                    validation['model_info'] = loading_test.get('model_info', {})
                else:
                    validation['compatibility_issues'].append(
                        f"Ultralytics loading failed: {loading_test.get('error', 'unknown error')}"
                    )
            else:
                validation['compatibility_issues'].append("Ultralytics not available")
            
            # Validate for expected type
            if expected_type and validation['model_info']:
                type_validation = cls._validate_model_type(validation['model_info'], expected_type)
                validation.update(type_validation)
            
            # Overall validation
            validation['is_valid'] = (
                len(validation['compatibility_issues']) == 0 and 
                validation['is_compatible']
            )
            
            return validation
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            validation['compatibility_issues'].append(f"Validation error: {str(e)}")
            return validation
    
    @classmethod
    def _test_ultralytics_loading(cls, model_path: Path) -> Dict[str, Any]:
        """Test loading model with Ultralytics."""
        result = {
            'success': False,
            'model_info': {},
            'error': None,
            'load_time': 0,
            'prediction_test': False
        }
        
        try:
            import time
            start_time = time.time()
            
            # Load model
            model = YOLO(str(model_path))
            result['load_time'] = time.time() - start_time
            
            # Extract model info
            result['model_info'] = {
                'task': getattr(model, 'task', 'unknown'),
                'nc': getattr(model.model, 'nc', 0) if hasattr(model, 'model') else 0,
                'names': getattr(model.model, 'names', {}) if hasattr(model, 'model') else {},
                'device': str(getattr(model, 'device', 'unknown')),
                'model_type': type(model).__name__
            }
            
            # Test prediction
            import numpy as np
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            results = model.predict(dummy_input, verbose=False)
            result['prediction_test'] = True
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    @classmethod
    def _validate_model_type(cls, model_info: Dict, expected_type: str) -> Dict[str, Any]:
        """Validate model for specific use case."""
        validation = {
            'type_compatible': False,
            'type_issues': [],
            'type_recommendations': []
        }
        
        nc = model_info.get('nc', 0)
        names = model_info.get('names', {})
        
        if expected_type == 'referee':
            # Referee detection should have 1 class
            if nc == 1:
                validation['type_compatible'] = True
            elif nc > 1:
                validation['type_issues'].append(f"Model has {nc} classes, expected 1 for referee detection")
                validation['type_recommendations'].append("Consider retraining with single 'referee' class")
            else:
                validation['type_issues'].append("Model has no classes defined")
        
        elif expected_type == 'signal':
            # Signal detection should have multiple classes
            expected_signals = ['armLeft', 'armRight', 'hits', 'leftServe', 'net', 'outside', 'rightServe', 'touched']
            
            if nc >= 2:
                validation['type_compatible'] = True
                # Check if signal names match expected
                if names:
                    missing_signals = [sig for sig in expected_signals if sig not in names.values()]
                    if missing_signals:
                        validation['type_recommendations'].append(
                            f"Model may be missing signal classes: {missing_signals}"
                        )
            else:
                validation['type_issues'].append(f"Model has {nc} classes, expected multiple for signal detection")
        
        return validation


def detect_and_validate_model(model_path: Path, expected_type: str = None) -> Dict[str, Any]:
    """
    Convenience function to detect and validate a model.
    
    Args:
        model_path: Path to model file
        expected_type: Expected model type
        
    Returns:
        Combined detection and validation results
    """
    # Detect version
    version_info = YOLOVersionDetector.detect_yolo_version(model_path)
    
    # Validate compatibility
    validation_info = ModelCompatibilityValidator.validate_model(model_path, expected_type)
    
    # Combine results
    return {
        'model_path': str(model_path),
        'version_detection': version_info,
        'compatibility_validation': validation_info,
        'summary': {
            'detected_version': version_info.get('detected_version'),
            'is_supported': version_info.get('supported', False),
            'is_compatible': validation_info.get('is_compatible', False),
            'is_valid': validation_info.get('is_valid', False),
            'confidence': version_info.get('confidence', 0.0),
            'main_issues': validation_info.get('compatibility_issues', []),
            'recommendations': validation_info.get('recommendations', [])
        }
    } 