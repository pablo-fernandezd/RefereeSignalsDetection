"""
Model Package for the Referee Detection System

This package contains AI model interfaces, inference engines, and related
functionality for referee detection and signal classification.
"""

from .inference_engine import InferenceEngine
from .referee_detector import RefereeDetector
from .signal_classifier import SignalClassifier

__all__ = [
    'InferenceEngine',
    'RefereeDetector', 
    'SignalClassifier'
] 