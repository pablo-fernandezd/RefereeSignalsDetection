"""
Routes Package for the Referee Detection System

This package contains all the API route handlers organized by functionality.
Each route module focuses on a specific domain (images, training, YouTube, etc.)
to maintain clear separation of concerns.
"""

from .image_routes import image_bp
from .training_routes import training_bp  
from .youtube_routes import youtube_bp
from .queue_routes import queue_bp

__all__ = [
    'image_bp',
    'training_bp',
    'youtube_bp', 
    'queue_bp'
] 