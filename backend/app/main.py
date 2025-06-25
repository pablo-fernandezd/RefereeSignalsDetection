"""
Main Application Module for Referee Detection System

This module serves as the main entry point for the Flask application,
organizing routes, initializing services, and configuring the application
with proper error handling and logging.
"""

import logging
import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS

# Add the parent directory to the Python path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

# Import with proper path handling
try:
    from config.settings import (
        FlaskConfig, CORSConfig, LoggingConfig, ModelConfig,
        ensure_directories
    )
    from routes.image_routes import image_bp
    from routes.training_routes import training_bp
    from routes.youtube_routes import youtube_bp
    from routes.queue_routes import queue_bp
    from routes.model_routes import model_bp
except ImportError:
    # Try alternative import paths
    sys.path.append(str(Path(__file__).parent))
    sys.path.append(str(Path(__file__).parent.parent / 'config'))
    
    from config.settings import (
        FlaskConfig, CORSConfig, LoggingConfig, ModelConfig,
        ensure_directories
    )
    from routes.image_routes import image_bp
    from routes.training_routes import training_bp
    from routes.youtube_routes import youtube_bp
    from routes.queue_routes import queue_bp
    from routes.model_routes import model_bp


def setup_logging():
    """
    Configure application logging with proper formatting and levels.
    """
    # Ensure log directory exists
    LoggingConfig.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, LoggingConfig.LOG_LEVEL),
        format=LoggingConfig.LOG_FORMAT,
        handlers=[
            logging.FileHandler(LoggingConfig.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)


def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Configure Flask settings
    app.config['SECRET_KEY'] = FlaskConfig.SECRET_KEY
    app.config['DEBUG'] = FlaskConfig.DEBUG
    
    # Setup CORS
    CORS(app, 
         methods=CORSConfig.METHODS,
         origins=CORSConfig.ORIGINS)
    
    # Ensure all required directories exist
    ensure_directories()
    
    # Setup logging
    setup_logging()
    
    # Register blueprints
    app.register_blueprint(image_bp, url_prefix='/api')
    app.register_blueprint(training_bp, url_prefix='/api')
    app.register_blueprint(youtube_bp, url_prefix='/api/youtube')
    app.register_blueprint(queue_bp, url_prefix='/api/queue')
    app.register_blueprint(model_bp)
    
    # Register error handlers
    register_error_handlers(app)
    
    logger = logging.getLogger(__name__)
    logger.info("Referee Detection System initialized successfully")
    logger.info(f"Debug mode: {app.config['DEBUG']}")
    
    return app


def register_error_handlers(app):
    """
    Register global error handlers for the application.
    
    Args:
        app: Flask application instance
    """
    @app.errorhandler(404)
    def not_found_error(error):
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger = logging.getLogger(__name__)
        logger.error(f"Internal server error: {error}")
        return {'error': 'Internal server error'}, 500
    
    @app.errorhandler(413)
    def file_too_large(error):
        return {'error': 'File too large'}, 413
    
    @app.errorhandler(400)
    def bad_request(error):
        return {'error': 'Bad request'}, 400


# Create the application instance
app = create_app()


@app.route('/api/health')
def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        dict: Health status information
    """
    return {
        'status': 'healthy',
        'service': 'Referee Detection System',
        'version': '2.0.0'
    }


@app.route('/api/info')
def system_info():
    """
    System information endpoint.
    
    Returns:
        dict: System configuration and status information
    """
    from models.inference_engine import InferenceEngine
    
    try:
        # Don't load models during startup for faster initialization
        model_info = {
            'status': 'Models will be loaded on first use',
            'referee_model': str(ModelConfig.REFEREE_MODEL_PATH.name),
            'signal_model': str(ModelConfig.SIGNAL_MODEL_PATH.name)
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to get model info: {e}")
        model_info = {'error': 'Failed to get model information'}
    
    return {
        'system': 'Referee Detection System',
        'version': '2.0.0',
        'debug_mode': app.config['DEBUG'],
        'models': model_info
    }


if __name__ == '__main__':
    # Run the application
    app.run(
        host=FlaskConfig.HOST,
        port=FlaskConfig.PORT,
        debug=FlaskConfig.DEBUG
    ) 