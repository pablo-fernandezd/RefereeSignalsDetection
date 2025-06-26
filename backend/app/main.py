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

# Import configurations and routes using relative imports
from config.settings import (
    FlaskConfig, CORSConfig, LoggingConfig, ModelConfig,
    ensure_directories
)
from routes.image_routes import image_bp
from routes.training_routes import training_bp
from routes.youtube_routes import youtube_bp
from routes.queue_routes import queue_bp
from routes.model_routes import model_bp

logger = logging.getLogger(__name__)


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
    
    # Set specific logger levels to reduce noise
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)


def create_app() -> Flask:
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Configure Flask settings
    app.config['SECRET_KEY'] = FlaskConfig.SECRET_KEY
    app.config['DEBUG'] = FlaskConfig.DEBUG
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    
    # Setup CORS
    CORS(app, 
         methods=CORSConfig.METHODS,
         origins=CORSConfig.ORIGINS)
    
    # Ensure all required directories exist
    ensure_directories()
    
    # Setup logging
    setup_logging()
    
    # Register blueprints with proper URL prefixes
    app.register_blueprint(image_bp, url_prefix='/api')
    app.register_blueprint(training_bp, url_prefix='/api')
    app.register_blueprint(youtube_bp, url_prefix='/api/youtube')
    app.register_blueprint(queue_bp, url_prefix='/api/queue')
    app.register_blueprint(model_bp, url_prefix='/api/models')
    
    # Register error handlers
    register_error_handlers(app)
    
    logger.info("Referee Detection System initialized successfully")
    logger.info(f"Debug mode: {app.config['DEBUG']}")
    logger.info(f"Host: {FlaskConfig.HOST}:{FlaskConfig.PORT}")
    
    return app


def register_error_handlers(app: Flask) -> None:
    """
    Register global error handlers for the application.
    
    Args:
        app: Flask application instance
    """
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 Not Found errors."""
        logger.warning(f"404 error: {error}")
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server errors."""
        logger.error(f"Internal server error: {error}", exc_info=True)
        return {'error': 'Internal server error'}, 500
    
    @app.errorhandler(413)
    def file_too_large(error):
        """Handle 413 Request Entity Too Large errors."""
        logger.warning(f"File too large: {error}")
        return {'error': 'File too large. Maximum size is 100MB.'}, 413
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors."""
        logger.warning(f"Bad request: {error}")
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
        'version': '2.0.0',
        'timestamp': LoggingConfig.LOG_FILE.parent.name
    }


@app.route('/api/info')
def system_info():
    """
    System information endpoint.
    
    Returns:
        dict: System configuration and status information
    """
    try:
        # Don't load models during startup for faster initialization
        model_info = {
            'status': 'Models will be loaded on first use',
            'referee_model': str(ModelConfig.REFEREE_MODEL_PATH.name),
            'signal_model': str(ModelConfig.SIGNAL_MODEL_PATH.name),
            'referee_model_exists': ModelConfig.REFEREE_MODEL_PATH.exists(),
            'signal_model_exists': ModelConfig.SIGNAL_MODEL_PATH.exists()
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        model_info = {'error': 'Failed to get model information'}
    
    return {
        'system': 'Referee Detection System',
        'version': '2.0.0',
        'debug_mode': app.config['DEBUG'],
        'models': model_info,
        'directories': {
            'upload_folder': str(DirectoryConfig.UPLOAD_FOLDER),
            'crops_folder': str(DirectoryConfig.CROPS_FOLDER),
            'referee_training': str(DirectoryConfig.REFEREE_TRAINING_DATA_FOLDER),
            'signal_training': str(DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER)
        }
    }


if __name__ == '__main__':
    # Run the application
    try:
        app.run(
            host=FlaskConfig.HOST,
            port=FlaskConfig.PORT,
            debug=FlaskConfig.DEBUG,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1) 