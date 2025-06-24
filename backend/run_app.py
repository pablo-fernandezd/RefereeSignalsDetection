#!/usr/bin/env python3
"""
Simple runner script for the Referee Detection Backend

This script sets up the Python path and runs the main application,
making it easier to start the backend without module import issues.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Add the app directory to Python path
app_dir = backend_dir / 'app'
sys.path.insert(0, str(app_dir))

# Now import and run the app
from app.main import app

if __name__ == '__main__':
    # Set environment variables if needed
    if 'PORT' not in os.environ:
        os.environ['PORT'] = '5000'
    
    print("Starting Referee Detection Backend...")
    print(f"Backend directory: {backend_dir}")
    print(f"App directory: {app_dir}")
    
    # Check if we want debug mode (for development)
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    use_reloader = debug_mode  # Only use reloader in debug mode
    
    print(f"Debug mode: {debug_mode}")
    print(f"Auto-reload: {use_reloader}")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=debug_mode,
        use_reloader=use_reloader
    ) 