#!/usr/bin/env python3
"""
Development runner script for the Referee Detection Backend

This script runs the backend in development mode with debug/auto-reload enabled.
WARNING: This will restart the server when files change, interrupting video processing!
Use run_production.py for stable video processing.
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
    
    print("Starting Referee Detection Backend (Development Mode)...")
    print(f"Backend directory: {backend_dir}")
    print(f"App directory: {app_dir}")
    print("Debug mode: ENABLED (auto-restart on file changes)")
    print("WARNING: This will interrupt video processing on restarts!")
    print("Use run_production.py for stable video processing.")
    
    # Run the application in development mode
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=True,
        use_reloader=True
    ) 