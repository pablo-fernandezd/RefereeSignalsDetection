#!/usr/bin/env python3
"""
Production runner script for the Referee Detection Backend

This script runs the backend in production mode without debug/auto-reload
to prevent interruptions during video processing.
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
    
    print("Starting Referee Detection Backend (Production Mode)...")
    print(f"Backend directory: {backend_dir}")
    print(f"App directory: {app_dir}")
    print("Debug mode: DISABLED (no auto-restart)")
    print("This prevents interruptions during video processing")
    
    # Run the application in production mode
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False,
        use_reloader=False
    ) 