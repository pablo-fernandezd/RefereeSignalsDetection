#!/usr/bin/env python3
"""
Setup script for test environment.
"""

import os
import sys
from pathlib import Path
import subprocess


def create_test_directories():
    """Create required test directories."""
    print("📁 Creating test directories...")
    
    directories = [
        "temp_uploads",
        "test_data", 
        "test_models",
        "model_registry/active",
        "model_registry/models",
        "static/uploads",
        "static/referee_crops",
        "static/signals",
        "logs"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created {directory}")


def install_test_dependencies():
    """Install test dependencies."""
    print("📦 Installing test dependencies...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'pytest>=7.0.0', 
            'pytest-cov>=4.0.0', 
            'pytest-mock>=3.10.0'
        ], check=True)
        print("   ✅ Test dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        return False


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is not supported")
        print("   📋 Please use Python 3.8 or higher")
        return False


def setup_environment():
    """Setup environment variables."""
    print("🔧 Setting up environment...")
    
    os.environ['TESTING'] = 'True'
    os.environ['FLASK_ENV'] = 'testing'
    
    # Add current directory to Python path
    backend_dir = Path(__file__).parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    print("   ✅ Environment configured")


def verify_test_setup():
    """Verify test setup is working."""
    print("🧪 Verifying test setup...")
    
    try:
        # Test import
        import pytest
        print("   ✅ pytest import successful")
        
        # Test basic functionality
        from conftest import app
        print("   ✅ conftest import successful")
        
        # Check if we can run a simple test
        result = subprocess.run([
            sys.executable, '-m', 'pytest', '--version'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ pytest version: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ pytest not working properly")
            return False
            
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Setup verification failed: {e}")
        return False


def run_sample_test():
    """Run a sample test to verify everything works."""
    print("🎯 Running sample test...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_utils.py::TestValidation::test_validate_confidence_threshold',
            '-v'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Sample test passed")
            return True
        else:
            print("   ❌ Sample test failed")
            print(f"   📋 Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Could not run sample test: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Referee Detection System Test Environment")
    print("=" * 60)
    
    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    print(f"📍 Working directory: {backend_dir}")
    
    success = True
    
    # Step 1: Check Python version
    if not check_python_version():
        success = False
    
    # Step 2: Create directories
    create_test_directories()
    
    # Step 3: Setup environment
    setup_environment()
    
    # Step 4: Install dependencies
    if not install_test_dependencies():
        success = False
    
    # Step 5: Verify setup
    if not verify_test_setup():
        success = False
    
    # Step 6: Run sample test
    if success and not run_sample_test():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Test environment setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run all tests: python -m pytest tests/ -v")
        print("   2. Run specific test: python -m pytest tests/test_api.py -v")
        print("   3. Run with coverage: python -m pytest tests/ --cov=app")
        print("   4. Use test runner: python run_tests.py")
    else:
        print("❌ Test environment setup failed!")
        print("\n📋 Please check the errors above and try again.")
        print("   💡 Common solutions:")
        print("   - Ensure Python 3.8+ is installed")
        print("   - Check internet connection for dependency installation")
        print("   - Verify write permissions in the backend directory")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 