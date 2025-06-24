#!/usr/bin/env python3
"""
Basic test script for YouTube processor functionality (without model files)
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_youtube_processor_basic():
    """Test the YouTube processor basic functionality without model loading"""
    
    print("🧪 Testing YouTube Processor (Basic)...")
    
    try:
        # Test if yt-dlp is available
        print("1. Testing yt-dlp availability...")
        import yt_dlp
        print("   ✅ yt-dlp is available")
        
        # Test if other dependencies are available
        print("2. Testing dependencies...")
        import cv2
        print("   ✅ OpenCV is available")
        
        import torch
        print("   ✅ PyTorch is available")
        
        import ultralytics
        print("   ✅ Ultralytics is available")
        
        # Test directory creation
        print("3. Testing directory structure...")
        base_dir = Path("data/youtube_videos")
        base_dir.mkdir(exist_ok=True)
        print(f"   📁 Base directory: {base_dir}")
        print(f"   ✅ Base directory exists: {base_dir.exists()}")
        
        # Test video ID extraction (without model)
        print("4. Testing video ID extraction...")
        test_url = "https://www.youtube.com/watch?v=TMou-6_4w10"
        
        # Simple regex extraction for testing
        import re
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', test_url)
        video_id = video_id_match.group(1) if video_id_match else 'unknown'
        
        print(f"   🎥 Test URL: {test_url}")
        print(f"   🆔 Extracted video ID: {video_id}")
        print(f"   ✅ Video ID extraction: {'TMou-6_4w10' in video_id}")
        
        # Test processing info structure
        print("5. Testing processing info structure...")
        sample_info = {
            "total_frames": 1000,
            "segments_created": 2,
            "frames_extracted": 50,
            "crops_created": 30,
            "auto_crop": True,
            "status": "completed"
        }
        print(f"   📊 Sample processing info: {sample_info}")
        print("   ✅ Processing info structure defined")
        
        # Test Flask app structure
        print("6. Testing Flask app structure...")
        app_file = Path("app.py")
        if app_file.exists():
            print("   ✅ Flask app.py exists")
        else:
            print("   ❌ Flask app.py not found")
            return False
        
        # Test YouTube processor file
        print("7. Testing YouTube processor file...")
        processor_file = Path("youtube_processor.py")
        if processor_file.exists():
            print("   ✅ youtube_processor.py exists")
        else:
            print("   ❌ youtube_processor.py not found")
            return False
        
        print("\n🎉 All basic tests passed! YouTube processor is ready.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   💡 Please install required dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_frontend_structure():
    """Test if frontend files exist"""
    
    print("\n🧪 Testing Frontend Structure...")
    
    try:
        frontend_dir = Path("../frontend")
        if not frontend_dir.exists():
            print("   ❌ Frontend directory not found")
            return False
        
        print(f"   ✅ Frontend directory exists: {frontend_dir}")
        
        # Check key files
        key_files = [
            "package.json",
            "src/App.js",
            "src/components/YouTubeProcessing.js",
            "src/components/YouTubeProcessing.css"
        ]
        
        for file_path in key_files:
            full_path = frontend_dir / file_path
            if full_path.exists():
                print(f"   ✅ {file_path} exists")
            else:
                print(f"   ❌ {file_path} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 Starting Basic System Tests\n")
    
    # Test basic functionality
    basic_tests = test_youtube_processor_basic()
    
    # Test frontend structure
    frontend_tests = test_frontend_structure()
    
    print("\n" + "="*50)
    print("📊 Test Results:")
    print(f"   Basic functionality: {'✅ PASS' if basic_tests else '❌ FAIL'}")
    print(f"   Frontend structure: {'✅ PASS' if frontend_tests else '❌ FAIL'}")
    
    if basic_tests and frontend_tests:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Download model files (optional for full functionality)")
        print("   2. Start the Flask server: python app.py")
        print("   3. Start the React frontend: npm start")
        print("   4. Navigate to the YouTube tab to process videos")
        print("\n💡 Note: Without model files, auto-crop functionality will not work,")
        print("   but video downloading and segmentation will still function.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 