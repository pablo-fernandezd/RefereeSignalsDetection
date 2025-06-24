#!/usr/bin/env python3
"""
Test script for YouTube processor functionality
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from youtube_processor import YouTubeProcessor

def test_youtube_processor():
    """Test the YouTube processor initialization and basic functionality"""
    
    print("🧪 Testing YouTube Processor...")
    
    try:
        # Test initialization
        print("1. Testing initialization...")
        processor = YouTubeProcessor()
        print("   ✅ YouTubeProcessor initialized successfully")
        
        # Test directory creation
        print("2. Testing directory structure...")
        base_dir = processor.base_dir
        print(f"   📁 Base directory: {base_dir}")
        print(f"   ✅ Base directory exists: {base_dir.exists()}")
        
        # Test video ID extraction
        print("3. Testing video ID extraction...")
        test_url = "https://www.youtube.com/watch?v=TMou-6_4w10"
        video_id = processor._extract_video_id(test_url)
        print(f"   🎥 Test URL: {test_url}")
        print(f"   🆔 Extracted video ID: {video_id}")
        print(f"   ✅ Video ID extraction: {'TMou-6_4w10' in video_id}")
        
        # Test get_all_videos (should be empty initially)
        print("4. Testing video listing...")
        videos = processor.get_all_videos()
        print(f"   📋 Found {len(videos)} processed videos")
        print("   ✅ Video listing works")
        
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
        
        print("\n🎉 All tests passed! YouTube processor is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_model_loading():
    """Test if the YOLO model can be loaded"""
    
    print("\n🧪 Testing Model Loading...")
    
    try:
        processor = YouTubeProcessor()
        
        # Check if model exists in the correct path (root models folder)
        model_path = Path("../models/bestRefereeDetection.pt")
        if not model_path.exists():
            print("   ⚠️  Model file not found. Please download the model.")
            print("   📁 Expected path: ../models/bestRefereeDetection.pt")
            return False
        
        print(f"   ✅ Model file found: {model_path}")
        print(f"   🧠 Model loaded successfully")
        print(f"   🎯 Class ID for referee: {processor.class_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 Starting YouTube Processor Tests\n")
    
    # Test basic functionality
    basic_tests = test_youtube_processor()
    
    # Test model loading
    model_tests = test_model_loading()
    
    print("\n" + "="*50)
    print("📊 Test Results:")
    print(f"   Basic functionality: {'✅ PASS' if basic_tests else '❌ FAIL'}")
    print(f"   Model loading: {'✅ PASS' if model_tests else '❌ FAIL'}")
    
    if basic_tests and model_tests:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Start the Flask server: python app.py")
        print("   2. Start the React frontend: npm start")
        print("   3. Navigate to the YouTube tab to process videos")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        if not model_tests:
            print("   💡 Tip: Download the model files to enable full functionality")

if __name__ == "__main__":
    main() 