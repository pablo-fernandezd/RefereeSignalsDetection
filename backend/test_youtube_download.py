#!/usr/bin/env python3
"""
Test script for YouTube download functionality
"""

import yt_dlp
import sys
from pathlib import Path

def test_youtube_download(url="https://www.youtube.com/watch?v=jNQXAC9IVRw"):
    """Test YouTube download with the same configuration as the main app."""
    
    print(f"Testing YouTube download for: {url}")
    print(f"yt-dlp version: {yt_dlp.version.__version__}")
    
    # Create test output directory
    output_dir = Path("test_downloads")
    output_dir.mkdir(exist_ok=True)
    
    # Enhanced configuration matching the main app
    ydl_opts = {
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'format': 'best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best',
        'quiet': False,  # Enable output for testing
        'noplaylist': True,
        # Enhanced options to handle YouTube restrictions
        'extractor_retries': 3,
        'fragment_retries': 3,
        'retry_sleep_functions': {'http': lambda n: min(4 ** n, 60)},
        'http_chunk_size': 10485760,  # 10MB chunks
        'no_warnings': False,
        'ignoreerrors': False,
        # User agent and headers to avoid detection
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'referer': 'https://www.youtube.com/',
        # Additional anti-detection measures
        'sleep_interval': 1,
        'max_sleep_interval': 5,
        'sleep_interval_requests': 1,
        # Fallback options
        'prefer_free_formats': True,
        'youtube_include_dash_manifest': False,
    }
    
    try:
        print("\n=== Testing primary download configuration ===")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First, try to extract info only
            print("Extracting video information...")
            info = ydl.extract_info(url, download=False)
            print(f"Title: {info.get('title', 'Unknown')}")
            print(f"Duration: {info.get('duration', 'Unknown')} seconds")
            print(f"Uploader: {info.get('uploader', 'Unknown')}")
            
            # Then try to download (uncomment to actually download)
            # print("Starting download...")
            # ydl.download([url])
            # print("✅ Primary download configuration works!")
            
        return True
        
    except Exception as primary_error:
        print(f"❌ Primary download failed: {primary_error}")
        
        print("\n=== Testing fallback download configuration ===")
        # Fallback options
        fallback_opts = {
            'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
            'format': 'worst[ext=mp4]/worst',  # Use worst quality as fallback
            'quiet': False,
            'noplaylist': True,
            'extractor_retries': 5,
            'fragment_retries': 5,
            'retry_sleep_functions': {'http': lambda n: min(8 ** n, 120)},
            'http_chunk_size': 1048576,  # 1MB chunks (smaller)
            'user_agent': 'yt-dlp/2025.6.9',  # Use default yt-dlp user agent
            'sleep_interval': 2,
            'max_sleep_interval': 10,
            'prefer_free_formats': True,
            'youtube_include_dash_manifest': False,
            'no_check_certificate': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                print("Extracting video information with fallback config...")
                info = ydl.extract_info(url, download=False)
                print(f"Title: {info.get('title', 'Unknown')}")
                print(f"Duration: {info.get('duration', 'Unknown')} seconds")
                print("✅ Fallback configuration works!")
                return True
                
        except Exception as fallback_error:
            print(f"❌ Fallback download also failed: {fallback_error}")
            print("\n=== Troubleshooting suggestions ===")
            print("1. Try updating yt-dlp: pip install --upgrade yt-dlp")
            print("2. Check your internet connection")
            print("3. Try a different YouTube video URL")
            print("4. Check if YouTube is blocking your IP")
            return False

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    success = test_youtube_download(url)
    sys.exit(0 if success else 1) 