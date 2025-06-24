"""
YouTube Video Processor for Referee Detection System

This module handles YouTube video downloading, processing, and frame extraction
with segment-based processing for robust resume capability.
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
import threading
import time
import subprocess

import cv2
import yt_dlp
from ultralytics import YOLO

# Import with proper path handling
import sys
from pathlib import Path

# Add necessary paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config.settings import DirectoryConfig, ModelConfig
except ImportError:
    from app.config.settings import DirectoryConfig, ModelConfig


logger = logging.getLogger(__name__)


class YouTubeProcessor:
    """Process YouTube videos for referee detection training with segment-based processing."""
    
    # Class-level lock to prevent multiple simultaneous downloads
    _download_lock = threading.Lock()
    _active_downloads = set()
    
    def __init__(self):
        self.MODEL_SIZE = 640
        self.CONFIDENCE_THRESHOLD = 0.5
        self.DEVICE = 'cuda' if ModelConfig.DEVICE == 'cuda' else 'cpu'
        self.SEGMENT_DURATION = 300  # 5 minutes per segment (300 seconds)
        self.model = None
        self.class_id = None
        self.progress_lock = threading.Lock()
        
        logger.info("YouTube processor initialized with segment-based processing")

    def _load_model(self):
        """Load YOLO model for referee detection (lazy loading)."""
        if self.model is None:
            try:
                model_path = ModelConfig.REFEREE_MODEL_PATH
                if not Path(model_path).exists():
                    logger.error(f"Model file not found: {model_path}")
                    return False
                
                logger.info("Loading referee detection model...")
                self.model = YOLO(str(model_path))
                self.model.to(self.DEVICE)
                self.class_id = self._get_referee_class_id()
                logger.info("Referee detection model loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False
        return True

    def _get_referee_class_id(self):
        """Get class ID for referee from model."""
        try:
            class_names = self.model.names
            for class_id, name in class_names.items():
                if 'referee' in name.lower():
                    return class_id
            return 0  # Default to first class if no referee class found
        except Exception:
            return 0

    def download_youtube_video(self, url, auto_label=True):
        """
        Download and process YouTube video with segment-based processing.
        
        Args:
            url: YouTube video URL
            auto_label: Whether to automatically create labels for detected referees
            
        Returns:
            dict: Processing results with folder_name and status
        """
        # Check if video is already being processed
        video_id = self._extract_video_id(url)
        if not video_id:
            return {'error': 'Invalid YouTube URL'}
        
        # Prevent multiple downloads of the same video
        with self._download_lock:
            if video_id in self._active_downloads:
                return {'error': f'Video {video_id} is already being processed'}
            self._active_downloads.add(video_id)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{video_id}_{timestamp}"
            video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
            
            # Create all necessary directories
            directories = ['original', 'frames', 'processed', 'crops', 'segments', 'thumbnails']
            for dir_name in directories:
                (video_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
            progress_file = video_dir / "progress.json"
            
            # Initialize progress file with segment-based structure
            initial_progress = {
                'status': 'downloading',
                'stage': 'initializing',
                'percent_download': 0.0,
                'percent': 0.0,  # Frontend expects 'percent' for downloading
                'percent_processing': 0.0,
                'current_segment': 0,
                'total_segments': 0,
                'completed_segments': 0,
                'current_frame': 0,
                'total_frames': 0,
                'frames_extracted': 0,
                'crops_created': 0,
                'labels_created': 0,
                'auto_label': auto_label,
                'video_id': video_id,
                'folder_name': folder_name,
                'segment_duration': self.SEGMENT_DURATION,
                'updated_at': datetime.now().isoformat()
            }
            
            # Write immediately with basic file operation to ensure it exists
            try:
                with open(progress_file, 'w') as f:
                    json.dump(initial_progress, f, indent=2)
                logger.info(f"Initialized progress file for {folder_name}")
            except Exception as e:
                logger.error(f"Failed to initialize progress file: {e}")
                return {'error': 'Failed to initialize progress tracking'}
            
            # Download video
            logger.info(f"Starting download for video: {video_id}")
            video_file = self._download_video(url, video_dir / "original", auto_label, progress_file)
            
            if not video_file:
                self._write_progress_safe(progress_file, {
                    'status': 'error',
                    'stage': 'download_failed',
                    'error': 'Failed to download video'
                })
                return {'error': 'Failed to download video'}
            
            # IMPORTANT: Remove from active downloads after download completes
            # This allows new downloads to start while this video is being processed
            with self._download_lock:
                self._active_downloads.discard(video_id)
                logger.info(f"Download completed for {video_id}, removed from active downloads")
            
            # Get video information
            video_info = self._get_video_info(video_file)
            if not video_info:
                self._write_progress_safe(progress_file, {
                    'status': 'error',
                    'stage': 'video_info_failed',
                    'error': 'Could not get video information'
                })
                return {'error': 'Could not get video information'}
            
            # Calculate segments and frames
            video_duration = video_info['duration']  # in seconds
            total_segments = max(1, int((video_duration + self.SEGMENT_DURATION - 1) // self.SEGMENT_DURATION))  # Ceiling division
            total_frames_to_extract = int(video_duration)  # 1 frame per second
            
            logger.info(f"Video duration: {video_duration}s, will create {total_segments} segments of {self.SEGMENT_DURATION}s each")
            logger.info(f"Total frames to extract: {total_frames_to_extract} (1 FPS)")
            
            # Update progress with segment information
            self._write_progress_safe(progress_file, {
                'status': 'processing',
                'stage': 'preparing_segments',
                'percent_download': 100.0,
                'percent_processing': 0.0,
                'current_segment': 0,
                'total_segments': total_segments,
                'completed_segments': 0,
                'total_frames': total_frames_to_extract,
                'video_duration': video_duration,
                'video_fps': video_info['fps']
            })
            
            # Process video by segments
            processing_result = self._process_video_by_segments(video_file, video_dir, auto_label, progress_file, video_info)
            
            # Final status update
            final_status = 'completed'
            if auto_label and processing_result['labels_created'] > 0:
                final_status = 'auto_labeling_ready'
            
            self._write_progress_safe(progress_file, {
                'status': final_status,
                'stage': 'completed',
                'percent_download': 100.0,
                'percent_processing': 100.0,
                'completed_segments': processing_result['completed_segments'],
                'frames_extracted': processing_result['frames_extracted'],
                'crops_created': processing_result['crops_created'],
                'labels_created': processing_result['labels_created'],
                'completed_at': datetime.now().isoformat()
            })
            
            logger.info(f"Successfully processed video {video_id}: {processing_result['frames_extracted']} frames, {processing_result['crops_created']} crops")
            
            return {
                'folder_name': folder_name,
                'status': final_status,
                'frames_extracted': processing_result['frames_extracted'],
                'crops_created': processing_result['crops_created'],
                'labels_created': processing_result['labels_created'],
                'completed_segments': processing_result['completed_segments']
            }
            
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            return {'error': str(e)}
            
        finally:
            # Ensure removal from active downloads in case of any error
            with self._download_lock:
                self._active_downloads.discard(video_id)

    def _extract_video_id(self, url):
        """Extract video ID from YouTube URL."""
        try:
            import re
            pattern = r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)'
            match = re.search(pattern, url)
            return match.group(1) if match else None
        except Exception:
            return None

    def _download_video(self, url, output_dir, auto_label, progress_file):
        """Download video from YouTube with progress tracking."""
        try:
            def progress_hook(d):
                try:
                    if d['status'] == 'downloading':
                        percent = 0.0
                        
                        # yt-dlp provides progress in different formats
                        if 'downloaded_bytes' in d and 'total_bytes' in d and d['total_bytes']:
                            # Most reliable method: calculate from bytes
                            percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                        elif 'downloaded_bytes' in d and 'total_bytes_estimate' in d and d['total_bytes_estimate']:
                            # Fallback: use estimated total bytes
                            percent = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 100
                        elif '_percent_str' in d and d['_percent_str']:
                            # Parse percentage string if available
                            percent_str = d['_percent_str'].strip().replace('%', '')
                            try:
                                percent = float(percent_str)
                            except ValueError:
                                percent = 0.0
                        
                        # Ensure percent is valid
                        percent = max(0.0, min(100.0, percent))
                        
                        # Update progress file
                        self._write_progress_safe(progress_file, {
                            'status': 'downloading',
                            'stage': 'downloading_video',
                            'percent_download': round(percent, 2),
                            'percent': round(percent, 2),  # Frontend expects this
                            'auto_label': auto_label
                        })
                        
                    elif d['status'] == 'finished':
                        self._write_progress_safe(progress_file, {
                            'status': 'processing',
                            'stage': 'download_completed',
                            'percent_download': 100.0,
                            'percent': 100.0,
                            'auto_label': auto_label
                        })
                        
                except Exception as e:
                    logger.warning(f"Error in progress hook: {e}")
            
            ydl_opts = {
                'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
                'merge_output_format': 'mp4',
                'cookiesfrombrowser': ('firefox',),
                'quiet': True,
                'noplaylist': True,
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'progress_hooks': [progress_hook]
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find the downloaded video file
            video_files = list(output_dir.glob("*.mp4"))
            if video_files:
                video_file = video_files[0]
                logger.info(f"Downloaded video: {video_file}")
                return video_file
            else:
                logger.error("No video file found after download")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            self._write_progress_safe(progress_file, {
                'status': 'error',
                'stage': 'download_failed',
                'error': str(e),
                'percent_download': 0.0,
                'percent': 0.0
            })
            return None

    def _get_video_info(self, video_path):
        """Get video information using FFprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            duration = float(info['format']['duration'])
            fps = eval(video_stream['r_frame_rate'])  # e.g., "30/1" -> 30.0
            
            return {
                'duration': duration,
                'fps': fps,
                'width': int(video_stream['width']),
                'height': int(video_stream['height'])
            }
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None

    def _process_video_by_segments(self, video_path, video_dir, auto_label, progress_file, video_info):
        """
        Process video by segments for robust resume capability.
        This is the core improvement - segment-based processing.
        """
        duration = video_info['duration']
        total_segments = max(1, int((duration + self.SEGMENT_DURATION - 1) // self.SEGMENT_DURATION))
        
        # Directories
        frames_dir = video_dir / "frames"
        processed_dir = video_dir / "processed"
        crops_dir = video_dir / "crops"
        segments_dir = video_dir / "segments"
        thumbnails_dir = video_dir / "thumbnails"

        # Check for resume: see which segments have been completed
        completed_segments = self._get_completed_segments(segments_dir, total_segments)
        start_segment = len(completed_segments)
        
        logger.info(f"Processing video: {total_segments} total segments")
        if start_segment > 0:
            logger.info(f"Resuming from segment {start_segment + 1} (found {start_segment} completed segments)")
        
        # Counters
        total_frames_extracted = len(list(frames_dir.glob("frame_*.jpg"))) if frames_dir.exists() else 0
        total_crops_created = len(list(crops_dir.glob("crop_*.jpg"))) if crops_dir.exists() else 0
        total_labels_created = 0
        
        # First step: Create video segments if they don't exist
        logger.info(f"Step 1: Creating {total_segments} video segments...")
        self._write_progress_safe(progress_file, {
            'status': 'processing',
            'stage': 'creating_segments',
            'percent_processing': 5.0,
            'current_segment': 0,
            'total_segments': total_segments,
            'completed_segments': 0
        })
        
        # Create video segments
        segments_created = 0
        for segment_idx in range(total_segments):
            segment_file = segments_dir / f"segment_{segment_idx + 1:03d}.mp4"
            if not segment_file.exists():
                segment_start_time = segment_idx * self.SEGMENT_DURATION
                segment_end_time = min((segment_idx + 1) * self.SEGMENT_DURATION, duration)
                segment_duration = segment_end_time - segment_start_time
                self._create_segment_video(video_path, segment_start_time, segment_duration, segment_file)
                logger.info(f"Created segment {segment_idx + 1}/{total_segments}")
            
            # Count existing segments (whether just created or already existed)
            if segment_file.exists():
                segments_created += 1
        
        # Update completed segments count after creating all segments
        self._write_progress_safe(progress_file, {
            'status': 'processing',
            'stage': 'creating_segments',
            'percent_processing': 10.0,
            'current_segment': 0,
            'total_segments': total_segments,
            'completed_segments': segments_created,
            'segments_created': segments_created
        })
        
        # Load model for referee detection if auto_label is enabled
        if auto_label:
            if not self._load_model():
                logger.warning("Could not load model for auto-labeling")
                auto_label = False
        
        try:
            # Step 2: Extract all frames at once using OpenCV (much faster)
            logger.info(f"Step 2: Extracting all frames at 1 FPS...")
            # Get current completed segments count
            current_completed_segments = len(self._get_completed_segments(segments_dir, total_segments))
            self._write_progress_safe(progress_file, {
                'status': 'processing',
                'stage': 'extracting_frames',
                'percent_processing': 15.0,
                'total_segments': total_segments,
                'completed_segments': current_completed_segments,
                'segments_created': current_completed_segments
            })
            
            # Extract all frames at once
            extraction_result = self._extract_all_frames_opencv(
                video_path, frames_dir, thumbnails_dir, progress_file, duration
            )
            
            if extraction_result['success']:
                total_frames_extracted = extraction_result['frames_extracted']
                logger.info(f"Successfully extracted {total_frames_extracted} frames using OpenCV")
            else:
                logger.error(f"Failed to extract frames: {extraction_result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'completed_segments': 0,
                    'frames_extracted': 0,
                    'crops_created': 0,
                    'labels_created': 0,
                    'error': extraction_result.get('error', 'Frame extraction failed')
                }
            
            # Step 3: Create auto-labels if enabled
            if auto_label and total_frames_extracted > 0:
                logger.info(f"Step 3: Creating auto-labels for {total_frames_extracted} frames...")
                current_completed_segments = len(self._get_completed_segments(segments_dir, total_segments))
                self._write_progress_safe(progress_file, {
                    'status': 'processing',
                    'stage': 'creating_auto_labels',
                    'percent_processing': 70.0,
                    'frames_extracted': total_frames_extracted,
                    'total_segments': total_segments,
                    'completed_segments': current_completed_segments,
                    'segments_created': current_completed_segments
                })
                
                labeling_result = self._create_auto_labels(
                    frames_dir, crops_dir, processed_dir, progress_file
                )
                total_crops_created = labeling_result['crops_created']
                total_labels_created = labeling_result['labels_created']
            
            # Final update
            completed_segments = len(self._get_completed_segments(segments_dir, total_segments))
            
            return {
                'success': True,
                'completed_segments': completed_segments,
                'frames_extracted': total_frames_extracted,
                'crops_created': total_crops_created,
                'labels_created': total_labels_created,
                'total_segments_processed': completed_segments
            }
            
        except Exception as e:
            logger.error(f"Error in segment processing: {e}")
            return {
                'success': False,
                'completed_segments': len(self._get_completed_segments(segments_dir, total_segments)),
                'frames_extracted': total_frames_extracted,
                'crops_created': total_crops_created,
                'labels_created': total_labels_created,
                'error': str(e)
            }

    def _get_completed_segments(self, segments_dir, total_segments):
        """Get list of completed segment files."""
        if not segments_dir.exists():
            return []
        
        completed = []
        for i in range(1, total_segments + 1):
            segment_file = segments_dir / f"segment_{i:03d}.mp4"
            if segment_file.exists():
                completed.append(i)
        
        return completed

    def _extract_frames_from_segment(self, video_path, segment_idx, start_time, duration, 
                                   frames_dir, thumbnails_dir, progress_file):
        """Extract frames from a single video segment at 1 FPS using OpenCV for speed."""
        try:
            frames_extracted = 0
            
            # Calculate frame indices for this segment (1 FPS)
            start_frame_idx = int(start_time)  # Frame index = seconds (1 FPS)
            end_frame_idx = int(start_time + duration)
            
            # Use OpenCV for much faster frame extraction
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return {'success': False, 'frames_extracted': 0, 'error': 'Could not open video'}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Extracting frames {start_frame_idx} to {end_frame_idx} from segment {segment_idx + 1}")
            
            # Extract frames for this segment
            for frame_idx in range(start_frame_idx, end_frame_idx):
                frame_filename = f"frame_{frame_idx:08d}.jpg"
                frame_path = frames_dir / frame_filename
                
                # Skip if frame already exists
                if frame_path.exists():
                    frames_extracted += 1
                    continue
                
                # Calculate the exact frame number in the video (1 FPS = every fps frames)
                video_frame_number = int(frame_idx * fps)
                
                # Skip if beyond video length
                if video_frame_number >= total_frames:
                    break
                
                # Seek to the frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_number)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Save full frame
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    frames_extracted += 1
                    
                    # Create thumbnail
                    thumbnail_path = thumbnails_dir / frame_filename
                    thumbnail = cv2.resize(frame, (320, 240))
                    cv2.imwrite(str(thumbnail_path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
                else:
                    logger.warning(f"Failed to read frame {frame_idx} (video frame {video_frame_number})")
            
            cap.release()
            
            logger.info(f"Extracted {frames_extracted} frames from segment {segment_idx + 1} using OpenCV")
            
            return {
                'success': True,
                'frames_extracted': frames_extracted
            }
            
        except Exception as e:
            logger.error(f"Error extracting frames from segment {segment_idx + 1}: {e}")
            return {
                'success': False,
                'frames_extracted': 0,
                'error': str(e)
            }

    def _extract_all_frames_opencv(self, video_path, frames_dir, thumbnails_dir, progress_file, duration):
        """Extract all frames at 1 FPS using OpenCV - much faster than individual FFmpeg calls."""
        try:
            frames_extracted = 0
            target_frames = int(duration)  # 1 FPS = 1 frame per second
            
            logger.info(f"Starting OpenCV frame extraction for {target_frames} frames...")
            
            # Open video with OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return {'success': False, 'frames_extracted': 0, 'error': 'Could not open video'}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            
            logger.info(f"Video properties: {fps:.2f} FPS, {total_frames} total frames, {video_duration:.1f}s duration")
            
            # Calculate frame interval (every N frames to get 1 FPS)
            frame_interval = int(fps)  # Extract every fps-th frame to get 1 FPS
            
            frame_number = 0
            extracted_count = 0
            
            while frame_number < total_frames and extracted_count < target_frames:
                # Check for pause signal every 100 frames
                if extracted_count % 100 == 0:
                    current_progress = self._read_json_safe(progress_file)
                    if current_progress and current_progress.get('status') == 'paused':
                        logger.info(f"Frame extraction paused at frame {extracted_count}")
                        break
                    
                    # Update progress
                    progress_percent = 15 + (extracted_count / target_frames) * 55  # 15-70% for extraction
                    # Get current progress data to preserve segment information
                    current_progress = self._read_json_safe(progress_file)
                    total_segments = current_progress.get('total_segments', 0) if current_progress else 0
                    completed_segments = current_progress.get('completed_segments', 0) if current_progress else 0
                    
                    self._write_progress_safe(progress_file, {
                        'status': 'processing',
                        'stage': 'extracting_frames',
                        'percent_processing': round(progress_percent, 2),
                        'frames_extracted': extracted_count,
                        'total_frames': target_frames,
                        'total_segments': total_segments,
                        'completed_segments': completed_segments,
                        'segments_created': completed_segments
                    })
                
                # Seek to the frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Calculate the timestamp for this frame
                    timestamp = frame_number / fps
                    frame_filename = f"frame_{int(timestamp):08d}.jpg"
                    frame_path = frames_dir / frame_filename
                    
                    # Skip if frame already exists
                    if not frame_path.exists():
                        # Save full frame
                        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        # Create thumbnail
                        thumbnail_path = thumbnails_dir / frame_filename
                        thumbnail = cv2.resize(frame, (320, 240))
                        cv2.imwrite(str(thumbnail_path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    extracted_count += 1
                    frames_extracted += 1
                else:
                    logger.warning(f"Failed to read frame at position {frame_number}")
                
                # Move to next frame (1 FPS interval)
                frame_number += frame_interval
            
            cap.release()
            
            logger.info(f"OpenCV extraction completed: {frames_extracted} frames extracted")
            
            return {
                'success': True,
                'frames_extracted': frames_extracted
            }
            
        except Exception as e:
            logger.error(f"Error in OpenCV frame extraction: {e}")
            return {
                'success': False,
                'frames_extracted': 0,
                'error': str(e)
            }

    def _create_auto_labels(self, frames_dir, crops_dir, processed_dir, progress_file):
        """Create automatic labels for all extracted frames using YOLO detection."""
        try:
            crops_created = 0
            labels_created = 0
            
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            total_frames = len(frame_files)
            
            logger.info(f"Creating auto-labels for {total_frames} frames...")
            
            for i, frame_path in enumerate(frame_files):
                # Update progress every 50 frames
                if i % 50 == 0:
                    progress = 70 + (i / total_frames) * 30  # 70-100% for labeling
                    # Get current progress data to preserve segment information
                    current_progress = self._read_json_safe(progress_file)
                    total_segments = current_progress.get('total_segments', 0) if current_progress else 0
                    completed_segments = current_progress.get('completed_segments', 0) if current_progress else 0
                    segments_created = current_progress.get('segments_created', 0) if current_progress else 0
                    frames_extracted = current_progress.get('frames_extracted', 0) if current_progress else 0
                    
                    self._write_progress_safe(progress_file, {
                        'status': 'processing',
                        'stage': 'creating_auto_labels',
                        'percent_processing': round(progress, 2),
                        'current_frame': i,
                        'total_frames': total_frames,
                        'frames_extracted': frames_extracted,
                        'total_segments': total_segments,
                        'completed_segments': completed_segments,  # Preserve the existing completed_segments
                        'segments_created': segments_created
                    })
                
                # Check for pause signal
                current_progress = self._read_json_safe(progress_file)
                if current_progress and current_progress.get('status') == 'paused':
                    logger.info(f"Auto-labeling paused at frame {i}")
                    break
                
                try:
                    frame = cv2.imread(str(frame_path))
                    if frame is not None:
                        # Get referee detection
                        bbox = self._get_referee_box(frame)
                        if bbox:
                            # Create crop
                            crop = self._auto_crop_frame(frame, bbox)
                            if crop is not None:
                                crop_filename = f"crop_{frame_path.stem.split('_')[1]}.jpg"
                                crop_path = crops_dir / crop_filename
                                cv2.imwrite(str(crop_path), crop)
                                crops_created += 1
                                
                                # Create YOLO label file
                                label_filename = f"{frame_path.stem}.txt"
                                label_path = frames_dir / label_filename
                                yolo_bbox = self._convert_to_yolo_format(bbox, frame.shape)
                                with open(label_path, 'w') as f:
                                    f.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                                labels_created += 1
                                
                                # Save processed frame with bbox
                                processed_frame = frame.copy()
                                x1, y1, x2, y2 = map(int, bbox)
                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                processed_path = processed_dir / frame_path.name
                                cv2.imwrite(str(processed_path), processed_frame)
                        
                except Exception as frame_error:
                    logger.warning(f"Auto-labeling failed for {frame_path.name}: {frame_error}")
            
            logger.info(f"Auto-labeling completed: {crops_created} crops, {labels_created} labels created")
            
            return {
                'success': True,
                'crops_created': crops_created,
                'labels_created': labels_created
            }
            
        except Exception as e:
            logger.error(f"Error in auto-labeling: {e}")
            return {
                'success': False,
                'crops_created': 0,
                'labels_created': 0,
                'error': str(e)
            }

    def _convert_to_yolo_format(self, bbox, frame_shape):
        """Convert bounding box to YOLO format (normalized center_x, center_y, width, height)."""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        
        # Calculate center and dimensions
        center_x = (x1 + x2) / 2.0 / w
        center_y = (y1 + y2) / 2.0 / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        
        return [center_x, center_y, width, height]

    def _process_single_segment(self, video_path, segment_idx, start_time, duration, 
                               frames_dir, crops_dir, processed_dir, thumbnails_dir, 
                               auto_crop, progress_file):
        """Process a single video segment - extract frames and create crops."""
        try:
            frames_extracted = 0
            crops_created = 0
            
            # Calculate frame indices for this segment (1 FPS)
            start_frame_idx = int(start_time)  # Frame index = seconds (1 FPS)
            end_frame_idx = int(start_time + duration)
            
            # Extract frames for this segment using FFmpeg
            for frame_idx in range(start_frame_idx, end_frame_idx):
                frame_filename = f"frame_{frame_idx:08d}.jpg"
                frame_path = frames_dir / frame_filename
                
                # Skip if frame already exists
                if frame_path.exists():
                    frames_extracted += 1
                    continue
                
                # Extract single frame at exact timestamp
                extract_cmd = [
                    'ffmpeg', '-y', '-loglevel', 'error',
                    '-ss', str(frame_idx),  # Seek to exact second
                    '-i', str(video_path),
                    '-vframes', '1',  # Extract exactly 1 frame
                    '-q:v', '2',  # High quality
                    str(frame_path)
                ]
                
                try:
                    subprocess.run(extract_cmd, check=True, capture_output=True)
                    frames_extracted += 1
                    
                    # Create thumbnail
                    thumbnail_path = thumbnails_dir / frame_filename
                    thumbnail_cmd = [
                        'ffmpeg', '-y', '-loglevel', 'error',
                        '-i', str(frame_path),
                        '-vf', 'scale=320:240',
                        '-q:v', '3',
                        str(thumbnail_path)
                    ]
                    subprocess.run(thumbnail_cmd, check=False, capture_output=True)
                    
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to extract frame {frame_idx}: {e}")
                    continue
            
            # Process auto-crop for extracted frames in this segment if enabled
            if auto_crop and self.model and frames_extracted > 0:
                for frame_idx in range(start_frame_idx, end_frame_idx):
                    frame_filename = f"frame_{frame_idx:08d}.jpg"
                    frame_path = frames_dir / frame_filename
                    crop_filename = f"crop_{frame_idx:08d}.jpg"
                    crop_path = crops_dir / crop_filename
                    
                    # Skip if frame doesn't exist or crop already exists
                    if not frame_path.exists() or crop_path.exists():
                        if crop_path.exists():
                            crops_created += 1
                        continue
                    
                    try:
                        frame = cv2.imread(str(frame_path))
                        if frame is not None:
                            bbox = self._get_referee_box(frame)
                            if bbox:
                                crop = self._auto_crop_frame(frame, bbox)
                                if crop is not None:
                                    cv2.imwrite(str(crop_path), crop)
                                    crops_created += 1
                                    
                                    # Save processed frame with bbox
                                    processed_frame = frame.copy()
                                    x1, y1, x2, y2 = map(int, bbox)
                                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    processed_path = processed_dir / frame_filename
                                    cv2.imwrite(str(processed_path), processed_frame)
                    except Exception as crop_error:
                        logger.warning(f"Auto-crop failed for frame {frame_idx}: {crop_error}")
            
            return {
                'success': True,
                'frames_extracted': frames_extracted,
                'crops_created': crops_created
            }
            
        except Exception as e:
            logger.error(f"Error processing segment {segment_idx + 1}: {e}")
            return {
                'success': False,
                'frames_extracted': 0,
                'crops_created': 0,
                'error': str(e)
            }

    def _create_segment_video(self, video_path, start_time, duration, output_path):
        """Create a segment video file to mark completion."""
        try:
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-ss', str(start_time),
                '-i', str(video_path),
                '-t', str(duration),
                '-c', 'copy',  # Copy streams without re-encoding (fast)
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.debug(f"Created segment video: {output_path}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to create segment video {output_path}: {e}")

    def _write_progress_safe(self, progress_file, progress_data):
        """Thread-safe progress file writing with backup and recovery."""
        with self.progress_lock:
            backup_file = progress_file.parent / f"{progress_file.stem}_backup.json"
            
            try:
                # Read existing progress to preserve values and prevent going backwards
                existing_progress = {}
                if progress_file.exists():
                    try:
                        with open(progress_file, 'r') as f:
                            existing_progress = json.load(f)
                    except:
                        # Try to read from backup
                        if backup_file.exists():
                            try:
                                with open(backup_file, 'r') as f:
                                    existing_progress = json.load(f)
                                logger.info(f"Recovered progress from backup: {existing_progress.get('status', 'unknown')}")
                            except:
                                pass
                
                # Always ensure we have basic structure with valid defaults
                base_progress = {
                    'status': 'downloading',  # Never use 'unknown' as default
                    'stage': 'initializing', 
                    'percent_download': 0.0,
                    'percent_processing': 0.0,
                    'current_frame': 0,
                    'total_frames': 0,
                    'frames_extracted': 0,
                    'crops_created': 0,
                    'segments_created': 0,
                    'auto_crop': False,
                    'updated_at': datetime.now().isoformat(),
                    'error': None
                }
                
                # Start with existing progress
                base_progress.update(existing_progress)
                
                # Update with new data
                base_progress.update(progress_data)
                base_progress['updated_at'] = datetime.now().isoformat()
                
                # Prevent progress from going backwards (except for status changes)
                if existing_progress:
                    # Don't let download progress go backwards
                    if 'percent_download' in existing_progress:
                        base_progress['percent_download'] = max(
                            existing_progress.get('percent_download', 0),
                            base_progress.get('percent_download', 0)
                        )
                    
                    # Don't let processing progress go backwards
                    if 'percent_processing' in existing_progress:
                        base_progress['percent_processing'] = max(
                            existing_progress.get('percent_processing', 0),
                            base_progress.get('percent_processing', 0)
                        )
                
                # First, create/update backup with the current state
                try:
                    with open(backup_file, 'w') as f:
                        json.dump(base_progress, f, indent=2)
                except:
                    pass  # Don't fail if backup fails
                
                # Write directly to main file
                with open(progress_file, 'w') as f:
                    json.dump(base_progress, f, indent=2)
                
            except Exception as e:
                logger.error(f"Could not write progress file: {e}")
                # Try simple write as fallback with guaranteed valid status
                try:
                    simple_progress = {
                        'status': progress_data.get('status', 'downloading'),  # Never default to 'unknown'
                        'stage': progress_data.get('stage', 'initializing'),
                        'percent_download': max(0.0, float(progress_data.get('percent_download', 0))),
                        'percent_processing': max(0.0, float(progress_data.get('percent_processing', 0))),
                        'current_frame': int(progress_data.get('current_frame', 0)),
                        'total_frames': int(progress_data.get('total_frames', 0)),
                        'frames_extracted': int(progress_data.get('frames_extracted', 0)),
                        'crops_created': int(progress_data.get('crops_created', 0)),
                        'segments_created': int(progress_data.get('segments_created', 0)),
                        'auto_crop': bool(progress_data.get('auto_crop', True)),
                        'updated_at': datetime.now().isoformat(),
                        'error': progress_data.get('error', None)
                    }
                    
                    # Try backup first
                    try:
                        with open(backup_file, 'w') as f:
                            json.dump(simple_progress, f, indent=2)
                    except:
                        pass
                    
                    # Then main file
                    with open(progress_file, 'w') as f:
                        json.dump(simple_progress, f, indent=2)
                    logger.info(f"Fallback progress write successful: {simple_progress['status']} - {simple_progress['stage']}")
                except Exception as fallback_error:
                    logger.error(f"Fallback progress write failed: {fallback_error}")
                    # Last resort: create minimal valid JSON
                    try:
                        minimal_progress = {
                            'status': 'downloading',
                            'stage': 'initializing',
                            'percent_download': 1.0,
                            'percent_processing': 0.0,
                            'updated_at': datetime.now().isoformat()
                        }
                        
                        # Backup first
                        try:
                            with open(backup_file, 'w') as f:
                                json.dump(minimal_progress, f)
                        except:
                            pass
                        
                        # Main file
                        with open(progress_file, 'w') as f:
                            json.dump(minimal_progress, f)
                    except:
                        pass  # Give up at this point

    def _create_segment_writer(self, video_dir, segment_num, fps):
        """Create video writer for segment."""
        try:
            segment_path = video_dir / "segments" / f"segment_{segment_num:03d}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            return cv2.VideoWriter(str(segment_path), fourcc, fps, (self.MODEL_SIZE, self.MODEL_SIZE))
        except Exception as e:
            logger.warning(f"Could not create segment writer: {e}")
            return None

    def _get_referee_box(self, frame):
        """Get referee bounding box from frame using YOLO model."""
        try:
            if not self.model:
                return None
                
            results = self.model(frame, device=self.DEVICE, verbose=False, conf=self.CONFIDENCE_THRESHOLD)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if box.cls == self.class_id and box.conf >= self.CONFIDENCE_THRESHOLD:
                            # Get pixel coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            return [x1, y1, x2, y2]
            return None
        except Exception as e:
            logger.warning(f"Referee detection failed: {e}")
            return None

    def _auto_crop_frame(self, frame, bbox):
        """Crop frame to referee bounding box with padding."""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding (20% of bbox size)
            width = x2 - x1
            height = y2 - y1
            padding_x = int(width * 0.2)
            padding_y = int(height * 0.2)
            
            # Ensure we don't go outside frame boundaries
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(w, x2 + padding_x)
            y2 = min(h, y2 + padding_y)
            
            crop = frame[y1:y2, x1:x2]
            return crop if crop.size > 0 else None
            
        except Exception as e:
            logger.warning(f"Crop extraction failed: {e}")
            return None

    def _read_json_safe(self, file_path):
        """Safely read JSON file with backup recovery."""
        backup_file = file_path.parent / f"{file_path.stem}_backup.json"
        
        # First try main file
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if data:  # Ensure it's not empty
                        return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read {file_path}: {e}")
        
        # If main file failed, try backup
        try:
            if backup_file.exists():
                with open(backup_file, 'r') as f:
                    data = json.load(f)
                    if data:
                        logger.info(f"Using backup progress file for {file_path.name}")
                        # Restore main file from backup
                        try:
                            with open(file_path, 'w') as f:
                                json.dump(data, f, indent=2)
                            logger.info(f"Restored main progress file from backup")
                        except:
                            pass  # Don't fail if we can't restore
                        return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read backup {backup_file}: {e}")
        
        return None

    def get_processing_status(self, folder_name):
        """Get processing status for a video with intelligent fallback detection."""
        try:
            video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
            progress_file = video_dir / "progress.json"
            
            progress = self._read_json_safe(progress_file)
            if progress:
                # Ensure we never return 'unknown' or 'pending' status
                if progress.get('status') in ['unknown', 'pending']:
                    progress['status'] = 'downloading'
                    progress['stage'] = progress.get('stage', 'initializing')
                
                # Add pending frames count
                progress['pending_frames'] = self._count_pending_frames(folder_name)
                progress['pending_signal_detections'] = self._count_pending_signal_detections(folder_name)
                return progress
            
            # Progress file is missing or corrupted - determine status from folder contents
            if video_dir.exists():
                frames_dir = video_dir / "frames"
                original_dir = video_dir / "original"
                crops_dir = video_dir / "crops"
                
                # Count actual content
                frame_files = list(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
                video_files = list(original_dir.glob("*.mp4")) if original_dir.exists() else []
                crop_files = list(crops_dir.glob("*.jpg")) if crops_dir.exists() else []
                
                if len(frame_files) > 0:
                    # Has frames extracted - determine completion level
                    pending_frames = self._count_pending_frames(folder_name)
                    if len(crop_files) > 0:
                        # Has crops, fully completed
                        return {
                            'status': 'completed',
                            'stage': 'completed',
                            'percent_download': 100.0,
                            'percent_processing': 100.0,
                            'current_frame': len(frame_files),
                            'total_frames': len(frame_files),
                            'frames_extracted': len(frame_files),
                            'crops_created': len(crop_files),
                            'pending_frames': pending_frames,
                            'pending_signal_detections': self._count_pending_signal_detections(folder_name),
                            'auto_crop': True,
                            'error': None,
                            'updated_at': datetime.now().isoformat()
                        }
                    else:
                        # Has frames but no crops, processing completed without auto-crop
                        return {
                            'status': 'completed',
                            'stage': 'frames_extracted',
                            'percent_download': 100.0,
                            'percent_processing': 100.0,
                            'current_frame': len(frame_files),
                            'total_frames': len(frame_files),
                            'frames_extracted': len(frame_files),
                            'crops_created': 0,
                            'pending_frames': pending_frames,
                            'pending_signal_detections': self._count_pending_signal_detections(folder_name),
                            'auto_crop': False,
                            'error': None,
                            'updated_at': datetime.now().isoformat()
                        }
                elif len(video_files) > 0:
                    # Has video file, currently processing frames
                    return {
                        'status': 'processing',
                        'stage': 'extracting_frames',
                        'percent_download': 100.0,
                        'percent_processing': 25.0,  # Processing started
                        'current_frame': 0,
                        'total_frames': 0,
                        'frames_extracted': 0,
                        'crops_created': 0,
                        'pending_frames': 0,
                        'auto_crop': True,
                        'error': None,
                        'updated_at': datetime.now().isoformat()
                    }
                else:
                    # Folder exists but no content, likely downloading
                    return {
                        'status': 'downloading',
                        'stage': 'downloading_video',
                        'percent_download': 15.0,  # Download in progress
                        'percent_processing': 0.0,
                        'current_frame': 0,
                        'total_frames': 0,
                        'frames_extracted': 0,
                        'crops_created': 0,
                        'pending_frames': 0,
                        'auto_crop': True,
                        'error': None,
                        'updated_at': datetime.now().isoformat()
                    }
            else:
                return {
                    'status': 'error',
                    'stage': 'folder_not_found',
                    'percent_download': 0,
                    'percent_processing': 0,
                    'current_frame': 0,
                    'total_frames': 0,
                    'frames_extracted': 0,
                    'crops_created': 0,
                    'pending_frames': 0,
                    'auto_crop': False,
                    'error': 'Video folder not found',
                    'updated_at': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error getting status for {folder_name}: {e}")
            return {
                'status': 'error',
                'stage': 'status_check_failed',
                'pending_frames': 0,
                'error': str(e)
            }

    def get_all_videos(self):
        """Get all processed videos with their status."""
        try:
            videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
            if not videos_dir.exists():
                return []
            
            videos = []
            for folder in videos_dir.iterdir():
                if folder.is_dir():
                    status = self.get_processing_status(folder.name)
                    videos.append({
                        'folder_name': folder.name,
                        'created_at': folder.stat().st_ctime,
                        **status
                    })
            
            # Sort by creation time (newest first)
            videos.sort(key=lambda x: x['created_at'], reverse=True)
            return videos
            
        except Exception as e:
            logger.error(f"Error getting all videos: {e}")
            return []

    def delete_video(self, folder_name):
        """Delete a processed video and all its files."""
        try:
            video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
            if video_dir.exists():
                import shutil
                shutil.rmtree(video_dir)
                logger.info(f"Deleted video folder: {folder_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting video {folder_name}: {e}")
            return False

    def pause_video_processing(self, folder_name):
        """Pause video processing."""
        try:
            video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
            progress_file = video_dir / "progress.json"
            
            current_progress = self._read_json_safe(progress_file)
            if current_progress:
                self._write_progress_safe(progress_file, {
                    **current_progress,
                    'status': 'paused',
                    'stage': 'paused_by_user',
                    'paused_at': datetime.now().isoformat()
                })
                logger.info(f"Paused video processing for {folder_name}")
                return {'status': 'success', 'message': 'Video processing paused'}
            else:
                return {'status': 'error', 'message': 'Video not found or not processing'}
                
        except Exception as e:
            logger.error(f"Error pausing video {folder_name}: {e}")
            return {'status': 'error', 'message': str(e)}

    def resume_video_processing(self, folder_name):
        """Resume paused video processing using segment-based approach."""
        try:
            video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
            progress_file = video_dir / "progress.json"
            
            current_progress = self._read_json_safe(progress_file)
            if not current_progress:
                return {'status': 'error', 'message': 'Video not found'}
            
            if current_progress.get('status') != 'paused':
                return {'status': 'error', 'message': 'Video is not paused'}
            
            # Find the original video file
            original_dir = video_dir / "original"
            video_files = list(original_dir.glob("*.mp4"))
            if not video_files:
                return {'status': 'error', 'message': 'Original video file not found'}
            
            video_file = video_files[0]
            
            # Get video info
            video_info = self._get_video_info(video_file)
            if not video_info:
                return {'status': 'error', 'message': 'Could not get video information'}
            
            # Resume processing in background
            def resume_processing():
                try:
                    # Update status to processing
                    self._write_progress_safe(progress_file, {
                        **current_progress,
                        'status': 'processing',
                        'stage': 'resuming_processing',
                        'resumed_at': datetime.now().isoformat()
                    })
                    
                    # Determine what stage to resume from based on current progress
                    frames_dir = video_dir / "frames"
                    crops_dir = video_dir / "crops"
                    processed_dir = video_dir / "processed"
                    
                    # Check current state
                    frames_extracted = len(list(frames_dir.glob("frame_*.jpg"))) if frames_dir.exists() else 0
                    crops_created = len(list(crops_dir.glob("crop_*.jpg"))) if crops_dir.exists() else 0
                    
                    auto_crop = current_progress.get('auto_crop', True)
                    
                    # If no frames extracted yet, do full processing
                    if frames_extracted == 0:
                        processing_result = self._process_video_by_segments(video_file, video_dir, auto_crop, progress_file, video_info)
                    else:
                        # Frames already extracted, check if we need to do auto-labeling
                        if auto_crop and crops_created == 0:
                            # Only do auto-labeling
                            logger.info(f"Resuming auto-labeling for {folder_name} with {frames_extracted} existing frames")
                            self._write_progress_safe(progress_file, {
                                **current_progress,
                                'status': 'processing',
                                'stage': 'creating_auto_labels',
                                'percent_processing': 70.0,
                                'frames_extracted': frames_extracted,
                                'crops_created': crops_created
                            })
                            
                            # Create auto labels from existing frames
                            result = self._create_auto_labels(frames_dir, crops_dir, processed_dir, progress_file)
                            crops_created = result.get('crops_created', crops_created)
                            
                            processing_result = {
                                'frames_extracted': frames_extracted,
                                'crops_created': crops_created,
                                'completed_segments': current_progress.get('completed_segments', 0)
                            }
                        else:
                            # Processing already complete or no auto-crop needed
                            processing_result = {
                                'frames_extracted': frames_extracted,
                                'crops_created': crops_created,
                                'completed_segments': current_progress.get('completed_segments', 0)
                            }
                    
                    # Mark as completed
                    final_status = 'completed'
                    if auto_crop and processing_result['crops_created'] > 0:
                        final_status = 'auto_labeling_ready'
                    
                    self._write_progress_safe(progress_file, {
                        **current_progress,
                        'status': final_status,
                        'stage': 'completed',
                        'percent_processing': 100.0,
                        'completed_segments': processing_result.get('completed_segments', 0),
                        'frames_extracted': processing_result['frames_extracted'],
                        'crops_created': processing_result['crops_created'],
                        'completed_at': datetime.now().isoformat()
                    })
                    
                    logger.info(f"Successfully resumed and completed processing for {folder_name}")
                    
                except Exception as e:
                    logger.error(f"Error during resume processing: {e}")
                    self._write_progress_safe(progress_file, {
                        **current_progress,
                        'status': 'error',
                        'stage': 'resume_failed',
                        'error': str(e)
                    })
            
            # Start processing in background thread
            import threading
            thread = threading.Thread(target=resume_processing)
            thread.start()
            
            logger.info(f"Resumed video processing for {folder_name}")
            return {'status': 'success', 'message': 'Video processing resumed'}
            
        except Exception as e:
            logger.error(f"Error resuming video {folder_name}: {e}")
            return {'status': 'error', 'message': str(e)}

    def confirm_referee_detection(self, folder_name, frame_name, is_correct):
        """Confirm or reject a referee detection from auto-labeled frames."""
        try:
            video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
            processed_dir = video_dir / 'processed'
            crops_dir = video_dir / 'crops'
            frames_dir = video_dir / 'frames'
            autolabeled_signal_dir = video_dir / 'autolabeled_signal_detection'
            
            # Find the processed frame
            processed_frame_path = processed_dir / frame_name
            if not processed_frame_path.exists():
                return {'success': False, 'error': 'Processed frame not found'}
            
            # Get corresponding crop and label files
            frame_base = frame_name.replace('.jpg', '')
            crop_name = f"crop_{frame_base.replace('frame_', '')}.jpg"
            label_name = f"{frame_base}.txt"
            
            crop_path = crops_dir / crop_name
            label_path = frames_dir / label_name
            
            if is_correct:
                # Process confirmed referee detection for signal detection
                autolabeled_signal_dir.mkdir(exist_ok=True)
                
                if crop_path.exists():
                    # Load the signal detection model and process the crop
                    signal_result = self._process_crop_for_signal_detection(crop_path)
                    
                    # Generate unique filename for autolabeled signal detection
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    signal_crop_filename = f"autolabeled_signal_{folder_name}_{frame_base}_{timestamp}.jpg"
                    signal_crop_path = autolabeled_signal_dir / signal_crop_filename
                    
                    # Copy crop to autolabeled signal detection folder
                    import shutil
                    shutil.copy2(str(crop_path), str(signal_crop_path))
                    
                    # Save signal detection metadata
                    metadata = {
                        'original_frame': frame_name,
                        'crop_filename': signal_crop_filename,
                        'signal_detection': signal_result,
                        'created_at': datetime.now().isoformat(),
                        'status': 'pending_confirmation'
                    }
                    
                    metadata_file = autolabeled_signal_dir / f"{signal_crop_filename.replace('.jpg', '.json')}"
                    with open(metadata_file, 'w') as f:
                        import json
                        json.dump(metadata, f, indent=2)
                    
                    # Remove original files
                    crop_path.unlink()
                    logger.info(f"Processed confirmed referee detection for signal: {signal_crop_filename}")
                
                # Remove from processed
                if processed_frame_path.exists():
                    processed_frame_path.unlink()
                
                logger.info(f"Confirmed referee detection for {frame_name} and processed for signal detection")
                return {
                    'success': True,
                    'message': 'Referee detection confirmed and processed for signal detection',
                    'action': 'confirmed'
                }
            else:
                # For incorrect detections, move original frame to manual labeling queue
                
                # Get the original frame (without bounding box)
                original_frame_path = frames_dir / frame_name
                if original_frame_path.exists():
                    # Copy to static uploads for manual labeling
                    uploads_dir = Path(DirectoryConfig.UPLOAD_FOLDER)
                    uploads_dir.mkdir(exist_ok=True)
                    
                    # Generate unique filename for manual queue
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    manual_filename = f"rejected_autolabel_{folder_name}_{frame_base}_{timestamp}.jpg"
                    manual_path = uploads_dir / manual_filename
                    
                    import shutil
                    shutil.copy2(str(original_frame_path), str(manual_path))
                    logger.info(f"Moved rejected frame to manual labeling queue: {manual_filename}")
                
                # Remove all related files
                if processed_frame_path.exists():
                    processed_frame_path.unlink()
                if crop_path.exists():
                    crop_path.unlink()
                if label_path.exists():
                    label_path.unlink()
                
                logger.info(f"Rejected referee detection for {frame_name} and moved to manual queue")
                return {
                    'success': True,
                    'message': 'Incorrect detection rejected and moved to manual labeling queue',
                    'action': 'rejected'
                }
                
        except Exception as e:
            logger.error(f"Error confirming referee detection for {folder_name}/{frame_name}: {e}")
            return {'success': False, 'error': str(e)}

    def confirm_signal_detection(self, folder_name, crop_filename, signal_class, is_correct):
        """Confirm signal detection for a confirmed referee crop."""
        try:
            video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
            confirmed_crops_dir = video_dir / 'confirmed_crops'
            signal_confirmed_dir = video_dir / 'signal_confirmed'
            
            # Find the confirmed crop
            crop_path = confirmed_crops_dir / crop_filename
            if not crop_path.exists():
                return {'success': False, 'error': 'Confirmed crop not found'}
            
            if is_correct and signal_class and signal_class != 'none':
                # Save to signal training data
                signal_training_dir = Path(DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER)
                signal_training_dir.mkdir(exist_ok=True)
                
                # Generate unique filename for signal training
                timestamp = int(datetime.now().timestamp() * 1000000)
                signal_filename = f"signal_{signal_class}_{timestamp}.jpg"
                signal_path = signal_training_dir / signal_filename
                
                # Copy the crop to signal training data
                import shutil
                shutil.copy2(crop_path, signal_path)
                
                # Create YOLO annotation file for signal
                signal_label_filename = signal_filename.replace('.jpg', '.txt')
                signal_label_path = signal_training_dir / signal_label_filename
                
                # Get signal class ID
                signal_class_id = self._get_signal_class_id(signal_class)
                
                # For now, use full image bbox (0.5, 0.5, 1.0, 1.0) - center at 50%, 50% with full width/height
                with open(signal_label_path, 'w') as f:
                    f.write(f"{signal_class_id} 0.5 0.5 1.0 1.0\n")
                
                logger.info(f"Saved signal training data: {signal_filename} (class: {signal_class})")
                
                # Move to signal_confirmed directory
                signal_confirmed_dir.mkdir(exist_ok=True)
                confirmed_signal_path = signal_confirmed_dir / crop_filename
                shutil.move(str(crop_path), str(confirmed_signal_path))
                
                return {
                    'success': True,
                    'message': f'Signal "{signal_class}" confirmed and saved to training data',
                    'action': 'confirmed'
                }
            elif signal_class == 'none' or not is_correct:
                # Save as negative signal sample
                signal_training_dir = Path(DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER)
                signal_training_dir.mkdir(exist_ok=True)
                
                # Generate unique filename for negative signal training
                timestamp = int(datetime.now().timestamp() * 1000000)
                signal_filename = f"signal_none_{timestamp}.jpg"
                signal_path = signal_training_dir / signal_filename
                
                # Copy the crop to signal training data
                import shutil
                shutil.copy2(crop_path, signal_path)
                
                # Create YOLO annotation file for negative signal (no signal detected)
                signal_label_filename = signal_filename.replace('.jpg', '.txt')
                signal_label_path = signal_training_dir / signal_label_filename
                
                # For negative samples, create empty annotation file or use 'none' class
                with open(signal_label_path, 'w') as f:
                    # Empty file for negative samples
                    pass
                
                logger.info(f"Saved negative signal training data: {signal_filename}")
                
                # Move to signal_confirmed directory
                signal_confirmed_dir.mkdir(exist_ok=True)
                confirmed_signal_path = signal_confirmed_dir / crop_filename
                shutil.move(str(crop_path), str(confirmed_signal_path))
                
                return {
                    'success': True,
                    'message': 'No signal confirmed and saved as negative sample',
                    'action': 'negative'
                }
            else:
                return {'success': False, 'error': 'Invalid signal class or confirmation status'}
                
        except Exception as e:
            logger.error(f"Error confirming signal detection for {folder_name}/{crop_filename}: {e}")
            return {'success': False, 'error': str(e)}

    def _get_signal_class_id(self, signal_class):
        """Get the class ID for a signal class name."""
        # This should match your signal model's class mapping
        signal_classes = {
            'armLeft': 0,
            'armRight': 1, 
            'hits': 2,
            'leftServe': 3,
            'net': 4,
            'outside': 5,
            'rightServe': 6,
            'touched': 7
        }
        return signal_classes.get(signal_class, 0)

    def _process_crop_for_signal_detection(self, crop_path):
        """Process a referee crop through the signal detection model."""
        try:
            from models.signal_classifier import SignalClassifier
            import cv2
            
            # Initialize signal classifier
            signal_classifier = SignalClassifier()
            
            # Load the crop image
            crop_image = cv2.imread(str(crop_path))
            if crop_image is None:
                return {'error': 'Could not load crop image'}
            
            # Run signal detection
            result = signal_classifier.detect_signal(crop_image)
            
            return {
                'predicted_class': result.get('predicted_class', 'none'),
                'confidence': result.get('confidence', 0.0),
                'bbox': result.get('bbox', []),
                'all_predictions': result.get('all_predictions', [])
            }
            
        except Exception as e:
            logger.error(f"Error processing crop for signal detection: {e}")
            return {
                'predicted_class': 'none',
                'confidence': 0.0,
                'bbox': [],
                'error': str(e)
            }

    def _count_pending_frames(self, folder_name):
        """Count frames that are processed but not yet confirmed/denied."""
        try:
            video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
            processed_dir = video_dir / 'processed'
            
            if not processed_dir.exists():
                return 0
            
            # Count all processed frames (these are pending confirmation)
            processed_files = list(processed_dir.glob('*.jpg'))
            return len(processed_files)
            
        except Exception as e:
            logger.error(f"Error counting pending frames for {folder_name}: {e}")
            return 0

    def _count_pending_signal_detections(self, folder_name):
        """Count signal detections that are pending confirmation."""
        try:
            video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
            signal_dir = video_dir / 'autolabeled_signal_detection'
            
            if not signal_dir.exists():
                return 0
            
            # Count pending signal detection files
            pending_files = []
            for json_file in signal_dir.glob('*.json'):
                try:
                    with open(json_file, 'r') as f:
                        import json
                        metadata = json.load(f)
                        if metadata.get('status') == 'pending_confirmation':
                            pending_files.append(json_file)
                except:
                    continue
            
            return len(pending_files)
            
        except Exception as e:
            logger.error(f"Error counting pending signal detections for {folder_name}: {e}")
            return 0

    def get_autolabeled_signal_detections(self, folder_name=None, page=1, per_page=10):
        """Get autolabeled signal detections for confirmation."""
        try:
            all_detections = []
            
            if folder_name:
                # Get detections for specific video
                video_dirs = [Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name]
            else:
                # Get detections for all videos
                videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
                video_dirs = [d for d in videos_dir.iterdir() if d.is_dir()]
            
            for video_dir in video_dirs:
                signal_dir = video_dir / 'autolabeled_signal_detection'
                if not signal_dir.exists():
                    continue
                
                for json_file in signal_dir.glob('*.json'):
                    try:
                        with open(json_file, 'r') as f:
                            import json
                            metadata = json.load(f)
                            
                        if metadata.get('status') == 'pending_confirmation':
                            # Get corresponding image file
                            image_file = signal_dir / metadata['crop_filename']
                            if image_file.exists():
                                detection = {
                                    'video_id': video_dir.name,
                                    'crop_filename': metadata['crop_filename'],
                                    'original_frame': metadata['original_frame'],
                                    'signal_detection': metadata['signal_detection'],
                                    'created_at': metadata['created_at'],
                                    'image_path': str(image_file)
                                }
                                all_detections.append(detection)
                    except Exception as e:
                        logger.error(f"Error reading signal detection metadata {json_file}: {e}")
                        continue
            
            # Sort by creation time (newest first)
            all_detections.sort(key=lambda x: x['created_at'], reverse=True)
            
            # Paginate results
            total = len(all_detections)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            detections = all_detections[start_idx:end_idx]
            
            return {
                'detections': detections,
                'total': total,
                'page': page,
                'per_page': per_page,
                'total_pages': (total + per_page - 1) // per_page
            }
            
        except Exception as e:
            logger.error(f"Error getting autolabeled signal detections: {e}")
            return {'detections': [], 'total': 0, 'page': page, 'per_page': per_page, 'total_pages': 0}

    def confirm_autolabeled_signal_detection(self, folder_name, crop_filename, signal_class, is_correct):
        """Confirm or modify an autolabeled signal detection."""
        try:
            video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
            signal_dir = video_dir / 'autolabeled_signal_detection'
            
            # Find the metadata file
            metadata_filename = crop_filename.replace('.jpg', '.json')
            metadata_file = signal_dir / metadata_filename
            image_file = signal_dir / crop_filename
            
            if not metadata_file.exists() or not image_file.exists():
                return {'success': False, 'error': 'Signal detection files not found'}
            
            # Read current metadata
            with open(metadata_file, 'r') as f:
                import json
                metadata = json.load(f)
            
            if is_correct and signal_class and signal_class != 'none':
                # Save to signal training data
                signal_training_dir = Path(DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER)
                signal_training_dir.mkdir(exist_ok=True)
                
                # Generate unique filename for signal training
                timestamp = int(datetime.now().timestamp() * 1000000)
                signal_filename = f"signal_{signal_class}_{timestamp}.jpg"
                signal_path = signal_training_dir / signal_filename
                
                # Copy the crop to signal training data
                import shutil
                shutil.copy2(str(image_file), str(signal_path))
                
                # Create YOLO annotation file for signal
                signal_label_filename = signal_filename.replace('.jpg', '.txt')
                signal_label_path = signal_training_dir / signal_label_filename
                
                # Get signal class ID
                signal_class_id = self._get_signal_class_id(signal_class)
                
                # Use detected bounding box if available, otherwise use full image
                bbox = metadata['signal_detection'].get('bbox', [])
                if bbox and len(bbox) == 4:
                    # Convert bbox to YOLO format (center_x, center_y, width, height)
                    x, y, w, h = bbox
                    center_x = x + w / 2
                    center_y = y + h / 2
                    # Normalize to image dimensions (assuming bbox is already normalized)
                    yolo_bbox = f"{signal_class_id} {center_x} {center_y} {w} {h}\n"
                else:
                    # Use full image bbox
                    yolo_bbox = f"{signal_class_id} 0.5 0.5 1.0 1.0\n"
                
                with open(signal_label_path, 'w') as f:
                    f.write(yolo_bbox)
                
                logger.info(f"Saved autolabeled signal training data: {signal_filename} (class: {signal_class})")
                
            elif signal_class == 'none' or not is_correct:
                # Save as negative signal sample
                signal_training_dir = Path(DirectoryConfig.SIGNAL_TRAINING_DATA_FOLDER)
                signal_training_dir.mkdir(exist_ok=True)
                
                # Generate unique filename for negative signal training
                timestamp = int(datetime.now().timestamp() * 1000000)
                signal_filename = f"signal_none_{timestamp}.jpg"
                signal_path = signal_training_dir / signal_filename
                
                # Copy the crop to signal training data
                import shutil
                shutil.copy2(str(image_file), str(signal_path))
                
                # Create empty annotation file for negative samples
                signal_label_filename = signal_filename.replace('.jpg', '.txt')
                signal_label_path = signal_training_dir / signal_label_filename
                
                with open(signal_label_path, 'w') as f:
                    # Empty file for negative samples
                    pass
                
                logger.info(f"Saved negative autolabeled signal training data: {signal_filename}")
            
            # Mark as confirmed and remove from pending
            metadata['status'] = 'confirmed'
            metadata['confirmed_at'] = datetime.now().isoformat()
            metadata['confirmed_signal_class'] = signal_class
            metadata['confirmed_as_correct'] = is_correct
            
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            return {
                'success': True,
                'message': f'Signal detection confirmed and saved to training data',
                'action': 'confirmed'
            }
            
        except Exception as e:
            logger.error(f"Error confirming autolabeled signal detection for {folder_name}/{crop_filename}: {e}")
            return {'success': False, 'error': str(e)} 