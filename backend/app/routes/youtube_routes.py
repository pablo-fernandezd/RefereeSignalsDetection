"""
YouTube Processing Routes

This module contains routes for YouTube video processing,
frame extraction, and automated labeling.
"""

from flask import Blueprint, request, jsonify, send_from_directory
from pathlib import Path
import sys
import threading
import json
from datetime import datetime

# Import with proper path handling
import sys
from pathlib import Path

# Add necessary paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from models.youtube_processor import YouTubeProcessor
    from config.settings import DirectoryConfig
except ImportError:
    # Try alternative import
    from app.models.youtube_processor import YouTubeProcessor
    from app.config.settings import DirectoryConfig

youtube_bp = Blueprint('youtube', __name__)

# Initialize YouTube processor
youtube_processor = YouTubeProcessor()

@youtube_bp.route('/process', methods=['POST'])
def process_youtube_video():
    """Process a YouTube video."""
    try:
        data = request.get_json()
        url = data.get('url')
        auto_label = data.get('auto_label', True)
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Validate YouTube URL
        if 'youtube.com' not in url and 'youtu.be' not in url:
            return jsonify({'error': 'Please provide a valid YouTube URL'}), 400
        
        processor = YouTubeProcessor()
        
        # Enhanced error handling for YouTube download issues
        try:
            result = processor.download_youtube_video(url, auto_label)
            
            if 'error' in result:
                error_msg = result['error']
                
                # Provide specific user-friendly error messages
                if 'HTTP Error 403' in error_msg or 'Forbidden' in error_msg:
                    return jsonify({
                        'error': 'YouTube access denied. This could be due to:\n'
                                '• Geographic restrictions\n'
                                '• Age-restricted content\n'
                                '• Private video\n'
                                '• YouTube anti-bot measures\n\n'
                                'Try a different video or wait a few minutes before retrying.'
                    }), 403
                
                elif 'HTTP Error 404' in error_msg or 'not found' in error_msg.lower():
                    return jsonify({
                        'error': 'Video not found. Please check the URL and make sure the video exists.'
                    }), 404
                
                elif 'nsig extraction failed' in error_msg or 'Precondition check failed' in error_msg:
                    return jsonify({
                        'error': 'YouTube download temporarily blocked. This is usually temporary.\n'
                                'Suggestions:\n'
                                '• Wait 10-15 minutes and try again\n'
                                '• Try a different video\n'
                                '• The system has been updated with latest fixes'
                    }), 429
                
                elif 'network' in error_msg.lower() or 'connection' in error_msg.lower():
                    return jsonify({
                        'error': 'Network connection issue. Please check your internet connection and try again.'
                    }), 503
                
                else:
                    return jsonify({
                        'error': f'Download failed: {error_msg}\n\n'
                                'The system uses the latest yt-dlp version with enhanced anti-detection measures.'
                    }), 500
            
            return jsonify(result)
            
        except Exception as download_error:
            logger.error(f"YouTube download error: {download_error}")
            return jsonify({
                'error': f'Unexpected error during download: {str(download_error)}\n\n'
                        'The system has been updated with the latest YouTube download fixes. '
                        'If this persists, try a different video or wait a few minutes.'
            }), 500
        
    except Exception as e:
        logger.error(f"Error in process_youtube_video: {e}")
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/status/<folder_name>', methods=['GET'])
def get_youtube_status(folder_name):
    """Get YouTube processing status for a specific video."""
    try:
        status = youtube_processor.get_processing_status(folder_name)
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/videos', methods=['GET'])
def get_all_youtube_videos():
    """Get list of all YouTube videos."""
    try:
        videos = youtube_processor.get_all_videos()
        
        # Ensure we return a list format expected by frontend
        if not isinstance(videos, list):
            videos = []
        
        # Ensure each video has all required fields with proper defaults
        formatted_videos = []
        for video in videos:
            formatted_video = {
                'folder_name': video.get('folder_name', ''),
                'created_at': video.get('created_at', 0),
                'status': video.get('status', 'downloading'),  # Never 'unknown'
                'stage': video.get('stage', 'initializing'),
                'percent_download': max(0.0, float(video.get('percent_download', 0))),
                'percent_processing': max(0.0, float(video.get('percent_processing', 0))),
                'current_frame': int(video.get('current_frame', 0)),
                'total_frames': int(video.get('total_frames', 0)),
                'frames_extracted': int(video.get('frames_extracted', 0)),
                'crops_created': int(video.get('crops_created', 0)),
                'segments_created': int(video.get('segments_created', 0)),
                'total_segments': int(video.get('total_segments', 0)),
                'completed_segments': int(video.get('completed_segments', 0)),
                'pending_frames': int(video.get('pending_frames', 0)),
                'pending_signal_detections': int(video.get('pending_signal_detections', 0)),
                'auto_label': bool(video.get('auto_label', True)),
                'updated_at': video.get('updated_at', datetime.now().isoformat()),
                'error': video.get('error', None)
            }
            formatted_videos.append(formatted_video)
        
        # Format the response properly
        response = {
            'videos': formatted_videos,
            'total': len(formatted_videos),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'videos': [],
            'total': 0,
            'status': 'error'
        }), 500

# Global auto-labeled frames routes
@youtube_bp.route('/autolabeled/all', methods=['GET'])
def get_all_autolabeled_frames():
    """Get all auto-labeled frames from all videos with pagination."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        # Get all video folders
        youtube_videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
        all_frames = []
        
        for video_folder in youtube_videos_dir.iterdir():
            if not video_folder.is_dir():
                continue
                
            # Check if video has auto-labeled frames
            processed_dir = video_folder / 'processed'
            if not processed_dir.exists():
                continue
                
            # Get video ID from folder name
            video_id = video_folder.name.split('_')[0]
            
            # Get all processed frames
            processed_frames = sorted([f.name for f in processed_dir.glob('*.jpg')])
            
            for frame_name in processed_frames:
                # Try to get confidence from label file if exists
                frame_base = frame_name.replace('.jpg', '')
                label_file = video_folder / 'frames' / f"{frame_base}.txt"
                confidence = None
                
                if label_file.exists():
                    try:
                        with open(label_file, 'r') as f:
                            line = f.readline().strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    # YOLO format has confidence as 6th value (index 5) if present
                                    confidence = float(parts[4]) if len(parts) > 5 else None
                    except:
                        pass
                
                all_frames.append({
                    'video_id': video_id,
                    'folder_name': video_folder.name,
                    'frame_name': frame_name,
                    'confidence': confidence,
                    'created_at': video_folder.stat().st_mtime
                })
        
        # Sort by creation time (newest first)
        all_frames.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Pagination
        total_frames = len(all_frames)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_frames = all_frames[start:end]
        
        return jsonify({
            'frames': paginated_frames,
            'total': total_frames,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_frames + per_page - 1) // per_page,
            'confirmed': 0,  # TODO: Track confirmed frames
            'pending': total_frames  # TODO: Track pending frames
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/autolabeled/confirm', methods=['POST'])
def confirm_autolabeled_frame():
    """Confirm or reject a global auto-labeled frame."""
    try:
        data = request.json
        frame_data = data.get('frame_data')
        is_correct = data.get('is_correct')
        
        if not frame_data or is_correct is None:
            return jsonify({'error': 'Frame data and is_correct are required'}), 400
        
        video_id = frame_data['video_id']
        folder_name = frame_data['folder_name']
        frame_name = frame_data['frame_name']
        
        # Use the existing confirm_referee_detection logic
        result = youtube_processor.confirm_referee_detection(folder_name, frame_name, is_correct)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Frame confirmed successfully',
                'action': 'confirmed' if is_correct else 'rejected'
            })
        else:
            return jsonify({'error': result.get('error', 'Failed to confirm frame')}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/autolabeled/image/<video_id>/<frame_name>')
def get_autolabeled_image(video_id, frame_name):
    """Serve auto-labeled frame image."""
    try:
        # Find the folder that matches this video_id
        youtube_videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
        
        for video_folder in youtube_videos_dir.iterdir():
            if video_folder.is_dir() and video_folder.name.startswith(video_id):
                processed_dir = video_folder / 'processed'
                image_path = processed_dir / frame_name
                
                if image_path.exists():
                    return send_from_directory(str(processed_dir), frame_name)
        
        return jsonify({'error': 'Image not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/frames', methods=['GET'])
def get_video_frames(folder_name):
    """Get list of frames for a video."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        frames_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name / 'frames'
        if not frames_dir.exists():
            return jsonify({'error': 'Frames not found'}), 404
        
        # Get all frame files
        frame_files = sorted([f.name for f in frames_dir.glob('*.jpg')])
        
        # Pagination
        start = (page - 1) * per_page
        end = start + per_page
        paginated_frames = frame_files[start:end]
        
        return jsonify({
            'frames': paginated_frames,
            'total': len(frame_files),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(frame_files) + per_page - 1) // per_page
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/crops', methods=['GET'])
def get_video_crops(folder_name):
    """Get list of crops for a video."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        crops_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name / 'crops'
        if not crops_dir.exists():
            return jsonify({'frames': [], 'total': 0})
        
        # Get all crop files
        crop_files = sorted([f.name for f in crops_dir.glob('*.jpg')])
        
        # Pagination
        start = (page - 1) * per_page
        end = start + per_page
        paginated_crops = crop_files[start:end]
        
        return jsonify({
            'frames': paginated_crops,
            'total': len(crop_files),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(crop_files) + per_page - 1) // per_page
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/thumbnails', methods=['GET'])
def get_video_thumbnails(folder_name):
    """Get list of thumbnails for a video."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        thumbnails_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name / 'thumbnails'
        if not thumbnails_dir.exists():
            return jsonify({'frames': [], 'total': 0})
        
        # Get all thumbnail files
        thumbnail_files = sorted([f.name for f in thumbnails_dir.glob('*.jpg')])
        
        # Pagination
        start = (page - 1) * per_page
        end = start + per_page
        paginated_thumbnails = thumbnail_files[start:end]
        
        return jsonify({
            'frames': paginated_thumbnails,
            'total': len(thumbnail_files),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(thumbnail_files) + per_page - 1) // per_page
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/processed', methods=['GET'])
def get_video_processed(folder_name):
    """Get list of processed frames for a video with confirmation status."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
        frames_dir = video_dir / 'frames'
        processed_dir = video_dir / 'processed'
        
        if not frames_dir.exists():
            return jsonify({'frames': [], 'total': 0, 'confirmed': 0, 'pending': 0})
        
        # Get all frame files (these are the frames that could potentially have auto-labels)
        all_frame_files = sorted([f.name for f in frames_dir.glob('frame_*.jpg')])
        
        # Get processed frames (pending confirmation)
        processed_files = set()
        if processed_dir.exists():
            processed_files = set([f.name for f in processed_dir.glob('*.jpg')])
        
        # Build frame data with status
        frame_data = []
        confirmed_count = 0
        pending_count = 0
        
        for frame_name in all_frame_files:
            # Check if this frame has a YOLO label (was auto-labeled)
            frame_base = frame_name.replace('.jpg', '')
            label_file = frames_dir / f"{frame_base}.txt"
            
            # Only include frames that were auto-labeled (have label files)
            if label_file.exists():
                if frame_name in processed_files:
                    # Frame is processed but not confirmed yet (pending)
                    status = 'pending'
                    pending_count += 1
                else:
                    # Frame was auto-labeled but no longer in processed (confirmed/denied)
                    status = 'confirmed'
                    confirmed_count += 1
                
                frame_data.append({
                    'frame_name': frame_name,
                    'status': status
                })
        
        # For the confirmation view, we only want pending frames
        pending_frames = [f for f in frame_data if f['status'] == 'pending']
        
        # Pagination on pending frames only
        total_pending = len(pending_frames)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_frames = [f['frame_name'] for f in pending_frames[start:end]]
        
        return jsonify({
            'frames': paginated_frames,
            'total': total_pending,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_pending + per_page - 1) // per_page if total_pending > 0 else 1,
            'confirmed': confirmed_count,
            'pending': pending_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/delete', methods=['DELETE'])
def delete_youtube_video(folder_name):
    """Delete a YouTube video and all its data."""
    try:
        result = youtube_processor.delete_video(folder_name)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/pause', methods=['POST'])
def pause_youtube_video(folder_name):
    """Pause YouTube video processing."""
    try:
        result = youtube_processor.pause_video_processing(folder_name)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/resume', methods=['POST'])
def resume_youtube_video(folder_name):
    """Resume YouTube video processing."""
    try:
        result = youtube_processor.resume_video_processing(folder_name)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/confirm_referee', methods=['POST'])
def confirm_referee_detection(folder_name):
    """Confirm or reject referee detection for a specific frame."""
    try:
        data = request.json
        frame_name = data.get('frame_name')
        is_correct = data.get('is_correct')
        
        if not frame_name or is_correct is None:
            return jsonify({'error': 'Frame name and is_correct are required'}), 400
        
        result = youtube_processor.confirm_referee_detection(folder_name, frame_name, is_correct)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Referee detection confirmed',
                'action': 'confirmed' if is_correct else 'rejected'
            })
        else:
            return jsonify({'error': result.get('error', 'Failed to confirm detection')}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/data/<video_id>/<asset_type>/<filename>')
def get_youtube_asset_file(video_id, asset_type, filename):
    """Get YouTube video asset files (frames, crops, thumbnails, processed)."""
    try:
        # Find the video folder
        youtube_videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
        video_folder = None
        
        for folder in youtube_videos_dir.iterdir():
            if folder.is_dir() and folder.name.startswith(video_id):
                video_folder = folder
                break
        
        if not video_folder:
            return jsonify({'error': 'Video not found'}), 404
        
        # Map asset types to directories
        asset_dirs = {
            'frames': 'frames',
            'crops': 'crops',
            'thumbnails': 'thumbnails',
            'processed': 'processed'
        }
        
        if asset_type not in asset_dirs:
            return jsonify({'error': 'Invalid asset type'}), 400
        
        asset_dir = video_folder / asset_dirs[asset_type]
        
        if not asset_dir.exists():
            return jsonify({'error': f'{asset_type.title()} directory not found'}), 404
        
        file_path = asset_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_from_directory(str(asset_dir), filename)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/confirmed_referees', methods=['GET'])
def get_confirmed_referees_for_signals(folder_name):
    """Get confirmed referee crops that are ready for signal confirmation."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        video_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER) / folder_name
        confirmed_dir = video_dir / 'confirmed_crops'
        
        if not confirmed_dir.exists():
            return jsonify({'crops': [], 'total': 0, 'confirmed': 0, 'pending': 0})
        
        # Get all confirmed crop files
        all_crop_files = sorted([f.name for f in confirmed_dir.glob('*.jpg')])
        
        # Pagination
        total = len(all_crop_files)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        crop_files = all_crop_files[start_idx:end_idx]
        
        # Count confirmed vs pending signal confirmations
        signal_confirmed_dir = video_dir / 'signal_confirmed'
        confirmed_signals = set()
        if signal_confirmed_dir.exists():
            confirmed_signals = {f.stem for f in signal_confirmed_dir.glob('*.jpg')}
        
        confirmed_count = len([f for f in all_crop_files if Path(f).stem in confirmed_signals])
        pending_count = total - confirmed_count
        
        return jsonify({
            'crops': crop_files,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page,
            'confirmed': confirmed_count,
            'pending': pending_count
        })
        
    except Exception as e:
        logger.error(f"Error getting confirmed referees for signals: {e}")
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/confirm_signal', methods=['POST'])
def confirm_signal_detection(folder_name):
    """Confirm signal detection for a confirmed referee crop."""
    try:
        data = request.get_json()
        crop_filename = data.get('crop_filename')
        signal_class = data.get('signal_class')
        is_correct = data.get('is_correct', True)
        
        if not crop_filename:
            return jsonify({'error': 'Missing crop_filename'}), 400
        
        result = youtube_processor.confirm_signal_detection(folder_name, crop_filename, signal_class, is_correct)
        
        if result['success']:
            return jsonify({'status': 'success', 'message': result['message']})
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        logger.error(f"Error confirming signal detection: {e}")
        return jsonify({'error': str(e)}), 500

# Signal detection workflow routes
@youtube_bp.route('/signal_detections/all', methods=['GET'])
def get_all_signal_detections():
    """Get all autolabeled signal detections for confirmation."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        result = youtube_processor.get_autolabeled_signal_detections(
            folder_name=None, page=page, per_page=per_page
        )
        
        return jsonify({
            'detections': result['detections'],
            'total': result['total'],
            'page': result['page'],
            'per_page': result['per_page'],
            'total_pages': result['total_pages'],
            'pending': result['total']
        })
        
    except Exception as e:
        logger.error(f"Error getting all signal detections: {e}")
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/signal_detections', methods=['GET'])
def get_video_signal_detections(folder_name):
    """Get signal detections for a specific video."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        result = youtube_processor.get_autolabeled_signal_detections(
            folder_name=folder_name, page=page, per_page=per_page
        )
        
        return jsonify({
            'detections': result['detections'],
            'total': result['total'],
            'page': result['page'],
            'per_page': result['per_page'],
            'total_pages': result['total_pages'],
            'pending': result['total']
        })
        
    except Exception as e:
        logger.error(f"Error getting signal detections for {folder_name}: {e}")
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/signal_detections/confirm', methods=['POST'])
def confirm_signal_detection_global():
    """Confirm or reject a signal detection globally."""
    try:
        data = request.json
        detection_data = data.get('detection_data')
        signal_class = data.get('signal_class')
        is_correct = data.get('is_correct', True)
        
        if not detection_data:
            return jsonify({'error': 'Detection data is required'}), 400
        
        folder_name = detection_data['video_id']
        crop_filename = detection_data['crop_filename']
        
        result = youtube_processor.confirm_autolabeled_signal_detection(
            folder_name, crop_filename, signal_class, is_correct
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'action': result['action']
            })
        else:
            return jsonify({'error': result.get('error', 'Failed to confirm signal detection')}), 500
            
    except Exception as e:
        logger.error(f"Error confirming signal detection: {e}")
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/confirm_signal_detection', methods=['POST'])
def confirm_video_signal_detection(folder_name):
    """Confirm signal detection for a specific video."""
    try:
        data = request.json
        crop_filename = data.get('crop_filename')
        signal_class = data.get('signal_class')
        is_correct = data.get('is_correct', True)
        
        if not crop_filename:
            return jsonify({'error': 'Crop filename is required'}), 400
        
        result = youtube_processor.confirm_autolabeled_signal_detection(
            folder_name, crop_filename, signal_class, is_correct
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'action': result['action']
            })
        else:
            return jsonify({'error': result.get('error', 'Failed to confirm signal detection')}), 500
            
    except Exception as e:
        logger.error(f"Error confirming signal detection for {folder_name}: {e}")
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/signal_detections/image/<video_id>/<crop_filename>')
def get_signal_detection_image(video_id, crop_filename):
    """Serve signal detection image."""
    try:
        # Find the folder that matches this video_id
        youtube_videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
        
        for video_folder in youtube_videos_dir.iterdir():
            if video_folder.is_dir() and video_folder.name.startswith(video_id):
                signal_dir = video_folder / 'autolabeled_signal_detection'
                image_path = signal_dir / crop_filename
                
                if image_path.exists():
                    return send_from_directory(str(signal_dir), crop_filename)
        
        return jsonify({'error': 'Image not found'}), 404
        
    except Exception as e:
        logger.error(f"Error serving signal detection image: {e}")
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/debug/status/<folder_name>', methods=['GET'])
def debug_get_processing_status(folder_name):
    """Debug endpoint to test get_processing_status method."""
    try:
        status = youtube_processor.get_processing_status(folder_name)
        return jsonify({
            'debug': True,
            'folder_name': folder_name,
            'status': status,
            'has_pending_frames': 'pending_frames' in status,
            'pending_frames_value': status.get('pending_frames', 'NOT_FOUND')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/autolabeled/discard', methods=['POST'])
def discard_autolabeled_frame():
    """Discard a global auto-labeled frame."""
    try:
        data = request.json
        frame_data = data.get('frame_data')
        
        if not frame_data:
            return jsonify({'error': 'Frame data is required'}), 400
        
        video_id = frame_data['video_id']
        folder_name = frame_data['folder_name']
        frame_name = frame_data['frame_name']
        
        # Find the video folder
        youtube_videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
        video_folder = None
        
        for folder in youtube_videos_dir.iterdir():
            if folder.is_dir() and folder.name == folder_name:
                video_folder = folder
                break
        
        if not video_folder:
            return jsonify({'error': 'Video folder not found'}), 404
        
        # Remove from processed folder
        processed_dir = video_folder / 'processed'
        frame_path = processed_dir / frame_name
        
        if frame_path.exists():
            frame_path.unlink()
            
        # Also remove any associated crop files
        crops_dir = video_folder / 'crops'
        frame_base = frame_name.replace('.jpg', '')
        crop_pattern = f"crop_{frame_base}*"
        
        for crop_file in crops_dir.glob(crop_pattern):
            crop_file.unlink()
        
        return jsonify({
            'success': True,
            'message': f'Frame "{frame_name}" has been discarded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/discard_frame', methods=['POST'])
def discard_video_frame(folder_name):
    """Discard an auto-labeled frame from a specific video."""
    try:
        data = request.json
        frame_name = data.get('frame_name')
        
        if not frame_name:
            return jsonify({'error': 'Frame name is required'}), 400
        
        # Find the video folder
        youtube_videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
        video_folder = youtube_videos_dir / folder_name
        
        if not video_folder.exists():
            return jsonify({'error': 'Video folder not found'}), 404
        
        # Remove from processed folder
        processed_dir = video_folder / 'processed'
        frame_path = processed_dir / frame_name
        
        if frame_path.exists():
            frame_path.unlink()
            
        # Also remove any associated crop files
        crops_dir = video_folder / 'crops'
        frame_base = frame_name.replace('.jpg', '')
        crop_pattern = f"crop_{frame_base}*"
        
        for crop_file in crops_dir.glob(crop_pattern):
            crop_file.unlink()
        
        return jsonify({
            'success': True,
            'message': f'Frame "{frame_name}" has been discarded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/signal_detections/discard', methods=['POST'])
def discard_signal_detection_global():
    """Discard a signal detection globally."""
    try:
        data = request.json
        detection_data = data.get('detection_data')
        
        if not detection_data:
            return jsonify({'error': 'Detection data is required'}), 400
        
        video_id = detection_data['video_id']
        crop_filename = detection_data['crop_filename']
        
        # Find the video folder
        youtube_videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
        video_folder = None
        
        for folder in youtube_videos_dir.iterdir():
            if folder.is_dir() and folder.name.startswith(video_id):
                video_folder = folder
                break
        
        if not video_folder:
            return jsonify({'error': 'Video folder not found'}), 404
        
        # Remove from autolabeled_signal_detection folder
        signal_detection_dir = video_folder / 'autolabeled_signal_detection'
        
        # Remove image file
        image_path = signal_detection_dir / crop_filename
        if image_path.exists():
            image_path.unlink()
            
        # Remove JSON file
        json_filename = crop_filename.replace('.jpg', '.json')
        json_path = signal_detection_dir / json_filename
        if json_path.exists():
            json_path.unlink()
        
        return jsonify({
            'success': True,
            'message': f'Signal detection "{crop_filename}" has been discarded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/discard_signal_detection', methods=['POST'])
def discard_video_signal_detection(folder_name):
    """Discard a signal detection from a specific video."""
    try:
        data = request.json
        crop_filename = data.get('crop_filename')
        
        if not crop_filename:
            return jsonify({'error': 'Crop filename is required'}), 400
        
        # Find the video folder
        youtube_videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
        video_folder = youtube_videos_dir / folder_name
        
        if not video_folder.exists():
            return jsonify({'error': 'Video folder not found'}), 404
        
        # Remove from autolabeled_signal_detection folder
        signal_detection_dir = video_folder / 'autolabeled_signal_detection'
        
        # Remove image file
        image_path = signal_detection_dir / crop_filename
        if image_path.exists():
            image_path.unlink()
            
        # Remove JSON file
        json_filename = crop_filename.replace('.jpg', '.json')
        json_path = signal_detection_dir / json_filename
        if json_path.exists():
            json_path.unlink()
        
        return jsonify({
            'success': True,
            'message': f'Signal detection "{crop_filename}" has been discarded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@youtube_bp.route('/video/<folder_name>/move_to_autolabel_confirmation', methods=['POST'])
def move_frames_to_autolabel_confirmation(folder_name):
    """Move frames to autolabel confirmation queue."""
    try:
        data = request.json
        frames = data.get('frames', [])
        
        if not frames:
            return jsonify({'error': 'No frames provided'}), 400
        
        # Find the video folder
        youtube_videos_dir = Path(DirectoryConfig.YOUTUBE_VIDEOS_FOLDER)
        video_folder = youtube_videos_dir / folder_name
        
        if not video_folder.exists():
            return jsonify({'error': 'Video folder not found'}), 404
        
        frames_dir = video_folder / 'frames'
        processed_dir = video_folder / 'processed'
        processed_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        already_pending = 0
        errors = []
        
        for frame_name in frames:
            try:
                source_path = frames_dir / frame_name
                dest_path = processed_dir / frame_name
                
                if dest_path.exists():
                    already_pending += 1
                    continue
                    
                if source_path.exists():
                    # Copy the frame to processed folder
                    import shutil
                    shutil.copy2(source_path, dest_path)
                    moved_count += 1
                else:
                    errors.append(f"Frame {frame_name} not found")
                    
            except Exception as e:
                errors.append(f"Error processing {frame_name}: {str(e)}")
        
        return jsonify({
            'success': True,
            'moved_count': moved_count,
            'already_pending': already_pending,
            'errors': errors,
            'message': f'Successfully moved {moved_count} frames to autolabel confirmation'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 