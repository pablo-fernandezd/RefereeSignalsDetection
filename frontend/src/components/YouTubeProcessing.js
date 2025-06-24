import React, { useState, useEffect, useRef, useCallback } from 'react';
import './YouTubeProcessing.css';

const BACKEND_URL = 'http://localhost:5000';

// Polling intervals based on status (in milliseconds)
const POLLING_INTERVALS = {
    downloading: 5000,     // 5 seconds for downloading
    segmenting: 20000,     // 20 seconds for segmenting
    processing: 20000,     // 20 seconds for processing frames
    autolabeling: 20000,   // 20 seconds for auto-labeling
    paused: 10000,         // 10 seconds for paused videos (faster refresh to show resume button)
    idle: 60000           // 60 seconds for completed/error videos
};

const YouTubeProcessing = ({ onViewAssets, showOnlyVideoList = false, refreshTrigger = 0 }) => {
    const [url, setUrl] = useState('');
    const [autoCrop, setAutoCrop] = useState(true);
    const [isProcessing, setIsProcessing] = useState(false);
    const [processingStatus, setProcessingStatus] = useState(null);
    const [videos, setVideos] = useState([]);
    const [selectedVideo, setSelectedVideo] = useState(null);
    const [videoDetails, setVideoDetails] = useState(null);
    const [videoStatuses, setVideoStatuses] = useState({});
    const [isTabFocused, setIsTabFocused] = useState(true);
    const [showAddForm, setShowAddForm] = useState(false);
    
    // Using refs to manage polling timeouts for individual videos
    const videoPollingTimeouts = useRef({});
    const videosListTimeout = useRef();
    const lastVideosListPoll = useRef(0);

    // Helper function to determine polling interval based on video status
    const getPollingInterval = useCallback((status, stage) => {
        if (!status) return POLLING_INTERVALS.idle;
        
        switch (status) {
            case 'downloading':
                return POLLING_INTERVALS.downloading;
            case 'processing':
                if (stage === 'creating_segments' || stage === 'preparing_segments') {
                    return POLLING_INTERVALS.segmenting;
                } else if (stage === 'extracting_frames' || stage === 'processing_frames') {
                    return POLLING_INTERVALS.processing;
                } else if (stage === 'creating_auto_labels') {
                    return POLLING_INTERVALS.autolabeling;
                }
                return POLLING_INTERVALS.processing; // Default for processing
            case 'paused':
                return POLLING_INTERVALS.paused;
            case 'completed':
            case 'error':
                return POLLING_INTERVALS.idle;
            default:
                return POLLING_INTERVALS.processing;
        }
    }, []);

    // Function to poll status for a specific video
    const pollVideoStatus = useCallback(async (folderName) => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/status/${folderName}`);
            const status = await response.json();
            
            setVideoStatuses(prev => ({
                ...prev,
                [folderName]: status
            }));

            // Update video details if this is the selected video
            if (selectedVideo === folderName) {
                setVideoDetails(status);
            }

            // Schedule next poll based on current status
            const interval = getPollingInterval(status.status, status.stage);
            
            // Clear existing timeout for this video
            if (videoPollingTimeouts.current[folderName]) {
                clearTimeout(videoPollingTimeouts.current[folderName]);
            }

            // Only continue polling if the video is still active or if tab is focused
            const isActiveStatus = ['downloading', 'processing', 'paused', 'resuming_processing'].includes(status.status);
            if (isActiveStatus || isTabFocused) {
                videoPollingTimeouts.current[folderName] = setTimeout(() => {
                    pollVideoStatus(folderName);
                }, interval);
            }

            console.log(`${folderName}: ${status.status} (${status.stage}) - next poll in ${interval/1000}s`);

        } catch (error) {
            console.error(`Error polling status for ${folderName}:`, error);
            // Retry with longer interval on error
            if (videoPollingTimeouts.current[folderName]) {
                clearTimeout(videoPollingTimeouts.current[folderName]);
            }
            videoPollingTimeouts.current[folderName] = setTimeout(() => {
                pollVideoStatus(folderName);
            }, POLLING_INTERVALS.idle);
        }
    }, [selectedVideo, getPollingInterval, isTabFocused]);

    // Function to fetch the full videos list (less frequent)
    const fetchVideosList = useCallback(async (force = false) => {
        const now = Date.now();
        
        // Only poll videos list when forced or when enough time has passed
        if (!force && (now - lastVideosListPoll.current) < 30000) {
            return;
        }
        lastVideosListPoll.current = now;

        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/videos`);
            const data = await response.json();
            const currentVideos = data.videos || [];
            
            setVideos(currentVideos);

            // Start individual polling for each video
            currentVideos.forEach(video => {
                // Only start polling if not already polling this video
                if (!videoPollingTimeouts.current[video.folder_name]) {
                    pollVideoStatus(video.folder_name);
                }
            });

            // Clean up polling for videos that no longer exist
            const currentFolderNames = new Set(currentVideos.map(v => v.folder_name));
            Object.keys(videoPollingTimeouts.current).forEach(folderName => {
                if (!currentFolderNames.has(folderName)) {
                    clearTimeout(videoPollingTimeouts.current[folderName]);
                    delete videoPollingTimeouts.current[folderName];
                }
            });

            console.log(`Videos list updated: ${currentVideos.length} videos`);

        } catch (error) {
            console.error('Error fetching videos list:', error);
        }
    }, [pollVideoStatus]);

    // Handle refresh trigger from navigation
    useEffect(() => {
        if (refreshTrigger > 0) {
            console.log('Navigation refresh triggered');
            fetchVideosList(true);
        }
    }, [refreshTrigger, fetchVideosList]);

    // Handle tab focus/blur to optimize polling
    useEffect(() => {
        const handleFocus = () => {
            setIsTabFocused(true);
            // Immediately refresh when tab becomes focused
            fetchVideosList(true);
        };
        
        const handleBlur = () => {
            setIsTabFocused(false);
        };

        window.addEventListener('focus', handleFocus);
        window.addEventListener('blur', handleBlur);

        return () => {
            window.removeEventListener('focus', handleFocus);
            window.removeEventListener('blur', handleBlur);
        };
    }, [fetchVideosList]);

    // Initial load and periodic videos list refresh
    useEffect(() => {
        // Initial load
        fetchVideosList(true);

        // Set up periodic videos list refresh (only when tab is focused)
        const scheduleNextVideosList = () => {
            if (videosListTimeout.current) {
                clearTimeout(videosListTimeout.current);
            }
            
            videosListTimeout.current = setTimeout(() => {
                if (isTabFocused) {
                    fetchVideosList();
                }
                scheduleNextVideosList(); // Schedule next refresh
            }, 60000); // Check every minute for new videos
        };

        scheduleNextVideosList();

        // Cleanup function
        return () => {
            // Clear all video polling timeouts
            Object.values(videoPollingTimeouts.current).forEach(timeout => {
                clearTimeout(timeout);
            });
            videoPollingTimeouts.current = {};
            
            // Clear videos list timeout
            if (videosListTimeout.current) {
                clearTimeout(videosListTimeout.current);
            }
        };
    }, [fetchVideosList, isTabFocused]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!url.trim()) return;

        setIsProcessing(true);
        setProcessingStatus('Starting processing...');

        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/process`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url.trim(), auto_label: autoCrop }),
            });

            const data = await response.json();
            
            if (response.ok && data.status === 'processing_started') {
                setProcessingStatus('Download started. You can monitor progress below.');
                setUrl(''); // Clear the input field on success
                
                // Immediately refresh the video list when a new video is added
                setTimeout(() => fetchVideosList(true), 1000);
                setTimeout(() => fetchVideosList(true), 3000);
                
            } else {
                setProcessingStatus(`Error: ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            setProcessingStatus(`Connection error: ${error.message}`);
        } finally {
            setIsProcessing(false);
        }
    };
    
    const handleVideoSelect = async (folderName) => {
        setSelectedVideo(folderName);
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/status/${folderName}`);
            const data = await response.json();
            setVideoDetails(data);
            
            // Update the video status in the main list to ensure consistency
            setVideoStatuses(prev => ({
                ...prev,
                [folderName]: data
            }));
        } catch (error) {
            console.error('Error loading video details:', error);
            setVideoDetails({ error: 'Could not load details.' });
        }
    };

    const formatStage = (stage) => {
        switch (stage) {
            case 'initializing':
                return 'Initializing';
            case 'downloading_video':
                return 'Downloading Video';
            case 'download_completed':
                return 'Download Completed';
            case 'preparing_segments':
                return 'Preparing Segments';
            case 'processing_frames':
                return 'Processing Frames';
            case 'extracting_frames':
                return 'Extracting Key Frames';
            case 'creating_crops':
                return 'Creating Referee Crops';
            case 'completed':
                return 'Completed';
            case 'paused_by_user':
                return 'Paused by User';
            case 'processing_interrupted':
                return 'Processing Interrupted';
            case 'resuming_processing':
                return 'Resuming Processing';
            default:
                // Handle segment-specific stages like 'processing_segment_15'
                if (stage && stage.startsWith('processing_segment_')) {
                    const segmentNum = stage.split('_')[2];
                    return `Processing Segment ${segmentNum}`;
                }
                return stage ? stage.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) : '';
        }
    };

    const formatVideoName = (folderName) => {
        // Extract video ID and timestamp from folder name
        const parts = folderName.split('_');
        if (parts.length >= 2) {
            const videoId = parts[0];
            const timestamp = parts[1];
            return `${videoId} (${timestamp})`;
        }
        return folderName;
    };

    const formatStatus = (status) => {
        if (status === 'completed') return 'Completed';
        if (status === 'auto_labeling_ready') return 'Auto-labeling Ready';
        if (status === 'processing') return 'Processing';
        if (status === 'paused') return 'Paused';
        if (status === 'downloading') return 'Downloading';
        if (status === 'error') return 'Error';
        return status;
    };

    const handleDeleteVideo = async (folderName, e) => {
        e.stopPropagation();
        if (!window.confirm(`Are you sure you want to delete the video "${formatVideoName(folderName)}"? This action cannot be undone.`)) {
            return;
        }

        try {
            await fetch(`${BACKEND_URL}/api/youtube/video/${folderName}/delete`, {
                method: 'DELETE',
            });
            setVideos(prev => prev.filter(video => video.folder_name !== folderName));
            if (selectedVideo === folderName) {
                setSelectedVideo(null);
                setVideoDetails(null);
            }
        } catch (error) {
            alert(`Error deleting video: ${error.message}`);
        }
    };

    const handlePauseVideo = async (folderName, e) => {
        e.stopPropagation();
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/video/${folderName}/pause`, {
                method: 'POST',
            });
            const result = await response.json();
            if (result.status === 'success') {
                // Immediately refresh the specific video status
                await pollVideoStatus(folderName);
                // Also refresh the full video list
                await fetchVideosList(true);
            } else {
                alert(`Error pausing video: ${result.message}`);
            }
        } catch (error) {
            alert(`Error pausing video: ${error.message}`);
        }
    };

    const handleResumeVideo = async (folderName, e) => {
        e.stopPropagation();
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/video/${folderName}/resume`, {
                method: 'POST',
            });
            const result = await response.json();
            if (result.status === 'success') {
                // Immediately refresh the specific video status
                await pollVideoStatus(folderName);
                // Also refresh the full video list
                await fetchVideosList(true);
            } else {
                alert(`Error resuming video: ${result.message}`);
            }
        } catch (error) {
            alert(`Error resuming video: ${error.message}`);
        }
    };

    const handleViewFrames = (folderName) => {
        onViewAssets(folderName, 'frames');
    };

    // Check if all videos are completed to stop unnecessary polling
    const areAllVideosCompleted = () => {
        return videos.length > 0 && videos.every(video => {
            const status = videoStatuses[video.folder_name] || video.info || {};
            return ['completed', 'auto_labeling_ready', 'error'].includes(status.status);
        });
    };

    return (
        <div className="youtube-processing">
            <div className="youtube-header">
                <h2>YouTube Video Processing</h2>
                {!showOnlyVideoList && (
                    <div className="header-actions">
                        <button 
                            onClick={() => setShowAddForm(!showAddForm)}
                            className="add-video-btn"
                        >
                            {showAddForm ? 'Cancel' : 'Add Video'}
                        </button>
                    </div>
                )}
            </div>
            
            {/* Form Section - Only show if not in assets-only mode and showAddForm is true */}
            {!showOnlyVideoList && showAddForm && (
                <div className="form-section">
                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label htmlFor="youtube-url">YouTube URL:</label>
                            <input
                                type="url"
                                id="youtube-url"
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                placeholder="https://www.youtube.com/watch?v=..."
                                required
                                disabled={isProcessing}
                            />
                        </div>
                        
                        <div className="form-group">
                            <label className="checkbox-label">
                                <input
                                    type="checkbox"
                                    checked={autoCrop}
                                    onChange={(e) => setAutoCrop(e.target.checked)}
                                    disabled={isProcessing}
                                />
                                Auto-label detected referees with YOLO annotations
                            </label>
                        </div>
                        
                        <button 
                            type="submit" 
                            disabled={isProcessing || !url.trim()}
                            className="submit-btn"
                        >
                            {isProcessing ? 'Processing...' : 'Process Video'}
                        </button>
                    </form>
                    
                    {processingStatus && (
                        <div className={`status-message ${isProcessing ? 'processing' : 'completed'}`}>
                            {processingStatus}
                        </div>
                    )}
                </div>
            )}

            {/* Videos List Section */}
            <div className="videos-section">
                <div className="videos-header">
                    <h3>Processed Videos</h3>
                    <button 
                        className="refresh-btn"
                        onClick={async () => {
                            try {
                                const response = await fetch(`${BACKEND_URL}/api/youtube/videos`);
                                const data = await response.json();
                                setVideos(data.videos || []);
                                await fetchVideosList(true);
                            } catch (error) {
                                console.error('Error refreshing videos:', error);
                            }
                        }}
                        title="Refresh video statuses"
                    >
                        🔄 Refresh
                    </button>
                </div>
                <div className="videos-container">
                    <div className="videos-list">
                        {videos.length === 0 ? (
                            <p className="no-videos">No processed videos</p>
                        ) : (
                            videos.map((video, index) => {
                                let status = videoStatuses[video.folder_name] || video.info || {};
                                // Use last known good info if status is empty or unknown
                                if (!status || !status.status || status.status === 'unknown' || Object.keys(status).length === 0) {
                                    status = video.info || { status: 'pending' };
                                }
                                
                                // Debug: Log status data for selected video
                                if (selectedVideo === video.folder_name) {
                                    console.log(`Status for ${video.folder_name}:`, status);
                                    console.log(`Video details:`, videoDetails);
                                }
                                let statusText = 'Unknown';
                                let percent = null;
                                let stage = status.stage || null;
                                let percentForDisplay = '';

                                if (status.status === 'downloading') {
                                    statusText = 'Downloading';
                                    // Use percent_download for downloading, fallback to percent
                                    percent = status.percent_download || status.percent || 0;
                                } else if (status.status === 'processing') {
                                    statusText = 'Processing';
                                    // Use percent_processing for processing
                                    percent = status.percent_processing || 0;
                                    stage = formatStage(status.stage);
                                } else if (status.status === 'paused') {
                                    statusText = 'Paused';
                                    percent = status.percent_processing || 0;
                                    stage = formatStage(status.stage);
                                } else if (status.status === 'completed') {
                                    statusText = 'Completed';
                                    percent = 100; // Always 100% for completed
                                } else if (status.status === 'auto_labeling_ready') {
                                    statusText = 'Auto-labeling Ready';
                                    percent = 100; // Always 100% for auto-labeling ready
                                } else if (status.status === 'finished') {
                                    statusText = 'Download finished, processing...';
                                    percent = 100; // Download is 100% complete
                                } else if (status.status === 'pending') {
                                    statusText = 'Pending';
                                    percent = 0; // 0% for pending
                                }

                                if (typeof percent === 'number' && isFinite(percent)) {
                                    percentForDisplay = percent.toFixed(1);
                                }

                                return (
                                    <div
                                        key={index}
                                        className={`video-item ${selectedVideo === video.folder_name ? 'selected' : ''}`}
                                        onClick={() => handleVideoSelect(video.folder_name)}
                                    >
                                        <div className="video-name">
                                            {formatVideoName(video.folder_name)}
                                        </div>
                                        <div className="video-status">
                                            <div className="status-indicator-wrapper">
                                                <div className={`status-indicator ${status.status}`}></div>
                                                <span className="status-text">
                                                    {statusText}
                                                    {stage && <span className="status-stage"> - {stage}</span>}
                                                </span>
                                            </div>
                                            {percentForDisplay && (
                                                <span className="status-percentage">{percentForDisplay}%</span>
                                            )}
                                        </div>
                                        {typeof percent === 'number' && isFinite(percent) && percent > 0 && (
                                            <div className={`progress-bar-container ${status.status} ${['processing', 'downloading', 'paused'].includes(status.status) ? 'active' : ''}`}>
                                                <div 
                                                    className="progress-bar" 
                                                    style={{ width: `${percent}%` }}
                                                    title={`${percentForDisplay}% - ${statusText}${stage ? ` (${stage})` : ''}`}
                                                >
                                                    {percent > 10 ? `${percentForDisplay}%` : ''}
                                                </div>
                                            </div>
                                        )}
                                        
                                        {(status.status === 'processing') && (
                                            <div className="video-progress-details">
                                                {/* Show segment progress if available */}
                                                {status.total_segments && (
                                                    <span>Segments: {status.completed_segments || 0} / {status.total_segments}</span>
                                                )}
                                                {status.total_frames && (
                                                    <span> | Frames: {status.frames_extracted || 0} / {status.total_frames}</span>
                                                )}
                                                {status.auto_label && (
                                                    <span> | Auto-labeled: {status.crops_created || 0}</span>
                                                )}
                                            </div>
                                        )}
                                        
                                                                {(status.status === 'completed' || status.status === 'auto_labeling_ready') && (
                            <div className="video-stats">
                                <span>Segments: {status.segments_created || video.info?.segments_created || 0}</span>
                                <span>Frames: {status.frames_extracted || video.info?.frames_extracted || 0}</span>
                                {(status.auto_label || video.info?.auto_label) && (
                                    <span>Auto-labeled: {status.crops_created || video.info?.crops_created || 0}</span>
                                )}
                                {(status.pending_frames || 0) > 0 && (
                                    <button 
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onViewAssets(video.folder_name, 'autolabeled_confirmation');
                                        }}
                                        className="control-btn confirm-btn"
                                        title="Confirm auto-labeled frames"
                                    >
                                        📋 {status.pending_frames}
                                    </button>
                                )}
                                {(status.pending_signal_detections || 0) > 0 && (
                                    <button 
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onViewAssets(video.folder_name, 'signal_detection_confirmation');
                                        }}
                                        className="control-btn signal-btn"
                                        title="Confirm signal detections"
                                    >
                                        🚦 {status.pending_signal_detections}
                                    </button>
                                )}
                            </div>
                        )}

                                        <div className="video-controls">
                                            {status.status === 'processing' && (
                                                <button 
                                                    onClick={(e) => handlePauseVideo(video.folder_name, e)}
                                                    className="control-btn pause-btn"
                                                    title="Pause processing"
                                                >
                                                    ⏸️
                                                </button>
                                            )}
                                            {status.status === 'paused' && (
                                                <button 
                                                    onClick={(e) => handleResumeVideo(video.folder_name, e)}
                                                    className="control-btn resume-btn"
                                                    title="Resume processing"
                                                >
                                                    ▶️
                                                </button>
                                            )}
                                            <button 
                                                onClick={(e) => handleDeleteVideo(video.folder_name, e)}
                                                className="control-btn delete-btn"
                                                title="Delete video"
                                            >
                                                🗑️
                                            </button>
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                    
                    {/* Video Details */}
                    {selectedVideo && videoDetails && (
                        <div className="video-details">
                            <h4>Video Details</h4>
                            <p><strong>Status:</strong> {formatStatus(videoDetails.status)} {videoDetails.status === 'processing' && `(${formatStage(videoDetails.stage)})`}</p>
                            
                            {/* Segment Information */}
                            {videoDetails.total_segments && (
                                <p><strong>Segments:</strong> {videoDetails.completed_segments || 0} / {videoDetails.total_segments} completed</p>
                            )}
                            {videoDetails.segment_duration && (
                                <p><strong>Segment Duration:</strong> {videoDetails.segment_duration / 60} minutes each</p>
                            )}
                            
                            {/* Frame and Processing Information */}
                                        {videoDetails.total_frames && <p><strong>Total frames:</strong> {videoDetails.total_frames.toLocaleString()}</p>}
            {videoDetails.frames_extracted && <p><strong>Frames extracted:</strong> {videoDetails.frames_extracted.toLocaleString()}</p>}
            {videoDetails.auto_label && <p><strong>Auto-labeled frames:</strong> {(videoDetails.crops_created || 0).toLocaleString()}</p>}
                            
                            {/* Video Information */}
                            {videoDetails.video_duration && (
                                <p><strong>Video Duration:</strong> {Math.round(videoDetails.video_duration / 60)} minutes</p>
                            )}
                            
                            {/* Enhanced Progress Bar for Processing */}
                            {videoDetails.status === 'processing' && videoDetails.percent_processing != null && (
                                <div className="video-progress">
                                    <div className={`progress-bar-container ${videoDetails.status} active`}>
                                        <div 
                                            className="progress-bar"
                                            style={{ width: `${videoDetails.percent_processing.toFixed(1)}%` }}
                                            title={`${videoDetails.percent_processing.toFixed(1)}% - ${formatStage(videoDetails.stage)}`}
                                        >
                                            {videoDetails.percent_processing > 10 ? `${videoDetails.percent_processing.toFixed(1)}%` : ''}
                                        </div>
                                    </div>
                                    <div className="progress-text">
                                        <span className="progress-stage">{formatStage(videoDetails.stage)}</span>
                                        <span className="progress-percentage">{videoDetails.percent_processing.toFixed(1)}%</span>
                                    </div>
                                </div>
                            )}

                            <p><strong>Auto-label:</strong> {
                                videoDetails.auto_label !== undefined 
                                    ? (videoDetails.auto_label ? 'Yes' : 'No') 
                                    : (videoDetails.crops_created > 0 ? 'Yes (inferred)' : 'Unknown')
                            }</p>

                            <div className="video-actions">
                                {videoDetails.status === 'paused' && (
                                    <button 
                                        onClick={() => handleResumeVideo(selectedVideo, { stopPropagation: () => {} })}
                                        className="btn-action resume-btn"
                                        title="Resume processing"
                                    >
                                        ▶️ Resume Processing
                                    </button>
                                )}
                                {videoDetails.status === 'processing' && (
                                    <button 
                                        onClick={() => handlePauseVideo(selectedVideo, { stopPropagation: () => {} })}
                                        className="btn-action pause-btn"
                                        title="Pause processing"
                                    >
                                        ⏸️ Pause Processing
                                    </button>
                                )}
                                <button onClick={() => handleViewFrames(selectedVideo)} className="btn-view">
                                    View Frames
                                </button>
                                {videoDetails.auto_label && (videoDetails.pending_frames || 0) > 0 && (
                                     <button onClick={() => onViewAssets(selectedVideo, 'autolabeled_confirmation')} className="btn-view primary">
                                        Confirm Auto-labeled Frames ({videoDetails.pending_frames || 0})
                                    </button>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default YouTubeProcessing;