import React, { useState, useEffect, useCallback } from 'react';
import './AutoLabeledConfirmation.css';
import SignalConfirmation from './SignalConfirmation';
import NotificationSystem from './NotificationSystem';

const BACKEND_URL = 'http://localhost:5000';

const AutoLabeledConfirmation = ({ videoId, onBack, isGlobalView = false }) => {
    const [autoLabeledFrames, setAutoLabeledFrames] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [totalFrames, setTotalFrames] = useState(0);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isProcessing, setIsProcessing] = useState(false);
    const [stats, setStats] = useState({ confirmed: 0, pending: 0, total: 0 });
    const [perPage] = useState(50);
    const [notifications, setNotifications] = useState([]);
    
    // Signal confirmation states
    const [showSignalConfirmation, setShowSignalConfirmation] = useState(false);
    const [confirmedReferees, setConfirmedReferees] = useState([]);
    const [currentSignalIndex, setCurrentSignalIndex] = useState(0);

    const addNotification = (message, type = 'info') => {
        const id = Date.now();
        setNotifications(prev => [...prev, { id, message, type }]);
        setTimeout(() => {
            setNotifications(prev => prev.filter(n => n.id !== id));
        }, 4000);
    };

    const fetchAutoLabeledFrames = useCallback(async (page = 1) => {
        setIsLoading(true);
        try {
            const endpoint = isGlobalView 
                ? `${BACKEND_URL}/api/youtube/autolabeled/all?page=${page}&per_page=${perPage}`
                : `${BACKEND_URL}/api/youtube/video/${videoId}/processed?page=${page}&per_page=${perPage}`;
            
            const response = await fetch(endpoint);
            if (!response.ok) throw new Error('Failed to fetch auto-labeled frames.');
            const data = await response.json();
            
            setAutoLabeledFrames(data.frames || []);
            setCurrentPage(data.page || 1);
            setTotalPages(data.total_pages || 1);
            setTotalFrames(data.total || 0);
            setStats({
                confirmed: data.confirmed || 0,
                pending: data.pending || data.total || 0,
                total: data.total || 0
            });
            setCurrentIndex(0);
            setError(null);
        } catch (err) {
            setError(err.message);
            setAutoLabeledFrames([]);
        } finally {
            setIsLoading(false);
        }
    }, [videoId, isGlobalView, perPage]);

    useEffect(() => {
        fetchAutoLabeledFrames(1);
    }, [fetchAutoLabeledFrames]);

    const handleBackClick = () => {
        console.log('AutoLabeledConfirmation: Back button clicked');
        if (onBack) {
            onBack();
        }
    };

    const handleEscapeKey = () => {
        console.log('AutoLabeledConfirmation: Escape key pressed');
        if (onBack) {
            onBack();
        }
    };

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyPress = (event) => {
            if (isProcessing) return;

            switch (event.key) {
                case 'ArrowLeft':
                    event.preventDefault();
                    handlePreviousFrame();
                    break;
                case 'ArrowRight':
                    event.preventDefault();
                    handleNextFrame();
                    break;
                case 'y':
                case 'Y':
                    event.preventDefault();
                    handleConfirmReferee(true);
                    break;
                case 'n':
                case 'N':
                    event.preventDefault();
                    handleConfirmReferee(false);
                    break;
                case 'd':
                case 'D':
                    event.preventDefault();
                    handleDiscardFrame();
                    break;
                case 's':
                case 'S':
                    event.preventDefault();
                    handleSkipFrame();
                    break;
                case 'Escape':
                    event.preventDefault();
                    handleEscapeKey();
                    break;
                default:
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyPress);
        return () => window.removeEventListener('keydown', handleKeyPress);
    }, [isProcessing, currentIndex, autoLabeledFrames.length, onBack]);

    const handleNextFrame = () => {
        if (currentIndex < autoLabeledFrames.length - 1) {
            setCurrentIndex(currentIndex + 1);
        } else if (currentPage < totalPages) {
            // Load next page
            fetchAutoLabeledFrames(currentPage + 1);
        }
    };

    const handlePreviousFrame = () => {
        if (currentIndex > 0) {
            setCurrentIndex(currentIndex - 1);
        } else if (currentPage > 1) {
            // Load previous page and go to last frame
            fetchAutoLabeledFrames(currentPage - 1).then(() => {
                setCurrentIndex(perPage - 1);
            });
        }
    };

    const handlePageChange = (newPage) => {
        if (newPage >= 1 && newPage <= totalPages) {
            fetchAutoLabeledFrames(newPage);
        }
    };

    const getCurrentFrameNumber = () => {
        return (currentPage - 1) * perPage + currentIndex + 1;
    };

    const handleDiscardFrame = async () => {
        if (!window.confirm("Are you sure you want to discard this frame? This action cannot be undone.")) {
            return;
        }

        setIsProcessing(true);
        const currentFrame = autoLabeledFrames[currentIndex];
        
        try {
            const endpoint = isGlobalView
                ? `${BACKEND_URL}/api/youtube/autolabeled/discard`
                : `${BACKEND_URL}/api/youtube/video/${videoId}/discard_frame`;
            
            const payload = isGlobalView
                ? { frame_data: currentFrame }
                : { frame_name: currentFrame };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Failed to discard frame');
            
            addNotification('Frame discarded successfully', 'success');
            
            // Update stats
            setStats(prev => ({
                ...prev,
                pending: prev.pending - 1
            }));
            
            // Move to next frame or handle end of frames
            if (currentIndex < autoLabeledFrames.length - 1) {
                handleNextFrame();
            } else if (currentPage < totalPages) {
                fetchAutoLabeledFrames(currentPage + 1);
            } else {
                // All frames processed, check for signal detections
                await checkForSignalDetections();
            }
            
        } catch (err) {
            addNotification(`Error discarding frame: ${err.message}`, 'error');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleConfirmReferee = async (isCorrect) => {
        setIsProcessing(true);
        const currentFrame = autoLabeledFrames[currentIndex];
        
        try {
            const endpoint = isGlobalView
                ? `${BACKEND_URL}/api/youtube/autolabeled/confirm`
                : `${BACKEND_URL}/api/youtube/video/${videoId}/confirm_referee`;
            
            const payload = isGlobalView
                ? { frame_data: currentFrame, is_correct: isCorrect }
                : { frame_name: currentFrame, is_correct: isCorrect };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Failed to confirm referee');
            
            addNotification(
                isCorrect ? 'Referee detection confirmed' : 'Referee detection rejected',
                'success'
            );
            
            // Update stats
            setStats(prev => ({
                ...prev,
                confirmed: prev.confirmed + 1,
                pending: prev.pending - 1
            }));
            
            // Move to next frame or handle end of frames
            if (currentIndex < autoLabeledFrames.length - 1) {
                handleNextFrame();
            } else if (currentPage < totalPages) {
                fetchAutoLabeledFrames(currentPage + 1);
            } else {
                // All referee confirmations done, check for signal detections
                await checkForSignalDetections();
            }
            
        } catch (err) {
            addNotification(`Error: ${err.message}`, 'error');
        } finally {
            setIsProcessing(false);
        }
    };

    const checkForSignalDetections = async () => {
        try {
            // Check if there are signal detections to confirm
            const endpoint = isGlobalView
                ? `${BACKEND_URL}/api/youtube/signal_detections/all?page=1&per_page=1`
                : `${BACKEND_URL}/api/youtube/video/${videoId}/signal_detections?page=1&per_page=1`;

            const response = await fetch(endpoint);
            if (response.ok) {
                const data = await response.json();
                if (data.total > 0) {
                    // Load all signal detections for confirmation
                    const allSignalsResponse = await fetch(
                        isGlobalView
                            ? `${BACKEND_URL}/api/youtube/signal_detections/all?page=1&per_page=${data.total}`
                            : `${BACKEND_URL}/api/youtube/video/${videoId}/signal_detections?page=1&per_page=${data.total}`
                    );
                    
                    if (allSignalsResponse.ok) {
                        const allSignalsData = await allSignalsResponse.json();
                        setConfirmedReferees(allSignalsData.detections);
                        setShowSignalConfirmation(true);
                        setCurrentSignalIndex(0);
                        return;
                    }
                }
            }
            
            // No signal detections found
            addNotification('All frames processed! Returning to previous view.', 'success');
            setTimeout(() => onBack(), 1500);
        } catch (error) {
            console.error('Error checking for signal detections:', error);
            addNotification('All frames processed! Returning to previous view.', 'success');
            setTimeout(() => onBack(), 1500);
        }
    };

    const handleSkipFrame = async () => {
        // Update stats for skipped frame
        setStats(prev => ({
            ...prev,
            pending: prev.pending - 1
        }));
        
        if (currentIndex < autoLabeledFrames.length - 1) {
            handleNextFrame();
        } else if (currentPage < totalPages) {
            fetchAutoLabeledFrames(currentPage + 1);
        } else {
            // All referee confirmations done, check for signal detections
            await checkForSignalDetections();
        }
    };

    // Signal confirmation handlers
    const handleSignalConfirm = async (signalData) => {
        const currentSignal = confirmedReferees[currentSignalIndex];
        
        try {
            const endpoint = isGlobalView
                ? `${BACKEND_URL}/api/youtube/signal_detections/confirm`
                : `${BACKEND_URL}/api/youtube/video/${currentSignal.video_id}/confirm_signal_detection`;

            const payload = isGlobalView
                ? {
                    detection_data: currentSignal,
                    signal_class: signalData.selected_class,
                    is_correct: signalData.correct
                }
                : {
                    crop_filename: currentSignal.crop_filename,
                    signal_class: signalData.selected_class,
                    is_correct: signalData.correct
                };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Failed to confirm signal');
            
            addNotification('Signal confirmed successfully', 'success');
            
            // Move to next signal or finish
            if (currentSignalIndex < confirmedReferees.length - 1) {
                setCurrentSignalIndex(currentSignalIndex + 1);
            } else {
                addNotification('All signals confirmed! Returning to previous view.', 'success');
                setTimeout(() => onBack(), 1500);
            }
            
        } catch (err) {
            addNotification(`Error confirming signal: ${err.message}`, 'error');
        }
    };

    const handleSignalCancel = () => {
        // Go back to referee confirmation
        setShowSignalConfirmation(false);
        setConfirmedReferees([]);
        setCurrentSignalIndex(0);
    };

    const currentFrame = autoLabeledFrames.length > 0 ? autoLabeledFrames[currentIndex] : null;
    const frameUrl = currentFrame ? 
        (isGlobalView 
            ? `${BACKEND_URL}/api/youtube/autolabeled/image/${currentFrame.video_id}/${currentFrame.frame_name}`
            : `${BACKEND_URL}/api/youtube/data/${videoId}/processed/${currentFrame}`
        ) : null;

    if (isLoading) return <div className="loading-confirmation">Loading frames pending confirmation...</div>;
    if (error) return <div className="error-confirmation">Error: {error}</div>;
    if (!currentFrame && !showSignalConfirmation) return <div className="no-frames">No frames pending confirmation found.</div>;

    // Show signal confirmation view
    if (showSignalConfirmation && confirmedReferees.length > 0) {
        const currentSignal = confirmedReferees[currentSignalIndex];
        return (
            <div className="autolabeled-confirmation-container">
                <NotificationSystem notifications={notifications} />
                <div className="confirmation-header">
                    <div className="header-left">
                        <h2>Signal Confirmation{videoId ? ` - ${videoId}` : ''}</h2>
                        <div className="header-subtitle">
                            Signal {currentSignalIndex + 1} of {confirmedReferees.length} • Frame: {currentSignal.original_frame}
                        </div>
                    </div>
                    <div className="header-right">
                        <button onClick={handleSignalCancel} className="back-button">
                            Back to Referee Confirmation
                        </button>
                    </div>
                </div>
                <SignalConfirmation
                    predictedClass={currentSignal.signal_detection?.predicted_class || 'none'}
                    confidence={currentSignal.signal_detection?.confidence || 0}
                    signalBbox={currentSignal.signal_detection?.bbox || []}
                    cropFilenameForSignal={currentSignal.crop_filename}
                    onConfirm={handleSignalConfirm}
                    onCancel={handleSignalCancel}
                    originalFilename={currentSignal.original_frame}
                    imageUrl={isGlobalView 
                        ? `${BACKEND_URL}/api/youtube/signal_detections/image/${currentSignal.video_id}/${currentSignal.crop_filename}`
                        : `${BACKEND_URL}/api/youtube/signal_detections/image/${videoId}/${currentSignal.crop_filename}`
                    }
                />
            </div>
        );
    }

    return (
        <div className="autolabeled-confirmation-container">
            <NotificationSystem notifications={notifications} />
            <div className="confirmation-header">
                <div className="header-left">
                    <h2>
                        {isGlobalView ? 'Confirm Pending Frames' : `Confirm Pending Frames - ${videoId}`}
                    </h2>
                    <div className="header-subtitle">
                        Frame {currentIndex + 1} of {autoLabeledFrames.length} • Page {currentPage} of {totalPages}
                    </div>
                </div>
                <div className="header-right">
                    <div className="confirmation-stats">
                        <div className="stat-item">
                            <span className="stat-label">Confirmed:</span>
                            <span className="stat-value">{stats.confirmed}</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">Pending:</span>
                            <span className="stat-value">{stats.pending}</span>
                        </div>
                    </div>
                    <button onClick={handleBackClick} className="back-button">
                        {isGlobalView ? 'Back to Dashboard' : 'Back to Video List'}
                    </button>
                </div>
            </div>

            <div className="confirmation-content">
                <div className="frame-display">
                    <img 
                        src={frameUrl} 
                        alt={`Frame ${currentFrame.frame_name || currentFrame}`} 
                        className="confirmation-frame" 
                    />
                    <div className="frame-info">
                        <div className="frame-name">
                            {isGlobalView ? `${currentFrame.video_id} - ${currentFrame.frame_name}` : currentFrame}
                        </div>
                        {isGlobalView && currentFrame.confidence && (
                            <div className="frame-confidence">
                                Confidence: {(currentFrame.confidence * 100).toFixed(1)}%
                            </div>
                        )}
                    </div>
                </div>

                <div className="confirmation-controls">
                    <h3>Is this referee detection correct?</h3>
                    <div className="keyboard-shortcuts">
                        <span>Shortcuts: Y = Correct, N = Incorrect, D = Discard, S = Skip, ← → = Navigate, Esc = Back</span>
                    </div>
                    <div className="confirmation-actions">
                        <button 
                            onClick={() => handleConfirmReferee(true)}
                            disabled={isProcessing}
                            className="confirm-btn correct"
                            title="Keyboard shortcut: Y"
                        >
                            ✓ Correct (Y)
                        </button>
                        <button 
                            onClick={() => handleConfirmReferee(false)}
                            disabled={isProcessing}
                            className="confirm-btn incorrect"
                            title="Keyboard shortcut: N"
                        >
                            ✗ Incorrect (N)
                        </button>
                        <button 
                            onClick={handleDiscardFrame}
                            disabled={isProcessing}
                            className="discard-btn"
                            title="Keyboard shortcut: D"
                        >
                            🗑️ Discard (D)
                        </button>
                        <button 
                            onClick={handleSkipFrame}
                            disabled={isProcessing}
                            className="skip-btn"
                            title="Keyboard shortcut: S"
                        >
                            Skip (S)
                        </button>
                    </div>

                    <div className="navigation-controls">
                        <button 
                            onClick={handlePreviousFrame}
                            disabled={currentIndex === 0 && currentPage === 1}
                            className="nav-btn"
                            title="Keyboard shortcut: ←"
                        >
                            ← Previous
                        </button>
                        <button 
                            onClick={handleNextFrame}
                            disabled={currentIndex === autoLabeledFrames.length - 1 && currentPage === totalPages}
                            className="nav-btn"
                            title="Keyboard shortcut: →"
                        >
                            Next →
                        </button>
                    </div>

                    <div className="pagination-controls">
                        <button 
                            onClick={() => handlePageChange(1)}
                            disabled={currentPage === 1}
                            className="page-btn"
                        >
                            First
                        </button>
                        <button 
                            onClick={() => handlePageChange(currentPage - 1)}
                            disabled={currentPage === 1}
                            className="page-btn"
                        >
                            Prev Page
                        </button>
                        <span className="page-info">
                            Page {currentPage} of {totalPages}
                        </span>
                        <button 
                            onClick={() => handlePageChange(currentPage + 1)}
                            disabled={currentPage === totalPages}
                            className="page-btn"
                        >
                            Next Page
                        </button>
                        <button 
                            onClick={() => handlePageChange(totalPages)}
                            disabled={currentPage === totalPages}
                            className="page-btn"
                        >
                            Last
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AutoLabeledConfirmation; 