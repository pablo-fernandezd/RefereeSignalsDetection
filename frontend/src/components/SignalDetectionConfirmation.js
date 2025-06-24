import React, { useState, useEffect, useCallback } from 'react';
import './AutoLabeledConfirmation.css';
import SignalConfirmation from './SignalConfirmation';
import NotificationSystem from './NotificationSystem';

const BACKEND_URL = 'http://localhost:5000';

const SignalDetectionConfirmation = ({ videoId, onBack, isGlobalView = false }) => {
    const [signalDetections, setSignalDetections] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [totalDetections, setTotalDetections] = useState(0);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isProcessing, setIsProcessing] = useState(false);
    const [stats, setStats] = useState({ confirmed: 0, pending: 0, total: 0 });
    const [perPage] = useState(10);
    const [notifications, setNotifications] = useState([]);

    const addNotification = (message, type = 'info') => {
        const id = Date.now();
        setNotifications(prev => [...prev, { id, message, type }]);
        setTimeout(() => {
            setNotifications(prev => prev.filter(n => n.id !== id));
        }, 4000);
    };

    const fetchSignalDetections = useCallback(async (page = 1) => {
        setIsLoading(true);
        try {
            const endpoint = isGlobalView 
                ? `${BACKEND_URL}/api/youtube/signal_detections/all?page=${page}&per_page=${perPage}`
                : `${BACKEND_URL}/api/youtube/video/${videoId}/signal_detections?page=${page}&per_page=${perPage}`;
            
            const response = await fetch(endpoint);
            if (!response.ok) throw new Error('Failed to fetch signal detections.');
            const data = await response.json();
            
            setSignalDetections(data.detections || []);
            setCurrentPage(data.page || 1);
            setTotalPages(data.total_pages || 1);
            setTotalDetections(data.total || 0);
            setStats({
                confirmed: 0, // TODO: Track confirmed detections
                pending: data.pending || data.total || 0,
                total: data.total || 0
            });
            setCurrentIndex(0);
            setError(null);
        } catch (err) {
            setError(err.message);
            setSignalDetections([]);
        } finally {
            setIsLoading(false);
        }
    }, [videoId, isGlobalView, perPage]);

    useEffect(() => {
        fetchSignalDetections(1);
    }, [fetchSignalDetections]);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyPress = (event) => {
            if (isProcessing) return;

            switch (event.key) {
                case 'ArrowLeft':
                    event.preventDefault();
                    handlePreviousDetection();
                    break;
                case 'ArrowRight':
                    event.preventDefault();
                    handleNextDetection();
                    break;
                case 'd':
                case 'D':
                    event.preventDefault();
                    handleDiscardDetection();
                    break;
                case 'Escape':
                    event.preventDefault();
                    if (onBack) {
                        onBack();
                    }
                    break;
                default:
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyPress);
        return () => window.removeEventListener('keydown', handleKeyPress);
    }, [isProcessing, currentIndex, signalDetections.length, onBack]);

    const handleNextDetection = () => {
        if (currentIndex < signalDetections.length - 1) {
            setCurrentIndex(currentIndex + 1);
        } else if (currentPage < totalPages) {
            // Load next page
            fetchSignalDetections(currentPage + 1);
        }
    };

    const handlePreviousDetection = () => {
        if (currentIndex > 0) {
            setCurrentIndex(currentIndex - 1);
        } else if (currentPage > 1) {
            // Load previous page and go to last detection
            fetchSignalDetections(currentPage - 1).then(() => {
                setCurrentIndex(perPage - 1);
            });
        }
    };

    const handlePageChange = (newPage) => {
        if (newPage >= 1 && newPage <= totalPages) {
            fetchSignalDetections(newPage);
        }
    };

    const getCurrentDetectionNumber = () => {
        return (currentPage - 1) * perPage + currentIndex + 1;
    };

    const handleDiscardDetection = async () => {
        if (!window.confirm("Are you sure you want to discard this signal detection? This action cannot be undone.")) {
            return;
        }

        setIsProcessing(true);
        const currentDetection = signalDetections[currentIndex];
        
        try {
            const endpoint = isGlobalView
                ? `${BACKEND_URL}/api/youtube/signal_detections/discard`
                : `${BACKEND_URL}/api/youtube/video/${currentDetection.video_id}/discard_signal_detection`;

            const payload = isGlobalView
                ? { detection_data: currentDetection }
                : { crop_filename: currentDetection.crop_filename };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Failed to discard signal detection');
            
            addNotification('Signal detection discarded successfully', 'success');
            
            // Update stats
            setStats(prev => ({
                ...prev,
                pending: prev.pending - 1
            }));
            
            // Remove the discarded detection from the current list
            const updatedDetections = signalDetections.filter((_, index) => index !== currentIndex);
            setSignalDetections(updatedDetections);
            
            // Handle navigation after discard
            if (updatedDetections.length === 0) {
                // No more detections on this page, try to load next page or go back
                if (currentPage < totalPages) {
                    fetchSignalDetections(currentPage + 1);
                } else if (currentPage > 1) {
                    fetchSignalDetections(currentPage - 1);
                } else {
                    addNotification('All signal detections processed! Returning to previous view.', 'success');
                    setTimeout(() => onBack(), 1500);
                }
            } else {
                // Adjust current index if needed
                if (currentIndex >= updatedDetections.length) {
                    setCurrentIndex(updatedDetections.length - 1);
                }
                // Update total count
                setTotalDetections(prev => prev - 1);
            }
            
        } catch (err) {
            addNotification(`Error discarding signal detection: ${err.message}`, 'error');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleSignalConfirm = async (signalData) => {
        setIsProcessing(true);
        const currentDetection = signalDetections[currentIndex];
        
        try {
            const endpoint = isGlobalView
                ? `${BACKEND_URL}/api/youtube/signal_detections/confirm`
                : `${BACKEND_URL}/api/youtube/video/${currentDetection.video_id}/confirm_signal_detection`;

            const payload = isGlobalView
                ? {
                    detection_data: currentDetection,
                    signal_class: signalData.selected_class,
                    is_correct: signalData.correct
                }
                : {
                    crop_filename: currentDetection.crop_filename,
                    signal_class: signalData.selected_class,
                    is_correct: signalData.correct
                };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Failed to confirm signal detection');
            
            addNotification('Signal detection confirmed successfully', 'success');
            
            // Update stats
            setStats(prev => ({
                ...prev,
                confirmed: prev.confirmed + 1,
                pending: prev.pending - 1
            }));
            
            // Remove the confirmed detection from the current list
            const updatedDetections = signalDetections.filter((_, index) => index !== currentIndex);
            setSignalDetections(updatedDetections);
            
            // Handle navigation after confirmation
            if (updatedDetections.length === 0) {
                // No more detections on this page, try to load next page or go back
                if (currentPage < totalPages) {
                    fetchSignalDetections(currentPage + 1);
                } else if (currentPage > 1) {
                    fetchSignalDetections(currentPage - 1);
                } else {
                    addNotification('All signal detections processed! Returning to previous view.', 'success');
                    setTimeout(() => onBack(), 1500);
                }
            } else {
                // Adjust current index if needed
                if (currentIndex >= updatedDetections.length) {
                    setCurrentIndex(updatedDetections.length - 1);
                }
                // Update total count
                setTotalDetections(prev => prev - 1);
            }
            
        } catch (err) {
            addNotification(`Error: ${err.message}`, 'error');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleBackClick = () => {
        console.log('SignalDetectionConfirmation: Back button clicked');
        if (onBack) {
            onBack();
        }
    };

    const handleSignalCancel = () => {
        onBack();
    };

    const currentDetection = signalDetections.length > 0 ? signalDetections[currentIndex] : null;

    if (isLoading) return <div className="loading-confirmation">Loading signal detections...</div>;
    if (error) return <div className="error-confirmation">Error: {error}</div>;
    if (!currentDetection) return <div className="no-frames">No signal detections found.</div>;

    return (
        <div className="autolabeled-confirmation-container">
            <NotificationSystem notifications={notifications} />
            <div className="confirmation-header">
                <div className="header-left">
                    <h2>
                        {isGlobalView ? 'Confirm Signal Detections' : `Confirm Signal Detections - ${videoId}`}
                    </h2>
                    <div className="header-subtitle">
                        Detection {getCurrentDetectionNumber()} of {totalDetections} • Page {currentPage} of {totalPages}
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
                <SignalConfirmation
                    predictedClass={currentDetection.signal_detection?.predicted_class || 'none'}
                    confidence={currentDetection.signal_detection?.confidence || 0}
                    signalBbox={currentDetection.signal_detection?.bbox || []}
                    cropFilenameForSignal={currentDetection.crop_filename}
                    onConfirm={handleSignalConfirm}
                    onCancel={handleSignalCancel}
                    onDiscard={handleDiscardDetection}
                    originalFilename={currentDetection.original_frame}
                    imageUrl={isGlobalView 
                        ? `${BACKEND_URL}/api/youtube/signal_detections/image/${currentDetection.video_id}/${currentDetection.crop_filename}`
                        : `${BACKEND_URL}/api/youtube/signal_detections/image/${videoId}/${currentDetection.crop_filename}`
                    }
                    showNavigationControls={true}
                />

                <div className="keyboard-shortcuts" style={{textAlign: 'center', margin: '10px 0', fontSize: '14px', color: '#666'}}>
                    <span>Shortcuts: Y = Correct (auto-submit), N = Incorrect, D = Discard, 1-9 = Quick class selection, ← → = Navigate, Esc = Back</span>
                </div>

                <div className="navigation-controls">
                    <button 
                        onClick={handlePreviousDetection}
                        disabled={currentIndex === 0 && currentPage === 1}
                        className="nav-btn"
                        title="Keyboard shortcut: ←"
                    >
                        ← Previous
                    </button>
                    <button 
                        onClick={handleNextDetection}
                        disabled={currentIndex === signalDetections.length - 1 && currentPage === totalPages}
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
    );
};

export default SignalDetectionConfirmation; 