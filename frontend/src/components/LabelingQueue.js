import React, { useState, useEffect, useCallback } from 'react';
import ManualCrop from './ManualCrop';
import SignalConfirmation from './SignalConfirmation';
import CropConfirmation from './CropConfirmation';
import NotificationSystem from './NotificationSystem';
import './LabelingQueue.css';

const BACKEND_URL = 'http://localhost:5000';

const LabelingQueue = ({ onBack }) => {
    const [pendingFiles, setPendingFiles] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    // Stage management for the new interactive flow
    const [stage, setStage] = useState('manual_referee'); // 'manual_referee', 'auto_referee_confirm', 'signal_confirm'
    const [autoDetectData, setAutoDetectData] = useState(null); // To store data from auto-detect
    const [cropData, setCropData] = useState(null); // After either manual or auto crop is confirmed
    const [signalResult, setSignalResult] = useState(null); // To store signal detection results
    const [isProcessing, setIsProcessing] = useState(false); // For disabling buttons during API calls
    const [notifications, setNotifications] = useState([]);

    const addNotification = (message, type = 'info') => {
        const id = Date.now();
        setNotifications(prev => [...prev, { id, message, type }]);
        setTimeout(() => {
            setNotifications(prev => prev.filter(n => n.id !== id));
        }, 4000);
    };

    // Keyboard shortcuts handler
    const handleKeyDown = useCallback((event) => {
        // Only handle shortcuts when not typing in input fields
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {
            return;
        }

        // Prevent default for our shortcuts
        if (event.key === 'Enter' || event.key.toLowerCase() === 'c' || event.key.toLowerCase() === 'y') {
            event.preventDefault();
        }

        if (isProcessing) return; // Don't handle shortcuts while processing

        switch (event.key) {
            case 'Enter':
                if (stage === 'auto_referee_confirm') {
                    handleAutoRefereeConfirm(true);
                } else if (stage === 'signal_confirm') {
                    // Trigger confirm with current predicted class
                    if (signalResult) {
                        handleSignalConfirm({
                            predicted_class: signalResult.predicted_class,
                            confidence: signalResult.confidence,
                            is_correct: true
                        });
                    }
                }
                break;
            case 'c':
            case 'C':
                if (stage === 'auto_referee_confirm') {
                    handleAutoRefereeConfirm(true);
                } else if (stage === 'signal_confirm') {
                    if (signalResult) {
                        handleSignalConfirm({
                            predicted_class: signalResult.predicted_class,
                            confidence: signalResult.confidence,
                            is_correct: true
                        });
                    }
                }
                break;
            case 'y':
            case 'Y':
                if (stage === 'auto_referee_confirm') {
                    handleAutoRefereeConfirm(true);
                } else if (stage === 'signal_confirm') {
                    if (signalResult) {
                        handleSignalConfirm({
                            predicted_class: signalResult.predicted_class,
                            confidence: signalResult.confidence,
                            is_correct: true
                        });
                    }
                }
                break;
            default:
                break;
        }
    }, [stage, isProcessing, signalResult]);

    // Add keyboard event listeners
    useEffect(() => {
        document.addEventListener('keydown', handleKeyDown);
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [handleKeyDown]);

    const fetchPendingFiles = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`${BACKEND_URL}/api/pending_images`);
            if (!response.ok) throw new Error('Failed to fetch pending images.');
            const data = await response.json();
            setPendingFiles(data.images);
            setCurrentIndex(0);
            setStage('manual_referee');
            setError(null);
        } catch (err) {
            setError(err.message);
            setPendingFiles([]);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchPendingFiles();
    }, [fetchPendingFiles]);

    const handleNextImage = () => {
        if (currentIndex < pendingFiles.length - 1) {
            setCurrentIndex(prev => prev + 1);
            setStage('manual_referee'); // Reset stage for the new image
            setAutoDetectData(null);
            setCropData(null);
            setSignalResult(null);
        } else {
            // Last image was processed, refresh the list
            addNotification("You've cleared the queue! Fetching new images.", 'success');
            fetchPendingFiles();
        }
    };

    const handleAutoDetectReferee = async () => {
        setIsProcessing(true);
        const currentFile = pendingFiles[currentIndex];
        try {
            const response = await fetch(`${BACKEND_URL}/api/process_queued_image_referee`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: currentFile })
            });
            const data = await response.json();
            if (!response.ok) {
                 if (data.action === 'deleted') {
                    addNotification(`Auto-detect failed: ${data.error}. The image has been removed from the queue.`, 'warning');
                    handleNextImage();
                 } else {
                    throw new Error(data.error || 'Failed to auto-detect referee.');
                 }
            } else {
                setAutoDetectData(data);
                setStage('auto_referee_confirm');
            }
        } catch (err) {
            addNotification(`Error: ${err.message}`, 'error');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleManualCropSubmit = async (data) => {
        setIsProcessing(true);
        const currentFile = pendingFiles[currentIndex];
        try {
            // 1. Submit manual crop
            const cropRes = await fetch(`${BACKEND_URL}/api/manual_crop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...data, original_filename: currentFile })
            });
            const cropResData = await cropRes.json();
            if (!cropRes.ok) throw new Error(cropResData.error || 'Failed to save manual crop.');

            // If the image was saved as a negative, just move to the next image.
            if (cropResData.action === 'saved_as_negative') {
                handleNextImage();
                return; // Stop processing for this image
            }

            // 2. Process the new crop for a signal
            const signalRes = await fetch(`${BACKEND_URL}/api/process_signal`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: cropResData.crop_filename_for_signal })
            });
            const signalData = await signalRes.json();
            if (!signalRes.ok) throw new Error(signalData.error || 'Failed to process signal.');

            setCropData(cropResData);
            setSignalResult(signalData);
            setStage('signal_confirm');

        } catch (err) {
            addNotification(`Error: ${err.message}`, 'error');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleAutoRefereeConfirm = async (confirmed) => {
        if (!confirmed) {
            setStage('manual_referee');
            setAutoDetectData(null);
            return;
        }

        setIsProcessing(true);
        try {
             // 1. Confirm the auto-crop
            const confirmRes = await fetch(`${BACKEND_URL}/api/confirm_crop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    original_filename: autoDetectData.filename,
                    crop_filename: autoDetectData.crop_filename,
                    bbox: autoDetectData.bbox,
                    class_id: 0, // Referee class ID
                    is_correct: true
                }),
            });
            const confirmData = await confirmRes.json();
            if (!confirmRes.ok) throw new Error(confirmData.error || 'Failed to confirm crop.');

            // Check for duplicate detection
            if (confirmData.status === 'warning' && confirmData.action === 'duplicate_detected') {
                addNotification(`⚠️ Duplicate Image Detected\n\n${confirmData.message}\n\nYou can continue with signal labeling, but the data will not be saved for training.`, 'warning');
                // Still proceed to signal labeling even for duplicates
                if (confirmData.crop_filename_for_signal) {
                    const signalRes = await fetch(`${BACKEND_URL}/api/process_signal`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename: confirmData.crop_filename_for_signal }),
                    });
                    const signalData = await signalRes.json();
                    if (!signalRes.ok) throw new Error(signalData.error || 'Failed to process signal.');

                    setCropData({ crop_filename_for_signal: confirmData.crop_filename_for_signal });
                    setSignalResult(signalData);
                    setStage('signal_confirm');
                }
                return;
            }

            // 2. Process for signal
            const signalRes = await fetch(`${BACKEND_URL}/api/process_signal`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: confirmData.crop_filename_for_signal }),
            });
            const signalData = await signalRes.json();
             if (!signalRes.ok) throw new Error(signalData.error || 'Failed to process signal.');

            setCropData({ crop_filename_for_signal: confirmData.crop_filename_for_signal });
            setSignalResult(signalData);
            setStage('signal_confirm');

        } catch (err) {
            addNotification(`Error: ${err.message}`, 'error');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleSignalConfirm = async (data) => {
        setIsProcessing(true);
         try {
            const response = await fetch(`${BACKEND_URL}/api/confirm_signal`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...data,
                    original_filename: pendingFiles[currentIndex],
                    crop_filename_for_signal: cropData.crop_filename_for_signal,
                })
            });
            const resData = await response.json();
            if (!response.ok) throw new Error(resData.error || 'Failed to confirm signal.');

            // Check for duplicate detection
            if (resData.status === 'warning' && resData.action === 'duplicate_detected') {
                addNotification(`⚠️ Duplicate Image Detected\n\n${resData.message}\n\nThank you for your labeling, but the data has not been saved for training.`, 'warning');
            } else {
            addNotification(resData.message || 'Signal confirmed!', 'success');
            }
            
            handleNextImage();

        } catch (err) {
            addNotification(`Error: ${err.message}`, 'error');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleDeleteImage = async () => {
        if (!window.confirm("Are you sure you want to permanently delete this image from the queue? This cannot be undone.")) {
            return;
        }
        
        setIsProcessing(true);
        const currentFile = pendingFiles[currentIndex];
        try {
            const response = await fetch(`${BACKEND_URL}/api/queue/image/${currentFile}`, {
                method: 'DELETE',
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Failed to delete image.');
            }
            addNotification(data.message || 'Image deleted.', 'success');
            handleNextImage(); // Move to the next one
        } catch (err) {
            addNotification(`Error: ${err.message}`, 'error');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleSkipImage = () => {
        handleNextImage();
    };

    const currentFile = pendingFiles.length > 0 ? pendingFiles[currentIndex] : null;

    if (isLoading) return <div className="loading-queue">Loading pending images...</div>;
    if (error) return <div className="error-queue">Error: {error}</div>;
    if (!currentFile) return <div className="all-done">All images have been labeled. Great work!</div>;

    return (
        <div className="labeling-queue-container">
            <NotificationSystem notifications={notifications} />
            <div className="queue-header">
                <div className="queue-info">
                    <h2>Label Pending Frames</h2>
                    <div className="queue-status">
                        <span>Image {currentIndex + 1} of {pendingFiles.length}</span>
                    </div>
                </div>
                <div className="queue-actions">
                    <button onClick={onBack} className="back-button">Back to Dashboard</button>
                </div>
            </div>

            {/* Keyboard shortcuts help */}
            <div className="keyboard-shortcuts-help">
                <span>⌨️ Shortcuts: <kbd>Enter</kbd>/<kbd>C</kbd>/<kbd>Y</kbd> to confirm</span>
            </div>

            <div className="labeling-stage">
                {stage === 'manual_referee' && (
                    <div className="manual-referee-stage">
                         <div className="autolabel-controls">
                             <button onClick={handleDeleteImage} disabled={isProcessing} className="delete-btn">
                                 Discard Image
                             </button>
                             <button onClick={handleAutoDetectReferee} disabled={isProcessing} className="autodetect-btn">
                                 {isProcessing ? 'Processing...' : 'Auto-detect Referee'}
                             </button>
                         </div>
                        <ManualCrop
                            key={currentFile}
                            filename={currentFile}
                            onSubmit={handleManualCropSubmit}
                            onCancel={handleSkipImage}
                        />
                    </div>
                )}

                {stage === 'auto_referee_confirm' && autoDetectData && (
                    <CropConfirmation
                        cropUrl={`${BACKEND_URL}${autoDetectData.crop_url}`}
                        onConfirm={handleAutoRefereeConfirm}
                        onProceedToSignal={handleAutoRefereeConfirm}
                        onSaveAndFinish={handleAutoRefereeConfirm}
                    />
                )}

                {stage === 'signal_confirm' && signalResult && cropData && (
                    <SignalConfirmation
                        predictedClass={signalResult.predicted_class}
                        confidence={signalResult.confidence}
                        signalBbox={signalResult.bbox_xywhn}
                        cropFilenameForSignal={cropData.crop_filename_for_signal}
                        onConfirm={handleSignalConfirm}
                        onCancel={() => setStage('manual_referee')}
                        originalFilename={currentFile}
                    />
                )}
            </div>
        </div>
    );
};

export default LabelingQueue; 