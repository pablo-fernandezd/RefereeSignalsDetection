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

    // We'll define the keyboard handler after the function definitions

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
            // Validate input data
            if (!data) {
                throw new Error('No crop data provided');
            }
            
            if (!currentFile) {
                throw new Error('No current file selected');
            }

            // 1. Submit manual crop
            const cropPayload = { 
                ...data, 
                original_filename: currentFile 
            };
            
            const cropRes = await fetch(`${BACKEND_URL}/api/manual_crop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cropPayload)
            });
            
            if (!cropRes.ok) {
                const errorData = await cropRes.json();
                throw new Error(errorData.error || `HTTP ${cropRes.status}: Failed to save manual crop`);
            }
            
            const cropResData = await cropRes.json();

            // If the image was saved as a negative, just move to the next image.
            if (cropResData.action === 'saved_as_negative') {
                addNotification('Image saved as negative sample (no referee)', 'info');
                handleNextImage();
                return;
            }

            // Check if we should proceed to signal detection
            if (data.proceedToSignal === false) {
                addNotification('Crop saved to training data', 'success');
                handleNextImage();
                return;
            }

            // 2. Process the new crop for a signal
            if (!cropResData.crop_filename_for_signal) {
                throw new Error('No crop filename returned from manual_crop. Response: ' + JSON.stringify(cropResData));
            }
            
            const signalRes = await fetch(`${BACKEND_URL}/api/process_signal`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: cropResData.crop_filename_for_signal })
            });
            
            if (!signalRes.ok) {
                const errorData = await signalRes.json();
                throw new Error(errorData.error || `HTTP ${signalRes.status}: Failed to process signal`);
            }
            
            const signalData = await signalRes.json();

            setCropData(cropResData);
            setSignalResult(signalData);
            setStage('signal_confirm');

        } catch (err) {
            console.error('Error in handleManualCropSubmit:', err);
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
            // Validate required data
            if (!data) {
                throw new Error('No signal confirmation data provided');
            }
            
            if (!pendingFiles[currentIndex]) {
                throw new Error('No current file selected');
            }
            
            if (!cropData?.crop_filename_for_signal) {
                throw new Error('No crop filename available for signal confirmation');
            }

            const payload = {
                selected_class: data.selected_class || 'none',
                confidence: data.confidence || 0.0,
                original_filename: pendingFiles[currentIndex],
                crop_filename_for_signal: cropData.crop_filename_for_signal,
            };

            const response = await fetch(`${BACKEND_URL}/api/confirm_signal`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}: Failed to confirm signal`);
            }
            
            const resData = await response.json();

            // Check for duplicate detection
            if (resData.status === 'warning' && resData.action === 'duplicate_detected') {
                addNotification(`⚠️ Duplicate Image Detected\n\n${resData.message}\n\nThank you for your labeling, but the data has not been saved for training.`, 'warning');
            } else {
                addNotification(resData.message || 'Signal confirmed!', 'success');
            }
            
            handleNextImage();

        } catch (err) {
            console.error('Error in handleSignalConfirm:', err);
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

    // Keyboard shortcuts handler - defined after all functions to avoid temporal dead zone
    const handleKeyDown = useCallback((event) => {
        // Only handle shortcuts when not typing in input fields
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {
            return;
        }

        if (isProcessing) return; // Don't handle shortcuts while processing

        // Handle shortcuts for manual_referee stage (top-level buttons only)
        if (stage === 'manual_referee') {
            if (event.key.toLowerCase() === 'd') {
                // D for Discard Image
                event.preventDefault();
                event.stopPropagation();
                handleDeleteImage();
            } else if (event.key.toLowerCase() === 'a') {
                // A for Auto-Detect Referee
                event.preventDefault();
                event.stopPropagation();
                handleAutoDetectReferee();
            }
            // Let ManualCrop component handle other shortcuts (Enter, Space, N, 1, 2)
            // Don't prevent default for other keys - let ManualCrop handle them
            return;
        }
        
        // For auto_referee_confirm and signal_confirm stages, 
        // let child components (CropConfirmation, SignalConfirmation) handle ALL shortcuts
        // Don't handle any shortcuts here to prevent conflicts
    }, [stage, isProcessing, handleDeleteImage, handleAutoDetectReferee]);

    // Add keyboard event listeners
    useEffect(() => {
        document.addEventListener('keydown', handleKeyDown);
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [handleKeyDown]);

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

            {/* Keyboard shortcuts help - conditional based on stage */}
            <div className="keyboard-shortcuts-help">
                {stage === 'manual_referee' && (
                    <span>⌨️ Shortcuts: <kbd>D</kbd> Discard | <kbd>A</kbd> Auto-Detect | <kbd>Enter</kbd> Save & Label | <kbd>Space</kbd> Skip | <kbd>N</kbd> No Referee</span>
                )}
                {stage === 'auto_referee_confirm' && (
                    <span>⌨️ Shortcuts: <kbd>Y</kbd> Yes | <kbd>N</kbd> No</span>
                )}
                {stage === 'signal_confirm' && (
                    <span>⌨️ Shortcuts: <kbd>Enter</kbd> Submit | <kbd>C</kbd> Correct | <kbd>Y</kbd> Accept Prediction</span>
                )}
            </div>

            <div className="labeling-stage">
                {stage === 'manual_referee' && (
                    <div className="manual-referee-stage">
                         <div className="autolabel-controls">
                             <button onClick={handleDeleteImage} disabled={isProcessing} className="delete-btn">
                                 <kbd>D</kbd> Discard Image
                             </button>
                             <button onClick={handleAutoDetectReferee} disabled={isProcessing} className="autodetect-btn">
                                 <kbd>A</kbd> {isProcessing ? 'Processing...' : 'Auto-detect Referee'}
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
                        onCancel={() => {
                            // When going back from signal detection, skip to next image
                            // because the original image has already been processed and moved to training data
                            addNotification("Crop already saved to training data. Moving to next image.", 'info');
                            handleNextImage();
                        }}
                        originalFilename={currentFile}
                    />
                )}
            </div>
        </div>
    );
};

export default LabelingQueue; 