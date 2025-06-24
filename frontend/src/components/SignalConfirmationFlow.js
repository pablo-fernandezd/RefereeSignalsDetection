import React, { useState, useEffect, useCallback } from 'react';
import './AutoLabeledConfirmation.css'; // Reuse the same styles

const BACKEND_URL = 'http://localhost:5000';

const SignalConfirmationFlow = ({ videoId, onBack }) => {
    const [confirmedCrops, setConfirmedCrops] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isProcessing, setIsProcessing] = useState(false);
    const [stats, setStats] = useState({ confirmed: 0, pending: 0, total: 0 });
    const [perPage] = useState(50);
    const [signalClasses, setSignalClasses] = useState([]);
    const [selectedSignalClass, setSelectedSignalClass] = useState('');

    // Fetch signal classes
    useEffect(() => {
        const fetchSignalClasses = async () => {
            try {
                const response = await fetch(`${BACKEND_URL}/api/signal_classes`);
                const data = await response.json();
                const classes = data.classes || [];
                if (!classes.includes('none')) {
                    classes.push('none');
                }
                setSignalClasses(classes);
            } catch (error) {
                console.error('Error fetching signal classes:', error);
                setSignalClasses(['none']); // Fallback
            }
        };
        fetchSignalClasses();
    }, []);

    const fetchConfirmedCrops = useCallback(async (page = 1) => {
        setIsLoading(true);
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/video/${videoId}/confirmed_referees?page=${page}&per_page=${perPage}`);
            if (!response.ok) throw new Error('Failed to fetch confirmed crops.');
            const data = await response.json();
            
            setConfirmedCrops(data.crops || []);
            setCurrentPage(data.page || 1);
            setTotalPages(data.total_pages || 1);
            setStats({
                confirmed: data.confirmed || 0,
                pending: data.pending || data.total || 0,
                total: data.total || 0
            });
            setCurrentIndex(0);
            setSelectedSignalClass(''); // Reset selection
            setError(null);
        } catch (err) {
            setError(err.message);
            setConfirmedCrops([]);
        } finally {
            setIsLoading(false);
        }
    }, [videoId, perPage]);

    useEffect(() => {
        fetchConfirmedCrops(1);
    }, [fetchConfirmedCrops]);

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
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    event.preventDefault();
                    const classIndex = parseInt(event.key) - 1;
                    if (classIndex < signalClasses.length) {
                        setSelectedSignalClass(signalClasses[classIndex]);
                    }
                    break;
                case 'Enter':
                    event.preventDefault();
                    if (selectedSignalClass) {
                        handleConfirmSignal();
                    }
                    break;
                case 'Escape':
                    event.preventDefault();
                    onBack();
                    break;
                default:
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyPress);
        return () => window.removeEventListener('keydown', handleKeyPress);
    }, [isProcessing, currentIndex, confirmedCrops.length, selectedSignalClass, signalClasses]);

    const handleNextFrame = () => {
        if (currentIndex < confirmedCrops.length - 1) {
            setCurrentIndex(currentIndex + 1);
            setSelectedSignalClass(''); // Reset selection
        } else if (currentPage < totalPages) {
            // Load next page
            fetchConfirmedCrops(currentPage + 1);
        }
    };

    const handlePreviousFrame = () => {
        if (currentIndex > 0) {
            setCurrentIndex(currentIndex - 1);
            setSelectedSignalClass(''); // Reset selection
        } else if (currentPage > 1) {
            // Load previous page and go to last frame
            fetchConfirmedCrops(currentPage - 1).then(() => {
                setCurrentIndex(perPage - 1);
            });
        }
    };

    const handlePageChange = (newPage) => {
        if (newPage >= 1 && newPage <= totalPages) {
            fetchConfirmedCrops(newPage);
        }
    };

    const handleConfirmSignal = async () => {
        if (!selectedSignalClass) {
            alert('Please select a signal class first.');
            return;
        }

        setIsProcessing(true);
        const currentCrop = confirmedCrops[currentIndex];
        
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/video/${videoId}/confirm_signal`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    crop_filename: currentCrop,
                    signal_class: selectedSignalClass,
                    is_correct: true
                })
            });
            
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Failed to confirm signal');
            
            // Update stats
            setStats(prev => ({
                ...prev,
                confirmed: prev.confirmed + 1,
                pending: prev.pending - 1
            }));
            
            // Move to next frame or handle end of frames
            if (currentIndex < confirmedCrops.length - 1) {
                handleNextFrame();
            } else if (currentPage < totalPages) {
                fetchConfirmedCrops(currentPage + 1);
            } else {
                alert('All crops processed! Returning to previous view.');
                onBack();
            }
            
        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            setIsProcessing(false);
        }
    };

    const currentCrop = confirmedCrops.length > 0 ? confirmedCrops[currentIndex] : null;
    const cropUrl = currentCrop ? `${BACKEND_URL}/api/youtube/data/${videoId}/confirmed_crops/${currentCrop}` : null;

    if (isLoading) return <div className="loading-confirmation">Loading confirmed crops for signal classification...</div>;
    if (error) return <div className="error-confirmation">Error: {error}</div>;
    if (!currentCrop) return <div className="no-frames">No confirmed crops found for signal classification.</div>;

    return (
        <div className="autolabeled-confirmation-container">
            <div className="confirmation-header">
                <h2>Signal Classification - {videoId}</h2>
                <div className="confirmation-stats">
                    <span>Crop {currentIndex + 1} of {confirmedCrops.length}</span>
                    <span>Page {currentPage} of {totalPages}</span>
                    <span>Confirmed: {stats.confirmed}</span>
                    <span>Pending: {stats.pending}</span>
                </div>
                <button onClick={onBack} className="back-button">
                    Back to Video List
                </button>
            </div>

            <div className="confirmation-content">
                <div className="frame-display">
                    <img 
                        src={cropUrl} 
                        alt={`Crop ${currentCrop}`} 
                        className="confirmation-frame" 
                    />
                    <div className="frame-info">
                        <div className="frame-name">{currentCrop}</div>
                    </div>
                </div>

                <div className="confirmation-controls">
                    <h3>What signal is the referee making?</h3>
                    <div className="keyboard-shortcuts">
                        <span>Shortcuts: 1-9 = Select Class, Enter = Confirm, ← → = Navigate, Esc = Back</span>
                    </div>
                    
                    <div className="signal-classes">
                        {signalClasses.map((signalClass, index) => (
                            <button
                                key={signalClass}
                                onClick={() => setSelectedSignalClass(signalClass)}
                                className={`signal-class-btn ${selectedSignalClass === signalClass ? 'selected' : ''}`}
                                title={`Keyboard shortcut: ${index + 1}`}
                            >
                                {index + 1}. {signalClass === 'none' ? 'No Signal' : signalClass.charAt(0).toUpperCase() + signalClass.slice(1)}
                            </button>
                        ))}
                    </div>

                    <div className="confirmation-actions">
                        <button 
                            onClick={handleConfirmSignal}
                            disabled={isProcessing || !selectedSignalClass}
                            className="confirm-btn correct"
                            title="Keyboard shortcut: Enter"
                        >
                            ✓ Confirm Signal (Enter)
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
                            disabled={currentIndex === confirmedCrops.length - 1 && currentPage === totalPages}
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

export default SignalConfirmationFlow; 