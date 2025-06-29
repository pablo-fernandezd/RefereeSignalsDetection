import React, { useState, useEffect, useCallback } from 
'
react
'
;
import 
'
./AutoLabeledConfirmation.css
'
;
import SignalConfirmation from 
'
./SignalConfirmation
'
;

const BACKEND_URL = 
'
http://localhost:5000
'
;

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
    
    // Signal confirmation states
    const [showSignalConfirmation, setShowSignalConfirmation] = useState(false);
    const [confirmedReferees, setConfirmedReferees] = useState([]);
    const [currentSignalIndex, setCurrentSignalIndex] = useState(0);

    const fetchAutoLabeledFrames = useCallback(async (page = 1) => {
        setIsLoading(true);
        try {
            const endpoint = isGlobalView 
                ? `${BACKEND_URL}/api/youtube/autolabeled/all?page=${page}&per_page=${perPage}`
                : `${BACKEND_URL}/api/youtube/video/${videoId}/processed?page=${page}&per_page=${perPage}`;
            
            const response = await fetch(endpoint);
            if (!response.ok) throw new Error(
'
Failed to fetch auto-labeled frames.
'
);
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
        console.log(
'
AutoLabeledConfirmation: Back button clicked
'
);
        if (onBack) {
            onBack();
        }
    };

    return (
        <div className="autolabeled-confirmation-container">
            <div className="confirmation-header">
                <button onClick={handleBackClick} className="back-button">
                    {isGlobalView ? 
'
Back to Dashboard
'
 : 
'
Back to Video List
'
}
                </button>
            </div>
        </div>
    );
};

export default AutoLabeledConfirmation;
