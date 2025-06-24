import React, { useState, useEffect, useMemo } from 'react';
import { useAppContext } from '../App';
import './VideoAssetViewer.css'; 

const BACKEND_URL = 'http://localhost:5000';
const ITEMS_PER_PAGE_OPTIONS = [50, 100, 200, 500];

const VideoAssetViewer = ({ videoId, assetType, onBack, onSignalDetected }) => {
    const { setSignalResultForApp, setCropFilenameForSignalApp } = useAppContext();
    const [assets, setAssets] = useState([]);
    const [selectedAssets, setSelectedAssets] = useState(new Set());
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    // Pagination state
    const [currentPage, setCurrentPage] = useState(1);
    const [itemsPerPage, setItemsPerPage] = useState(ITEMS_PER_PAGE_OPTIONS[0]);

    // This state will track the correct source folder ('thumbnails' or 'frames')
    const [assetSourceType, setAssetSourceType] = useState(assetType === 'frames' ? 'thumbnails' : assetType);

    useEffect(() => {
        // Clear local storage on component mount to ensure fresh state on reload
        localStorage.clear();

        const fetchAssets = async () => {
            setIsLoading(true);
            let initialFetchType = assetType === 'frames' ? 'thumbnails' : assetType;
            setAssetSourceType(initialFetchType); // Assume success initially

            try {
                const response = await fetch(`${BACKEND_URL}/api/youtube/video/${videoId}/${initialFetchType}`);
                if (!response.ok) {
                    // If thumbnails fail (e.g., for older videos), try fetching original frames
                    if (initialFetchType === 'thumbnails') {
                        console.warn("Thumbnails not found, falling back to full frames.");
                        setAssetSourceType('frames'); // THIS IS THE FIX: Set the correct source for fallback
                        const fallbackResponse = await fetch(`${BACKEND_URL}/api/youtube/video/${videoId}/frames`);
                        if (!fallbackResponse.ok) throw new Error(`HTTP error! Status: ${fallbackResponse.status}`);
                        const data = await fallbackResponse.json();
                        setAssets(data.frames || []);
                    } else {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                } else {
                    const data = await response.json();
                    setAssets(data[initialFetchType] || []);
                }
            } catch (e) {
                setError(e.message);
            } finally {
                setIsLoading(false);
            }
        };
        fetchAssets();
    }, [videoId, assetType]);

    // Memoized calculation for paginated assets
    const paginatedAssets = useMemo(() => {
        const startIndex = (currentPage - 1) * itemsPerPage;
        return assets.slice(startIndex, startIndex + itemsPerPage);
    }, [assets, currentPage, itemsPerPage]);

    const totalPages = Math.ceil(assets.length / itemsPerPage);

    // --- Selection Handlers ---
    const handleSelect = (asset) => {
        const newSelection = new Set(selectedAssets);
        newSelection.has(asset) ? newSelection.delete(asset) : newSelection.add(asset);
        setSelectedAssets(newSelection);
    };
    
    const handleSelectPage = () => {
        const pageAssets = new Set(paginatedAssets);
        const newSelection = new Set(selectedAssets);
        
        // Check if every item on the current page is already selected
        const allOnPageSelected = paginatedAssets.every(asset => selectedAssets.has(asset));

        if (allOnPageSelected) {
            // If all are selected, deselect them
            pageAssets.forEach(asset => newSelection.delete(asset));
        } else {
            // Otherwise, select them all
            pageAssets.forEach(asset => newSelection.add(asset));
        }
        setSelectedAssets(newSelection);
    };

    const handleSelectAll = () => {
        // If all assets are already selected, deselect all. Otherwise, select all.
        if (selectedAssets.size === assets.length) {
            setSelectedAssets(new Set());
        } else {
            setSelectedAssets(new Set(assets));
        }
    };

    const handleDeselectAll = () => {
        setSelectedAssets(new Set());
    };

    const handleMoveToAutolabelConfirmation = async () => {
        if (selectedAssets.size === 0) {
            return alert('Please select at least one frame.');
        }
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/video/${videoId}/move_to_autolabel_confirmation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames: Array.from(selectedAssets) }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Failed to move frames to autolabel confirmation.');
            }
            alert(`Frames moved to autolabel confirmation:\n- ${data.moved_count} frames moved successfully.\n- ${data.already_pending} frames were already pending.\n- ${data.errors?.length || 0} errors.`);
            // Clear selection after successfully moving frames
            handleDeselectAll();
        } catch (e) {
            alert(`Error: ${e.message}`);
        }
    };

    const handleAutolabelSelected = async () => {
        if (selectedAssets.size === 0) {
            return alert('Please select at least one frame.');
        }
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/video/${videoId}/autolabel_frames`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames: Array.from(selectedAssets) }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Failed to start autolabeling process.');
            }
            alert(`Autolabeling process started:\n- ${data.newly_added_to_queue} new frames added to confirmation queue.\n- ${data.already_pending} frames were already pending.\n- ${data.errors.length} errors.`);
            // Clear selection after successfully adding to the queue
            handleDeselectAll();
        } catch (e) {
            alert(`Error: ${e.message}`);
        }
    };

    // --- API Handlers ---
    const handleSubmitForLabeling = async () => {
        if (selectedAssets.size === 0) return alert('Please select at least one frame.');
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/video/${videoId}/label_frames`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames: Array.from(selectedAssets) }),
            });
            const data = await response.json();
            
            // Provide more detailed feedback
            const successCount = data.processed_count || 0;
            const duplicateCount = data.duplicates || 0;
            const errorCount = data.errors ? data.errors.length : 0;
            alert(`Labeling complete.\n- ${successCount} new frames processed.\n- ${duplicateCount} duplicates found.\n- ${errorCount} errors.`);

            if (response.ok) {
                // Deselect all assets upon successful submission
                handleDeselectAll();
            }
        } catch (e) {
            alert(`Error: ${e.message}`);
        }
    };

    const handleAddToTraining = async (asset) => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/video/${videoId}/add_to_training`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame_name: asset }),
            });
            const data = await response.json();
            alert(data.message || data.error || 'Request sent.');
        } catch (e) {
            alert(`Error: ${e.message}`);
        }
    };

    const handleDetectSignals = async (asset) => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/youtube/video/${videoId}/detect_signals`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame_name: asset }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to detect signals.');

            // Use the callback to lift state up to App.js
            onSignalDetected({
                result: data,
                crop_filename_for_signal: data.crop_filename_for_signal
            });
            
        } catch (e) {
            alert(`Error: ${e.message}`);
        }
    };
    
    if (isLoading) return <div className="loading">Loading {assetType}...</div>;
    if (error) return <div className="error-message">Error: {error}</div>;

    return (
        <div className="asset-viewer">
            {/* Header */}
            <div className="asset-viewer-header">
                <h2>Viewing {assetType} for {videoId}</h2>
                <button onClick={onBack} className="nav-btn">Back to Video List</button>
            </div>

            {/* Controls */}
            {assetType === 'frames' && (
                <div className="asset-viewer-controls">
                    <div className="selection-controls">
                        <button onClick={handleSelectPage}>Select Page</button>
                        <button onClick={handleSelectAll}>
                            {selectedAssets.size === assets.length ? 'Deselect All' : `Select All (${assets.length})`}
                        </button>
                        <button onClick={handleDeselectAll} disabled={selectedAssets.size === 0}>Clear Selection</button>
                    </div>
                    <div className="action-buttons">
                        <button onClick={handleSubmitForLabeling} disabled={selectedAssets.size === 0} className="action-btn">
                            Add {selectedAssets.size} to Manual Queue
                        </button>
                        <button onClick={handleAutolabelSelected} disabled={selectedAssets.size === 0} className="autolabel-btn">
                            Autolabel {selectedAssets.size} Selected
                        </button>
                        <button onClick={handleMoveToAutolabelConfirmation} disabled={selectedAssets.size === 0} className="move-to-confirmation-btn">
                            Move {selectedAssets.size} to Autolabel Confirmation
                        </button>
                    </div>
                    <div className="pagination-controls">
                         <button onClick={() => setCurrentPage(p => Math.max(1, p - 1))} disabled={currentPage === 1}>Prev</button>
                        <span>Page {currentPage} of {totalPages}</span>
                        <button onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))} disabled={currentPage === totalPages}>Next</button>
                        <select value={itemsPerPage} onChange={e => { setItemsPerPage(Number(e.target.value)); setCurrentPage(1); }}>
                            {ITEMS_PER_PAGE_OPTIONS.map(size => <option key={size} value={size}>{size}/page</option>)}
                        </select>
                    </div>
                </div>
            )}
            
            {/* Grid */}
            <div className="asset-grid">
                {paginatedAssets.map(asset => {
                    // Use the correct source type for the URL
                    const assetUrl = `${BACKEND_URL}/api/youtube/data/${videoId}/${assetSourceType}/${asset}`;
                    return (
                        <div key={asset} className={`asset-item ${selectedAssets.has(asset) ? 'selected' : ''}`} onClick={() => assetType === 'frames' && handleSelect(asset)}>
                            <img src={assetUrl} alt={asset} loading="lazy"/>
                            <div className="asset-name">{asset}</div>
                            {assetType === 'frames' && <input type="checkbox" checked={selectedAssets.has(asset)} readOnly />}
                            {assetType === 'processed' && (
                                <div className="asset-item-actions">
                                    <button onClick={(e) => { e.stopPropagation(); handleAddToTraining(asset); }}>Add to Training</button>
                                    <button onClick={(e) => { e.stopPropagation(); handleDetectSignals(asset); }}>Detect Signals</button>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default VideoAssetViewer; 