import React, { useState, useEffect, createContext, useContext } from 'react';
import UploadForm from './components/UploadForm';
import CropConfirmation from './components/CropConfirmation';
import ManualCrop from './components/ManualCrop';
import SignalConfirmation from './components/SignalConfirmation';
import YouTubeProcessing from './components/YouTubeProcessing';
import VideoAssetViewer from './components/VideoAssetViewer';
import AutoLabeledConfirmation from './components/AutoLabeledConfirmation';
import SignalConfirmationFlow from './components/SignalConfirmationFlow';
import SignalDetectionConfirmation from './components/SignalDetectionConfirmation';
import LabelingQueue from './components/LabelingQueue';
import './App.css';

const BACKEND_URL = 'http://localhost:5000';

const AppContext = createContext();
export const useAppContext = () => useContext(AppContext);

function App() {
    const [currentView, setCurrentView] = useState('dashboard');
    const [step, setStep] = useState(0);
    const [uploadData, setUploadData] = useState(null);
    const [imageFile, setImageFile] = useState(null);
    const [cropUrl, setCropUrl] = useState(null);
    const [cropFilename, setCropFilename] = useState(null);
    const [originalFilename, setOriginalFilename] = useState(null);
    const [signalResult, setSignalResult] = useState(null);
    const [cropFilenameForSignal, setCropFilenameForSignal] = useState(null);
    const [refereeCounts, setRefereeCounts] = useState({ positive: 0, negative: 0 });
    const [signalClasses, setSignalClasses] = useState([]);
    const [moveRefereeMsg, setMoveRefereeMsg] = useState('');
    const [moveSignalMsg, setMoveSignalMsg] = useState('');
    const [deleteRefereeMsg, setDeleteRefereeMsg] = useState('');
    const [deleteSignalMsg, setDeleteSignalMsg] = useState('');
    const [signalClassCounts, setSignalClassCounts] = useState({});
    const [viewingAssets, setViewingAssets] = useState(null); 
    const [pendingImageCount, setPendingImageCount] = useState(0);
    const [pendingAutolabelCount, setPendingAutolabelCount] = useState(0);
    const [pendingSignalDetectionCount, setPendingSignalDetectionCount] = useState(0);
    const [showGlobalAutoLabeledConfirmation, setShowGlobalAutoLabeledConfirmation] = useState(false);
    const [showGlobalSignalDetectionConfirmation, setShowGlobalSignalDetectionConfirmation] = useState(false);
    const [theme, setTheme] = useState(() => {
        const savedTheme = localStorage.getItem('theme');
        return savedTheme || 'light';
    });
    const [refreshTrigger, setRefreshTrigger] = useState(0);

    // Apply theme to document
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    }, [theme]);

    const toggleTheme = () => {
        setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
    };

    const fetchDashboardData = () => {
        fetch(`${BACKEND_URL}/api/referee_training_count`)
            .then(res => res.json())
            .then(data => setRefereeCounts({ positive: data.positive_count || 0, negative: data.negative_count || 0 }));
        fetch(`${BACKEND_URL}/api/signal_classes`)
            .then(res => res.json())
            .then(data => {
                // Add 'none' class for negative samples if not already present
                const classes = data.classes || [];
                if (!classes.includes('none')) {
                    classes.push('none');
                }
                setSignalClasses(classes);
            });
        fetch(`${BACKEND_URL}/api/signal_class_counts`)
            .then(res => res.json())
            .then(data => setSignalClassCounts(data || {}));
        fetch(`${BACKEND_URL}/api/pending_images`)
            .then(res => res.json())
            .then(data => setPendingImageCount(data.count || 0));
        // Fetch actual auto-labeled frames count
        fetch(`${BACKEND_URL}/api/youtube/autolabeled/all?page=1&per_page=1`)
            .then(res => res.json())
            .then(data => setPendingAutolabelCount(data.total || 0))
            .catch(error => {
                console.error('Error fetching auto-labeled count:', error);
                setPendingAutolabelCount(0);
            });
        // Fetch signal detections count
        fetch(`${BACKEND_URL}/api/youtube/signal_detections/all?page=1&per_page=1`)
            .then(res => res.json())
            .then(data => setPendingSignalDetectionCount(data.total || 0))
            .catch(error => {
                console.error('Error fetching signal detections count:', error);
                setPendingSignalDetectionCount(0);
            });
    };

    useEffect(() => {
        fetchDashboardData();
    }, []);

    const setSignalResultForApp = (data) => {
        setSignalResult(data);
        setStep(2);
        setCurrentView('upload');
    };

    const setCropFilenameForSignalApp = (filename) => {
        setCropFilenameForSignal(filename);
    };

    const resetUploadFlow = () => {
        setStep(0);
        setUploadData(null);
        setImageFile(null);
        setCropUrl(null);
        setCropFilename(null);
        setOriginalFilename(null);
        setSignalResult(null);
        setCropFilenameForSignal(null);
    };

    const handleUpload = (data, file) => {
        setUploadData(data);
        setImageFile(file);
        setOriginalFilename(data.filename);
        if (data.crop_url) {
            setCropUrl(BACKEND_URL + data.crop_url);
            setCropFilename(data.crop_filename);
            setStep(1);
        } else if (data.error && data.filename) {
            alert(data.error + ". Please manually crop.");
            setStep(3);
        }
    };

    const handleCropConfirm = async (confirmed) => {
        if (confirmed) {
            // Old simple workflow - just proceed to manual crop
            setStep(3);
        } else {
            setStep(3);
        }
    };

    const handleProceedToSignal = async (confirmed) => {
        if (confirmed) {
            // Confirm the crop and proceed to signal labeling
            const res = await fetch(`${BACKEND_URL}/api/confirm_crop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    original_filename: originalFilename,
                    crop_filename: cropFilename,
                    bbox: uploadData.bbox
                }),
            });
            const data = await res.json();
            
            // Check for duplicate detection
            if (data.status === 'warning' && data.action === 'duplicate_detected') {
                alert(`⚠️ Duplicate Image Detected\n\n${data.message}\n\nYou can continue with signal labeling, but the data will not be saved for training.`);
                // Still proceed to signal labeling even for duplicates
                if (data.crop_filename_for_signal) {
                    setCropFilenameForSignal(data.crop_filename_for_signal);
                    const signalRes = await fetch(`${BACKEND_URL}/api/process_crop_for_signal`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ crop_filename_for_signal: data.crop_filename_for_signal }),
                    });
                    const signalData = await signalRes.json();
                    setSignalResult(signalData);
                    setStep(2);
                }
                return;
            }
            
            if (data.status === 'ok' && data.crop_filename_for_signal) {
                setCropFilenameForSignal(data.crop_filename_for_signal);
                const signalRes = await fetch(`${BACKEND_URL}/api/process_crop_for_signal`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ crop_filename_for_signal: data.crop_filename_for_signal }),
                });
                const signalData = await signalRes.json();
                setSignalResult(signalData);
                setStep(2);
            } else {
                alert('Error: ' + (data.error || 'Failed to confirm crop'));
            }
        }
    };

    const handleSaveAndFinish = async (confirmed) => {
        if (confirmed) {
            // Confirm the crop and finish without signal labeling
            const res = await fetch(`${BACKEND_URL}/api/confirm_crop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    original_filename: originalFilename,
                    crop_filename: cropFilename,
                    bbox: uploadData.bbox
                }),
            });
            const data = await res.json();
            
            // Check for duplicate detection
            if (data.status === 'warning' && data.action === 'duplicate_detected') {
                alert(`⚠️ Duplicate Image Detected\n\n${data.message}\n\nThe image has not been saved to training data.`);
                resetUploadFlow();
                fetchDashboardData();
                setCurrentView('dashboard');
                return;
            }
            
            if (data.status === 'ok') {
                alert('Referee crop saved to training data!');
                resetUploadFlow();
                fetchDashboardData();
                setCurrentView('dashboard');
            } else {
                alert('Error: ' + (data.error || 'Failed to save crop'));
            }
        }
    };

    const handleSignalConfirm = async ({ correct, selected_class, signal_bbox_yolo, original_filename }) => {
        const res = await fetch(`${BACKEND_URL}/api/confirm_signal`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                crop_filename_for_signal: cropFilenameForSignal,
                correct,
                selected_class,
                signal_bbox_yolo: signal_bbox_yolo,
                original_filename: original_filename || originalFilename,
            }),
        });
        const data = await res.json();
        
        // Check for duplicate detection
        if (data.status === 'warning' && data.action === 'duplicate_detected') {
            alert(`⚠️ Duplicate Image Detected\n\n${data.message}\n\nThank you for your labeling, but the data has not been saved for training.`);
        } else if (data.status === 'success') {
            alert(data.message || 'Thank you for your feedback!');
        } else {
            alert('Error: ' + (data.error || 'Failed to confirm signal'));
        }
        
        resetUploadFlow();
        fetchDashboardData();
        setCurrentView('dashboard');
    };

    const handleManualCrop = async ({ bbox, class_id, proceedToSignal = true }) => {
        const res = await fetch(`${BACKEND_URL}/api/manual_crop`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                original_filename: originalFilename,
                bbox, 
                class_id,
                proceedToSignal
            })
        });
        const data = await res.json();
        
        if (data.status === 'ok') {
            if (data.action === 'saved_as_negative') {
                // Image was saved as negative sample (no referee)
                alert('Image saved as negative sample (no referee detected).');
                resetUploadFlow();
                fetchDashboardData();
                setCurrentView('dashboard');
            } else if (data.crop_filename_for_signal && proceedToSignal) {
                // Proceed to signal labeling
                setCropFilenameForSignal(data.crop_filename_for_signal);
                const signalRes = await fetch(`${BACKEND_URL}/api/process_crop_for_signal`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ crop_filename_for_signal: data.crop_filename_for_signal })
                });
                const signalResultData = await signalRes.json();
                setSignalResult(signalResultData);
                setStep(2);
            } else if (data.crop_filename_for_signal) {
                // Manual crop successful, but user chose to finish without signal labeling
                alert('Manual crop saved to training data!');
                resetUploadFlow();
                fetchDashboardData();
                setCurrentView('dashboard');
            } else {
                alert('Manual crop saved successfully!');
                resetUploadFlow();
                fetchDashboardData();
                setCurrentView('dashboard');
            }
        } else {
            alert('Error saving manual crop: ' + (data.error || 'Unknown error'));
            resetUploadFlow();
        }
    };

    const handleMoveReferee = async () => {
        setMoveRefereeMsg('Moving...');
        const res = await fetch(`${BACKEND_URL}/api/move_referee_training`, { method: 'POST' });
        const data = await res.json();
        setMoveRefereeMsg(`Moved ${data.moved.length} files to ${data.dst}`);
        fetchDashboardData();
    };

    const handleMoveSignal = async () => {
        setMoveSignalMsg('Moving...');
        const res = await fetch(`${BACKEND_URL}/api/move_signal_training`, { method: 'POST' });
        const data = await res.json();
        setMoveSignalMsg(`Moved ${data.moved.length} files to ${data.dst}`);
    };

    const handleDeleteRefereeTrainingData = async () => {
        if (!window.confirm('Are you sure you want to delete all referee training data? This action cannot be undone.')) {
            return;
        }
        
        setDeleteRefereeMsg('Deleting...');
        try {
            const res = await fetch(`${BACKEND_URL}/api/delete_referee_training_data`, { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'success') {
                setDeleteRefereeMsg(`Deleted ${data.deleted_count} referee training files`);
                fetchDashboardData(); // Refresh counts
            } else {
                setDeleteRefereeMsg(`Error: ${data.error || 'Failed to delete referee training data'}`);
            }
        } catch (error) {
            setDeleteRefereeMsg(`Error: ${error.message}`);
        }
    };

    const handleDeleteSignalTrainingData = async () => {
        if (!window.confirm('Are you sure you want to delete all signal training data? This action cannot be undone.')) {
            return;
        }
        
        setDeleteSignalMsg('Deleting...');
        try {
            const res = await fetch(`${BACKEND_URL}/api/delete_signal_training_data`, { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'success') {
                setDeleteSignalMsg(`Deleted ${data.deleted_count} signal training files`);
                fetchDashboardData(); // Refresh counts
            } else {
                setDeleteSignalMsg(`Error: ${data.error || 'Failed to delete signal training data'}`);
            }
        } catch (error) {
            setDeleteSignalMsg(`Error: ${error.message}`);
        }
    };

    const handleNavigate = (screen) => {
        resetUploadFlow();
        setViewingAssets(null);
        
        // If we're in a confirmation view and user clicks Dashboard, properly handle the back navigation
        if (showGlobalAutoLabeledConfirmation || showGlobalSignalDetectionConfirmation) {
            setShowGlobalAutoLabeledConfirmation(false);
            setShowGlobalSignalDetectionConfirmation(false);
        }
        
        if (screen === 'dashboard') {
            fetchDashboardData();
        } else if (screen === 'youtube' || screen === 'assets') {
            // Trigger video list refresh when navigating to YouTube Processing or Video Assets
            setRefreshTrigger(prev => prev + 1);
        }
        setCurrentView(screen);
    };

    const handleViewAssets = (videoId, assetType) => {
        setViewingAssets({ videoId, assetType });
    };

    const handleExitAssetView = () => {
        setViewingAssets(null);
    };
    
    const handleSignalDetection = (data) => {
        setSignalResultForApp(data.result);
        setCropFilenameForSignalApp(data.crop_filename_for_signal);
        setCurrentView('upload');
    };

    const renderScreen = () => {
        if (showGlobalAutoLabeledConfirmation) {
            return (
                <AutoLabeledConfirmation
                    videoId={null}
                    isGlobalView={true}
                    onBack={() => {
                        setShowGlobalAutoLabeledConfirmation(false);
                        fetchDashboardData(); // Refresh dashboard data
                        setCurrentView('dashboard'); // Ensure we're on dashboard view
                    }}
                />
            );
        }

        if (showGlobalSignalDetectionConfirmation) {
            return (
                <SignalDetectionConfirmation
                    videoId={null}
                    isGlobalView={true}
                    onBack={() => {
                        setShowGlobalSignalDetectionConfirmation(false);
                        fetchDashboardData(); // Refresh dashboard data
                        setCurrentView('dashboard'); // Ensure we're on dashboard view
                    }}
                />
            );
        }

        if (viewingAssets) {
            if (viewingAssets.assetType === 'autolabeled_confirmation') {
                return (
                    <AutoLabeledConfirmation
                        videoId={viewingAssets.videoId}
                        onBack={handleExitAssetView}
                    />
                );
            }
            if (viewingAssets.assetType === 'signal_confirmation') {
                return (
                    <SignalConfirmationFlow
                        videoId={viewingAssets.videoId}
                        onBack={handleExitAssetView}
                    />
                );
            }
            if (viewingAssets.assetType === 'signal_detection_confirmation') {
                return (
                    <SignalDetectionConfirmation
                        videoId={viewingAssets.videoId}
                        onBack={handleExitAssetView}
                    />
                );
            }
            return (
                <VideoAssetViewer
                    videoId={viewingAssets.videoId}
                    assetType={viewingAssets.assetType}
                    onBack={handleExitAssetView}
                    onSignalDetected={handleSignalDetection}
                />
            );
        }

        switch (currentView) {
            case 'dashboard':
                return <Dashboard />;
            case 'upload':
                return <UploadFlow />;
            case 'youtube':
                return <YouTubeProcessing onViewAssets={handleViewAssets} refreshTrigger={refreshTrigger} />;
            case 'assets':
                return <YouTubeProcessing onViewAssets={handleViewAssets} showOnlyVideoList={true} refreshTrigger={refreshTrigger} />;
            case 'labelingQueue':
                return <LabelingQueue onBack={() => handleNavigate('dashboard')} />;
            default:
                return <Dashboard />;
        }
    };

    const UploadFlow = () => {
        switch (step) {
            case 1: return (
                <CropConfirmation 
                    cropUrl={cropUrl} 
                    onConfirm={handleCropConfirm}
                    onProceedToSignal={handleProceedToSignal}
                    onSaveAndFinish={handleSaveAndFinish}
                />
            );
            case 2: return <SignalConfirmation {...signalResult} cropFilenameForSignal={cropFilenameForSignal} onConfirm={handleSignalConfirm} onCancel={() => setStep(3)} originalFilename={originalFilename}/>;
            case 3: return <ManualCrop imageFile={imageFile} onSubmit={handleManualCrop} onCancel={resetUploadFlow} />;
            default: return <UploadForm onUpload={handleUpload} />;
        }
    };

    const Dashboard = () => {
        const totalReferee = refereeCounts.positive + refereeCounts.negative;
        const totalSignals = Object.values(signalClassCounts || {}).reduce((a, b) => a + b, 0);

        return (
            <div className="dashboard">
                <h2>Dashboard - Referee & Signal Detection</h2>
                <div className="main-actions">
                    <button className="nav-button" onClick={() => handleNavigate('labelingQueue')}>
                        Label Pending Frames ({pendingImageCount})
                    </button>
                    <button className="nav-button autolabel" onClick={() => setShowGlobalAutoLabeledConfirmation(true)}>
                        Confirm Autolabeled Frames ({pendingAutolabelCount})
                    </button>
                    <button className="nav-button signal-detection" onClick={() => setShowGlobalSignalDetectionConfirmation(true)}>
                        Confirm Signal Detections ({pendingSignalDetectionCount})
                    </button>
                </div>
                <div className="stats-container">
                    <div className="stat-card">
                        <h3>Referee Training Set ({totalReferee} Total)</h3>
                        <ul className="signal-classes">
                            <li>
                                <span>Referees Detected</span>
                                <span className="count">
                                    {refereeCounts.positive} ({totalReferee > 0 ? ((refereeCounts.positive / totalReferee) * 100).toFixed(1) : 0}%)
                                </span>
                            </li>
                            <li>
                                <span>Negative Samples</span>
                                <span className="count">
                                    {refereeCounts.negative} ({totalReferee > 0 ? ((refereeCounts.negative / totalReferee) * 100).toFixed(1) : 0}%)
                                </span>
                            </li>
                        </ul>
                        <button onClick={handleMoveReferee} className="action-button">
                            Move referee images to global training folder
                        </button>
                        <button onClick={handleDeleteRefereeTrainingData} className="action-button delete-button" style={{backgroundColor: '#dc3545', marginTop: '10px'}}>
                            🗑️ Delete All Referee Training Data
                        </button>
                        <div className="message">{moveRefereeMsg}</div>
                        {deleteRefereeMsg && <div className="message" style={{color: deleteRefereeMsg.startsWith('Error') ? '#dc3545' : '#28a745'}}>{deleteRefereeMsg}</div>}
                    </div>
                    
                    <div className="stat-card">
                        <h3>Signal Training Set ({totalSignals} Total)</h3>
                        <ul className="signal-classes">
                            {signalClasses.map(cls => (
                                <li key={cls}>
                                    <span>
                                        {cls === 'none' ? 'No Signal (Negative Samples)' : cls.charAt(0).toUpperCase() + cls.slice(1)}
                                    </span>
                                    <span className="count">
                                        {signalClassCounts[cls] || 0} ({totalSignals > 0 ? (((signalClassCounts[cls] || 0) / totalSignals) * 100).toFixed(1) : 0}%)
                                    </span>
                                </li>
                            ))}
                        </ul>
                        <button onClick={handleMoveSignal} className="action-button">
                            Move signal images to global training folder
                        </button>
                        <button onClick={handleDeleteSignalTrainingData} className="action-button delete-button" style={{backgroundColor: '#dc3545', marginTop: '10px'}}>
                            🗑️ Delete All Signal Training Data
                        </button>
                        <div className="message">{moveSignalMsg}</div>
                        {deleteSignalMsg && <div className="message" style={{color: deleteSignalMsg.startsWith('Error') ? '#dc3545' : '#28a745'}}>{deleteSignalMsg}</div>}
                    </div>
                </div>
            </div>
        );
    };

    return (
        <AppContext.Provider value={{ setSignalResultForApp, setCropFilenameForSignalApp }}>
            <div className="App">
                <nav className="nav-bar">
                    <div className="nav-container">
                        <h1 className="nav-title">Referee Detection System</h1>
                        <div className="nav-buttons">
                            <button 
                                className={`nav-button ${currentView === 'dashboard' ? 'active' : ''}`}
                                onClick={() => handleNavigate('dashboard')}
                            >
                                Dashboard
                            </button>
                            <button 
                                className={`nav-button ${currentView === 'upload' ? 'active' : ''}`}
                                onClick={() => handleNavigate('upload')}
                            >
                                Upload Image
                            </button>
                            <button 
                                className={`nav-button ${currentView === 'youtube' ? 'active' : ''}`}
                                onClick={() => handleNavigate('youtube')}
                            >
                                YouTube Processing
                            </button>
                            <button 
                                className={`nav-button ${currentView === 'assets' ? 'active' : ''}`}
                                onClick={() => handleNavigate('assets')}
                            >
                                Video Assets
                            </button>
                            <button 
                                className="theme-toggle"
                                onClick={toggleTheme}
                                title={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
                            >
                                {theme === 'light' ? '🌙' : '☀️'}
                            </button>
                        </div>
                    </div>
                </nav>

                <div className="main-content">
                    {renderScreen()}
                </div>
            </div>
        </AppContext.Provider>
    );
}

export default App;
