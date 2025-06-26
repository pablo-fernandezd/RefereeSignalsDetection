import React, { useState, useEffect } from 'react';
import './ModelTraining.css';

// API base URL configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const ModelTraining = () => {
    const [activeTab, setActiveTab] = useState('smart-selection');
    const [models, setModels] = useState([]);
    const [trainingSessions, setTrainingSessions] = useState([]);
    const [loading, setLoading] = useState(false);
    
    // Smart Base Model Selection
    const [selectionForm, setSelectionForm] = useState({
        model_type: 'referee',
        dataset_size: 100,
        performance_priority: 'balanced'
    });
    const [recommendations, setRecommendations] = useState([]);
    
    // Transfer Learning
    const [transferForm, setTransferForm] = useState({
        model_type: 'referee',
        base_model_id: '',
        experiment_name: '',
        epochs: 100,
        batch_size: 16,
        learning_rate: 0.001,
        optimizer: 'AdamW',
        freeze_backbone: false,
        freeze_epochs: 10
    });
    
    // Training Progress
    const [activeTraining, setActiveTraining] = useState(null);
    const [progressData, setProgressData] = useState(null);
    
    // Model Comparison
    const [comparisonModels, setComparisonModels] = useState([]);
    const [comparisonResults, setComparisonResults] = useState(null);

    useEffect(() => {
        loadModels();
        loadTrainingSessions();
    }, []);

    const loadModels = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/list`);
            const data = await response.json();
            if (data.status === 'success') {
                setModels(data.models);
            }
        } catch (error) {
            console.error('Error loading models:', error);
        }
    };

    const loadTrainingSessions = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/training/training_sessions`);
            const data = await response.json();
            if (data.status === 'success') {
                setTrainingSessions(data.sessions);
            }
        } catch (error) {
            console.error('Error loading training sessions:', error);
        }
    };

    const getSmartRecommendations = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/api/training/smart_base_selection`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(selectionForm)
            });
            const data = await response.json();
            if (data.status === 'success') {
                setRecommendations(data.recommendations);
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error getting recommendations:', error);
            alert('Failed to get recommendations');
        }
        setLoading(false);
    };

    const startTransferLearning = async () => {
        if (!transferForm.base_model_id) {
            alert('Please select a base model');
            return;
        }

        setLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/api/training/transfer_learning/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...transferForm,
                    training_config: {
                        epochs: transferForm.epochs,
                        batch_size: transferForm.batch_size,
                        learning_rate: transferForm.learning_rate,
                        optimizer: transferForm.optimizer,
                        freeze_backbone: transferForm.freeze_backbone,
                        freeze_epochs: transferForm.freeze_epochs
                    }
                })
            });
            const data = await response.json();
            if (data.status === 'success') {
                alert('Transfer learning started successfully!');
                setActiveTraining(data.session_id);
                setActiveTab('progress');
                startProgressPolling(data.session_id);
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error starting transfer learning:', error);
            alert('Failed to start transfer learning');
        }
        setLoading(false);
    };

    const startProgressPolling = (sessionId) => {
        const pollProgress = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/api/training/progress/${sessionId}`);
                const data = await response.json();
                if (data.status === 'success') {
                    setProgressData(data.progress);
                    
                    // Continue polling if training is still active
                    if (data.progress.status === 'training') {
                        setTimeout(pollProgress, 5000); // Poll every 5 seconds
                    }
                }
            } catch (error) {
                console.error('Error polling progress:', error);
            }
        };
        
        pollProgress();
    };

    const compareModels = async () => {
        if (comparisonModels.length < 2) {
            alert('Please select at least 2 models to compare');
            return;
        }

        setLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/api/training/model_comparison`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_ids: comparisonModels,
                    metrics: ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score']
                })
            });
            const data = await response.json();
            if (data.status === 'success') {
                setComparisonResults(data);
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error comparing models:', error);
            alert('Failed to compare models');
        }
        setLoading(false);
    };

    const formatDuration = (minutes) => {
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
    };

    return (
        <div className="model-training">
            <div className="header">
                <h1>Enhanced Model Training</h1>
                <p>Phase 3.1: Smart Transfer Learning & Advanced Training Features</p>
            </div>

            <div className="tabs">
                <button 
                    className={activeTab === 'smart-selection' ? 'active' : ''}
                    onClick={() => setActiveTab('smart-selection')}
                >
                    Smart Base Selection
                </button>
                <button 
                    className={activeTab === 'transfer-learning' ? 'active' : ''}
                    onClick={() => setActiveTab('transfer-learning')}
                >
                    Transfer Learning
                </button>
                <button 
                    className={activeTab === 'progress' ? 'active' : ''}
                    onClick={() => setActiveTab('progress')}
                >
                    Training Progress
                </button>
                <button 
                    className={activeTab === 'comparison' ? 'active' : ''}
                    onClick={() => setActiveTab('comparison')}
                >
                    Model Comparison
                </button>
            </div>

            {activeTab === 'smart-selection' && (
                <div className="smart-selection-tab">
                    <h3>Smart Base Model Selection</h3>
                    <p>Get AI-powered recommendations for the best base model based on your dataset and requirements.</p>
                    
                    <div className="selection-form">
                        <div className="form-row">
                            <div className="form-group">
                                <label>Model Type:</label>
                                <select 
                                    value={selectionForm.model_type}
                                    onChange={(e) => setSelectionForm({...selectionForm, model_type: e.target.value})}
                                >
                                    <option value="referee">Referee Detection</option>
                                    <option value="signal">Signal Classification</option>
                                    <option value="player">Player Detection</option>
                                    <option value="ball">Ball Detection</option>
                                </select>
                            </div>
                            
                            <div className="form-group">
                                <label>Dataset Size:</label>
                                <input 
                                    type="number"
                                    value={selectionForm.dataset_size}
                                    onChange={(e) => setSelectionForm({...selectionForm, dataset_size: parseInt(e.target.value)})}
                                    min="1"
                                    placeholder="Number of training images"
                                />
                            </div>
                            
                            <div className="form-group">
                                <label>Performance Priority:</label>
                                <select 
                                    value={selectionForm.performance_priority}
                                    onChange={(e) => setSelectionForm({...selectionForm, performance_priority: e.target.value})}
                                >
                                    <option value="speed">Speed (Fast Inference)</option>
                                    <option value="balanced">Balanced</option>
                                    <option value="accuracy">Accuracy (High Performance)</option>
                                </select>
                            </div>
                        </div>
                        
                        <button 
                            className="get-recommendations-btn"
                            onClick={getSmartRecommendations}
                            disabled={loading}
                        >
                            {loading ? 'Getting Recommendations...' : 'Get Smart Recommendations'}
                        </button>
                    </div>

                    {recommendations.length > 0 && (
                        <div className="recommendations">
                            <h4>Recommended Models</h4>
                            <div className="recommendations-grid">
                                {recommendations.map((model, index) => (
                                    <div key={model.name} className={`recommendation-card ${index === 0 ? 'best' : ''}`}>
                                        <div className="recommendation-header">
                                            <h5>{model.name}</h5>
                                            {index === 0 && <span className="best-badge">Best Match</span>}
                                        </div>
                                        <div className="recommendation-details">
                                            <p><strong>Size:</strong> {model.size}</p>
                                            <p><strong>Parameters:</strong> {model.params}</p>
                                            <p><strong>Speed:</strong> {model.speed}</p>
                                            <p className="reason"><strong>Why:</strong> {model.recommendation_reason}</p>
                                        </div>
                                        <button 
                                            className="select-model-btn"
                                            onClick={() => {
                                                setTransferForm({...transferForm, base_model_id: model.name});
                                                setActiveTab('transfer-learning');
                                            }}
                                        >
                                            Use This Model
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'transfer-learning' && (
                <div className="transfer-learning-tab">
                    <h3>Transfer Learning Configuration</h3>
                    <p>Configure and start transfer learning with a pre-trained base model.</p>
                    
                    <div className="transfer-form">
                        <div className="form-section">
                            <h4>Model Configuration</h4>
                            <div className="form-row">
                                <div className="form-group">
                                    <label>Model Type:</label>
                                    <select 
                                        value={transferForm.model_type}
                                        onChange={(e) => setTransferForm({...transferForm, model_type: e.target.value})}
                                    >
                                        <option value="referee">Referee Detection</option>
                                        <option value="signal">Signal Classification</option>
                                        <option value="player">Player Detection</option>
                                        <option value="ball">Ball Detection</option>
                                    </select>
                                </div>
                                
                                <div className="form-group">
                                    <label>Base Model:</label>
                                    <select 
                                        value={transferForm.base_model_id}
                                        onChange={(e) => setTransferForm({...transferForm, base_model_id: e.target.value})}
                                    >
                                        <option value="">Select a base model...</option>
                                        {models.filter(m => m.source === 'ultralytics' || m.source === 'legacy' || m.source === 'upload').map(model => (
                                            <option key={model.model_id} value={model.model_id}>
                                                {model.version} ({model.file_size_mb.toFixed(1)}MB) 
                                                {model.yolo_version ? ` [${model.yolo_version.toUpperCase()}]` : ''} 
                                                [{model.source === 'legacy' ? 'Legacy' : model.source === 'upload' ? 'Custom' : 'Ultralytics'}]
                                                {model.compatibility_status === 'incompatible' ? ' ⚠️' : model.compatibility_status === 'warning' ? ' ⚠️' : model.compatibility_status === 'compatible' ? ' ✅' : ''}
                                            </option>
                                        ))}
                                    </select>
                                    {transferForm.base_model_id && (
                                        <div style={{marginTop: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)'}}>
                                            {(() => {
                                                const selectedModel = models.find(m => m.model_id === transferForm.base_model_id);
                                                if (!selectedModel) return null;
                                                
                                                const status = selectedModel.compatibility_status;
                                                const issues = selectedModel.validation_results?.issues || [];
                                                
                                                if (status === 'incompatible') {
                                                    return <span style={{color: 'var(--error-color)'}}>⚠️ Incompatible: {issues.join(', ')}</span>;
                                                } else if (status === 'warning') {
                                                    return <span style={{color: 'var(--warning-color)'}}>⚠️ Warning: May have compatibility issues</span>;
                                                } else if (status === 'compatible') {
                                                    return <span style={{color: 'var(--success-color)'}}>✅ Compatible for training</span>;
                                                } else {
                                                    return <span>❓ Compatibility unknown</span>;
                                                }
                                            })()}
                                        </div>
                                    )}
                                </div>
                                
                                <div className="form-group">
                                    <label>Experiment Name:</label>
                                    <input 
                                        type="text"
                                        value={transferForm.experiment_name}
                                        onChange={(e) => setTransferForm({...transferForm, experiment_name: e.target.value})}
                                        placeholder="Optional experiment name"
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="form-section">
                            <h4>Training Parameters</h4>
                            <div className="form-row">
                                <div className="form-group">
                                    <label>Epochs:</label>
                                    <input 
                                        type="number"
                                        value={transferForm.epochs}
                                        onChange={(e) => setTransferForm({...transferForm, epochs: parseInt(e.target.value)})}
                                        min="1"
                                        max="1000"
                                    />
                                </div>
                                
                                <div className="form-group">
                                    <label>Batch Size:</label>
                                    <select 
                                        value={transferForm.batch_size}
                                        onChange={(e) => setTransferForm({...transferForm, batch_size: parseInt(e.target.value)})}
                                    >
                                        <option value={8}>8</option>
                                        <option value={16}>16</option>
                                        <option value={32}>32</option>
                                        <option value={64}>64</option>
                                    </select>
                                </div>
                                
                                <div className="form-group">
                                    <label>Learning Rate:</label>
                                    <input 
                                        type="number"
                                        step="0.0001"
                                        value={transferForm.learning_rate}
                                        onChange={(e) => setTransferForm({...transferForm, learning_rate: parseFloat(e.target.value)})}
                                        min="0.0001"
                                        max="0.1"
                                    />
                                </div>
                                
                                <div className="form-group">
                                    <label>Optimizer:</label>
                                    <select 
                                        value={transferForm.optimizer}
                                        onChange={(e) => setTransferForm({...transferForm, optimizer: e.target.value})}
                                    >
                                        <option value="AdamW">AdamW</option>
                                        <option value="Adam">Adam</option>
                                        <option value="SGD">SGD</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div className="form-section">
                            <h4>Transfer Learning Options</h4>
                            <div className="form-row">
                                <div className="form-group checkbox-group">
                                    <label>
                                        <input 
                                            type="checkbox"
                                            checked={transferForm.freeze_backbone}
                                            onChange={(e) => setTransferForm({...transferForm, freeze_backbone: e.target.checked})}
                                        />
                                        Freeze Backbone (recommended for small datasets)
                                    </label>
                                </div>
                                
                                {transferForm.freeze_backbone && (
                                    <div className="form-group">
                                        <label>Freeze Epochs:</label>
                                        <input 
                                            type="number"
                                            value={transferForm.freeze_epochs}
                                            onChange={(e) => setTransferForm({...transferForm, freeze_epochs: parseInt(e.target.value)})}
                                            min="1"
                                            max="50"
                                        />
                                    </div>
                                )}
                            </div>
                        </div>
                        
                        <button 
                            className="start-training-btn"
                            onClick={startTransferLearning}
                            disabled={loading || !transferForm.base_model_id}
                        >
                            {loading ? 'Starting Training...' : 'Start Transfer Learning'}
                        </button>
                    </div>
                </div>
            )}

            {activeTab === 'progress' && (
                <div className="progress-tab">
                    <h3>Training Progress</h3>
                    
                    {!progressData ? (
                        <div className="no-training">
                            <p>No active training session. Start a transfer learning session to see progress here.</p>
                        </div>
                    ) : (
                        <div className="progress-dashboard">
                            <div className="progress-overview">
                                <div className="progress-card">
                                    <h4>Training Status</h4>
                                    <div className={`status-badge ${progressData.status}`}>
                                        {progressData.status.toUpperCase()}
                                    </div>
                                    <div className="progress-bar">
                                        <div 
                                            className="progress-fill"
                                            style={{ width: `${progressData.progress_percentage}%` }}
                                        ></div>
                                    </div>
                                    <p>{progressData.current_epoch} / {progressData.total_epochs} epochs ({progressData.progress_percentage}%)</p>
                                </div>
                                
                                <div className="progress-card">
                                    <h4>Time</h4>
                                    <p><strong>Elapsed:</strong> {formatDuration(progressData.elapsed_time_minutes)}</p>
                                    <p><strong>Remaining:</strong> {formatDuration(progressData.estimated_remaining_minutes)}</p>
                                </div>
                                
                                <div className="progress-card">
                                    <h4>Current Metrics</h4>
                                    <div className="metrics-grid">
                                        <div>Loss: {progressData.current_metrics.train_loss.toFixed(3)}</div>
                                        <div>Val Loss: {progressData.current_metrics.val_loss.toFixed(3)}</div>
                                        <div>mAP50: {progressData.current_metrics.mAP50.toFixed(3)}</div>
                                        <div>Precision: {progressData.current_metrics.precision.toFixed(3)}</div>
                                    </div>
                                </div>
                            </div>
                            
                            <div className="training-charts">
                                <div className="chart-container">
                                    <h4>Training Loss Curve</h4>
                                    <div className="simple-chart">
                                        <p>Epoch {progressData.current_epoch}: Train Loss = {progressData.current_metrics.train_loss.toFixed(3)}</p>
                                        <p>Best Val Loss: {progressData.best_metrics.val_loss.toFixed(3)} (Epoch {progressData.best_metrics.epoch})</p>
                                    </div>
                                </div>
                                
                                <div className="chart-container">
                                    <h4>Performance Metrics</h4>
                                    <div className="metrics-comparison">
                                        <div className="metric-row">
                                            <span>Current mAP50:</span>
                                            <span>{progressData.current_metrics.mAP50.toFixed(3)}</span>
                                        </div>
                                        <div className="metric-row">
                                            <span>Best mAP50:</span>
                                            <span>{progressData.best_metrics.mAP50.toFixed(3)}</span>
                                        </div>
                                        <div className="metric-row">
                                            <span>Current F1:</span>
                                            <span>{progressData.current_metrics.f1_score.toFixed(3)}</span>
                                        </div>
                                        <div className="metric-row">
                                            <span>Best F1:</span>
                                            <span>{progressData.best_metrics.f1_score.toFixed(3)}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'comparison' && (
                <div className="comparison-tab">
                    <h3>Model Performance Comparison</h3>
                    <p>Compare the performance of multiple models side by side.</p>
                    
                    <div className="comparison-setup">
                        <h4>Select Models to Compare</h4>
                        <div className="model-selection">
                            {models.map(model => (
                                <label key={model.model_id} className="model-checkbox">
                                    <input 
                                        type="checkbox"
                                        checked={comparisonModels.includes(model.model_id)}
                                        onChange={(e) => {
                                            if (e.target.checked) {
                                                setComparisonModels([...comparisonModels, model.model_id]);
                                            } else {
                                                setComparisonModels(comparisonModels.filter(id => id !== model.model_id));
                                            }
                                        }}
                                    />
                                    <span>{model.version} ({model.model_type})</span>
                                </label>
                            ))}
                        </div>
                        
                        <button 
                            className="compare-btn"
                            onClick={compareModels}
                            disabled={loading || comparisonModels.length < 2}
                        >
                            {loading ? 'Comparing...' : 'Compare Models'}
                        </button>
                    </div>

                    {comparisonResults && (
                        <div className="comparison-results">
                            <h4>Comparison Results</h4>
                            
                            {comparisonResults.summary.best_overall && (
                                <div className="best-model-highlight">
                                    <h5>🏆 Best Overall Model</h5>
                                    <p>Model ID: {comparisonResults.summary.best_overall.model_id}</p>
                                    <p>Average Score: {comparisonResults.summary.best_overall.average_score.toFixed(3)}</p>
                                </div>
                            )}
                            
                            <div className="comparison-table">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>Type</th>
                                            <th>Size (MB)</th>
                                            <th>mAP50</th>
                                            <th>mAP50-95</th>
                                            <th>Precision</th>
                                            <th>Recall</th>
                                            <th>F1 Score</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {comparisonResults.comparison_results.map(model => (
                                            <tr key={model.model_id}>
                                                <td>{model.model_name}</td>
                                                <td>{model.model_type}</td>
                                                <td>{model.file_size_mb.toFixed(1)}</td>
                                                <td>{model.performance_metrics.mAP50?.toFixed(3) || 'N/A'}</td>
                                                <td>{model.performance_metrics.mAP50_95?.toFixed(3) || 'N/A'}</td>
                                                <td>{model.performance_metrics.precision?.toFixed(3) || 'N/A'}</td>
                                                <td>{model.performance_metrics.recall?.toFixed(3) || 'N/A'}</td>
                                                <td>{model.performance_metrics.f1_score?.toFixed(3) || 'N/A'}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ModelTraining;
