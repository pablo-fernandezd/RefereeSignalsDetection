import React, { useState, useEffect, useCallback } from 'react';
import './ModelTraining.css';

const ModelTraining = () => {
  // State management
  const [modelType, setModelType] = useState('referee');
  const [datasetStats, setDatasetStats] = useState(null);
  const [modelVersions, setModelVersions] = useState([]);
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Dataset splitting state (Roboflow-style single bar)
  const [trainSplit, setTrainSplit] = useState(0.7);
  const [valSplit, setValSplit] = useState(0.2);
  const [testSplit, setTestSplit] = useState(0.1);
  const [datasetPrepared, setDatasetPrepared] = useState(false);
  const [preparedDataset, setPreparedDataset] = useState(null);
  
  // Training configuration state
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 100,
    batch_size: 16,
    learning_rate: 0.001,
    optimizer: 'AdamW',
    use_pretrained: true,
    data_augmentation: true
  });
  
  // Data augmentation state
  const [augmentationOptions, setAugmentationOptions] = useState({});
  const [augmentationConfig, setAugmentationConfig] = useState({});
  const [showAugmentationDetails, setShowAugmentationDetails] = useState(false);
  
  // Signal classes management state
  const [signalClasses, setSignalClasses] = useState([]);
  const [newClassName, setNewClassName] = useState('');
  const [editingClasses, setEditingClasses] = useState(false);
  
  // Real-time updates
  const [autoRefresh, setAutoRefresh] = useState(false);
  
  // Tooltip state
  const [activeTooltip, setActiveTooltip] = useState(null);

  // API base URL
  const API_BASE = 'http://localhost:5000/api';

  // Fetch functions
  const fetchDatasetStats = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/dataset_stats/${modelType}`);
      const data = await response.json();
      if (data.status === 'success') {
        setDatasetStats(data.stats);
      } else {
        setError(data.error || 'Failed to fetch dataset stats');
      }
    } catch (err) {
      setError('Network error fetching dataset stats');
    }
  }, [modelType]);

  const fetchModelVersions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/model_versions?model_type=${modelType}`);
      const data = await response.json();
      if (data.status === 'success') {
        setModelVersions(data.versions);
      } else {
        setError(data.error || 'Failed to fetch model versions');
      }
    } catch (err) {
      setError('Network error fetching model versions');
    }
  }, [modelType]);

  const fetchTrainingSessions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/training_sessions`);
      const data = await response.json();
      if (data.status === 'success') {
        setTrainingSessions(data.sessions);
        // Find active session
        const active = data.sessions.find(s => s.status === 'training');
        setActiveSession(active);
      } else {
        setError(data.error || 'Failed to fetch training sessions');
      }
    } catch (err) {
      setError('Network error fetching training sessions');
    }
  }, []);

  const fetchTrainingMetrics = useCallback(async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE}/training_metrics/${sessionId}`);
      const data = await response.json();
      if (data.status === 'success') {
        setTrainingMetrics(data);
      } else {
        setError(data.error || 'Failed to fetch training metrics');
      }
    } catch (err) {
      setError('Network error fetching training metrics');
    }
  }, []);

  const fetchSignalClasses = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/model_classes/${modelType}`);
      const data = await response.json();
      if (data.status === 'success') {
        setSignalClasses(data.classes);
      }
    } catch (err) {
      console.error('Failed to fetch signal classes:', err);
    }
  }, [modelType]);

  const fetchAugmentationOptions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/augmentation_options`);
      const data = await response.json();
      if (data.status === 'success') {
        setAugmentationOptions(data.augmentations);
        setAugmentationConfig(data.defaults);
      }
    } catch (err) {
      console.error('Failed to fetch augmentation options:', err);
    }
  }, []);

  // Initial data loading
  useEffect(() => {
    fetchDatasetStats();
    fetchModelVersions();
    fetchTrainingSessions();
    fetchSignalClasses();
    fetchAugmentationOptions();
  }, [fetchDatasetStats, fetchModelVersions, fetchTrainingSessions, fetchSignalClasses, fetchAugmentationOptions]);

  // Auto-refresh for active training sessions
  useEffect(() => {
    let interval;
    if (autoRefresh && activeSession) {
      interval = setInterval(() => {
        fetchTrainingSessions();
        fetchTrainingMetrics(activeSession.session_id);
      }, 5000); // Refresh every 5 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, activeSession, fetchTrainingSessions, fetchTrainingMetrics]);

  // Roboflow-style split handler with proper handle positioning and null safety
  const handleSplitChange = (handleType, event) => {
    // Prevent errors if event or target is null
    if (!event || !event.target || !event.target.parentElement) {
      return;
    }
    
    try {
      const rect = event.target.parentElement.getBoundingClientRect();
      if (!rect || rect.width === 0) {
        return;
      }
      
      const x = event.clientX - rect.left;
      const percentage = Math.max(0, Math.min(1, x / rect.width));
      
      if (handleType === 'train-val') {
        // First handle controls train/val boundary
        const newTrainSplit = percentage;
        const remaining = Math.max(0, 1.0 - newTrainSplit);
        
        // Prevent division by zero
        const currentValTest = valSplit + testSplit;
        const valTestRatio = currentValTest > 0 ? valSplit / currentValTest : 0.5;
        
        const newValSplit = remaining * valTestRatio;
        const newTestSplit = remaining * (1 - valTestRatio);
        
        setTrainSplit(Math.max(0, Math.min(1, newTrainSplit)));
        setValSplit(Math.max(0, Math.min(1, newValSplit)));
        setTestSplit(Math.max(0, Math.min(1, newTestSplit)));
        
      } else if (handleType === 'val-test') {
        // Second handle controls val/test boundary
        const valTestBoundary = percentage;
        const availableForValTest = Math.max(0, 1.0 - trainSplit);
        const newValSplit = Math.max(0, Math.min(availableForValTest, (valTestBoundary - trainSplit)));
        const newTestSplit = Math.max(0, availableForValTest - newValSplit);
        
        setValSplit(Math.max(0, newValSplit));
        setTestSplit(Math.max(0, newTestSplit));
      }
    } catch (error) {
      console.warn('Error in split change:', error);
    }
  };

  // Handle mouse drag for split bar with improved event handling
  const handleMouseDown = (handleType) => (event) => {
    if (!event) return;
    
    event.preventDefault();
    event.stopPropagation();
    
    const handleMouseMove = (e) => {
      if (!e) return;
      
      // Create a synthetic event that targets the split bar container
      const splitBarContainer = document.querySelector('.split-bar-track');
      if (!splitBarContainer) return;
      
      const syntheticEvent = {
        clientX: e.clientX,
        target: {
          parentElement: splitBarContainer
        }
      };
      
      handleSplitChange(handleType, syntheticEvent);
    };
    
    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  // Handle direct click on split bar
  const handleSplitBarClick = (event) => {
    if (!event || !event.target) return;
    
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    
    const x = event.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    
    // Determine which handle is closer and move that one
    const trainEnd = trainSplit;
    const valEnd = trainSplit + valSplit;
    
    const distanceToTrainHandle = Math.abs(percentage - trainEnd);
    const distanceToValHandle = Math.abs(percentage - valEnd);
    
    if (distanceToTrainHandle < distanceToValHandle) {
      // Move train handle
      const syntheticEvent = {
        clientX: event.clientX,
        target: {
          parentElement: event.currentTarget
        }
      };
      handleSplitChange('train-val', syntheticEvent);
    } else {
      // Move val handle
      const syntheticEvent = {
        clientX: event.clientX,
        target: {
          parentElement: event.currentTarget
        }
      };
      handleSplitChange('val-test', syntheticEvent);
    }
  };

  // Prepare dataset
  const prepareDataset = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/prepare_dataset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_type: modelType,
          train_split: trainSplit,
          val_split: valSplit,
          test_split: testSplit
        })
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setPreparedDataset(data.dataset);
        setDatasetPrepared(true);
        setError(null);
      } else {
        setError(data.error || 'Failed to prepare dataset');
      }
    } catch (err) {
      setError('Network error preparing dataset');
    } finally {
      setLoading(false);
    }
  };

  // Start training
  const startTraining = async () => {
    if (!datasetPrepared) {
      setError('Please prepare dataset first');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/start_training`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_type: modelType,
          dataset_id: preparedDataset.dataset_id,
          augmentation_config: augmentationConfig,
          ...trainingConfig
        })
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setError(null);
        setAutoRefresh(true);
        fetchTrainingSessions();
      } else {
        setError(data.error || 'Failed to start training');
      }
    } catch (err) {
      setError('Network error starting training');
    } finally {
      setLoading(false);
    }
  };

  // Deploy model version
  const deployModel = async (versionId) => {
    try {
      const response = await fetch(`${API_BASE}/deploy_model/${versionId}`, {
        method: 'POST'
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        fetchModelVersions();
        setError(null);
        alert(data.message + '\n\n' + data.description);
      } else {
        setError(data.error || 'Failed to deploy model');
      }
    } catch (err) {
      setError('Network error deploying model');
    }
  };

  // Download model
  const downloadModel = async (versionId) => {
    try {
      const response = await fetch(`${API_BASE}/download_model/${versionId}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${versionId}.pt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        setError('Failed to download model');
      }
    } catch (err) {
      setError('Network error downloading model');
    }
  };

  // Update signal classes
  const updateSignalClasses = async () => {
    try {
      const response = await fetch(`${API_BASE}/update_signal_classes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          classes: signalClasses
        })
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setEditingClasses(false);
        setError(null);
        fetchDatasetStats(); // Refresh stats to show updated classes
      } else {
        setError(data.error || 'Failed to update signal classes');
      }
    } catch (err) {
      setError('Network error updating signal classes');
    }
  };

  // Add new signal class
  const addSignalClass = () => {
    if (newClassName.trim() && !signalClasses.includes(newClassName.trim())) {
      setSignalClasses([...signalClasses, newClassName.trim()]);
      setNewClassName('');
    }
  };

  // Remove signal class
  const removeSignalClass = (className) => {
    setSignalClasses(signalClasses.filter(c => c !== className));
  };

  // Tooltip component
  const Tooltip = ({ content, children, id }) => (
    <div 
      className="tooltip-container"
      onMouseEnter={() => setActiveTooltip(id)}
      onMouseLeave={() => setActiveTooltip(null)}
    >
      {children}
      {activeTooltip === id && (
        <div className="tooltip">
          {content}
        </div>
      )}
    </div>
  );

  // Calculate split positions for visualization
  const trainWidth = trainSplit * 100;
  const valWidth = valSplit * 100;
  const testWidth = testSplit * 100;

  // Loading component
  const LoadingSpinner = ({ message = "Loading..." }) => (
    <div className="loading-container">
      <div className="loading-spinner">
        <div className="spinner-ring"></div>
        <div className="spinner-ring"></div>
        <div className="spinner-ring"></div>
        <div className="spinner-ring"></div>
      </div>
      <p className="loading-message">{message}</p>
    </div>
  );

  return (
    <div className="model-training">
      <div className="training-header">
        <h2>Model Training Workflow</h2>
        <div className="model-type-selector">
          <label>Model Type:</label>
          <select 
            value={modelType} 
            onChange={(e) => setModelType(e.target.value)}
            disabled={loading}
          >
            <option value="referee">Referee Detection</option>
            <option value="signal">Signal Detection</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className="training-content">
        {/* Dataset Statistics */}
        <div className="section dataset-section">
          <h3>Dataset Statistics</h3>
          {datasetStats ? (
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Total Images:</span>
                <span className="stat-value">{datasetStats.total_images}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Total Labels:</span>
                <span className="stat-value">{datasetStats.total_labels}</span>
              </div>
              {Object.keys(datasetStats.classes).length > 0 && (
                <div className="class-distribution">
                  <h4>Class Distribution:</h4>
                  {Object.entries(datasetStats.classes).map(([className, count]) => (
                    <div key={className} className="class-item">
                      <span>{className}:</span>
                      <span>{count}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <LoadingSpinner message="Loading dataset statistics..." />
          )}
        </div>

        {/* Signal Classes Management */}
        {modelType === 'signal' && (
          <div className="section classes-section">
            <h3>Signal Classes Management</h3>
            <div className="classes-container">
              {editingClasses ? (
                <div className="classes-editor">
                  <div className="current-classes">
                    {signalClasses.map((className, index) => (
                      <div key={index} className="class-tag">
                        <span>{className}</span>
                        <button 
                          onClick={() => removeSignalClass(className)}
                          className="remove-class"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                  <div className="add-class">
                    <input
                      type="text"
                      value={newClassName}
                      onChange={(e) => setNewClassName(e.target.value)}
                      placeholder="New class name"
                      onKeyPress={(e) => e.key === 'Enter' && addSignalClass()}
                    />
                    <button onClick={addSignalClass}>Add</button>
                  </div>
                  <div className="classes-actions">
                    <button onClick={updateSignalClasses} className="save-classes">
                      Save Classes
                    </button>
                    <button 
                      onClick={() => setEditingClasses(false)}
                      className="cancel-edit"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className="classes-display">
                  <div className="current-classes">
                    {signalClasses.map((className, index) => (
                      <span key={index} className="class-tag">{className}</span>
                    ))}
                  </div>
                  <button 
                    onClick={() => setEditingClasses(true)}
                    className="edit-classes"
                  >
                    Edit Classes
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Roboflow-style Dataset Splitting */}
        <div className="section splitting-section">
          <h3>Dataset Splitting</h3>
          <div className="split-container">
            <div className="roboflow-split-bar">
              <div className="split-labels">
                <div className="split-label train-label">
                  <span>Train</span>
                  <span>{Math.round(trainSplit * 100)}%</span>
                </div>
                <div className="split-label val-label">
                  <span>Valid</span>
                  <span>{Math.round(valSplit * 100)}%</span>
                </div>
                <div className="split-label test-label">
                  <span>Test</span>
                  <span>{Math.round(testSplit * 100)}%</span>
                </div>
              </div>
              
              <div className="split-bar-container">
                <div 
                  className="split-bar-track"
                  onClick={handleSplitBarClick}
                >
                  <div 
                    className="split-segment train-segment"
                    style={{ width: `${trainWidth}%` }}
                  />
                  <div 
                    className="split-segment val-segment"
                    style={{ width: `${valWidth}%` }}
                  />
                  <div 
                    className="split-segment test-segment"
                    style={{ width: `${testWidth}%` }}
                  />
                  
                  {/* Draggable handles - only show if split > 0 */}
                  {trainSplit > 0 && (
                    <div
                      className="split-handle train-handle"
                      style={{ left: `${Math.max(0, trainWidth)}%` }}
                      onMouseDown={handleMouseDown('train-val')}
                    />
                  )}
                  {valSplit > 0 && trainSplit + valSplit < 1 && (
                    <div
                      className="split-handle val-handle"
                      style={{ left: `${Math.max(0, Math.min(100, trainWidth + valWidth))}%` }}
                      onMouseDown={handleMouseDown('val-test')}
                    />
                  )}
                </div>
              </div>
              
              <div className="split-numbers">
                <span className={trainSplit === 0 ? 'zero-split' : ''}>
                  Train: {Math.round(trainSplit * 100)}%
                </span>
                <span className={valSplit === 0 ? 'zero-split' : ''}>
                  Valid: {Math.round(valSplit * 100)}%
                </span>
                <span className={testSplit === 0 ? 'zero-split' : ''}>
                  Test: {Math.round(testSplit * 100)}%
                </span>
              </div>
            </div>
            
            {preparedDataset && (
              <div className="prepared-dataset-info">
                <h4>Prepared Dataset: {preparedDataset.dataset_id}</h4>
                <div className="dataset-splits">
                  <div>Train: {preparedDataset.splits.train.count} images</div>
                  <div>Validation: {preparedDataset.splits.val.count} images</div>
                  <div>Test: {preparedDataset.splits.test.count} images</div>
                </div>
              </div>
            )}
            
            <button 
              onClick={prepareDataset}
              disabled={loading}
              className="prepare-dataset-btn"
            >
              {loading ? 'Preparing...' : 'Prepare Dataset'}
            </button>
          </div>
        </div>

        {/* Data Augmentation Configuration */}
        <div className="section augmentation-section">
          <h3>
            Data Augmentation
            <Tooltip 
              content="Data augmentation techniques help improve model generalization by creating variations of training images"
              id="augmentation-tooltip"
            >
              <span className="info-icon">ℹ️</span>
            </Tooltip>
          </h3>
          
          <div className="augmentation-toggle">
            <label>
              <input
                type="checkbox"
                checked={trainingConfig.data_augmentation}
                onChange={(e) => setTrainingConfig({
                  ...trainingConfig,
                  data_augmentation: e.target.checked
                })}
              />
              Enable Data Augmentation
            </label>
            <button 
              onClick={() => setShowAugmentationDetails(!showAugmentationDetails)}
              className="toggle-details-btn"
            >
              {showAugmentationDetails ? 'Hide Details' : 'Show Details'}
            </button>
          </div>

          {showAugmentationDetails && trainingConfig.data_augmentation && (
            <div className="augmentation-options">
              {Object.entries(augmentationOptions).map(([key, info]) => (
                <div key={key} className="augmentation-option">
                  <label>
                    <input
                      type="checkbox"
                      checked={augmentationConfig[key] || false}
                      onChange={(e) => setAugmentationConfig({
                        ...augmentationConfig,
                        [key]: e.target.checked
                      })}
                    />
                    <span className="option-name">{info.name}</span>
                  </label>
                  <Tooltip content={info.description} id={`aug-${key}`}>
                    <span className="info-icon">ℹ️</span>
                  </Tooltip>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Training Configuration */}
        <div className="section config-section">
          <h3>Training Configuration</h3>
          <div className="config-grid">
            <div className="config-item">
              <label>
                Epochs:
                <Tooltip 
                  content="Number of complete passes through the training dataset. More epochs = longer training but potentially better results."
                  id="epochs-tooltip"
                >
                  <span className="info-icon">ℹ️</span>
                </Tooltip>
              </label>
              <input
                type="number"
                value={trainingConfig.epochs}
                onChange={(e) => setTrainingConfig({
                  ...trainingConfig,
                  epochs: parseInt(e.target.value)
                })}
                min="1"
                max="1000"
              />
            </div>
            
            <div className="config-item">
              <label>
                Batch Size:
                <Tooltip 
                  content="Number of images processed together. Larger batches = more memory usage but potentially more stable training."
                  id="batch-tooltip"
                >
                  <span className="info-icon">ℹ️</span>
                </Tooltip>
              </label>
              <select
                value={trainingConfig.batch_size}
                onChange={(e) => setTrainingConfig({
                  ...trainingConfig,
                  batch_size: parseInt(e.target.value)
                })}
              >
                <option value={4}>4</option>
                <option value={8}>8</option>
                <option value={16}>16</option>
                <option value={32}>32</option>
              </select>
            </div>
            
            <div className="config-item">
              <label>
                Learning Rate:
                <Tooltip 
                  content="Controls how much the model weights are updated during training. Lower = more stable but slower learning."
                  id="lr-tooltip"
                >
                  <span className="info-icon">ℹ️</span>
                </Tooltip>
              </label>
              <select
                value={trainingConfig.learning_rate}
                onChange={(e) => setTrainingConfig({
                  ...trainingConfig,
                  learning_rate: parseFloat(e.target.value)
                })}
              >
                <option value={0.0001}>0.0001</option>
                <option value={0.0005}>0.0005</option>
                <option value={0.001}>0.001</option>
                <option value={0.005}>0.005</option>
              </select>
            </div>
            
            <div className="config-item">
              <label>
                Optimizer:
                <Tooltip 
                  content="AdamW: Best for most cases, handles weight decay well. Adam: Good general purpose. SGD: Traditional, requires careful tuning."
                  id="optimizer-tooltip"
                >
                  <span className="info-icon">ℹ️</span>
                </Tooltip>
              </label>
              <select
                value={trainingConfig.optimizer}
                onChange={(e) => setTrainingConfig({
                  ...trainingConfig,
                  optimizer: e.target.value
                })}
              >
                <option value="AdamW">AdamW (Recommended)</option>
                <option value="Adam">Adam</option>
                <option value="SGD">SGD</option>
              </select>
            </div>
            
            <div className="config-item checkbox-item">
              <label>
                <input
                  type="checkbox"
                  checked={trainingConfig.use_pretrained}
                  onChange={(e) => setTrainingConfig({
                    ...trainingConfig,
                    use_pretrained: e.target.checked
                  })}
                />
                Use Pretrained Model
                <Tooltip 
                  content="Start with a model already trained on general objects. Recommended for faster training and better results."
                  id="pretrained-tooltip"
                >
                  <span className="info-icon">ℹ️</span>
                </Tooltip>
              </label>
            </div>
          </div>
          
          <button 
            onClick={startTraining}
            disabled={loading || !datasetPrepared}
            className="start-training-btn"
          >
            {loading ? 'Starting...' : 'Start Training'}
          </button>
        </div>

        {/* Active Training Session */}
        {activeSession && (
          <div className="section active-training-section">
            <h3>Active Training Session</h3>
            <div className="session-info">
              <div className="session-header">
                <h4>{activeSession.session_id}</h4>
                <div className="auto-refresh">
                  <label>
                    <input
                      type="checkbox"
                      checked={autoRefresh}
                      onChange={(e) => setAutoRefresh(e.target.checked)}
                    />
                    Auto-refresh
                  </label>
                </div>
              </div>
              
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ width: `${activeSession.progress_percentage}%` }}
                ></div>
                <span className="progress-text">
                  Epoch {activeSession.current_epoch}/{activeSession.total_epochs} 
                  ({activeSession.progress_percentage}%)
                </span>
              </div>
              
              <div className="metrics-grid">
                <div className="metric-item">
                  <span className="metric-label">Training Loss:</span>
                  <span className="metric-value">
                    {activeSession.current_metrics.train_loss.toFixed(4)}
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Validation Loss:</span>
                  <span className="metric-value">
                    {activeSession.current_metrics.val_loss.toFixed(4)}
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Precision:</span>
                  <span className="metric-value">
                    {(activeSession.current_metrics.precision * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Recall:</span>
                  <span className="metric-value">
                    {(activeSession.current_metrics.recall * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">F1 Score:</span>
                  <span className="metric-value">
                    {(activeSession.current_metrics.f1_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">mAP@50:</span>
                  <span className="metric-value">
                    {(activeSession.current_metrics.mAP50 * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              
              <div className="duration-info">
                Training Duration: {activeSession.duration_minutes} minutes
              </div>
            </div>
          </div>
        )}

        {/* Model Versions */}
        <div className="section versions-section">
          <h3>
            Model Versions
            <Tooltip 
              content="Compare different trained models. Deploy to make a model active for auto-labeling. Download to save model files locally."
              id="versions-tooltip"
            >
              <span className="info-icon">ℹ️</span>
            </Tooltip>
          </h3>
          <div className="versions-list">
            {modelVersions.map((version) => (
              <div key={version.version_id} className="version-card">
                <div className="version-header">
                  <h4>{version.version_id}</h4>
                  {version.is_active && <span className="active-badge">Active</span>}
                  <span className="version-date">
                    {new Date(version.created_at).toLocaleDateString()}
                  </span>
                </div>
                
                <div className="version-metrics">
                  <div className="metrics-row">
                    <div className="metric">
                      <span>Precision:</span>
                      <span>{(version.performance_metrics.precision * 100).toFixed(1)}%</span>
                    </div>
                    <div className="metric">
                      <span>Recall:</span>
                      <span>{(version.performance_metrics.recall * 100).toFixed(1)}%</span>
                    </div>
                    <div className="metric">
                      <span>F1:</span>
                      <span>{(version.performance_metrics.f1_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="metric">
                      <span>mAP@50:</span>
                      <span>{(version.performance_metrics.mAP50 * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  
                  <div className="training-info">
                    <span>Epochs: {version.performance_metrics.epochs_trained}</span>
                    <span>Batch Size: {version.training_config.batch_size}</span>
                    <span>LR: {version.training_config.learning_rate}</span>
                    <span>Time: {version.performance_metrics.training_time_minutes}min</span>
                    <span>Size: {version.file_size_mb}MB</span>
                  </div>
                </div>
                
                <div className="version-actions">
                  {!version.is_active && (
                    <Tooltip 
                      content="Make this model active for auto-labeling and inference"
                      id={`deploy-${version.version_id}`}
                    >
                      <button 
                        onClick={() => deployModel(version.version_id)}
                        className="deploy-btn"
                      >
                        Deploy
                      </button>
                    </Tooltip>
                  )}
                  <Tooltip 
                    content="Download model weights file to your computer"
                    id={`download-${version.version_id}`}
                  >
                    <button 
                      onClick={() => downloadModel(version.version_id)}
                      className="download-btn"
                    >
                      Download
                    </button>
                  </Tooltip>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Training History */}
        <div className="section history-section">
          <h3>Training History</h3>
          <div className="sessions-list">
            {trainingSessions.map((session) => (
              <div key={session.session_id} className="session-card">
                <div className="session-header">
                  <h4>{session.session_id}</h4>
                  <span className={`status-badge ${session.status}`}>
                    {session.status}
                  </span>
                </div>
                
                <div className="session-details">
                  <div>Model: {session.model_type}</div>
                  <div>Duration: {session.duration_minutes} minutes</div>
                  <div>Epochs: {session.current_epoch}/{session.total_epochs}</div>
                </div>
                
                {session.status === 'completed' && (
                  <div className="final-metrics">
                    <div>Final Precision: {(session.current_metrics.precision * 100).toFixed(1)}%</div>
                    <div>Final Recall: {(session.current_metrics.recall * 100).toFixed(1)}%</div>
                    <div>Final mAP@50: {(session.current_metrics.mAP50 * 100).toFixed(1)}%</div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelTraining;
