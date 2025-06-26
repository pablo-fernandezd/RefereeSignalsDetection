import React, { useState, useEffect } from 'react';
import './ModelManagement.css';

// API base URL configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const ModelManagement = () => {
    const [models, setModels] = useState([]);
    const [availableModels, setAvailableModels] = useState({});
    const [workflowTypes, setWorkflowTypes] = useState([]);
    const [registryStats, setRegistryStats] = useState({});
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('overview');
    const [selectedModelType, setSelectedModelType] = useState('');
    const [uploadFile, setUploadFile] = useState(null);
    const [downloadingModel, setDownloadingModel] = useState(null);
    const [editingModel, setEditingModel] = useState(null);
    const [editFormData, setEditFormData] = useState({
        description: '',
        tags: '',
        performance_metrics: {},
        yolo_version: '',
        yolo_architecture: ''
    });
    const [supportedVersions, setSupportedVersions] = useState({});
    const [compatibilityReport, setCompatibilityReport] = useState(null);
    const [detectingVersion, setDetectingVersion] = useState(null);

    useEffect(() => {
        console.log('ModelManagement component mounted, API_BASE_URL:', API_BASE_URL);
        loadData();
        loadSupportedVersions();
        loadCompatibilityReport();
    }, []);

    const loadData = async () => {
        setLoading(true);
        try {
            await Promise.all([
                loadModels(),
                loadRegistryStats(),
                loadAvailableModels(),
                loadWorkflowTypes()
            ]);
        } catch (error) {
            console.error('Error loading data:', error);
        }
        setLoading(false);
    };

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

    const loadRegistryStats = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/registry/stats`);
            const data = await response.json();
            if (data.status === 'success') {
                setRegistryStats(data.stats);
            }
        } catch (error) {
            console.error('Error loading registry stats:', error);
        }
    };

    const loadAvailableModels = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/ultralytics/available`);
            const data = await response.json();
            if (data.status === 'success') {
                setAvailableModels(data.available_models);
            }
        } catch (error) {
            console.error('Error loading available models:', error);
        }
    };

    const loadWorkflowTypes = async () => {
        try {
            console.log('Loading workflow types from:', `${API_BASE_URL}/api/models/workflow/types`);
            const response = await fetch(`${API_BASE_URL}/api/models/workflow/types`);
            console.log('Workflow types response status:', response.status);
            const data = await response.json();
            console.log('Workflow types data:', data);
            if (data.status === 'success') {
                setWorkflowTypes(data.workflow_types);
                console.log('Set workflow types:', data.workflow_types);
            } else {
                console.error('Workflow types error:', data.error);
            }
        } catch (error) {
            console.error('Error loading workflow types:', error);
        }
    };

    const importLegacyModels = async () => {
        try {
            console.log('Importing legacy models from:', `${API_BASE_URL}/api/models/import/legacy`);
            const response = await fetch(`${API_BASE_URL}/api/models/import/legacy`, {
                method: 'POST'
            });
            console.log('Legacy import response status:', response.status);
            const data = await response.json();
            console.log('Legacy import data:', data);
            if (data.status === 'success') {
                alert('Legacy models imported successfully!');
                loadData();
            } else {
                alert(`Error: ${data.error}`);
                console.error('Legacy import error:', data.error);
            }
        } catch (error) {
            console.error('Error importing legacy models:', error);
            alert('Failed to import legacy models');
        }
    };

    const downloadBaseModel = async (modelName, modelType) => {
        setDownloadingModel(modelName);
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/base/download/${modelName}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_type: modelType,
                    description: `Base ${modelName} model for ${modelType} detection`
                })
            });
            const data = await response.json();
            if (data.status === 'success') {
                alert(`${modelName} downloaded successfully!`);
                loadData();
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error downloading model:', error);
            alert('Failed to download model');
        }
        setDownloadingModel(null);
    };

    const deployModel = async (modelId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/deploy/${modelId}`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.status === 'success') {
                alert('Model deployed successfully!');
                loadData();
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error deploying model:', error);
            alert('Failed to deploy model');
        }
    };

    const deleteModel = async (modelId, modelName) => {
        if (!window.confirm(`Are you sure you want to delete the model "${modelName}"?\n\nThis action cannot be undone and will permanently remove the model file.`)) {
            return;
        }
        
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/delete/${modelId}`, {
                method: 'DELETE'
            });
            const data = await response.json();
            if (data.status === 'success') {
                alert('Model deleted successfully!');
                loadData();
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error deleting model:', error);
            alert('Failed to delete model');
        }
    };

    const uploadModel = async (e) => {
        e.preventDefault();
        if (!uploadFile || !selectedModelType) {
            alert('Please select a file and model type');
            return;
        }

        const formData = new FormData();
        formData.append('file', uploadFile);
        formData.append('model_type', selectedModelType);
        formData.append('description', `Custom uploaded ${selectedModelType} model`);

        try {
            const response = await fetch(`${API_BASE_URL}/api/models/upload`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.status === 'success') {
                alert('Model uploaded successfully!');
                setUploadFile(null);
                setSelectedModelType('');
                loadData();
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error uploading model:', error);
            alert('Failed to upload model');
        }
    };

    const startEditModel = (model) => {
        setEditingModel(model.model_id);
        setEditFormData({
            description: model.description || '',
            tags: (model.tags || []).join(', '),
            performance_metrics: model.performance_metrics || {},
            yolo_version: model.yolo_version || '',
            yolo_architecture: model.yolo_architecture || ''
        });
    };

    const cancelEdit = () => {
        setEditingModel(null);
        setEditFormData({
            description: '',
            tags: '',
            performance_metrics: {},
            yolo_version: '',
            yolo_architecture: ''
        });
    };

    const saveEdit = async (modelId) => {
        try {
            const tagsArray = editFormData.tags.split(',').map(tag => tag.trim()).filter(tag => tag);
            
            const response = await fetch(`${API_BASE_URL}/api/models/edit/${modelId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    description: editFormData.description,
                    tags: tagsArray,
                    performance_metrics: editFormData.performance_metrics,
                    yolo_version: editFormData.yolo_version,
                    yolo_architecture: editFormData.yolo_architecture
                })
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                alert('Model updated successfully!');
                setEditingModel(null);
                loadData();
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error updating model:', error);
            alert('Failed to update model');
        }
    };

    const formatFileSize = (sizeInMB) => {
        if (sizeInMB < 1) {
            return `${(sizeInMB * 1024).toFixed(1)} KB`;
        }
        return `${sizeInMB.toFixed(1)} MB`;
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleDateString() + ' ' + new Date(dateString).toLocaleTimeString();
    };

    const loadSupportedVersions = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/supported_versions`);
            const data = await response.json();
            if (data.status === 'success') {
                setSupportedVersions(data.version_info);
            }
        } catch (error) {
            console.error('Error loading supported versions:', error);
        }
    };

    const loadCompatibilityReport = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/compatibility_report`);
            const data = await response.json();
            if (data.status === 'success') {
                setCompatibilityReport(data.compatibility_report);
            }
        } catch (error) {
            console.error('Error loading compatibility report:', error);
        }
    };

    const detectModelVersion = async (modelId) => {
        setDetectingVersion(modelId);
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/detect_version/${modelId}`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.status === 'success') {
                alert(`Version detected: ${data.detection_results.yolo_version || 'unknown'}`);
                loadData();
                loadCompatibilityReport();
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error detecting version:', error);
            alert('Failed to detect version');
        }
        setDetectingVersion(null);
    };

    const validateModelForTraining = async (modelId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/validate_for_training/${modelId}`);
            const data = await response.json();
            if (data.status === 'success') {
                const message = data.can_train 
                    ? `✅ Model is ready for training!\n\n${data.message}`
                    : `❌ Model cannot be used for training:\n\n${data.message}`;
                alert(message);
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error validating model:', error);
            alert('Failed to validate model');
        }
    };

    if (loading) {
        return (
            <div className="model-management">
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <p>Loading model registry...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="model-management">
            <div className="header">
                <h1>Model Registry</h1>
                <p>Manage YOLO models for referee detection and signal classification</p>
            </div>

            <div className="tabs">
                <button 
                    className={activeTab === 'overview' ? 'active' : ''}
                    onClick={() => setActiveTab('overview')}
                >
                    Overview
                </button>
                <button 
                    className={activeTab === 'models' ? 'active' : ''}
                    onClick={() => setActiveTab('models')}
                >
                    My Models
                </button>
                <button 
                    className={activeTab === 'download' ? 'active' : ''}
                    onClick={() => setActiveTab('download')}
                >
                    Download Base Models
                </button>
                <button 
                    className={activeTab === 'upload' ? 'active' : ''}
                    onClick={() => setActiveTab('upload')}
                >
                    Upload Custom Model
                </button>
                <button 
                    className={activeTab === 'workflows' ? 'active' : ''}
                    onClick={() => setActiveTab('workflows')}
                >
                    Workflow Types
                </button>
                <button 
                    className={activeTab === 'compatibility' ? 'active' : ''}
                    onClick={() => setActiveTab('compatibility')}
                >
                    Compatibility Report
                </button>
            </div>

            {activeTab === 'overview' && (
                <div className="overview-tab">
                    <div className="stats-grid">
                        <div className="stat-card">
                            <h3>Total Models</h3>
                            <div className="stat-value">{registryStats.total_models || 0}</div>
                        </div>
                        <div className="stat-card">
                            <h3>Active Models</h3>
                            <div className="stat-value">{Object.keys(registryStats.active_models || {}).length}</div>
                        </div>
                        <div className="stat-card">
                            <h3>Storage Used</h3>
                            <div className="stat-value">{formatFileSize(registryStats.total_size_mb || 0)}</div>
                        </div>
                        <div className="stat-card">
                            <h3>Model Types</h3>
                            <div className="stat-value">{Object.keys(registryStats.models_by_type || {}).length}</div>
                        </div>
                    </div>

                    <div className="quick-actions">
                        <h3>Quick Actions</h3>
                        <button className="action-btn primary" onClick={importLegacyModels}>
                            Import Existing Models
                        </button>
                        <button className="action-btn" onClick={() => setActiveTab('download')}>
                            Download YOLO Models
                        </button>
                        <button className="action-btn" onClick={() => setActiveTab('upload')}>
                            Upload Custom Model
                        </button>
                    </div>

                    <div className="model-types-overview">
                        <h3>Models by Type</h3>
                        {Object.entries(registryStats.models_by_type || {}).map(([type, count]) => (
                            <div key={type} className="type-summary">
                                <span className="type-name">{type}</span>
                                <span className="type-count">{count} models</span>
                                <span className="active-indicator">
                                    {registryStats.active_models?.[type] ? '✅ Active' : '⚪ No Active Model'}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {activeTab === 'models' && (
                <div className="models-tab">
                    <div className="models-header">
                        <h3>My Models ({models.length})</h3>
                        <button onClick={loadModels} className="refresh-btn">Refresh</button>
                    </div>
                    
                    <div className="models-grid">
                        {models.map(model => (
                            <div key={model.model_id} className={`model-card ${model.is_active ? 'active' : ''}`}>
                                {editingModel === model.model_id ? (
                                    // Edit mode
                                    <div className="model-edit-form">
                                        <div className="model-header">
                                            <h4>Edit: {model.version}</h4>
                                            <div className="model-badges">
                                                <span className={`badge ${model.source}`}>{model.source}</span>
                                                {model.is_active && <span className="badge active">Active</span>}
                                            </div>
                                        </div>
                                        
                                        <div className="edit-form">
                                            <div className="form-group">
                                                <label>Description:</label>
                                                <textarea
                                                    value={editFormData.description}
                                                    onChange={(e) => setEditFormData({...editFormData, description: e.target.value})}
                                                    placeholder="Model description..."
                                                    rows={3}
                                                />
                                            </div>
                                            
                                            <div className="form-group">
                                                <label>Tags (comma-separated):</label>
                                                <input
                                                    type="text"
                                                    value={editFormData.tags}
                                                    onChange={(e) => setEditFormData({...editFormData, tags: e.target.value})}
                                                    placeholder="tag1, tag2, tag3..."
                                                />
                                            </div>
                                            
                                            <div className="form-group">
                                                <label>YOLO Version:</label>
                                                <select
                                                    value={editFormData.yolo_version}
                                                    onChange={(e) => setEditFormData({...editFormData, yolo_version: e.target.value})}
                                                >
                                                    <option value="">Select YOLO version...</option>
                                                    {Object.keys(supportedVersions).map(version => (
                                                        <option key={version} value={version}>{version.toUpperCase()}</option>
                                                    ))}
                                                </select>
                                            </div>
                                            
                                            {editFormData.yolo_version && supportedVersions[editFormData.yolo_version] && (
                                                <div className="form-group">
                                                    <label>Architecture:</label>
                                                    <select
                                                        value={editFormData.yolo_architecture}
                                                        onChange={(e) => setEditFormData({...editFormData, yolo_architecture: e.target.value})}
                                                    >
                                                        <option value="">Select architecture...</option>
                                                        {supportedVersions[editFormData.yolo_version].architecture_names?.map(arch => (
                                                            <option key={arch} value={arch}>{arch}</option>
                                                        ))}
                                                    </select>
                                                </div>
                                            )}
                                            
                                            <div className="edit-actions">
                                                <button 
                                                    className="save-btn"
                                                    onClick={() => saveEdit(model.model_id)}
                                                >
                                                    Save
                                                </button>
                                                <button 
                                                    className="cancel-btn"
                                                    onClick={cancelEdit}
                                                >
                                                    Cancel
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    // View mode
                                    <>
                                        <div className="model-header">
                                            <h4>{model.version}</h4>
                                            <div className="model-badges">
                                                <span className={`badge ${model.source}`}>{model.source}</span>
                                                {model.is_active && <span className="badge active">Active</span>}
                                            </div>
                                        </div>
                                        
                                        <div className="model-info">
                                            <p><strong>Type:</strong> {model.model_type}</p>
                                            <p><strong>Size:</strong> {formatFileSize(model.file_size_mb)}</p>
                                            <p><strong>Created:</strong> {formatDate(model.created_at)}</p>
                                            {model.yolo_version && (
                                                <p><strong>YOLO Version:</strong> {model.yolo_version.toUpperCase()}
                                                {model.yolo_architecture && ` (${model.yolo_architecture})`}
                                                </p>
                                            )}
                                            {model.compatibility_status && (
                                                <p><strong>Compatibility:</strong> 
                                                    <span style={{
                                                        color: model.compatibility_status === 'compatible' ? 'var(--success-color)' :
                                                               model.compatibility_status === 'warning' ? 'var(--warning-color)' :
                                                               model.compatibility_status === 'incompatible' ? 'var(--error-color)' : 'var(--text-secondary)',
                                                        fontWeight: 'bold',
                                                        marginLeft: '8px'
                                                    }}>
                                                        {model.compatibility_status === 'compatible' ? '✅ Compatible' :
                                                         model.compatibility_status === 'warning' ? '⚠️ Warning' :
                                                         model.compatibility_status === 'incompatible' ? '❌ Incompatible' : '❓ Unknown'}
                                                    </span>
                                                </p>
                                            )}
                                            {model.description && <p><strong>Description:</strong> {model.description}</p>}
                                            {model.tags && model.tags.length > 0 && (
                                                <div className="model-tags">
                                                    <strong>Tags:</strong>
                                                    {model.tags.map(tag => (
                                                        <span key={tag} className="tag">{tag}</span>
                                                    ))}
                                                </div>
                                            )}
                                        </div>

                                        {model.validation_results && (model.validation_results.issues?.length > 0 || model.validation_results.recommendations?.length > 0) && (
                                            <div className="model-validation">
                                                <h5>Validation Results</h5>
                                                {model.validation_results.issues?.length > 0 && (
                                                    <div className="validation-issues">
                                                        <strong>Issues:</strong>
                                                        <ul>
                                                            {model.validation_results.issues.map((issue, index) => (
                                                                <li key={index} style={{color: 'var(--error-color)'}}>{issue}</li>
                                                            ))}
                                                        </ul>
                                                    </div>
                                                )}
                                                {model.validation_results.recommendations?.length > 0 && (
                                                    <div className="validation-recommendations">
                                                        <strong>Recommendations:</strong>
                                                        <ul>
                                                            {model.validation_results.recommendations.map((rec, index) => (
                                                                <li key={index} style={{color: 'var(--warning-color)'}}>{rec}</li>
                                                            ))}
                                                        </ul>
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {model.performance_metrics && Object.keys(model.performance_metrics).length > 0 && (
                                            <div className="model-metrics">
                                                <h5>Performance Metrics</h5>
                                                {Object.entries(model.performance_metrics).map(([key, value]) => (
                                                    <div key={key} className="metric">
                                                        <span>{key}:</span>
                                                        <span>{typeof value === 'number' ? value.toFixed(3) : value}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        <div className="model-actions">
                                            {!model.is_active && (
                                                <button 
                                                    className="deploy-btn"
                                                    onClick={() => deployModel(model.model_id)}
                                                >
                                                    Deploy
                                                </button>
                                            )}
                                            <button 
                                                className="download-btn"
                                                onClick={() => window.open(`${API_BASE_URL}/api/models/download/${model.model_id}`, '_blank')}
                                            >
                                                Download
                                            </button>
                                            <button 
                                                className="edit-btn"
                                                onClick={() => startEditModel(model)}
                                            >
                                                Edit
                                            </button>
                                            <button 
                                                className="btn btn-secondary"
                                                onClick={() => detectModelVersion(model.model_id)}
                                                disabled={detectingVersion === model.model_id}
                                                title="Auto-detect YOLO version"
                                            >
                                                {detectingVersion === model.model_id ? 'Detecting...' : 'Detect Version'}
                                            </button>
                                            <button 
                                                className="btn btn-warning"
                                                onClick={() => validateModelForTraining(model.model_id)}
                                                title="Check if model can be used for training"
                                            >
                                                Validate Training
                                            </button>
                                            {!model.is_active && (
                                                <button 
                                                    className="delete-btn"
                                                    onClick={() => deleteModel(model.model_id, model.version)}
                                                >
                                                    Delete
                                                </button>
                                            )}
                                        </div>
                                    </>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {activeTab === 'download' && (
                <div className="download-tab">
                    <h3>Download YOLO Base Models</h3>
                    <p>Download pre-trained YOLO models from Ultralytics for training new models</p>
                    
                    <div className="model-selector">
                        <label>Select Model Type:</label>
                        <select 
                            value={selectedModelType} 
                            onChange={(e) => setSelectedModelType(e.target.value)}
                        >
                            <option value="">Choose model type...</option>
                            {workflowTypes.map(workflow => (
                                <option key={workflow.type} value={workflow.type}>
                                    {workflow.name}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="available-models-grid">
                        {Object.entries(availableModels).map(([modelName, modelInfo]) => (
                            <div key={modelName} className="available-model-card">
                                <h4>{modelName}</h4>
                                <div className="model-details">
                                    <p><strong>Size:</strong> {modelInfo.size}</p>
                                    <p><strong>Parameters:</strong> {modelInfo.params}</p>
                                    <p><strong>Description:</strong> {modelInfo.description}</p>
                                </div>
                                <button 
                                    className="download-model-btn"
                                    disabled={!selectedModelType || downloadingModel === modelName}
                                    onClick={() => downloadBaseModel(modelName, selectedModelType)}
                                >
                                    {downloadingModel === modelName ? 'Downloading...' : 'Download'}
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {activeTab === 'upload' && (
                <div className="upload-tab">
                    <h3>Upload Custom Model</h3>
                    <p>Upload your own trained .pt model files</p>
                    
                    <form onSubmit={uploadModel} className="upload-form">
                        <div className="form-group">
                            <label>Model Type:</label>
                            <select 
                                value={selectedModelType} 
                                onChange={(e) => setSelectedModelType(e.target.value)}
                                required
                            >
                                <option value="">Choose model type...</option>
                                {workflowTypes.map(workflow => (
                                    <option key={workflow.type} value={workflow.type}>
                                        {workflow.name}
                                    </option>
                                ))}
                            </select>
                        </div>
                        
                        <div className="form-group">
                            <label>Model File (.pt):</label>
                            <input 
                                type="file" 
                                accept=".pt"
                                onChange={(e) => setUploadFile(e.target.files[0])}
                                required
                            />
                        </div>
                        
                        <button type="submit" className="upload-btn">
                            Upload Model
                        </button>
                    </form>
                </div>
            )}

            {activeTab === 'workflows' && (
                <div className="workflows-tab">
                    <h3>Available Workflow Types</h3>
                    {workflowTypes.length === 0 ? (
                        <div className="no-workflows">
                            <p>No workflow types available.</p>
                            <p>Workflow types are used to categorize different model training and deployment workflows.</p>
                        </div>
                    ) : (
                        <div className="workflows-list">
                            {workflowTypes.map((workflow, index) => (
                                <div key={index} className="workflow-item">
                                    <h4>{workflow.name || `Workflow ${index + 1}`}</h4>
                                    <p>{workflow.description || 'No description available'}</p>
                                    {workflow.supported_models && (
                                        <div className="supported-models">
                                            <strong>Supported Models:</strong> {workflow.supported_models.join(', ')}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'compatibility' && (
                <div className="compatibility-tab">
                    <div className="compatibility-header">
                        <h3>Model Compatibility Report</h3>
                        <button onClick={loadCompatibilityReport} className="refresh-btn">Refresh Report</button>
                    </div>
                    
                    {compatibilityReport ? (
                        <>
                            <div className="compatibility-summary">
                                <div className="summary-grid">
                                    <div className="summary-card compatible">
                                        <h4>Compatible</h4>
                                        <div className="count">{compatibilityReport.compatible}</div>
                                        <div className="percentage">
                                            {((compatibilityReport.compatible / compatibilityReport.total_models) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    <div className="summary-card warning">
                                        <h4>Warning</h4>
                                        <div className="count">{compatibilityReport.warning}</div>
                                        <div className="percentage">
                                            {((compatibilityReport.warning / compatibilityReport.total_models) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    <div className="summary-card incompatible">
                                        <h4>Incompatible</h4>
                                        <div className="count">{compatibilityReport.incompatible}</div>
                                        <div className="percentage">
                                            {((compatibilityReport.incompatible / compatibilityReport.total_models) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    <div className="summary-card unknown">
                                        <h4>Unknown</h4>
                                        <div className="count">{compatibilityReport.unknown}</div>
                                        <div className="percentage">
                                            {((compatibilityReport.unknown / compatibilityReport.total_models) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="version-breakdown">
                                <h4>By YOLO Version</h4>
                                <div className="version-grid">
                                    {Object.entries(compatibilityReport.by_version).map(([version, data]) => (
                                        <div key={version} className="version-card">
                                            <h5>{version.toUpperCase()}</h5>
                                            <div className="version-stats">
                                                <div>Total: {data.count}</div>
                                                <div>Compatible: {data.compatible}</div>
                                                <div className="compatibility-rate">
                                                    Rate: {((data.compatible / data.count) * 100).toFixed(1)}%
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {compatibilityReport.issues_summary.length > 0 && (
                                <div className="common-issues">
                                    <h4>Common Issues</h4>
                                    <ul className="issues-list">
                                        {compatibilityReport.issues_summary.map((issue, index) => (
                                            <li key={index} className="issue-item">{issue}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            <div className="detailed-report">
                                <h4>Detailed Model Report</h4>
                                <div className="models-table">
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Model</th>
                                                <th>Type</th>
                                                <th>YOLO Version</th>
                                                <th>Status</th>
                                                <th>Issues</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {compatibilityReport.models_detail.map((model) => (
                                                <tr key={model.model_id} className={`status-${model.compatibility_status}`}>
                                                    <td>{model.version}</td>
                                                    <td>{model.model_type}</td>
                                                    <td>
                                                        {model.yolo_version ? model.yolo_version.toUpperCase() : 'Unknown'}
                                                        {model.yolo_architecture && <br />}
                                                        {model.yolo_architecture && <small>({model.yolo_architecture})</small>}
                                                    </td>
                                                    <td>
                                                        <span className={`status-badge ${model.compatibility_status}`}>
                                                            {model.compatibility_status === 'compatible' ? '✅ Compatible' :
                                                             model.compatibility_status === 'warning' ? '⚠️ Warning' :
                                                             model.compatibility_status === 'incompatible' ? '❌ Incompatible' : '❓ Unknown'}
                                                        </span>
                                                    </td>
                                                    <td>
                                                        {model.issues.length > 0 ? (
                                                            <ul className="mini-issues-list">
                                                                {model.issues.map((issue, idx) => (
                                                                    <li key={idx}>{issue}</li>
                                                                ))}
                                                            </ul>
                                                        ) : (
                                                            <span className="no-issues">No issues</span>
                                                        )}
                                                    </td>
                                                    <td>
                                                        <button 
                                                            className="btn btn-sm"
                                                            onClick={() => detectModelVersion(model.model_id)}
                                                            disabled={detectingVersion === model.model_id}
                                                        >
                                                            {detectingVersion === model.model_id ? 'Detecting...' : 'Re-detect'}
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </>
                    ) : (
                        <div className="loading-report">
                            <p>Loading compatibility report...</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ModelManagement;
