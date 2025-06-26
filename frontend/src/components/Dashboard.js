/**
 * Dashboard Component
 * 
 * Main dashboard for the Referee Detection System showing training data
 * statistics and providing access to various system functions.
 */

import React, { useEffect } from 'react';
import { useDashboardData, useTrainingData } from '../hooks/useApi';
import './Dashboard.css';

const Dashboard = ({ 
  onNavigate, 
  onShowGlobalAutoLabeledConfirmation, 
  onShowGlobalSignalDetectionConfirmation 
}) => {
  const { dashboardData, loading, error, fetchDashboardData } = useDashboardData();
  const { messages, actions, loading: trainingLoading } = useTrainingData();

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  const handleRefreshData = () => {
    fetchDashboardData();
  };

  const handleActionWithRefresh = async (action) => {
    try {
      await action();
      // Refresh dashboard data after successful action
      await fetchDashboardData();
    } catch (error) {
      console.error('Action failed:', error);
    }
  };

  const { refereeCounts, signalClasses, signalClassCounts, pendingCounts } = dashboardData;
  const totalReferee = refereeCounts.positive + refereeCounts.negative;
  const totalSignals = Object.values(signalClassCounts || {}).reduce((a, b) => a + b, 0);

  if (loading && totalReferee === 0) {
    return (
      <div className="dashboard loading">
        <div className="loading-spinner">Loading dashboard data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard error">
        <div className="error-message">
          <h3>Error loading dashboard data</h3>
          <p>{error.message}</p>
          <button onClick={handleRefreshData} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Dashboard - Referee & Signal Detection</h2>
        <button 
          onClick={handleRefreshData} 
          className={`refresh-button ${loading ? 'loading' : ''}`}
          disabled={loading}
        >
          {loading ? '⏳' : '🔄'} Refresh
        </button>
      </div>

      <div className="main-actions">
        <button 
          className="nav-button" 
          onClick={() => onNavigate('labelingQueue')}
          disabled={pendingCounts.images === 0}
        >
          Label Pending Frames ({pendingCounts.images})
        </button>
        <button 
          className="nav-button autolabel" 
          onClick={onShowGlobalAutoLabeledConfirmation}
          disabled={pendingCounts.autolabeled === 0}
        >
          Confirm Autolabeled Frames ({pendingCounts.autolabeled})
        </button>
        <button 
          className="nav-button signal-detection" 
          onClick={onShowGlobalSignalDetectionConfirmation}
          disabled={pendingCounts.signalDetections === 0}
        >
          Confirm Signal Detections ({pendingCounts.signalDetections})
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
          
          <div className="action-buttons">
            <button 
              onClick={() => handleActionWithRefresh(actions.moveReferee)} 
              className="action-button"
              disabled={trainingLoading.moveReferee || totalReferee === 0}
            >
              {trainingLoading.moveReferee ? 'Moving...' : 'Move to Global Training Folder'}
            </button>
            <button 
              onClick={() => handleActionWithRefresh(actions.deleteReferee)} 
              className="action-button delete-button"
              disabled={trainingLoading.deleteReferee || totalReferee === 0}
              style={{backgroundColor: '#dc3545', marginTop: '10px'}}
            >
              {trainingLoading.deleteReferee ? 'Deleting...' : '🗑️ Delete All Referee Training Data'}
            </button>
          </div>
          
          {messages.moveReferee && (
            <div className="message">
              {messages.moveReferee}
            </div>
          )}
          {messages.deleteReferee && (
            <div 
              className="message" 
              style={{color: messages.deleteReferee.startsWith('Error') ? '#dc3545' : '#28a745'}}
            >
              {messages.deleteReferee}
            </div>
          )}
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
          
          <div className="action-buttons">
            <button 
              onClick={() => handleActionWithRefresh(actions.moveSignal)} 
              className="action-button"
              disabled={trainingLoading.moveSignal || totalSignals === 0}
            >
              {trainingLoading.moveSignal ? 'Moving...' : 'Move to Global Training Folder'}
            </button>
            <button 
              onClick={() => handleActionWithRefresh(actions.deleteSignal)} 
              className="action-button delete-button"
              disabled={trainingLoading.deleteSignal || totalSignals === 0}
              style={{backgroundColor: '#dc3545', marginTop: '10px'}}
            >
              {trainingLoading.deleteSignal ? 'Deleting...' : '🗑️ Delete All Signal Training Data'}
            </button>
          </div>
          
          {messages.moveSignal && (
            <div className="message">
              {messages.moveSignal}
            </div>
          )}
          {messages.deleteSignal && (
            <div 
              className="message" 
              style={{color: messages.deleteSignal.startsWith('Error') ? '#dc3545' : '#28a745'}}
            >
              {messages.deleteSignal}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;