/**
 * Custom React Hook for API Operations
 * 
 * This hook provides a unified interface for making API calls with
 * loading states, error handling, and automatic state management.
 */

import { useState, useCallback } from 'react';
import apiService, { APIError } from '../services/api';
import { ERROR_MESSAGES } from '../constants';

/**
 * Custom hook for API operations with state management
 * @param {Object} options - Configuration options
 * @returns {Object} - API state and methods
 */
export const useApi = (options = {}) => {
  const {
    onSuccess = () => {},
    onError = () => {},
    retryOnError = false,
    showErrorAlert = true
  } = options;

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const execute = useCallback(async (apiCall, ...args) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await apiCall(...args);
      setData(result);
      onSuccess(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof APIError 
        ? err.message 
        : ERROR_MESSAGES.UNKNOWN_ERROR;
      
      setError(errorMessage);
      
      if (showErrorAlert) {
        alert(`Error: ${errorMessage}`);
      }
      
      onError(err);
      
      if (retryOnError) {
        console.warn('Retrying API call due to error:', errorMessage);
        // Implement retry logic here if needed
      }
      
      throw err;
    } finally {
      setLoading(false);
    }
  }, [onSuccess, onError, retryOnError, showErrorAlert]);

  const reset = useCallback(() => {
    setLoading(false);
    setError(null);
    setData(null);
  }, []);

  return {
    loading,
    error,
    data,
    execute,
    reset
  };
};

/**
 * Hook specifically for image upload operations
 */
export const useImageUpload = () => {
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const { loading, error, data, execute, reset } = useApi({
    onSuccess: () => setUploadProgress(100),
    onError: () => setUploadProgress(0)
  });

  const uploadImage = useCallback(async (file) => {
    setUploadProgress(0);
    
    // Simulate upload progress (in a real app, you'd track actual progress)
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => Math.min(prev + 10, 90));
    }, 100);

    try {
      const result = await execute(apiService.uploadImage, file);
      clearInterval(progressInterval);
      setUploadProgress(100);
      return result;
    } catch (error) {
      clearInterval(progressInterval);
      setUploadProgress(0);
      throw error;
    }
  }, [execute]);

  return {
    loading,
    error,
    data,
    uploadProgress,
    uploadImage,
    reset
  };
};

/**
 * Hook for dashboard data management
 */
export const useDashboardData = () => {
  const [dashboardData, setDashboardData] = useState({
    refereeCounts: { positive: 0, negative: 0 },
    signalClasses: [],
    signalClassCounts: {},
    pendingImageCount: 0,
    pendingAutolabelCount: 0
  });

  const { loading, error, execute } = useApi({
    onSuccess: (data) => {
      if (data) {
        setDashboardData(prev => ({ ...prev, ...data }));
      }
    }
  });

  const fetchDashboardData = useCallback(async () => {
    try {
      const [
        refereeCountsRes,
        signalClassesRes,
        signalCountsRes,
        pendingImagesRes,
        pendingAutolabelRes
      ] = await Promise.all([
        execute(apiService.getRefereeTrainingCount),
        execute(apiService.getSignalClasses),
        execute(apiService.getSignalClassCounts),
        execute(apiService.getPendingImages),
        execute(apiService.getPendingAutoLabelCount)
      ]);

      const newData = {
        refereeCounts: {
          positive: refereeCountsRes.positive_count || 0,
          negative: refereeCountsRes.negative_count || 0
        },
        signalClasses: signalClassesRes.classes || [],
        signalClassCounts: signalCountsRes.counts || {},
        pendingImageCount: pendingImagesRes.count || 0,
        pendingAutolabelCount: pendingAutolabelRes.count || 0
      };

      setDashboardData(newData);
      return newData;
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      throw error;
    }
  }, [execute]);

  return {
    dashboardData,
    loading,
    error,
    fetchDashboardData,
    setDashboardData
  };
};

/**
 * Hook for YouTube video processing
 */
export const useYouTubeProcessing = () => {
  const [videos, setVideos] = useState([]);
  const [processingStatus, setProcessingStatus] = useState({});

  const { loading, error, execute } = useApi();

  const processVideo = useCallback(async (url, autoCrop = true) => {
    return execute(apiService.processYouTubeVideo, { url, auto_crop: autoCrop });
  }, [execute]);

  const fetchVideos = useCallback(async () => {
    const result = await execute(apiService.getAllYouTubeVideos);
    setVideos(result || []);
    return result;
  }, [execute]);

  const getVideoStatus = useCallback(async (folderName) => {
    const status = await execute(apiService.getYouTubeStatus, folderName);
    setProcessingStatus(prev => ({ ...prev, [folderName]: status }));
    return status;
  }, [execute]);

  const deleteVideo = useCallback(async (folderName) => {
    await execute(apiService.deleteYouTubeVideo, folderName);
    setVideos(prev => prev.filter(video => video.folder_name !== folderName));
  }, [execute]);

  return {
    videos,
    processingStatus,
    loading,
    error,
    processVideo,
    fetchVideos,
    getVideoStatus,
    deleteVideo
  };
};

/**
 * Hook for training data operations
 */
export const useTrainingData = () => {
  const { loading, error, execute } = useApi();

  const moveRefereeTraining = useCallback(async () => {
    return execute(apiService.moveRefereeTraining);
  }, [execute]);

  const moveSignalTraining = useCallback(async () => {
    return execute(apiService.moveSignalTraining);
  }, [execute]);

  return {
    loading,
    error,
    moveRefereeTraining,
    moveSignalTraining
  };
};

export default useApi; 