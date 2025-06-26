/**
 * Custom Hooks for API State Management
 * 
 * These hooks encapsulate API calls and state management,
 * providing a clean interface for components.
 */

import { useState, useCallback } from 'react';
import { imageApi, trainingApi, youtubeApi, handleApiError } from '../services/api';

/**
 * Generic hook for API calls with loading and error states
 */
export function useApiCall(apiFunction) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const execute = useCallback(async (...args) => {
    try {
      setLoading(true);
      setError(null);
      const result = await apiFunction(...args);
      setData(result);
      return result;
    } catch (err) {
      const errorInfo = handleApiError(err);
      setError(errorInfo);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [apiFunction]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
}

/**
 * Hook for dashboard data management
 */
export function useDashboardData() {
  const [dashboardData, setDashboardData] = useState({
    refereeCounts: { positive: 0, negative: 0 },
    signalClasses: [],
    signalClassCounts: {},
    pendingCounts: {
      images: 0,
      autolabeled: 0,
      signalDetections: 0
    }
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchDashboardData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch all dashboard data concurrently
      const [
        refereeCountsData,
        signalClassesData,
        signalCountsData,
        pendingImagesData,
        autolabeledData,
        signalDetectionsData
      ] = await Promise.allSettled([
        trainingApi.getRefereeTrainingCount(),
        trainingApi.getSignalClasses(),
        trainingApi.getSignalClassCounts(),
        imageApi.getPendingImages(),
        youtubeApi.getAutolabeledCount(),
        youtubeApi.getSignalDetectionsCount()
      ]);

      // Process results
      const refereeCounts = refereeCountsData.status === 'fulfilled' 
        ? { positive: refereeCountsData.value.positive_count || 0, negative: refereeCountsData.value.negative_count || 0 }
        : { positive: 0, negative: 0 };

      const signalClasses = signalClassesData.status === 'fulfilled'
        ? signalClassesData.value.classes || []
        : [];

      const signalClassCounts = signalCountsData.status === 'fulfilled'
        ? signalCountsData.value || {}
        : {};

      const pendingImages = pendingImagesData.status === 'fulfilled'
        ? pendingImagesData.value.count || 0
        : 0;

      const autolabeledCount = autolabeledData.status === 'fulfilled'
        ? autolabeledData.value.total || 0
        : 0;

      const signalDetectionsCount = signalDetectionsData.status === 'fulfilled'
        ? signalDetectionsData.value.total || 0
        : 0;

      // Ensure 'none' class is included
      if (!signalClasses.includes('none')) {
        signalClasses.push('none');
      }

      setDashboardData({
        refereeCounts,
        signalClasses,
        signalClassCounts,
        pendingCounts: {
          images: pendingImages,
          autolabeled: autolabeledCount,
          signalDetections: signalDetectionsCount
        }
      });

    } catch (err) {
      const errorInfo = handleApiError(err, 'Failed to fetch dashboard data');
      setError(errorInfo);
      console.error('Dashboard data fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    dashboardData,
    loading,
    error,
    fetchDashboardData,
    refetchData: fetchDashboardData
  };
}

/**
 * Hook for image upload workflow
 */
export function useImageUpload() {
  const [uploadState, setUploadState] = useState({
    step: 0,
    uploadData: null,
    imageFile: null,
    cropUrl: null,
    cropFilename: null,
    originalFilename: null,
    signalResult: null,
    cropFilenameForSignal: null
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const uploadImage = useCallback(async (file) => {
    try {
      setLoading(true);
      setError(null);
      
      const result = await imageApi.uploadImage(file);
      
      setUploadState(prev => ({
        ...prev,
        uploadData: result,
        imageFile: file,
        originalFilename: result.filename,
        cropUrl: result.crop_url ? `${window.location.origin}${result.crop_url}` : null,
        cropFilename: result.crop_filename,
        step: result.crop_url ? 1 : 3 // Skip to manual crop if no auto-crop
      }));

      return result;
    } catch (err) {
      const errorInfo = handleApiError(err, 'Failed to upload image');
      setError(errorInfo);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const confirmCrop = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const result = await imageApi.confirmCrop(
        uploadState.originalFilename,
        uploadState.cropFilename,
        uploadState.uploadData.bbox
      );

      if (result.status === 'warning' && result.action === 'duplicate_detected') {
        // Handle duplicate
        if (result.crop_filename_for_signal) {
          const signalResult = await imageApi.processCropForSignal(result.crop_filename_for_signal);
          setUploadState(prev => ({
            ...prev,
            cropFilenameForSignal: result.crop_filename_for_signal,
            signalResult,
            step: 2
          }));
        }
        return { type: 'duplicate', ...result };
      }

      if (result.status === 'ok' && result.crop_filename_for_signal) {
        const signalResult = await imageApi.processCropForSignal(result.crop_filename_for_signal);
        setUploadState(prev => ({
          ...prev,
          cropFilenameForSignal: result.crop_filename_for_signal,
          signalResult,
          step: 2
        }));
        return { type: 'success', ...result };
      }

      throw new Error(result.error || 'Failed to confirm crop');
    } catch (err) {
      const errorInfo = handleApiError(err, 'Failed to confirm crop');
      setError(errorInfo);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [uploadState.originalFilename, uploadState.cropFilename, uploadState.uploadData]);

  const createManualCrop = useCallback(async (bbox, classId = 0, proceedToSignal = true) => {
    try {
      setLoading(true);
      setError(null);

      const result = await imageApi.createManualCrop(
        uploadState.originalFilename,
        bbox,
        classId,
        proceedToSignal
      );

      if (result.status === 'warning' && result.action === 'duplicate_detected') {
        return { type: 'duplicate', ...result };
      }

      if (result.status === 'ok') {
        if (result.action === 'saved_as_negative') {
          return { type: 'negative', ...result };
        }

        if (result.crop_filename_for_signal && proceedToSignal) {
          const signalResult = await imageApi.processCropForSignal(result.crop_filename_for_signal);
          setUploadState(prev => ({
            ...prev,
            cropFilenameForSignal: result.crop_filename_for_signal,
            signalResult,
            step: 2
          }));
          return { type: 'success_with_signal', ...result };
        }

        return { type: 'success', ...result };
      }

      throw new Error(result.error || 'Failed to create manual crop');
    } catch (err) {
      const errorInfo = handleApiError(err, 'Failed to create manual crop');
      setError(errorInfo);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [uploadState.originalFilename]);

  const confirmSignal = useCallback(async ({ correct, selectedClass, signalBboxYolo, originalFilename }) => {
    try {
      setLoading(true);
      setError(null);

      const result = await imageApi.confirmSignal({
        cropFilenameForSignal: uploadState.cropFilenameForSignal,
        correct,
        selectedClass,
        signalBboxYolo,
        originalFilename: originalFilename || uploadState.originalFilename
      });

      return result;
    } catch (err) {
      const errorInfo = handleApiError(err, 'Failed to confirm signal');
      setError(errorInfo);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [uploadState.cropFilenameForSignal, uploadState.originalFilename]);

  const resetUploadFlow = useCallback(() => {
    setUploadState({
      step: 0,
      uploadData: null,
      imageFile: null,
      cropUrl: null,
      cropFilename: null,
      originalFilename: null,
      signalResult: null,
      cropFilenameForSignal: null
    });
    setError(null);
  }, []);

  const setStep = useCallback((step) => {
    setUploadState(prev => ({ ...prev, step }));
  }, []);

  return {
    uploadState,
    loading,
    error,
    uploadImage,
    confirmCrop,
    createManualCrop,
    confirmSignal,
    resetUploadFlow,
    setStep
  };
}

/**
 * Hook for training data operations
 */
export function useTrainingData() {
  const [messages, setMessages] = useState({
    moveReferee: '',
    moveSignal: '',
    deleteReferee: '',
    deleteSignal: ''
  });

  const moveRefereeTraining = useApiCall(trainingApi.moveRefereeTraining);
  const moveSignalTraining = useApiCall(trainingApi.moveSignalTraining);
  const deleteRefereeTraining = useApiCall(trainingApi.deleteRefereeTraining);
  const deleteSignalTraining = useApiCall(trainingApi.deleteSignalTraining);

  const handleMoveReferee = useCallback(async () => {
    try {
      setMessages(prev => ({ ...prev, moveReferee: 'Moving...' }));
      const result = await moveRefereeTraining.execute();
      setMessages(prev => ({ 
        ...prev, 
        moveReferee: `Moved ${result.moved?.length || 0} files to ${result.dst || 'destination'}` 
      }));
      return result;
    } catch (err) {
      setMessages(prev => ({ ...prev, moveReferee: `Error: ${err.message}` }));
      throw err;
    }
  }, [moveRefereeTraining]);

  const handleMoveSignal = useCallback(async () => {
    try {
      setMessages(prev => ({ ...prev, moveSignal: 'Moving...' }));
      const result = await moveSignalTraining.execute();
      setMessages(prev => ({ 
        ...prev, 
        moveSignal: `Moved ${result.moved?.length || 0} files to ${result.dst || 'destination'}` 
      }));
      return result;
    } catch (err) {
      setMessages(prev => ({ ...prev, moveSignal: `Error: ${err.message}` }));
      throw err;
    }
  }, [moveSignalTraining]);

  const handleDeleteReferee = useCallback(async () => {
    if (!window.confirm('Are you sure you want to delete all referee training data? This action cannot be undone.')) {
      return;
    }

    try {
      setMessages(prev => ({ ...prev, deleteReferee: 'Deleting...' }));
      const result = await deleteRefereeTraining.execute();
      setMessages(prev => ({ 
        ...prev, 
        deleteReferee: `Deleted ${result.deleted_count || 0} referee training files` 
      }));
      return result;
    } catch (err) {
      setMessages(prev => ({ ...prev, deleteReferee: `Error: ${err.message}` }));
      throw err;
    }
  }, [deleteRefereeTraining]);

  const handleDeleteSignal = useCallback(async () => {
    if (!window.confirm('Are you sure you want to delete all signal training data? This action cannot be undone.')) {
      return;
    }

    try {
      setMessages(prev => ({ ...prev, deleteSignal: 'Deleting...' }));
      const result = await deleteSignalTraining.execute();
      setMessages(prev => ({ 
        ...prev, 
        deleteSignal: `Deleted ${result.deleted_count || 0} signal training files` 
      }));
      return result;
    } catch (err) {
      setMessages(prev => ({ ...prev, deleteSignal: `Error: ${err.message}` }));
      throw err;
    }
  }, [deleteSignalTraining]);

  return {
    messages,
    actions: {
      moveReferee: handleMoveReferee,
      moveSignal: handleMoveSignal,
      deleteReferee: handleDeleteReferee,
      deleteSignal: handleDeleteSignal
    },
    loading: {
      moveReferee: moveRefereeTraining.loading,
      moveSignal: moveSignalTraining.loading,
      deleteReferee: deleteRefereeTraining.loading,
      deleteSignal: deleteSignalTraining.loading
    }
  };
} 