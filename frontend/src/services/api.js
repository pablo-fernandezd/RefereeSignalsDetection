/**
 * API Service Layer
 * 
 * Centralized API communication with error handling, retry logic,
 * and proper response processing.
 */

import { API_CONFIG } from '../constants';

/**
 * Custom error class for API-related errors
 */
export class ApiError extends Error {
  constructor(message, status, response) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.response = response;
  }
}

/**
 * Base API client with common functionality
 */
class ApiClient {
  constructor(baseURL = API_CONFIG.BASE_URL) {
    this.baseURL = baseURL;
    this.timeout = API_CONFIG.TIMEOUT;
  }

  /**
   * Make HTTP request with error handling
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new ApiError(
          errorData.error || `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          errorData
        );
      }

      // Handle different content types
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
      
      return response;
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      
      // Handle network errors, timeouts, etc.
      throw new ApiError(
        error.message || 'Network error occurred',
        0,
        null
      );
    }
  }

  /**
   * GET request
   */
  async get(endpoint, params = {}) {
    const searchParams = new URLSearchParams(params);
    const url = searchParams.toString() ? `${endpoint}?${searchParams}` : endpoint;
    return this.request(url, { method: 'GET' });
  }

  /**
   * POST request
   */
  async post(endpoint, data = null, options = {}) {
    const config = { method: 'POST', ...options };
    
    if (data) {
      if (data instanceof FormData) {
        // Remove Content-Type header for FormData (let browser set it)
        delete config.headers?.['Content-Type'];
        config.body = data;
      } else {
        config.body = JSON.stringify(data);
      }
    }
    
    return this.request(endpoint, config);
  }

  /**
   * DELETE request
   */
  async delete(endpoint) {
    return this.request(endpoint, { method: 'DELETE' });
  }
}

// Create API client instance
const apiClient = new ApiClient();

/**
 * Image Processing API
 */
export const imageApi = {
  /**
   * Upload image for processing
   */
  async uploadImage(file) {
    const formData = new FormData();
    formData.append('image', file);
    return apiClient.post(API_CONFIG.ENDPOINTS.UPLOAD, formData);
  },

  /**
   * Get pending images for labeling
   */
  async getPendingImages() {
    return apiClient.get(API_CONFIG.ENDPOINTS.PENDING_IMAGES);
  },

  /**
   * Confirm crop selection
   */
  async confirmCrop(originalFilename, cropFilename, bbox) {
    return apiClient.post(API_CONFIG.ENDPOINTS.CONFIRM_CROP, {
      original_filename: originalFilename,
      crop_filename: cropFilename,
      bbox
    });
  },

  /**
   * Create manual crop
   */
  async createManualCrop(originalFilename, bbox, classId = 0, proceedToSignal = true) {
    return apiClient.post(API_CONFIG.ENDPOINTS.MANUAL_CROP, {
      original_filename: originalFilename,
      bbox,
      class_id: classId,
      proceedToSignal
    });
  },

  /**
   * Process crop for signal detection
   */
  async processCropForSignal(cropFilename) {
    return apiClient.post(API_CONFIG.ENDPOINTS.PROCESS_CROP_FOR_SIGNAL, {
      crop_filename_for_signal: cropFilename
    });
  },

  /**
   * Confirm signal classification
   */
  async confirmSignal({
    cropFilenameForSignal,
    correct,
    selectedClass,
    signalBboxYolo,
    originalFilename
  }) {
    return apiClient.post(API_CONFIG.ENDPOINTS.CONFIRM_SIGNAL, {
      crop_filename_for_signal: cropFilenameForSignal,
      correct,
      selected_class: selectedClass,
      signal_bbox_yolo: signalBboxYolo,
      original_filename: originalFilename
    });
  }
};

/**
 * Training Data API
 */
export const trainingApi = {
  /**
   * Get referee training count
   */
  async getRefereeTrainingCount() {
    return apiClient.get(API_CONFIG.ENDPOINTS.REFEREE_TRAINING_COUNT);
  },

  /**
   * Get signal classes
   */
  async getSignalClasses() {
    return apiClient.get(API_CONFIG.ENDPOINTS.SIGNAL_CLASSES);
  },

  /**
   * Get signal class counts
   */
  async getSignalClassCounts() {
    return apiClient.get(API_CONFIG.ENDPOINTS.SIGNAL_CLASS_COUNTS);
  },

  /**
   * Move referee training data
   */
  async moveRefereeTraining() {
    return apiClient.post(API_CONFIG.ENDPOINTS.MOVE_REFEREE_TRAINING);
  },

  /**
   * Move signal training data
   */
  async moveSignalTraining() {
    return apiClient.post(API_CONFIG.ENDPOINTS.MOVE_SIGNAL_TRAINING);
  },

  /**
   * Delete referee training data
   */
  async deleteRefereeTraining() {
    return apiClient.post(API_CONFIG.ENDPOINTS.DELETE_REFEREE_TRAINING);
  },

  /**
   * Delete signal training data
   */
  async deleteSignalTraining() {
    return apiClient.post(API_CONFIG.ENDPOINTS.DELETE_SIGNAL_TRAINING);
  }
};

/**
 * YouTube Processing API
 */
export const youtubeApi = {
  /**
   * Get autolabeled frames count
   */
  async getAutolabeledCount() {
    return apiClient.get(`${API_CONFIG.ENDPOINTS.YOUTUBE_AUTOLABELED}?page=1&per_page=1`);
  },

  /**
   * Get signal detections count
   */
  async getSignalDetectionsCount() {
    return apiClient.get(`${API_CONFIG.ENDPOINTS.YOUTUBE_SIGNAL_DETECTIONS}?page=1&per_page=1`);
  }
};

/**
 * Utility function to build image URLs
 */
export const buildImageUrl = (path) => {
  if (!path) return null;
  return path.startsWith('http') ? path : `${API_CONFIG.BASE_URL}${path}`;
};

/**
 * Error handler utility
 */
export const handleApiError = (error, fallbackMessage = 'An error occurred') => {
  console.error('API Error:', error);
  
  if (error instanceof ApiError) {
    return {
      message: error.message,
      status: error.status,
      response: error.response
    };
  }
  
  return {
    message: error.message || fallbackMessage,
    status: 0,
    response: null
  };
};

// Export everything
export default {
  imageApi,
  trainingApi,
  youtubeApi,
  buildImageUrl,
  handleApiError,
  ApiError
}; 