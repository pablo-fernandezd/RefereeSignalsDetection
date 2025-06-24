/**
 * API Service for Referee Detection System
 * 
 * This module provides a centralized interface for all backend API communication,
 * including error handling, request retries, and response formatting.
 */

import { API_CONFIG, API_ENDPOINTS, ERROR_MESSAGES } from '../constants';

/**
 * Custom error class for API-related errors
 */
class APIError extends Error {
  constructor(message, status = null, data = null) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.data = data;
  }
}

/**
 * Main API service class
 */
class APIService {
  constructor() {
    this.baseURL = API_CONFIG.BASE_URL;
    this.timeout = API_CONFIG.TIMEOUT;
    this.retryAttempts = API_CONFIG.RETRY_ATTEMPTS;
  }

  /**
   * Make a HTTP request with error handling and retries
   * @param {string} endpoint - API endpoint
   * @param {Object} options - Fetch options
   * @param {number} retryCount - Current retry count
   * @returns {Promise<Object>} - Response data
   */
  async makeRequest(endpoint, options = {}, retryCount = 0) {
    const url = `${this.baseURL}${endpoint}`;
    
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      timeout: this.timeout,
      ...options
    };

    // Remove Content-Type for FormData
    if (options.body instanceof FormData) {
      delete defaultOptions.headers['Content-Type'];
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(url, {
        ...defaultOptions,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new APIError(
          errorData.error || `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          errorData
        );
      }

      return await response.json();
    } catch (error) {
      // Handle network errors and retries
      if (error.name === 'AbortError') {
        error.message = ERROR_MESSAGES.NETWORK_ERROR;
      }

      if (retryCount < this.retryAttempts && this.shouldRetry(error)) {
        console.warn(`Request failed, retrying... (${retryCount + 1}/${this.retryAttempts})`);
        await this.delay(1000 * (retryCount + 1)); // Exponential backoff
        return this.makeRequest(endpoint, options, retryCount + 1);
      }

      throw error;
    }
  }

  /**
   * Determine if a request should be retried
   * @param {Error} error - The error that occurred
   * @returns {boolean} - Whether to retry the request
   */
  shouldRetry(error) {
    // Retry on network errors or specific HTTP status codes
    return error.name === 'AbortError' || 
           error.name === 'TypeError' ||
           (error.status >= 500 && error.status < 600);
  }

  /**
   * Delay helper for retries
   * @param {number} ms - Milliseconds to delay
   * @returns {Promise} - Promise that resolves after delay
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Image Processing API calls
  async uploadImage(file) {
    const formData = new FormData();
    formData.append('image', file);

    return this.makeRequest(API_ENDPOINTS.UPLOAD, {
      method: 'POST',
      body: formData
    });
  }

  async confirmCrop(data) {
    return this.makeRequest(API_ENDPOINTS.CONFIRM_CROP, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async manualCrop(data) {
    return this.makeRequest(API_ENDPOINTS.MANUAL_CROP, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async processSignal(data) {
    return this.makeRequest(API_ENDPOINTS.PROCESS_SIGNAL, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async confirmSignal(data) {
    return this.makeRequest(API_ENDPOINTS.CONFIRM_SIGNAL, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  // Queue Management API calls
  async getPendingImages() {
    return this.makeRequest(API_ENDPOINTS.PENDING_IMAGES);
  }

  async processQueuedImageReferee(data) {
    return this.makeRequest(API_ENDPOINTS.PROCESS_QUEUED_IMAGE_REFEREE, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async processCropForSignal(data) {
    return this.makeRequest(API_ENDPOINTS.PROCESS_CROP_FOR_SIGNAL, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async deleteQueuedImage(filename) {
    return this.makeRequest(`${API_ENDPOINTS.DELETE_QUEUED_IMAGE}/${filename}`, {
      method: 'DELETE'
    });
  }

  // Training Data API calls
  async getRefereeTrainingCount() {
    return this.makeRequest(API_ENDPOINTS.REFEREE_TRAINING_COUNT);
  }

  async getSignalClasses() {
    return this.makeRequest(API_ENDPOINTS.SIGNAL_CLASSES);
  }

  async getSignalClassCounts() {
    return this.makeRequest(API_ENDPOINTS.SIGNAL_CLASS_COUNTS);
  }

  async moveRefereeTraining() {
    return this.makeRequest(API_ENDPOINTS.MOVE_REFEREE_TRAINING, {
      method: 'POST'
    });
  }

  async moveSignalTraining() {
    return this.makeRequest(API_ENDPOINTS.MOVE_SIGNAL_TRAINING, {
      method: 'POST'
    });
  }

  // YouTube Processing API calls
  async processYouTubeVideo(data) {
    return this.makeRequest(API_ENDPOINTS.YOUTUBE_PROCESS, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async getYouTubeStatus(folderName) {
    return this.makeRequest(`${API_ENDPOINTS.YOUTUBE_STATUS}/${folderName}`);
  }

  async getAllYouTubeVideos() {
    return this.makeRequest(API_ENDPOINTS.YOUTUBE_VIDEOS);
  }

  async getVideoFrames(folderName, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const endpoint = `${API_ENDPOINTS.YOUTUBE_VIDEO_FRAMES}/${folderName}/frames${queryString ? '?' + queryString : ''}`;
    return this.makeRequest(endpoint);
  }

  async getVideoCrops(folderName, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const endpoint = `${API_ENDPOINTS.YOUTUBE_VIDEO_FRAMES}/${folderName}/crops${queryString ? '?' + queryString : ''}`;
    return this.makeRequest(endpoint);
  }

  async getVideoThumbnails(folderName, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const endpoint = `${API_ENDPOINTS.YOUTUBE_VIDEO_FRAMES}/${folderName}/thumbnails${queryString ? '?' + queryString : ''}`;
    return this.makeRequest(endpoint);
  }

  async deleteYouTubeVideo(folderName) {
    return this.makeRequest(`${API_ENDPOINTS.YOUTUBE_VIDEO_DELETE}/${folderName}/delete`, {
      method: 'DELETE'
    });
  }

  async labelYouTubeFrames(videoId, data) {
    return this.makeRequest(`${API_ENDPOINTS.YOUTUBE_LABEL_FRAMES}/${videoId}/label_frames`, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async addYouTubeFrameToTraining(videoId, data) {
    return this.makeRequest(`${API_ENDPOINTS.YOUTUBE_ADD_TO_TRAINING}/${videoId}/add_to_training`, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async detectYouTubeFrameSignals(videoId, data) {
    return this.makeRequest(`${API_ENDPOINTS.YOUTUBE_DETECT_SIGNALS}/${videoId}/detect_signals`, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async autoLabelYouTubeFrames(videoId, data) {
    return this.makeRequest(`${API_ENDPOINTS.YOUTUBE_AUTOLABEL_FRAMES}/${videoId}/autolabel_frames`, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  // Auto-labeling API calls
  async getPendingAutoLabelCount() {
    return this.makeRequest(API_ENDPOINTS.AUTOLABEL_PENDING_COUNT);
  }

  // File serving helpers
  getCropImageUrl(filename) {
    return `${this.baseURL}${API_ENDPOINTS.CROP_IMAGE}/${filename}`;
  }

  getRefereeCropImageUrl(filename) {
    return `${this.baseURL}${API_ENDPOINTS.REFEREE_CROP_IMAGE}/${filename}`;
  }

  getUploadedImageUrl(filename) {
    return `${this.baseURL}${API_ENDPOINTS.UPLOADED_IMAGE}/${filename}`;
  }

  getYouTubeAssetUrl(videoId, assetType, filename) {
    return `${this.baseURL}${API_ENDPOINTS.YOUTUBE_ASSET_FILE}/${videoId}/${assetType}/${filename}`;
  }
}

// Create and export a singleton instance
const apiService = new APIService();

export default apiService;
export { APIError }; 