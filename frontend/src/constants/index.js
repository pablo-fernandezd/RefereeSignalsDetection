/**
 * Frontend Constants and Configuration
 * 
 * This file contains all constants used throughout the frontend application,
 * including API endpoints, configuration values, and UI constants.
 */

// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:5000',
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3
};

// API Endpoints
export const API_ENDPOINTS = {
  // Image processing endpoints
  UPLOAD: '/api/upload',
  CONFIRM_CROP: '/api/confirm_crop',
  MANUAL_CROP: '/api/manual_crop',
  PROCESS_SIGNAL: '/api/process_signal',
  CONFIRM_SIGNAL: '/api/confirm_signal',
  
  // Queue management endpoints
  PENDING_IMAGES: '/api/pending_images',
  PROCESS_QUEUED_IMAGE_REFEREE: '/api/process_queued_image_referee',
  PROCESS_CROP_FOR_SIGNAL: '/api/process_crop_for_signal',
  DELETE_QUEUED_IMAGE: '/api/queue/image',
  
  // Training data endpoints
  REFEREE_TRAINING_COUNT: '/api/referee_training_count',
  SIGNAL_CLASSES: '/api/signal_classes',
  SIGNAL_CLASS_COUNTS: '/api/signal_class_counts',
  MOVE_REFEREE_TRAINING: '/api/move_referee_training',
  MOVE_SIGNAL_TRAINING: '/api/move_signal_training',
  
  // YouTube processing endpoints
  YOUTUBE_PROCESS: '/api/youtube/process',
  YOUTUBE_STATUS: '/api/youtube/status',
  YOUTUBE_VIDEOS: '/api/youtube/videos',
  YOUTUBE_VIDEO_FRAMES: '/api/youtube/video',
  YOUTUBE_VIDEO_DELETE: '/api/youtube/video',
  YOUTUBE_LABEL_FRAMES: '/api/youtube/video',
  YOUTUBE_ADD_TO_TRAINING: '/api/youtube/video',
  YOUTUBE_DETECT_SIGNALS: '/api/youtube/video',
  YOUTUBE_AUTOLABEL_FRAMES: '/api/youtube/video',
  
  // File serving endpoints
  CROP_IMAGE: '/api/crop',
  REFEREE_CROP_IMAGE: '/api/referee_crop_image',
  UPLOADED_IMAGE: '/api/uploads',
  YOUTUBE_ASSET_FILE: '/api/youtube/data',
  
  // Auto-labeling endpoints
  AUTOLABEL_PENDING_COUNT: '/api/autolabel/pending_count'
};

// Referee Detection Classes
export const REFEREE_CLASSES = [
  { name: 'referee', id: 0, displayName: 'Referee' },
  { name: 'none', id: -1, displayName: 'No Referee' }
];

// Signal Classes (these should match the backend model)
export const SIGNAL_CLASSES = [
  'armLeft', 'armRight', 'hits', 'leftServe', 
  'net', 'outside', 'rightServe', 'touched', 'none'
];

// UI Constants
export const UI_CONSTANTS = {
  // Navigation
  NAVIGATION_ITEMS: [
    { key: 'dashboard', label: 'Dashboard', icon: '📊' },
    { key: 'upload', label: 'Upload Image', icon: '📤' },
    { key: 'youtube', label: 'YouTube Processing', icon: '🎥' },
    { key: 'assets', label: 'Video Assets', icon: '📁' },
    { key: 'labelingQueue', label: 'Labeling Queue', icon: '🏷️' }
  ],
  
  // Themes
  THEMES: {
    LIGHT: 'light',
    DARK: 'dark'
  },
  
  // File upload limits
  MAX_FILE_SIZE: 100 * 1024 * 1024, // 100MB
  ALLOWED_IMAGE_TYPES: ['image/jpeg', 'image/jpg', 'image/png'],
  
  // Processing states
  PROCESSING_STATES: {
    IDLE: 'idle',
    UPLOADING: 'uploading',
    PROCESSING: 'processing',
    COMPLETED: 'completed',
    ERROR: 'error'
  },
  
  // Workflow steps
  UPLOAD_WORKFLOW_STEPS: {
    UPLOAD: 0,
    CROP_CONFIRMATION: 1,
    SIGNAL_CONFIRMATION: 2,
    MANUAL_CROP: 3
  }
};

// YouTube Processing Constants
export const YOUTUBE_CONSTANTS = {
  PROCESSING_STAGES: {
    DOWNLOADING: 'downloading',
    EXTRACTING_FRAMES: 'extracting_frames',
    DETECTING_REFEREES: 'detecting_referees',
    SEGMENTING: 'segmenting',
    COMPLETED: 'completed'
  },
  
  ASSET_TYPES: {
    FRAMES: 'frames',
    CROPS: 'crops',
    PROCESSED: 'processed',
    THUMBNAILS: 'thumbnails',
    SEGMENTS: 'segments'
  },
  
  POLLING_INTERVAL: 2000, // 2 seconds
  MAX_POLLING_ATTEMPTS: 300 // Stop polling after 10 minutes
};

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error occurred. Please check your connection.',
  UPLOAD_FAILED: 'Failed to upload image. Please try again.',
  PROCESSING_FAILED: 'Processing failed. Please try again.',
  INVALID_FILE_TYPE: 'Invalid file type. Please upload a valid image.',
  FILE_TOO_LARGE: 'File is too large. Maximum size is 100MB.',
  YOUTUBE_URL_INVALID: 'Please enter a valid YouTube URL.',
  UNKNOWN_ERROR: 'An unknown error occurred. Please try again.'
};

// Success Messages
export const SUCCESS_MESSAGES = {
  UPLOAD_SUCCESS: 'Image uploaded successfully!',
  CROP_SAVED: 'Crop saved to training data!',
  SIGNAL_CONFIRMED: 'Signal confirmation saved!',
  YOUTUBE_PROCESSING_STARTED: 'YouTube video processing started!',
  TRAINING_DATA_MOVED: 'Training data moved successfully!'
};

// Local Storage Keys
export const STORAGE_KEYS = {
  THEME: 'referee_detection_theme',
  USER_PREFERENCES: 'referee_detection_preferences',
  RECENT_UPLOADS: 'referee_detection_recent_uploads'
};

// Animation and Timing Constants
export const ANIMATION_CONSTANTS = {
  FADE_DURATION: 300,
  SLIDE_DURATION: 250,
  BOUNCE_DURATION: 500,
  NOTIFICATION_TIMEOUT: 5000,
  POLLING_INTERVAL: 2000
};

// Color Scheme
export const COLOR_SCHEME = {
  PRIMARY: '#007bff',
  SECONDARY: '#6c757d',
  SUCCESS: '#28a745',
  WARNING: '#ffc107',
  DANGER: '#dc3545',
  INFO: '#17a2b8',
  LIGHT: '#f8f9fa',
  DARK: '#343a40'
};

export default {
  API_CONFIG,
  API_ENDPOINTS,
  REFEREE_CLASSES,
  SIGNAL_CLASSES,
  UI_CONSTANTS,
  YOUTUBE_CONSTANTS,
  ERROR_MESSAGES,
  SUCCESS_MESSAGES,
  STORAGE_KEYS,
  ANIMATION_CONSTANTS,
  COLOR_SCHEME
}; 