import React, { useState, useEffect } from 'react';
import './SignalConfirmation.css';

const SIGNAL_CLASSES = [
  'armLeft', 'armRight', 'hits', 'leftServe', 'net', 'outside', 'rightServe', 'touched', 'none'
];

const SignalConfirmation = ({ predictedClass, confidence, signalBbox, cropFilenameForSignal, onConfirm, onCancel, originalFilename, imageUrl, onDiscard }) => {
  const [isCorrect, setIsCorrect] = useState(true);
  const [selectedClass, setSelectedClass] = useState(predictedClass || 'none');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Add keyboard shortcuts for faster navigation
  useEffect(() => {
    const handleKeyPress = (e) => {
      // Don't process keyboard shortcuts if already submitting
      if (isSubmitting) return;
      
      // Don't process if user is typing in an input field
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
        return;
      }
      
      if (e.key === 'y' || e.key === 'Y') {
        // 'Y' for Yes/Correct - accept prediction and submit immediately
        e.preventDefault();
        e.stopPropagation();
        if (!isSubmitting) {
          setIsCorrect(true);
          // Small delay to ensure state is updated, then submit
          setTimeout(() => {
            handleSubmit();
          }, 50);
        }
      } else if (e.key === 'n' || e.key === 'N') {
        // 'N' for No/Incorrect - just set state
        e.preventDefault();
        if (!isSubmitting) {
          setIsCorrect(false);
        }
      } else if (e.key === 'Enter') {
        // Enter to submit - prevent default to avoid double submission
        e.preventDefault();
        e.stopPropagation();
        if (!isSubmitting) {
          handleSubmit();
        }
      } else if (e.key === 'd' || e.key === 'D') {
        // 'D' for Discard
        e.preventDefault();
        handleDiscard();
      } else if (e.key >= '1' && e.key <= '9') {
        // Number keys 1-9 for quick class selection
        e.preventDefault();
        const classIndex = parseInt(e.key) - 1;
        if (classIndex < SIGNAL_CLASSES.length) {
          setSelectedClass(SIGNAL_CLASSES[classIndex]);
          setIsCorrect(false); // Automatically set to incorrect when manually selecting
        }
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [isSubmitting]);

  const handleSubmit = () => {
    // Prevent duplicate submissions
    if (isSubmitting) {
      return;
    }
    
    setIsSubmitting(true);
    
    // Simple logic: just send the selected class
    const finalSelectedClass = isCorrect ? (predictedClass || 'none') : selectedClass;

    onConfirm({
      selected_class: finalSelectedClass,
      confidence: confidence || 0.0,
      signal_bbox_yolo: signalBbox
    });
    
    // Reset submitting state after a delay
    setTimeout(() => {
      setIsSubmitting(false);
    }, 500);
  };

  const handleCorrectClick = () => {
    setIsCorrect(true);
    // Don't auto-submit when clicking - let user press Enter or Y key
  };

  const handleBackToCrop = () => {
    if (onCancel) onCancel();
  };

  const handleDiscard = () => {
    if (onDiscard) {
      onDiscard();
    }
  };

  // Use provided imageUrl or construct the URL for the referee crop image
  const refereeCropImageUrl = imageUrl || (cropFilenameForSignal ? `http://localhost:5000/api/referee_crop_image/${cropFilenameForSignal}` : null);

  return (
    <div className="signal-confirmation-container">
      <h3>Hand Signal Labeling</h3>
      <div className="keyboard-shortcuts">
        <span>Shortcuts: <kbd>Y</kbd> Accept & Submit | <kbd>N</kbd> Choose Class | <kbd>Enter</kbd> Submit | <kbd>D</kbd> Discard | <kbd>1-9</kbd> Quick class selection</span>
      </div>
      {refereeCropImageUrl && (
        <div className="referee-crop-display">
          <h4>Referee Crop:</h4>
          <img src={refereeCropImageUrl} alt="Referee Crop" className="crop-image" />
        </div>
      )}
      
      <div className="prediction-info">
        <p className="signal-prediction-info">
          <strong>AI Prediction:</strong> {predictedClass || 'none'} 
          {confidence && <span className="confidence"> (Confidence: {(confidence * 100).toFixed(1)}%)</span>}
        </p>
      </div>

      <div className="signal-feedback-section">
        <h4>Accept AI prediction or choose correct class:</h4>
        <div className="signal-feedback-options">
          <label className="feedback-option" onClick={() => !isSubmitting && handleCorrectClick()}>
            <input
              type="radio"
              checked={isCorrect}
              onChange={() => {}} // Prevent onChange from triggering
              disabled={isSubmitting}
            />
            <span><kbd>Y</kbd> ✓ Accept "{predictedClass || 'none'}"</span>
          </label>
          <label className="feedback-option" onClick={() => !isSubmitting && setIsCorrect(false)}>
            <input
              type="radio"
              checked={!isCorrect}
              onChange={() => {}} // Prevent onChange from triggering
              disabled={isSubmitting}
            />
            <span><kbd>N</kbd> ✗ Choose different class</span>
          </label>
        </div>
      </div>

      {!isCorrect && (
        <div className="signal-correct-class-selection">
          <h4>Select the correct signal:</h4>
          <select 
            value={selectedClass} 
            onChange={(e) => setSelectedClass(e.target.value)} 
            className="class-select"
            disabled={isSubmitting}
          >
            {SIGNAL_CLASSES.map((cls, index) => (
              <option key={cls} value={cls}>
                {index + 1}. {cls === 'none' ? 'No Signal (Negative Sample)' : cls.charAt(0).toUpperCase() + cls.slice(1)}
              </option>
            ))}
          </select>
          <div className="class-shortcuts">
            <span>Use number keys <kbd>1</kbd>-<kbd>9</kbd> for quick selection</span>
          </div>
        </div>
      )}

      <div className="action-buttons">
        <button onClick={() => handleSubmit()} className="submit-button" disabled={isSubmitting}>
          <kbd>Enter</kbd> {isSubmitting ? 'Saving...' : 'Save Signal Label'}
        </button>
        {onDiscard && (
          <button onClick={() => handleDiscard()} className="discard-button">
            <kbd>D</kbd> Discard Detection
          </button>
        )}
        <button onClick={() => handleBackToCrop()} className="back-button">
          Skip to Next Image
        </button>
      </div>
    </div>
  );
};

export default SignalConfirmation; 