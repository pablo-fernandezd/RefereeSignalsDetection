import React, { useState, useEffect } from 'react';
import './SignalConfirmation.css';

const SIGNAL_CLASSES = [
  'armLeft', 'armRight', 'hits', 'leftServe', 'net', 'outside', 'rightServe', 'touched', 'none'
];

const SignalConfirmation = ({ predictedClass, confidence, signalBbox, cropFilenameForSignal, onConfirm, onCancel, originalFilename, imageUrl, onDiscard }) => {
  const [isCorrect, setIsCorrect] = useState(true);
  const [selectedClass, setSelectedClass] = useState(predictedClass || 'none');

  // Add keyboard shortcuts for faster navigation
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'y' || e.key === 'Y') {
        // 'Y' for Yes/Correct - auto submit
        e.preventDefault();
        setIsCorrect(true);
        // Auto-submit when marking as correct
        setTimeout(() => {
          handleSubmit(true);
        }, 100);
      } else if (e.key === 'n' || e.key === 'N') {
        // 'N' for No/Incorrect
        e.preventDefault();
        setIsCorrect(false);
      } else if (e.key === 'Enter') {
        // Enter to submit
        e.preventDefault();
        handleSubmit();
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
  }, []);

  const handleSubmit = (forceCorrect = false) => {
    // Determine the final selected class
    let finalSelectedClass;
    const finalIsCorrect = forceCorrect || isCorrect;
    
    if (finalIsCorrect) {
      // If user says prediction is correct, use the predicted class or default to 'none'
      finalSelectedClass = predictedClass || 'none';
    } else {
      // If user says prediction is incorrect, use their manually selected class
      finalSelectedClass = selectedClass;
    }

    onConfirm({
      correct: finalIsCorrect,
      selected_class: finalSelectedClass,
      signal_bbox_yolo: signalBbox,
      original_filename: originalFilename
    });
  };

  const handleCorrectClick = () => {
    setIsCorrect(true);
    // Auto-submit when clicking correct
    setTimeout(() => {
      handleSubmit(true);
    }, 100);
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
        <span>Shortcuts: <kbd>Y</kbd> Correct | <kbd>N</kbd> Incorrect | <kbd>D</kbd> Discard | <kbd>1-9</kbd> Quick class selection | <kbd>Enter</kbd> Submit</span>
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
        <h4>Is this prediction correct?</h4>
        <div className="signal-feedback-options">
          <label className="feedback-option">
            <input
              type="radio"
              checked={isCorrect}
              onChange={handleCorrectClick}
            />
            <span><kbd>Y</kbd> ✓ Correct</span>
          </label>
          <label className="feedback-option">
            <input
              type="radio"
              checked={!isCorrect}
              onChange={() => setIsCorrect(false)}
            />
            <span><kbd>N</kbd> ✗ Incorrect</span>
          </label>
        </div>
      </div>

      <div className="signal-correct-class-selection">
        <h4>{isCorrect ? 'Confirmed signal:' : 'Select the correct signal:'}</h4>
        <select 
          value={isCorrect ? (predictedClass || 'none') : selectedClass} 
          onChange={e => setSelectedClass(e.target.value)} 
          className="class-select"
          disabled={isCorrect}
        >
          {SIGNAL_CLASSES.map((cls, index) => (
            <option key={cls} value={cls}>
              {index + 1}. {cls === 'none' ? 'No Signal (Negative Sample)' : cls.charAt(0).toUpperCase() + cls.slice(1)}
            </option>
          ))}
        </select>
        {!isCorrect && (
          <div className="class-shortcuts">
            <span>Use number keys <kbd>1</kbd>-<kbd>9</kbd> for quick selection</span>
          </div>
        )}
      </div>

      <div className="action-buttons">
        <button onClick={handleSubmit} className="submit-button">
          <kbd>Enter</kbd> Save Signal Label
        </button>
        {onDiscard && (
          <button onClick={handleDiscard} className="discard-button">
            <kbd>D</kbd> Discard Detection
          </button>
        )}
        <button onClick={handleBackToCrop} className="back-button">
          Back to Manual Crop
        </button>
      </div>
    </div>
  );
};

export default SignalConfirmation; 