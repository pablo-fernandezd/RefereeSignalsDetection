import React, { useState, useEffect } from 'react';
import './CropConfirmation.css';

const CropConfirmation = ({ cropUrl, onConfirm, onProceedToSignal, onSaveAndFinish }) => {
  const [selectedAction, setSelectedAction] = useState('signal');

  // Add keyboard shortcuts for faster navigation
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'y' || e.key === 'Y') {
        // 'Y' for Yes - proceed with selected action
        e.preventDefault();
        handleConfirm();
      } else if (e.key === 'n' || e.key === 'N') {
        // 'N' for No - reject crop
        e.preventDefault();
        onConfirm(false);
      } else if (e.key === '1') {
        // '1' for Signal labeling
        e.preventDefault();
        setSelectedAction('signal');
      } else if (e.key === '2') {
        // '2' for Save and finish
        e.preventDefault();
        setSelectedAction('save');
      } else if (e.key === 'Enter') {
        // Enter to confirm
        e.preventDefault();
        handleConfirm();
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [selectedAction]);

  const handleConfirm = () => {
    if (selectedAction === 'signal') {
      onProceedToSignal(true);
    } else if (selectedAction === 'save') {
      onSaveAndFinish(true);
    }
  };

  return (
    <div className="crop-confirmation-container">
      <h3>Is the referee crop correct?</h3>
      <div className="keyboard-shortcuts">
        <span>Shortcuts: <kbd>Y</kbd> Yes | <kbd>N</kbd> No | <kbd>1</kbd> Signal | <kbd>2</kbd> Save | <kbd>Enter</kbd> Confirm</span>
      </div>
      <img src={cropUrl} alt="Referee Crop" className="crop-preview" />
      
      <div className="action-selection">
        <h4>What would you like to do next?</h4>
        <div className="action-options">
          <label className="action-option">
            <input
              type="radio"
              name="action"
              value="signal"
              checked={selectedAction === 'signal'}
              onChange={(e) => setSelectedAction(e.target.value)}
            />
            <div className="option-content">
                <strong>Label Hand Signals <kbd>1</kbd></strong>
              <span>Proceed to label referee's hand signals in this crop</span>
            </div>
          </label>
          
          <label className="action-option">
            <input
              type="radio"
              name="action"
              value="save"
              checked={selectedAction === 'save'}
              onChange={(e) => setSelectedAction(e.target.value)}
            />
            <div className="option-content">
                <strong>Save and Finish <kbd>2</kbd></strong>
              <span>Save this referee crop to training data and finish</span>
            </div>
          </label>
        </div>
      </div>

      <div className="crop-confirmation-buttons">
        <button className="confirm-button" onClick={handleConfirm}>
          <kbd>Y</kbd> {selectedAction === 'signal' ? 'Proceed to Signal Labeling' : 'Save and Finish'}
        </button>
        <button className="reject-button" onClick={() => onConfirm(false)}>
          <kbd>N</kbd> Crop is Wrong - Manual Crop
        </button>
      </div>
    </div>
  );
};

export default CropConfirmation; 