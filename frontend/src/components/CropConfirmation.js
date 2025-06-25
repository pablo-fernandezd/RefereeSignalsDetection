import React, { useState, useEffect } from 'react';
import './CropConfirmation.css';

const CropConfirmation = ({ cropUrl, onConfirm, onProceedToSignal, onSaveAndFinish }) => {
  // Simplified keyboard shortcuts for auto-detect confirmation
  useEffect(() => {
    const handleKeyPress = (e) => {
      // Don't process if user is typing in an input field
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
        return;
      }

      if (e.key.toLowerCase() === 'y') {
        // Y for Yes - proceed to signal detection (same as old Enter)
        e.preventDefault();
        e.stopPropagation();
        onProceedToSignal(true);
      } else if (e.key.toLowerCase() === 'n') {
        // N for No - reject crop (same as old Space)
        e.preventDefault();
        e.stopPropagation();
        onConfirm(false);
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [onConfirm, onProceedToSignal]);

  return (
    <div className="crop-confirmation-container">
      <h3>Is the referee crop correct?</h3>
      <div className="keyboard-shortcuts">
        <span>Shortcuts: <kbd>Y</kbd> Yes | <kbd>N</kbd> No</span>
      </div>
      <img src={cropUrl} alt="Referee Crop" className="crop-preview" />

      <div className="crop-confirmation-buttons">
        <button className="confirm-button" onClick={() => onProceedToSignal(true)}>
          <kbd>Y</kbd> Yes - Proceed to Signal Detection
        </button>
        <button className="reject-button" onClick={() => onConfirm(false)}>
          <kbd>N</kbd> No - This crop is incorrect
        </button>
      </div>
    </div>
  );
};

export default CropConfirmation; 