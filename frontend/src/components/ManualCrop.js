import React, { useRef, useState } from 'react';
import './ManualCrop.css'; // Import the CSS file

const REFEREE_CLASSES = [
  { name: 'referee', id: 0 }, // Assuming 'referee' is class 0 in your YOLO model
  { name: 'none', id: -1 } // Special value for no detection/class
];

const ManualCrop = ({ filename, imageFile, onSubmit, onCancel }) => {
  const canvasRef = useRef();
  const [rect, setRect] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const [start, setStart] = useState(null);
  const [selectedClass, setSelectedClass] = useState(REFEREE_CLASSES[0].name);
  const [workflowAction, setWorkflowAction] = useState('signal');

  // Draw image and rectangle
  React.useEffect(() => {
    const source = imageFile || filename;
    if (!source) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new window.Image();
    img.crossOrigin = "Anonymous"; // Allow loading cross-origin images

    img.onload = () => {
      // Set canvas dimensions to image natural size to avoid distortion
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      if (rect) {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      }
    };
    
    if (typeof source === 'string') {
      img.src = `http://localhost:5000/api/uploads/${source}`;
    } else { // Assumes it's a File object
      img.src = URL.createObjectURL(source);
    }

  }, [filename, imageFile, rect]);

  const handleNoReferee = () => {
    // Call onSubmit with null bbox and -1 class_id to signify a negative sample
    onSubmit({ bbox: null, class_id: -1, proceedToSignal: false });
  };

  const getScaledCoords = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    return { x, y };
  };

  const handleMouseDown = (e) => {
    setDrawing(true);
    const { x, y } = getScaledCoords(e);
    setStart({ x, y });
    setRect(null); // Reset rect on new draw
  };
  
  const handleMouseUp = (e) => {
    setDrawing(false);
    const { x: x2, y: y2 } = getScaledCoords(e);
    // Ensure width/height are positive
    setRect({
      x: Math.min(start.x, x2),
      y: Math.min(start.y, y2),
      w: Math.abs(x2 - start.x),
      h: Math.abs(y2 - start.y)
    });
  };

  const handleMouseMove = (e) => {
    if (!drawing) return;
    const { x: x2, y: y2 } = getScaledCoords(e);
    setRect({
      x: Math.min(start.x, x2),
      y: Math.min(start.y, y2),
      w: Math.abs(x2 - start.x),
      h: Math.abs(y2 - start.y)
    });
  };

  const handleSubmit = () => {
    if (!rect || rect.w === 0 || rect.h === 0) return alert('Please draw a valid rectangle.');
    
    // Find the class_id for the selected class name
    const selectedClassObj = REFEREE_CLASSES.find(cls => cls.name === selectedClass);
    if (!selectedClassObj || selectedClassObj.id === -1) { // If 'none' selected
      // No label will be sent, just acknowledge the manual interaction
      onSubmit({ bbox: [], class_id: -1, proceedToSignal: false }); // Send empty bbox and -1 for 'none'
      return;
    }

    onSubmit({
      bbox: [Math.round(rect.x), Math.round(rect.y), Math.round(rect.x + rect.w), Math.round(rect.y + rect.h)],
      class_id: selectedClassObj.id,
      proceedToSignal: workflowAction === 'signal'
    });
  };

  return (
    <div className="manual-crop-container">
      <div className="controls-and-canvas">
        <div className="canvas-wrapper">
            <canvas ref={canvasRef} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} />
        </div>
        <div className="crop-controls">
          <p>Draw a rectangle around the referee or select an option.</p>
          
          <div className="class-selector">
            <label htmlFor="ref-class">Class: </label>
            <select id="ref-class" value={selectedClass} onChange={(e) => setSelectedClass(e.target.value)}>
              {REFEREE_CLASSES.map(cls => <option key={cls.id} value={cls.name}>{cls.name}</option>)}
            </select>
          </div>

          {selectedClass === 'referee' && (
            <div className="workflow-selection">
              <h4>After saving the crop:</h4>
              <div className="workflow-options">
                <label className="workflow-option">
                  <input
                    type="radio"
                    name="workflow"
                    value="signal"
                    checked={workflowAction === 'signal'}
                    onChange={(e) => setWorkflowAction(e.target.value)}
                  />
                  <div className="option-content">
                    <strong>Label Hand Signals</strong>
                    <span>Proceed to label referee's hand signals</span>
                  </div>
                </label>
                
                <label className="workflow-option">
                  <input
                    type="radio"
                    name="workflow"
                    value="save"
                    checked={workflowAction === 'save'}
                    onChange={(e) => setWorkflowAction(e.target.value)}
                  />
                  <div className="option-content">
                    <strong>Save and Finish</strong>
                    <span>Save crop to training data and finish</span>
                  </div>
                </label>
              </div>
            </div>
          )}

          <div className="action-buttons">
            <button onClick={handleSubmit} disabled={!rect} className="submit-button">
              {selectedClass === 'referee' 
                ? (workflowAction === 'signal' ? 'Save & Label Signals' : 'Save & Finish')
                : 'Submit'
              }
            </button>
            <button onClick={onCancel} className="cancel-button">Skip Image</button>
            <button onClick={handleNoReferee} className="negative-button">No Referee in Image</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ManualCrop; 