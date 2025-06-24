import React, { useRef } from 'react';
import './UploadForm.css';

const UploadForm = ({ onUpload }) => {
  const fileInput = useRef();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const file = fileInput.current.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('image', file);
    const res = await fetch('http://localhost:5000/api/upload', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    onUpload(data, file);
  };

  return (
    <form onSubmit={handleSubmit} className="upload-form">
      <label>Upload a referee image:</label>
      <input type="file" accept="image/*" ref={fileInput} />
      <button type="submit">Upload</button>
    </form>
  );
};

export default UploadForm; 