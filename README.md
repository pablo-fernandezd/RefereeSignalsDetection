# Referee Detection System

A computer vision system designed to detect and track referees in sports videos using YOLO (You Only Look Once) object detection, with a focus on volleyball officiating and signal classification. The system now includes advanced YouTube video processing capabilities for automated data collection and analysis.

## Acknowledgments

This project has been developed in collaboration with the Real Federación Española de Voleibol (RFEVB), who granted permission for the use of their official match videos for AI training purposes. This collaboration has been crucial for the development and validation of the system.

## Training Dataset

The model has been trained using a carefully curated dataset of referee signals from official volleyball matches. The dataset includes the following signal classes:

| Signal Class | Image Count |
|--------------|-------------|
| Point (Left Arm) | 381 |
| Point (Right Arm) | 282 |
| Service Fault (Left Serve) | 185 |
| Service Fault (Right Serve) | 161 |
| Net Touch | 164 |
| Ball Outside | 113 |
| Four Hits | 52 |
| Ball Touched | 48 |
| **Subtotal (Listed Classes)** | **1,386** |

**Note:** The training images are currently private and not publicly available due to copyright restrictions. Access to the dataset requires explicit permission from the RFEVB.

## Training Metrics

![Training Metrics](docs/training_metrics.png)

**Explanation:**

- **Model Performance (mAP):**
  The top plot shows the evolution of the main performance metric: mAP (mean Average Precision), both overall and at an IoU threshold of 0.5, over 350 epochs. The model quickly reaches a high mAP and maintains it throughout training, indicating strong and stable detection and classification performance for referee signals.

- **Box Loss, Class Loss, Object Loss:**
  The three lower plots show the evolution of the loss values during model training. All losses decrease rapidly in the early epochs and stabilize at low values, with occasional spikes. This indicates that the model is learning effectively and converging well, with only minor fluctuations.

These metrics reflect that the model is able to effectively learn to detect and classify referee signals, achieving high and stable performance across a long training period.

## System Architecture

![System Flowchart](docs/flowchart.svg)

## Overview

This system processes video files to detect and track referees, creating segmented video clips that focus on the referee's movements. The system uses a trained YOLO model optimized for referee detection and includes features for video segmentation and processing. It represents a significant advancement in automated sports analysis technology.

## Key Features

- Real-time referee detection using YOLO model
- Video segmentation into configurable time intervals
- Automatic video processing pipeline
- GPU acceleration support (CUDA)
- Frame tracking and persistence
- Automatic file management for processed videos
- Temporal consistency mechanisms for robust video analysis
- Dual-model architecture for detection and classification tasks
- **NEW: YouTube video processing and automated data collection**
- **NEW: Web-based annotation interface with three-screen architecture**
- **NEW: Automated cropping with configurable margins**

## Research Contributions

### Technical Contributions
- A dual-model architecture that effectively separates detection and classification tasks
- Novel implementation of temporal consistency mechanisms for robust video analysis
- Empirical evaluation of data augmentation strategies for sports-specific gesture recognition
- Insights into precision-recall trade-offs in signal classification models
- **NEW: Automated YouTube video processing pipeline for scalable data collection**

### Practical Applications
- Automated volleyball match analysis system
- Enhanced broadcast capabilities for all competition levels
- Foundation for automated graphics generation and statistics compilation
- Pathway toward comprehensive match analysis through unsupervised learning
- **NEW: Web-based annotation system for collaborative data labeling**

## Impact

### Sports Broadcasting
- Reduces personnel requirements for professional-quality productions
- Enables comprehensive statistics and graphics for viewer engagement
- Makes advanced broadcast capabilities accessible to all competition levels
- Creates opportunities for new interactive viewing experiences

### Officiating and Analysis
- Provides consistent and objective recording of match events
- Enhances training resources for referees through automated signal analysis
- Generates comprehensive match statistics without manual data entry
- Identifies patterns and insights that might be missed in manual analysis

## Requirements

### System Requirements
- FFmpeg (required for video processing)
- CUDA-capable GPU (optional, for GPU acceleration)

### Python Dependencies
- Python 3.x
- OpenCV (cv2)
- PyTorch
- Ultralytics YOLO
- Flask
- Flask-CORS
- yt-dlp
- Pillow
- PyYAML

### Node.js Dependencies
- Node.js 14+ (for frontend development)
- npm or yarn

## Web Application

This project now includes a comprehensive web application for referee and signal detection, enabling interactive annotation and data collection for model retraining. The web application features a three-screen architecture built with a Flask backend and a React frontend.

### Application Screens

#### 1. Dashboard
- Overview of training data statistics
- Referee training image counts
- Signal class distribution
- Data management tools for moving training data to global folders

#### 2. Image Upload & Processing
- **Image Upload:** Users can upload images for analysis
- **Referee Detection & Cropping:** The system automatically detects referees and proposes a crop with 20% margins
- **Signal Detection & Confirmation:** After referee confirmation, the cropped image is processed for signal detection
- **Manual Correction:** Users can manually adjust crops and correct signal predictions
- **Data Collection:** All confirmed and corrected data is saved with YOLO-formatted labels

#### 3. YouTube Processing (NEW)
- **Video URL Input:** Process YouTube videos directly by URL
- **Automated Processing Pipeline:**
  - Video download using yt-dlp
  - Segmentation into 10-minute segments
  - Frame extraction every 30 seconds
  - Optional auto-crop with 20% margins
- **Progress Tracking:** Real-time monitoring of processing status
- **Result Visualization:** Browse segments, frames, and crops
- **Organized Storage:** Structured folder organization by video

### Backend Architecture

#### Core API (`backend/app.py`)
- **Duplicate Prevention:** MD5 hash-based deduplication mechanism
- **File Management:** Timestamp-based unique filenames
- **Signal Processing:** Complete signal detection and confirmation pipeline
- **YouTube Processing:** New endpoints for video processing and status tracking

#### YouTube Processor (`backend/youtube_processor.py`)
- **Video Download:** Automated YouTube video downloading
- **Processing Pipeline:** Segmentation, frame extraction, and auto-cropping
- **Status Management:** Real-time processing status tracking
- **File Organization:** Structured storage system

### Frontend Architecture

#### React Application (`frontend/src/`)
- **Navigation System:** Three-screen navigation with active state management
- **Upload Form:** Drag-and-drop image upload with progress feedback
- **Crop Confirmation:** Interactive crop review and confirmation
- **Signal Confirmation:** Signal prediction review with visual feedback
- **YouTube Processing:** Complete video processing interface
- **Responsive Design:** Mobile and desktop optimized interface

## Installation

### 1. Install FFmpeg
- **Windows**: Download from [FFmpeg official website](https://ffmpeg.org/download.html) or install via Chocolatey:
  ```bash
  choco install ffmpeg
  ```
- **Linux**:
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```

### 2. Clone the repository
```bash
git clone https://github.com/pablo-fernandezd/RefereeSinglasDetection.git
cd RefereeDetection
```

### 3. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Install Node.js dependencies
```bash
cd ../frontend
npm install
```

### 5. Download model files (optional but recommended)
Download the trained models and place them in the `models/` directory:
- `bestRefereeDetection.pt`: Referee detection model
- `bestSignalsDetection.pt`: Signal classification model

**Note:** Without the model files, the auto-crop functionality will not work, but video downloading and segmentation will still function.

## Usage

### Starting the Application

1. **Start the backend server:**
```bash
cd backend
python app.py
```
The Flask server will be available at `http://localhost:5000`

2. **Start the frontend development server:**
```bash
cd frontend
npm start
```
The React application will be available at `http://localhost:3000`

### Using the Application

#### Dashboard
- View training data statistics
- Monitor referee and signal training counts
- Move training data to global folders

#### Image Processing
1. Navigate to "Upload Image" tab
2. Upload an image containing a referee
3. Review and confirm the automatic crop
4. Review and confirm the signal detection
5. Data is automatically saved for training

#### YouTube Processing
1. Navigate to "YouTube" tab
2. Enter a YouTube video URL
3. Configure auto-crop settings (optional)
4. Monitor processing progress
5. Browse results in organized folders

## Project Structure

```
RefereeDetection/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── youtube_processor.py   # YouTube processing module
│   ├── models/
│   │   └── inference.py       # Model inference functions
│   ├── data/
│   │   ├── referee_training_data/  # Referee training images
│   │   ├── signal_training_data/   # Signal training images
│   │   └── youtube_videos/         # Processed YouTube videos
│   ├── static/
│   │   ├── uploads/           # Temporary upload storage
│   │   ├── referee_crops/     # Referee crop images
│   │   └── signals/           # Signal images
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── UploadForm.js
│   │   │   ├── CropConfirmation.js
│   │   │   ├── SignalConfirmation.js
│   │   │   ├── ManualCrop.js
│   │   │   └── YouTubeProcessing.js
│   │   ├── App.js             # Main application
│   │   └── App.css            # Application styles
│   └── package.json           # Node.js dependencies
├── models/
│   ├── bestRefereeDetection.pt    # Referee detection model
│   └── bestSignalsDetection.pt    # Signal classification model
├── data/
│   ├── input_videos/          # Input video directory
│   ├── processed_videos/      # Processed video output
│   └── used_videos/           # Processed input files
└── README.md
```

## Configuration

### Backend Configuration
The system can be configured through parameters in the respective modules:

- **YouTube Processing** (`youtube_processor.py`):
  - `SEGMENT_DURATION`: Duration of video segments in seconds (default: 600)
  - `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.7)
  - `MODEL_SIZE`: Input size for the model (default: 640)

- **Video Processing** (`main.py`):
  - `DEVICE`: 'cuda' for GPU acceleration or 'cpu' for CPU processing
  - `SEGMENT_DURATION`: Duration of video segments in seconds (default: 3600)

### Frontend Configuration
- **API Endpoints**: Configured in component files for backend communication
- **Polling Intervals**: Configurable for real-time status updates

## YouTube Processing Details

### Processing Pipeline
1. **Video Download**: Uses yt-dlp for reliable YouTube video downloading
2. **Segmentation**: Splits videos into 10-minute segments for manageable processing
3. **Frame Extraction**: Extracts frames every 30 seconds for analysis
4. **Auto-cropping**: Optional automatic referee detection and cropping with 20% margins
5. **Storage Organization**: Creates structured folder hierarchy for each video

### Folder Structure
```
data/youtube_videos/
├── video_id_timestamp/
│   ├── original/           # Original downloaded video
│   ├── segments/           # 10-minute video segments
│   ├── frames/             # Extracted frames (every 30s)
│   ├── crops/              # Auto-cropped referee images
│   ├── processed/          # Additional processed files
│   └── processing_info.json # Processing metadata
```

### API Endpoints
- `POST /api/youtube/process` - Start video processing
- `GET /api/youtube/videos` - List processed videos
- `GET /api/youtube/status/<folder>` - Get processing status
- `GET /api/youtube/video/<folder>/frames` - List extracted frames
- `GET /api/youtube/video/<folder>/crops` - List auto-crops
- `GET /api/youtube/video/<folder>/segments` - List video segments

## Testing

Run the test suite to verify system functionality:

```bash
cd backend
python test_youtube_basic.py
```

This will test:
- Dependency availability
- Directory structure creation
- Video ID extraction
- File organization
- Frontend component structure

## Future Development

The system is designed to evolve toward:
- Unsupervised learning approaches for direct match event detection
- Enhanced viewer experiences through automated graphics generation
- More comprehensive match analysis capabilities
- Integration with broader sports technology ecosystems
- **Advanced YouTube processing with multiple video sources**
- **Real-time processing capabilities for live streams**
- **Enhanced auto-cropping with multiple referee detection**
- **Integration with cloud storage for scalable processing**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Pablo Fernandez
