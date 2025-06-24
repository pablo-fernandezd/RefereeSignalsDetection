import os
import cv2
import torch
from datetime import datetime
from ultralytics import YOLO

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SIZE = 640  # Model input size
SEGMENT_DURATION = 3600  # 1 hour in seconds
CONFIDENCE_THRESHOLD = 0.7


class RefereeProcessor:
    def __init__(self):
        # Load trained model from the models directory
        self.model = YOLO('../models/bestRefereeDetection.pt').to(DEVICE)
        self.model.fuse()

        # CUDA optimizations
        if DEVICE == 'cuda':
            self.model.half()  # Use FP16 precision
            torch.backends.cudnn.benchmark = True

        # Get referee class ID from model metadata
        self.class_id = self._get_referee_class_id()

    def _get_referee_class_id(self):
        """Retrieves the class ID for 'referee' from model names"""
        if 'referee' in self.model.names.values():
            return [k for k, v in self.model.names.items() if v == 'referee'][0]
        return 0  # Default to first class if not found

    # Use corrected relative paths for data directories
    def process_videos(self, input_dir='../data/input_videos', output_dir='../data/processed_videos', used_dir='../data/used_videos'):
        """Main processing pipeline for video files"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(used_dir, exist_ok=True)

        for video_file in self._get_video_files(input_dir):
            video_path = os.path.join(input_dir, video_file)
            self._process_single_video(video_path, output_dir, used_dir)

    def _get_video_files(self, directory):
        """Get list of .mp4 files in target directory"""
        return [f for f in os.listdir(directory) if f.endswith('.mp4')]

    def _process_single_video(self, video_path, output_dir, used_dir):
        """Process individual video file"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) # Keep original FPS retrieval
        if fps <= 0: fps = 30 # Handle potential invalid FPS

        # Video segmentation setup
        frames_per_segment = int(SEGMENT_DURATION * fps) if SEGMENT_DURATION > 0 else 0
        segment_counter = 1
        current_frame = 0
        video_writer = None
        last_valid_frame = None
        target_size = (MODEL_SIZE, MODEL_SIZE) # Define target size

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame through detection pipeline
                processed_frame = self._process_frame(frame, last_valid_frame)

                if processed_frame is not None:
                    # --- Ensure frame matches target size before writing ---
                    if processed_frame.shape[0] != target_size[1] or processed_frame.shape[1] != target_size[0]:
                       processed_frame = cv2.resize(processed_frame, target_size)
                    last_valid_frame = processed_frame
                    current_frame += 1

                    # Manage video segmentation
                    # Check frames_per_segment only if it's positive
                    if frames_per_segment > 0 and current_frame % frames_per_segment == 0:
                        self._close_writer(video_writer)
                        # Pass target_size to writer creation
                        video_writer = self._create_new_writer(video_path, output_dir, segment_counter, fps, target_size)
                        segment_counter += 1

                    # Initialize writer if needed
                    if video_writer is None:
                         # Pass target_size to writer creation
                        video_writer = self._create_new_writer(video_path, output_dir, segment_counter, fps, target_size)

                    # Write processed frame only if writer is valid
                    if video_writer:
                       video_writer.write(processed_frame)
                # --- Keep original behavior: if no detection, nothing is written unless last_valid used ---
                # --- Consider if writing last_valid_frame on non-detection is desired from original ---
                # elif last_valid_frame is not None and video_writer is not None:
                #    video_writer.write(last_valid_frame) # Optional: write last valid frame if needed

        finally:
            # Cleanup resources
            self._close_writer(video_writer)
            cap.release()
            # Use the original move function (or adapt slightly if needed)
            self._move_processed_file(video_path, used_dir)

    def _process_frame(self, frame, last_valid):
        """Full frame processing pipeline"""
        # Prepare tensor for model input (resizes frame)
        frame_tensor = self._prepare_frame_tensor(frame)

        # Run model inference
        results = self.model.track(
            frame_tensor,
            conf=CONFIDENCE_THRESHOLD,
            classes=[self.class_id],
            verbose=False,
            persist=True # Keep persist=True for tracking
        )[0]

        # Handle detection and cropping
        return self._handle_frame_cropping(frame, results, last_valid)

    def _prepare_frame_tensor(self, frame):
        """Convert frame to optimized tensor format"""
        # Resize to model input size
        resized_frame = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE))
        # Convert to RGB and move to device
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_frame).to(DEVICE)

        # Apply device-specific optimizations
        tensor = tensor.half() if DEVICE == 'cuda' else tensor.float()

        # Normalize and reshape tensor (HWC to BCHW)
        tensor = tensor.permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0)

    def _handle_frame_cropping(self, frame, results, last_valid):
        """Handle referee detection and image cropping"""
        if results.boxes is not None and results.boxes.xyxy.shape[0] > 0:
            # Extract bounding box coordinates (relative to MODEL_SIZE)
            x1_rel, y1_rel, x2_rel, y2_rel = results.boxes.xyxy[0].cpu().numpy()

             # Scale coordinates back to original frame dimensions
            h, w = frame.shape[:2]
            x1 = int(x1_rel * w / MODEL_SIZE)
            y1 = int(y1_rel * h / MODEL_SIZE)
            x2 = int(x2_rel * w / MODEL_SIZE)
            y2 = int(y2_rel * h / MODEL_SIZE)

            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Crop and resize if dimensions are valid
            if x1 < x2 and y1 < y2:
                cropped = frame[y1:y2, x1:x2]
                # Resize cropped image to the standard MODEL_SIZE
                return cv2.resize(cropped, (MODEL_SIZE, MODEL_SIZE))

        return last_valid  # Return previous valid frame if no detection or invalid crop

    # Added target_size parameter consistency
    def _create_new_writer(self, video_path, output_dir, segment_num, fps, target_size):
        """Initialize new video writer for segment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_{timestamp}_part{segment_num:03d}.mp4" # Use 3 digits for part number
        output_path = os.path.join(output_dir, output_filename)

        # Define codec and create writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, target_size) # Use target_size
        if not writer.isOpened():
             print(f"Error: Could not open video writer for {output_path}")
             return None
        return writer


    def _close_writer(self, writer):
        """Safely close video writer"""
        if writer is not None:
            writer.release()

    # Reverted to simpler os.rename, assuming no cross-filesystem moves needed
    def _move_processed_file(self, src_path, dest_dir):
         """Move processed file to completed directory"""
         os.makedirs(dest_dir, exist_ok=True)
         dest_path = os.path.join(dest_dir, os.path.basename(src_path))
         try:
              if os.path.exists(dest_path):
                  os.remove(dest_path) # Remove if exists
              os.rename(src_path, dest_path) # Use simple rename
              print(f"Moved {os.path.basename(src_path)} to {dest_dir}")
         except OSError as e:
              print(f"Error moving file {src_path} to {dest_path}: {e}")
              # Consider adding shutil.move as fallback or retry logic here if needed

if __name__ == "__main__":
    processor = RefereeProcessor()
    processor.process_videos()
    print("Processing finished.") # Keep final print