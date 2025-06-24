import os
import cv2
import torch
from ultralytics import YOLO

# Cargar modelos solo una vez (singleton)
referee_model = None
signal_model = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SIZE = 640
CONFIDENCE_THRESHOLD = 0.7

REFEREE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'bestRefereeDetection.pt')
SIGNAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'bestSignalsDetection.pt')

# Clases de señales (ajusta según tu modelo)
SIGNAL_CLASSES = ['armLeft', 'armRight', 'hits', 'leftServe', 'net', 'outside', 'rightServe', 'touched']

def load_models():
    global referee_model, signal_model
    if referee_model is None:
        referee_model = YOLO(REFEREE_MODEL_PATH).to(DEVICE)
        referee_model.fuse()
    if signal_model is None:
        signal_model = YOLO(SIGNAL_MODEL_PATH).to(DEVICE)
        signal_model.fuse()


def detect_referee(image_path, crop_save_path=None):
    """
    Detecta al árbitro en la imagen y devuelve el bounding box y el crop.
    Si crop_save_path se especifica, guarda el crop en esa ruta.
    """
    load_models()
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] detect_referee: Failed to load image from {image_path}")
        return {'detected': False}

    results = referee_model(img, conf=CONFIDENCE_THRESHOLD)[0]
    if results.boxes is not None and results.boxes.xyxy.shape[0] > 0:
        # Tomar la primera detección
        x1, y1, x2, y2 = map(int, results.boxes.xyxy[0].cpu().numpy())
        crop = img[y1:y2, x1:x2]
        
        # Resize the cropped image to MODEL_SIZE for consistency with signal model input
        if crop.shape[0] > 0 and crop.shape[1] > 0: # Ensure valid crop dimensions
            resized_crop = cv2.resize(crop, (MODEL_SIZE, MODEL_SIZE))
        else:
            print(f"[WARNING] detect_referee: Invalid crop dimensions for {image_path}")
            return {'detected': False}

        if crop_save_path is not None:
            print(f"[DEBUG] detect_referee: Attempting to write crop to {crop_save_path}")
            write_result = cv2.imwrite(crop_save_path, resized_crop)
            print(f"[DEBUG] detect_referee: cv2.imwrite returned {write_result}")
            if not os.path.exists(crop_save_path):
                print(f"[ERROR] detect_referee: Failed to write crop to {crop_save_path}")
                return {'detected': False}
            
        return {
            'bbox': [x1, y1, x2, y2],
            'crop_path': crop_save_path,
            'detected': True
        }
    return {'detected': False}


def detect_signal(crop_path):
    """
    Detecta la señal en el crop del árbitro y devuelve la clase y confianza.
    """
    load_models()
    
    print(f"[DEBUG] detect_signal: Receiving crop_path: {crop_path}")
    img = cv2.imread(crop_path)
    
    if img is None:
        print(f"[DEBUG] detect_signal: Failed to load image from {crop_path}")
        return {'predicted_class': None, 'confidence': 0.0, 'bbox_xywhn': None}

    # The image should already be MODEL_SIZE x MODEL_SIZE from detect_referee / manual_crop
    print(f"[DEBUG] detect_signal: Image loaded. Shape: {img.shape}")
    
    results = signal_model(img, conf=CONFIDENCE_THRESHOLD)[0]
    
    print(f"[DEBUG] detect_signal: Raw model results: {results}")

    if results.boxes is not None and results.boxes.cls.shape[0] > 0:
        # Tomar la predicción con mayor confianza
        idx = int(results.boxes.cls[0].cpu().numpy())
        conf = float(results.boxes.conf[0].cpu().numpy())
        class_name = SIGNAL_CLASSES[idx] if idx < len(SIGNAL_CLASSES) else str(idx)
        
        # Obtener el bounding box en formato normalizado (xywhn)
        bbox_xywhn = results.boxes.xywhn[0].cpu().numpy().tolist() # Normalized x_center, y_center, width, height

        print(f"[DEBUG] detect_signal: Detected class: {class_name}, Confidence: {conf}, Bbox (xywhn): {bbox_xywhn}")
        return {
            'predicted_class': class_name,
            'confidence': conf,
            'bbox_xywhn': bbox_xywhn # Add the normalized bounding box
        }
    print(f"[DEBUG] detect_signal: No signal detected with confidence > {CONFIDENCE_THRESHOLD}")
    return {'predicted_class': None, 'confidence': 0.0, 'bbox_xywhn': None} # Also return None for bbox 