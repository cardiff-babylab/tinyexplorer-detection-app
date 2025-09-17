import os
import cv2
import numpy as np
import threading
import time
from typing import List, Dict, Optional, Callable
import csv
from datetime import datetime
import requests
import sys

# Conditional imports to avoid conflicts between environments
YOLO_AVAILABLE = False
RETINAFACE_AVAILABLE = False

# Try to import YOLO (ultralytics + torch)
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
    print("YOLO/Ultralytics loaded successfully", file=sys.stderr)
except ImportError as e:
    print(f"YOLO/Ultralytics not available: {e}", file=sys.stderr)

# Try to import RetinaFace (cross-platform; depends on installed packages)
try:
    from retinaface import RetinaFace  # type: ignore
    RETINAFACE_AVAILABLE = True
    print("RetinaFace loaded successfully", file=sys.stderr)
except ImportError as e:
    print(f"RetinaFace not available: {e}", file=sys.stderr)

def safe_imread(file_path: str):
    """Robust image reader that handles Windows paths with spaces/unicode."""
    try:
        img = cv2.imread(file_path)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
    except Exception:
        pass
    return None

def safe_imwrite(file_path: str, image) -> bool:
    """Robust image writer that handles Windows paths with spaces/unicode."""
    try:
        return cv2.imwrite(file_path, image)
    except Exception:
        try:
            ext = os.path.splitext(file_path)[1] or '.jpg'
            ok, encoded = cv2.imencode(ext, image)
            if ok:
                encoded.tofile(file_path)
                return True
        except Exception:
            return False
    return False

class FaceDetectionProcessor:
    def __init__(self, progress_callback: Optional[Callable] = None, completion_callback: Optional[Callable] = None):
        self.model = None
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.is_processing = False
        self.results = []
        self.model_type = "YOLO"
        self.current_model_path = None
        
        # Status symbols for better user feedback
        self.status_symbols = {
            "info": "ℹ️",
            "success": "✅",
            "error": "❌",
            "warning": "⚠️",
            "processing": "⏳",
            "folder": "📁",
            "image": "🖼️",
            "video": "🎬",
            "detection": "🔍",
            "face": "👤",
            "complete": "🏁"
        }
        
        # Set up a writable directory for storing downloaded models when running from a read-only filesystem
        # like AppImage. Uses FACE_MODEL_DIR if provided, otherwise XDG_DATA_HOME or ~/.local/share.
        self._model_dir = None
    
    def _get_model_dir(self) -> str:
        """Return a writable directory path for model files and ensure it exists."""
        if self._model_dir is not None:
            return self._model_dir
        try:
            base_dir = os.environ.get("FACE_MODEL_DIR")
            if not base_dir:
                data_home = os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share"))
                base_dir = os.path.join(data_home, "TinyExplorerFaceDetection", "models")
            os.makedirs(base_dir, exist_ok=True)
            self._model_dir = base_dir
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['folder']} Using model directory: {base_dir}")
            return base_dir
        except Exception:
            # Fallback to current working directory as last resort
            self._model_dir = os.getcwd()
            return self._model_dir
        
    def _download_face_model(self, model_name: str) -> bool:
        """Download face detection model from GitHub releases"""
        face_models_urls = {
            "yolov8n-face.pt": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
            "yolov8m-face.pt": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8m-face.pt",
            "yolov8l-face.pt": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8l-face.pt",
            "yolov11m-face.pt": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt",
            "yolov11l-face.pt": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11l-face.pt",
            "yolov12l-face.pt": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov12l-face.pt"
        }
        
        if model_name not in face_models_urls:
            return False
            
        try:
            url = face_models_urls[model_name]
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['info']} Downloading {model_name} from GitHub...")
            
            # Download with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Always write to a writable model directory
            model_filename = os.path.basename(model_name)
            save_dir = self._get_model_dir()
            save_path = os.path.join(save_dir, model_filename)
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and self.progress_callback:
                            progress = (downloaded / total_size) * 100
                            self.progress_callback(f"{self.status_symbols['processing']} Downloading {model_name}: {progress:.1f}%")
            
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['success']} Downloaded {model_name} successfully to {save_path}")
            # Update current model path
            self.current_model_path = save_path
            return True
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['error']} Error downloading {model_name}: {str(e)}")
            return False
        
    def load_model(self, model_path: str = "yolov8n.pt"):
        """Load model for face detection (YOLO, RetinaFace, or OpenCV)"""
        try:
            if model_path.lower() == "retinaface":
                # This is the RetinaFace case (moved here for clarity)
                self.model_type = "RetinaFace"
                self.current_model_path = model_path
                
                # Ensure RetinaFace is actually available before proceeding
                if not RETINAFACE_AVAILABLE:
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['error']} RetinaFace is not available in this environment")
                        self.progress_callback(f"{self.status_symbols['info']} Ensure 'retina-face' and 'tensorflow' are installed in the active Python environment")
                    return False
                
                # RetinaFace doesn't need explicit loading - it loads models on first use
                # Just verify it's available and working
                try:
                    # This will trigger model download if not already present
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['info']} RetinaFace will download models on first use if needed")
                except Exception as e:
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['warning']} RetinaFace initialization warning: {str(e)}")
                
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['success']} RetinaFace model ready")
                return True
                
            else:
                self.model_type = "YOLO"
                self.current_model_path = model_path
                
                # Always use writable directory for model storage
                if not os.path.isabs(model_path):
                    # Check if model exists in our writable directory
                    candidate_path = os.path.join(self._get_model_dir(), os.path.basename(model_path))
                    resolved_model_path = candidate_path
                else:
                    resolved_model_path = model_path
                
                # Check if model file exists locally, otherwise download it
                if not os.path.exists(resolved_model_path):
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['info']} Model {os.path.basename(model_path)} not found - downloading...")
                    
                    # Download face-specific models from GitHub
                    if "-face.pt" in model_path.lower():
                        if not self._download_face_model(model_path):
                            if self.progress_callback:
                                self.progress_callback(f"{self.status_symbols['error']} Failed to download face model {model_path}")
                            return False
                        # After successful download, use the path in our model directory
                        resolved_model_path = os.path.join(self._get_model_dir(), os.path.basename(model_path))
                    else:
                        if self.progress_callback:
                            self.progress_callback(f"{self.status_symbols['info']} Standard YOLO model will be downloaded automatically...")
                else:
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['success']} Found existing model: {os.path.basename(resolved_model_path)}")
                
                # Check if YOLO is available before loading
                if not YOLO_AVAILABLE:
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['error']} YOLO/Ultralytics is not available in this environment")
                        self.progress_callback(f"{self.status_symbols['info']} Please use RetinaFace model or switch to YOLO environment")
                    return False
                
                # YOLO automatically downloads standard models if they don't exist
                # Use MPS for Apple Silicon Macs, CPU otherwise (avoid CUDA issues)
                if torch.backends.mps.is_available():
                    device = 'mps'
                elif torch.cuda.is_available():
                    device = 'cuda:0'
                else:
                    device = 'cpu'
                
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['processing']} Loading YOLO model: {resolved_model_path}")
                
                # Load YOLO model (removed signal timeout as it doesn't work in subprocess)
                self.model = YOLO(resolved_model_path)
                # Move to device in a separate try-catch to handle device issues
                try:
                    self.model = self.model.to(device)
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['info']} Using device: {device}")
                except Exception as device_error:
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['warning']} Device {device} failed, falling back to CPU: {str(device_error)}")
                    self.model = self.model.to('cpu')
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['success']} YOLO model loaded successfully: {resolved_model_path}")
                return True
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['error']} Error loading model: {str(e)}")
            return False
    
    def process_image(self, image_path: str, confidence_threshold: float = 0.5, save_results: bool = False, result_folder: str = None) -> List[Dict]:
        """Process a single image for face detection"""
        try:
            # Load image
            image = safe_imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            detections = []
            
            if self.model_type == "RetinaFace":
                detections = self._process_with_retinaface(image, image_path, confidence_threshold, save_results, result_folder)
            else:
                detections = self._process_with_yolo(image, image_path, confidence_threshold, save_results, result_folder)
            
            # Log completion of individual image processing
            if self.progress_callback:
                num_faces = len(detections) if detections else 0
                if num_faces > 0:
                    self.progress_callback(f"{self.status_symbols['face']} Found {num_faces} face(s) in {os.path.basename(image_path)}")
                else:
                    self.progress_callback(f"{self.status_symbols['complete']} No faces detected in {os.path.basename(image_path)}")
            
            return detections
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['error']} Error processing {os.path.basename(image_path)}: {str(e)}")
            return []
    
    def process_folder(self, folder_path: str, confidence_threshold: float = 0.5, model_name: str = "yolov8n.pt", save_results: bool = False, results_folder: str = None):
        """Process all images and videos in a folder or single file"""
        self.is_processing = True
        self.results = []
        # Store confidence threshold for summary export
        self.current_confidence = confidence_threshold
        # Normalize incoming paths to handle spaces and mixed separators reliably
        folder_path = os.path.normpath(folder_path)
        if results_folder:
            results_folder = os.path.normpath(results_folder)
        
        # Send processing started event
        if self.completion_callback:
            self.completion_callback({
                'status': 'processing_started',
                'folder_path': folder_path,
                'model': model_name,
                'confidence': confidence_threshold
            })
            # Small delay to ensure event is processed before continuing
            import time
            time.sleep(0.1)
        
        try:
            # Create result folder if saving results
            result_folder = None
            if save_results:
                if results_folder:
                    # Use the user-specified results folder
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_folder = os.path.join(results_folder, f"face_detection_results_{timestamp}")
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['folder']} Creating results folder: {result_folder}")
                else:
                    # Fall back to current directory with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_folder = os.path.join(os.getcwd(), f"face_detection_results_{timestamp}")
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['folder']} Creating default results folder: {result_folder}")
                
                os.makedirs(result_folder, exist_ok=True)
                os.makedirs(os.path.join(result_folder, "results"), exist_ok=True)
            
            # Load the specified model
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['info']} Loading model: {model_name}...")
            if not self.load_model(model_name):
                return
            
            # Get all image and video files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            video_extensions = ['.mp4', '.avi', '.mov']
            image_files = []
            video_files = []
            
            # Check if the path is a file or directory
            if os.path.isfile(folder_path):
                # Single file processing
                if any(folder_path.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(folder_path)
                elif any(folder_path.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(folder_path)
            else:
                # Directory processing
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(file_path)
                        elif any(file.lower().endswith(ext) for ext in video_extensions):
                            video_files.append(file_path)
            
            total_files = len(image_files) + len(video_files)
            if total_files == 0:
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['warning']} No image or video files found in the specified location")
                return
            
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['folder']} Found {len(image_files)} images and {len(video_files)} videos to process")
            
            # Process each image
            for i, image_path in enumerate(image_files):
                if not self.is_processing:
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['warning']} Processing stopped by user")
                    break
                    
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['image']} Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                try:
                    detections = self.process_image(image_path, confidence_threshold, save_results, result_folder)
                    if detections:
                        self.results.extend(detections)
                        if self.progress_callback:
                            self.progress_callback(f"{self.status_symbols['face']} Added {len(detections)} detections to results (total: {len(self.results)})")
                    
                    # Progress update after each image
                    progress_percent = ((i + 1) / len(image_files)) * 100
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['complete']} Image {i+1}/{len(image_files)} complete ({progress_percent:.1f}%)")
                    
                    # Emit per-image completion event
                    if self.completion_callback:
                        self.completion_callback({
                            'status': 'image_completed',
                            'image_index': i + 1,
                            'total_images': len(image_files),
                            'progress_percent': progress_percent,
                            'detections_in_image': len(detections) if detections else 0,
                            'total_detections': len(self.results),
                            'image_path': os.path.basename(image_path)
                        })
                    
                    # Small delay to prevent overwhelming the UI
                    time.sleep(0.05)
                        
                except Exception as image_error:
                    if self.progress_callback:
                        self.progress_callback(f"{self.status_symbols['error']} Failed to process {os.path.basename(image_path)}: {str(image_error)}")
                    continue  # Continue with next image
            
            # Process each video
            for i, video_path in enumerate(video_files):
                if not self.is_processing:
                    break
                    
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['video']} Processing video {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
                
                if result_folder:
                    video_detections = self.process_video(video_path, confidence_threshold, result_folder)
                    self.results.extend(video_detections)
            
            # Export results to CSV if saving results
            if save_results and result_folder and self.results:
                csv_path = os.path.join(result_folder, "detection_results.csv")
                self.export_results_to_csv(self.results, csv_path)
                
                # Export summary statistics 
                summary_csv_path = os.path.join(result_folder, "summary.csv")
                self.export_summary_to_csv(folder_path, image_files, video_files, result_folder, summary_csv_path)
            
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['complete']} Processing complete. Found {len(self.results)} face detections across {total_files} files")
                if save_results and result_folder:
                    self.progress_callback(f"{self.status_symbols['folder']} Results saved to: {result_folder}")
                # Log technical completion message to console only
                print(f"{self.status_symbols['success']} All processing finished - setting is_processing = False", file=sys.stderr)
            
            # Emit completion event
            if self.completion_callback:
                self.completion_callback({
                    'status': 'completed',
                    'results_count': len(self.results),
                    'total_files': total_files,
                    'results_folder': result_folder if save_results else None
                })
                    
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['error']} Error during processing: {str(e)}")
            # Log technical message to console only
            print(f"{self.status_symbols['warning']} Setting is_processing = False due to error", file=sys.stderr)
            
            # Emit error completion event
            if self.completion_callback:
                self.completion_callback({
                    'status': 'error',
                    'error': str(e),
                    'results_count': len(self.results)
                })
        finally:
            # Always reset the processing flag, regardless of success or failure
            # Log technical messages to console only
            print(f"{self.status_symbols['info']} FINALLY block: Setting is_processing = False (was {self.is_processing})", file=sys.stderr)
            self.is_processing = False
            print(f"{self.status_symbols['success']} is_processing is now: {self.is_processing}", file=sys.stderr)
            
            # Always emit final completion event in finally block
            if self.completion_callback:
                self.completion_callback({
                    'status': 'finished',
                    'is_processing': self.is_processing,
                    'results_count': len(self.results)
                })
    
    def stop_processing(self):
        """Stop the current processing"""
        self.is_processing = False
        if self.progress_callback:
            self.progress_callback("Processing stopped by user")
    
    def get_results(self) -> List[Dict]:
        """Get the current results"""
        return self.results
    
    def _process_with_yolo(self, image, image_path: str, confidence_threshold: float, save_results: bool, result_folder: str) -> List[Dict]:
        """Process image with YOLO model"""
        import shutil
        
        if self.progress_callback:
            self.progress_callback(f"{self.status_symbols['processing']} Running YOLO inference on {os.path.basename(image_path)}...")
        
        # Run inference using predict method (like old_script.py)
        # Ensure Ultralytics writes to a writable directory (avoid default './runs' in read-only filesystems)
        runs_project_dir = os.path.join(self._get_model_dir(), "runs")
        os.makedirs(runs_project_dir, exist_ok=True)
        results = self.model.predict(
            source=image_path,
            conf=confidence_threshold,
            save=save_results,
            save_txt=save_results,
            save_conf=save_results,
            project=runs_project_dir,
            name="predict",
            exist_ok=True
        )
        
        if self.progress_callback:
            self.progress_callback(f"{self.status_symbols['success']} YOLO inference completed for {os.path.basename(image_path)}")
        
        # Copy results to the result folder if save_results is True
        if results and results[0].save_dir and save_results and result_folder:
            yolo_result_dir = results[0].save_dir
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['folder']} Results saved to {yolo_result_dir}")
            
            for item in os.listdir(yolo_result_dir):
                s = os.path.join(yolo_result_dir, item)
                d = os.path.join(result_folder, "results", item)
                if os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.move(s, d)
                else:
                    if os.path.exists(d):
                        os.remove(d)
                    shutil.move(s, d)
            
            # Delete original YOLO results directory
            shutil.rmtree(yolo_result_dir)
            
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['complete']} Results copied to final location")
        
        # Extract face detections from the YOLO results
        detections = []
        
        # Use the results from the YOLO prediction we already made
        if results and len(results) > 0:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['detection']} Extracting detections from YOLO results...")
            
            # Get detections directly from YOLO results object
            result = results[0]
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['info']} DEBUG: result.boxes = {result.boxes}")
                if result.boxes is not None:
                    self.progress_callback(f"{self.status_symbols['info']} DEBUG: len(result.boxes) = {len(result.boxes)}")
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes in xyxy format
                confidences = result.boxes.conf.cpu().numpy()  # Get confidences
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    confidence = confidences[i]
                    
                    # Convert to center coordinates and width/height (like the old method)
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2
                    
                    detections.append({
                        'x': float(x_center),
                        'y': float(y_center),
                        'width': float(width),
                        'height': float(height),
                        'confidence': float(confidence),
                        'image_path': image_path
                    })
                    
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['face']} Extracted {len(detections)} face detections from YOLO results")
            else:
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['info']} No faces detected in YOLO results")
        
        # Legacy fallback: try to read from saved text files if direct extraction failed
        label_file = None
        if len(detections) == 0:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['warning']} Direct extraction failed, trying text file fallback...")
            
            if save_results and result_folder:
                label_file = os.path.join(result_folder, "results", "labels", os.path.splitext(os.path.basename(image_path))[0] + ".txt")
            elif results and results[0].save_dir:
                # Use the save_dir from the first prediction
                label_file = os.path.join(results[0].save_dir, "labels", os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        
        if label_file and os.path.exists(label_file):
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['info']} Reading detections from text file: {label_file}")
            with open(label_file, 'r') as lf:
                lines = lf.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 6:  # class_id, x_center, y_center, width, height, confidence
                        _, x_center, y_center, width, height, confidence = map(float, parts[:6])
                        detections.append({
                            'x': float(x_center),
                            'y': float(y_center),
                            'width': float(width),
                            'height': float(height),
                            'confidence': float(confidence),
                            'image_path': image_path
                        })
        
        # Log completion of YOLO processing
        if self.progress_callback:
            self.progress_callback(f"{self.status_symbols['complete']} YOLO processing completed for {os.path.basename(image_path)} - found {len(detections)} faces")
            self.progress_callback(f"{self.status_symbols['info']} DEBUG: Returning detections: {detections}")
        
        return detections
    
    def _process_with_retinaface(self, image, image_path: str, confidence_threshold: float, save_results: bool, result_folder: str) -> List[Dict]:
        """Process image with RetinaFace model"""
        detections = []
        
        try:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['processing']} Running RetinaFace inference on {os.path.basename(image_path)}...")
            
            # Import locally to avoid NameError when global import is unavailable
            from retinaface import RetinaFace as _RetinaFace
            
            # Use RetinaFace for detection - it expects image path or numpy array
            face_detections = _RetinaFace.detect_faces(image_path, threshold=confidence_threshold)
            
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['success']} RetinaFace inference completed for {os.path.basename(image_path)}")
            
            if face_detections:
                result_img = image.copy() if save_results else None
                
                for key, detection in face_detections.items():
                    facial_area = detection["facial_area"]
                    confidence = detection["score"]
                    
                    # RetinaFace returns coordinates as [x1, y1, x2, y2]
                    x1, y1, x2, y2 = facial_area
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1
                    
                    detections.append({
                        'x': float(x),
                        'y': float(y),
                        'width': float(w),
                        'height': float(h),
                        'confidence': float(confidence),
                        'image_path': image_path
                    })
                    
                    # Draw bounding box for visualization
                    if save_results and result_img is not None:
                        cv2.rectangle(result_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                        cv2.putText(result_img, f"{confidence:.3f}", (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Save result image
                if save_results and result_img is not None and result_folder:
                    result_img_dir = os.path.join(result_folder, "results")
                    os.makedirs(result_img_dir, exist_ok=True)
                    result_img_path = os.path.join(result_img_dir, os.path.basename(image_path))
                    safe_imwrite(result_img_path, result_img)
                    
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['face']} RetinaFace found {len(detections)} face(s)")
            else:
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['info']} RetinaFace found no faces above confidence threshold")
                    
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['error']} RetinaFace processing error: {str(e)}")
        
        return detections
    
    # Removed OpenCV Haar Cascade fallback per product requirements
    
    def _save_image_with_boxes(self, image, detections: List[Dict], image_path: str, result_folder: str):
        """Save image with bounding boxes drawn"""
        try:
            result_img = image.copy()
            
            for detection in detections:
                x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
                confidence = detection['confidence']
                
                # Draw bounding box
                cv2.rectangle(result_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(result_img, f"{confidence:.2f}", (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save result image
            result_img_dir = os.path.join(result_folder, "results")
            os.makedirs(result_img_dir, exist_ok=True)
            result_img_path = os.path.join(result_img_dir, os.path.basename(image_path))
            safe_imwrite(result_img_path, result_img)
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['error']} Error saving result image: {str(e)}")
    
    def process_video(self, video_path: str, confidence_threshold: float = 0.5, result_folder: str = None) -> List[Dict]:
        """Process video for face detection by sampling frames"""
        try:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['video']} Processing video: {os.path.basename(video_path)}")
            
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps
            
            # Calculate frames to skip (1 frame per second)
            frames_to_skip = int(fps)
            
            frames_with_faces = 0
            processed_frames = 0
            all_detections = []
            
            for frame_idx in range(0, frame_count, frames_to_skip):
                if not self.is_processing:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frames += 1
                
                # Save frame temporarily
                temp_frame_path = os.path.join(result_folder, f'temp_frame_{frame_idx}.jpg')
                safe_imwrite(temp_frame_path, frame)
                
                # Process frame
                detections = self.process_image(temp_frame_path, confidence_threshold, save_results=False)
                
                # Clean up temp frame
                os.remove(temp_frame_path)
                
                if detections:
                    frames_with_faces += 1
                    for detection in detections:
                        detection['frame_idx'] = frame_idx
                        detection['timestamp'] = frame_idx / fps
                    all_detections.extend(detections)
                
                # Emit per-frame completion event
                if self.completion_callback:
                    frame_progress = (processed_frames / (frame_count // frames_to_skip)) * 100
                    self.completion_callback({
                        'status': 'frame_completed',
                        'frame_index': frame_idx,
                        'processed_frames': processed_frames,
                        'total_frames': frame_count // frames_to_skip,
                        'progress_percent': frame_progress,
                        'detections_in_frame': len(detections) if detections else 0,
                        'total_detections': len(all_detections),
                        'timestamp': frame_idx / fps,
                        'video_path': os.path.basename(video_path)
                    })
            
            cap.release()
            
            face_percentage = (frames_with_faces / processed_frames) * 100 if processed_frames > 0 else 0
            
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['complete']} Video processing complete. {frames_with_faces}/{processed_frames} frames with faces ({face_percentage:.1f}%)")
            
            return all_detections
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['error']} Error processing video: {str(e)}")
            return []
    
    def export_results_to_csv(self, results: List[Dict], output_path: str):
        """Export detection results to CSV file"""
        try:
            if not results:
                if self.progress_callback:
                    self.progress_callback(f"{self.status_symbols['warning']} No results to export")
                return False
            
            # Group results by image
            image_results = {}
            for detection in results:
                image_path = detection['image_path']
                if image_path not in image_results:
                    image_results[image_path] = []
                image_results[image_path].append(detection)
            
            # Prepare CSV data
            csv_data = []
            max_faces = max(len(detections) for detections in image_results.values())
            
            # Create headers
            headers = ['filename', 'face_detected', 'face_count']
            for i in range(max_faces):
                headers.extend([f'face_{i+1}_x', f'face_{i+1}_y', f'face_{i+1}_width', f'face_{i+1}_height', f'face_{i+1}_confidence'])
            
            # Create rows
            for image_path, detections in image_results.items():
                row = [os.path.basename(image_path), 1 if detections else 0, len(detections)]
                
                for detection in detections:
                    row.extend([detection['x'], detection['y'], detection['width'], detection['height'], detection['confidence']])
                
                # Pad row to match headers length
                while len(row) < len(headers):
                    row.append('')
                
                csv_data.append(row)
            
            # Write CSV file
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(csv_data)
            
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['success']} Results exported to {output_path}")
            
            return True
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['error']} Error exporting results: {str(e)}")
            return False
    
    def export_summary_to_csv(self, folder_path: str, image_files: List[str], video_files: List[str], result_folder: str, output_path: str):
        """Export summary statistics to CSV file (like old_script.py)"""
        try:
            summary_headers = ['path', 'type', 'total_processed_frames', 'total_duration', 'processed_frames_with_faces', 'face_percentage', 'model', 'confidence_threshold']
            summary_data = []
            
            # Calculate summary for images
            if image_files:
                images_with_faces = 0
                # Count images that have face detections
                image_paths_with_faces = set()
                for detection in self.results:
                    if any(detection['image_path'].endswith(os.path.basename(img)) for img in image_files):
                        image_paths_with_faces.add(detection['image_path'])
                
                images_with_faces = len(image_paths_with_faces)
                face_percentage_images = (images_with_faces / len(image_files)) * 100 if len(image_files) > 0 else 0
                
                summary_images = [
                    folder_path,
                    'image(s)', 
                    len(image_files),
                    'N/A',
                    images_with_faces,
                    face_percentage_images,
                    self.current_model_path or 'Unknown',
                    getattr(self, 'current_confidence', 'Unknown')
                ]
                summary_data.append(summary_images)
            
            # Calculate summary for videos  
            if video_files:
                # Count video frames with faces
                video_frames_with_faces = 0
                total_video_frames = 0
                total_duration = 0
                
                for video_file in video_files:
                    try:
                        cap = cv2.VideoCapture(video_file)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        duration = frame_count / fps if fps > 0 else 0
                        total_duration += duration
                        
                        # Estimate processed frames (1 per second)
                        frames_to_skip = max(1, int(fps))
                        processed_frames_this_video = frame_count // frames_to_skip
                        total_video_frames += processed_frames_this_video
                        cap.release()
                    except Exception as e:
                        if self.progress_callback:
                            self.progress_callback(f"{self.status_symbols['warning']} Could not get video info for {video_file}: {str(e)}")
                
                # Count detections from video frames
                for detection in self.results:
                    if 'frame_idx' in detection:  # This indicates it's from a video
                        video_frames_with_faces += 1
                
                face_percentage_videos = (video_frames_with_faces / total_video_frames) * 100 if total_video_frames > 0 else 0
                
                summary_videos = [
                    folder_path,
                    'video(s)',
                    total_video_frames,
                    total_duration,
                    video_frames_with_faces,
                    face_percentage_videos,
                    self.current_model_path or 'Unknown',
                    getattr(self, 'current_confidence', 'Unknown')
                ]
                summary_data.append(summary_videos)
            
            # Write summary CSV
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(summary_headers)
                writer.writerows(summary_data)
            
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['success']} Summary exported to {output_path}")
            
            return True
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"{self.status_symbols['error']} Error exporting summary: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models (limited to match old_script.py)"""
        models = []
        
        # Check if we're in development or packaged mode and whether known env folders exist
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        yolo_env_path = os.path.join(parent_dir, "yolo-env")
        retinaface_env_path = os.path.join(parent_dir, "retinaface-env")
        
        has_yolo_env = os.path.exists(yolo_env_path)
        has_retinaface_env = os.path.exists(retinaface_env_path)
        
        # Also detect common conda env locations so we can list models even if current env differs
        conda_env_dirs = []
        try:
            if os.name == 'nt':
                candidates = [
                    os.path.join(os.environ.get('USERPROFILE', ''), 'miniconda3', 'envs'),
                    os.path.join(os.environ.get('USERPROFILE', ''), 'anaconda3', 'envs'),
                    r'C:\\Miniconda3\\envs',
                    r'C:\\ProgramData\\Miniconda3\\envs',
                    r'C:\\Anaconda3\\envs',
                ]
            else:
                candidates = [
                    os.path.join(os.environ.get('HOME', ''), 'miniconda3', 'envs'),
                    os.path.join(os.environ.get('HOME', ''), 'anaconda3', 'envs'),
                    '/opt/homebrew/miniconda3/envs',
                    '/opt/homebrew/anaconda3/envs',
                ]
            for d in candidates:
                if d and os.path.exists(d):
                    conda_env_dirs.append(d)
        except Exception:
            pass
        
        has_yolo_env_conda = any(os.path.exists(os.path.join(d, 'electron-python-yolo')) for d in conda_env_dirs)
        has_retinaface_env_conda = any(os.path.exists(os.path.join(d, 'electron-python-retinaface')) for d in conda_env_dirs)
        
        has_yolo_sources = has_yolo_env or has_yolo_env_conda
        has_retina_sources = has_retinaface_env or has_retinaface_env_conda
        
        # If we detect either environment (bundled or conda), show corresponding models regardless of current env
        if has_yolo_sources or has_retina_sources:
            print(f"Detected model environments - YOLO: {has_yolo_sources}, RetinaFace: {has_retina_sources}", file=sys.stderr)
            
            if has_yolo_sources:
                models.extend([
                    "yolov8n-face.pt",
                    "yolov8m-face.pt", 
                    "yolov8l-face.pt",
                    "yolov11m-face.pt",
                    "yolov11l-face.pt",
                    "yolov12l-face.pt"
                ])
                if has_yolo_env:
                    print(f"YOLO environment detected at {yolo_env_path}, adding YOLO models", file=sys.stderr)
                if has_yolo_env_conda:
                    print("YOLO conda environment detected, adding YOLO models", file=sys.stderr)
            
            if has_retina_sources:
                models.append("RetinaFace")
                if has_retinaface_env:
                    print(f"RetinaFace environment detected at {retinaface_env_path}, adding RetinaFace", file=sys.stderr)
                if has_retinaface_env_conda:
                    print("RetinaFace conda environment detected, adding RetinaFace", file=sys.stderr)
        else:
            # Single environment mode - only show models for available frameworks in the active environment
            if YOLO_AVAILABLE:
                models.extend([
                    "yolov8n-face.pt",
                    "yolov8m-face.pt", 
                    "yolov8l-face.pt",
                    "yolov11m-face.pt",
                    "yolov11l-face.pt",
                    "yolov12l-face.pt"
                ])
            if RETINAFACE_AVAILABLE:
                models.append("RetinaFace")
        
        return models