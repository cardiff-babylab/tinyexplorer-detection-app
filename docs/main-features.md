# Main Features

## File and Folder Selection
- Browse and select individual image or video files
- Browse and select entire folders containing multiple images and videos

## Model Selection
Choose from multiple face detection models:

- **YOLOv8n-face (Nano):** fastest inference, smallest size (~2.7 MB); lower accuracy; ideal for real‑time or limited resources.
- **YOLOv8m-face (Medium):** balanced speed and accuracy (~27.3 MB); solid default for most tasks.
- **YOLOv8l-face (Large):** highest accuracy within v8 (~59.2 MB); slower inference; best for high precision.
- **YOLOv11m-face (Medium):** newer generation with improved accuracy/speed trade‑offs; good general‑purpose choice on modern hardware.
- **YOLOv11l-face (Large):** higher accuracy variant; increased compute and memory cost.
- **YOLOv12l-face (Large):** latest large model; highest accuracy and resource use; recommended for offline batch processing.
- **RetinaFace:** alternative architecture with facial landmarks; good speed/accuracy for feature localization. Note: available on Apple Silicon (arm64) macOS only. Source: [serengil/retinaface](https://github.com/serengil/retinaface).

The app automatically downloads required model weights when needed.

### Model Sources
- YOLO face weights: [cardiff-babylab/tinyexplorer-detection-app releases](https://github.com/cardiff-babylab/tinyexplorer-detection-app/releases/tag/v1.0.0-models) (originally from [akanametov/yolo-face](https://github.com/akanametov/yolo-face))
- RetinaFace implementation: [serengil/retinaface](https://github.com/serengil/retinaface)

## Confidence Threshold Adjustment
- Adjustable slider from 0.0 to 1.0
- Default confidence values tailored to each model

## Face Recognition Process
- Works with images and videos
- Batch processing for multiple files in a folder
- Real-time progress bar and percentage display
- Detailed logging of the recognition process

## Results and Output
- Timestamped results folder
- CSV output with detailed detection data
- Summary CSV with overall statistics
- Visual results saved for images and video frames
- Results folder opens automatically when processing completes

## User Interface
- Intuitive GUI with file/folder selection, model choice, and confidence adjustment
- Real-time status updates in the window
- Error handling and user notifications

## Supported File Formats
**Images**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

**Videos**
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)

You can select individual files or folders containing these formats for processing.
