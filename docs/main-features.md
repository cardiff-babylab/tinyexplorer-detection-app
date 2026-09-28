# Main Features

## File and Folder Selection
- Browse and select individual image or video files
- Browse and select entire folders containing multiple images and videos

## Model Selection
Choose from multiple models:

#### Face Detection
- **YOLOv8n-face (Nano):** fastest inference, smallest size (~2.7 MB); lower accuracy; ideal for real-time or limited resources.
- **YOLOv8m-face (Medium):** balanced speed and accuracy (~27.3 MB); solid default for most tasks.
- **YOLOv8l-face (Large):** highest accuracy within v8 (~59.2 MB); slower inference; best for high precision.
- **YOLOv11m-face (Medium):** newer generation with improved accuracy/speed trade-off; good general-purpose choice on modern hardware.
- **YOLOv11l-face (Large):** higher accuracy variant; increased compute and memory cost.
- **YOLOv12l-face (Large):** latest large model; highest accuracy and resource use; recommended for offline batch processing.
- **RetinaFace:** alternative architecture with facial landmarks; good speed/accuracy for feature localization. Note: available on Apple Silicon (arm64) macOS only. Source: https://github.com/serengil/retinaface.

#### Hand Detection
- **HandObject (100DOH baseline) - Hand_object_detector model by ddshan, trained on 100DOH dataset**
- **HandObject (100DOH TinyExplorer-tuned) - TinyExplored-Tuned version of the 100DOH hand_object_detector, targeted at detecting infant hands with ownership classification integrated**

#### Automatic Speech Recognition
- **Whisper (OpenAI)**
- **Faster Whisper**
- **WhisperX**
Available in sizes from `tiny` to `large-v3-turbo`. Smaller sizes are faster but have lower accuracy; larger sizes are more accurate but computationally heavier.

### Model Sources
- YOLO face weights: [cardiff-babylab/tinyexplorer-detection-app releases](https://github.com/cardiff-babylab/tinyexplorer-detection-app/releases/tag/v1.0.0-models) (originally from [akanametov/yolo-face](https://github.com/akanametov/yolo-face))
- RetinaFace implementation: [serengil/retinaface](https://github.com/serengil/retinaface)
- Hand detection weights: [cardiff-babylab/tinyexplorer-detection-app releases](https://github.com/cardiff-babylab/tinyexplorer-detection-app/releases/tag/handobj-weights-v1) (HandObject / 100DOH Faster R‑CNN, originally from [ddshan/hand_object_detector](https://github.com/ddshan/hand_object_detector)). TinyExplorer-Tuned version at: [100DOH-TinyExplorer-Tuned](https://github.com/CraigThomp1/100DOH-TinyExplorer-Tuned-hand-detection/tree/main)
- Whisper (OpenAI): [openai/whisper](https://github.com/openai/whisper)
- Faster Whisper: [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- WhisperX: [m-bain/whisperx](https://github.com/m-bain/whisperx)

The app automatically downloads required model weights when needed.


## Sampling Rate (for face and hand detection)
- Current sampling rate of 1 frame per second (1fps)
- Coming soon: adjustable slider for sampling rate
  
## Confidence Threshold Adjustment
- Adjustable slider from 0.0 to 1.0
- Default confidence values tailored to each model

## Pipeline details
- Face and hand detection works with images and videos
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
