# TinyExplorer DetectionApp

> ⚠️ **BETA STATUS** ⚠️
> 
> This software is currently in beta testing. Features and functionality may change as development continues.  
> **Coming Soon:** Windows version release planned for the near future. Currently available for MacOS only.

<div align="center">
  <img src="docs/assets/screenshots/app-main-interface.png" alt="TinyExplorer DetectionApp Interface" />
  <br>
  <em>Main application interface showing file selection, model options, and confidence threshold controls</em>
</div>

## Overview

The TinyExplorer DetectionApp is a user-friendly graphical interface designed specifically for developmental psychologists working with infants and young children. This toolbox integrates state-of-the-art open-source face recognition algorithms into an easy-to-use software package, streamlining the process of analyzing facial data in developmental research.

## Features

- Simple graphical user interface for easy operation
- Integration of cutting-edge face recognition models
- Batch processing capabilities for efficient analysis of large datasets
- Customizable confidence thresholds for detection accuracy


## Face Recognition Models

Choose from multiple face detection models included in the app:

- YOLOv8n-face (Nano): fastest inference, smallest size (~2.7 MB); lower accuracy; ideal for real‑time or limited resources.
- YOLOv8m-face (Medium): balanced speed and accuracy (~27.3 MB); solid default for most tasks.
- YOLOv8l-face (Large): highest accuracy within v8 (~59.2 MB); slower inference; best for high precision.
- YOLOv11m-face (Medium): newer generation with improved accuracy/speed trade‑offs; good general‑purpose choice on modern hardware.
- YOLOv11l-face (Large): higher accuracy variant; increased compute and memory cost.
- YOLOv12l-face (Large): latest large model; highest accuracy and resource use; recommended for offline batch processing.
- RetinaFace: alternative architecture with facial landmarks; good speed/accuracy for feature localization. Note: available on Apple Silicon (arm64) macOS only.

The app automatically downloads required model weights when needed.

### Model Sources
- YOLO face weights: https://github.com/akanametov/yolo-face
- RetinaFace implementation: https://github.com/serengil/retinaface
  
## Value for Developmental Psychologists

This toolbox addresses several key needs in developmental psychology research:

- Efficiency: Automates the time-consuming process of manual face detection in video and image data.
- Accessibility: Provides a user-friendly interface, making advanced face recognition technology accessible to researchers without extensive programming experience.
- Flexibility: Allows researchers to easily switch between different face recognition models to suit their specific research needs.
- Reproducibility: Ensures consistent application of face detection criteria across studies, enhancing research reproducibility.

## Development and Contributions

This toolbox is actively developed by the Cardiff University BabyLab, a research group dedicated to exploring attentional and motor skills in young children and their impact on learning in everyday settings. We welcome contributions from the developmental psychology community to enhance and expand the capabilities of this toolbox.

If you have ideas for new features, improvements, or bug fixes, please feel free to:

- Submit a pull request
- Open an issue with your suggestion
- Contact us directly with your ideas

## Getting Started

### Install a Prebuilt Release (recommended)
- Download the latest installer for your OS from the [Releases page](https://github.com/andreifoldes/tinyexplorer-facedetectionapp/releases).
- Run the installer and launch the app.

### Build and Run Locally (fallback)
If no release exists for your system or the installer doesn't work, you can build locally.

- Ensure [Node.js](https://nodejs.org/) and [npm](https://www.npmjs.com/) are installed.
- Install dependencies: `npm install`
- Start in development: `npm run start`
- Build a distributable package: `npm run build`

### See Also (Documentation)
- Basic Usage: https://cardiff-babylab.github.io/tinyexplorer-facedetectionapp/getting-started
- Supported File Formats: https://cardiff-babylab.github.io/tinyexplorer-facedetectionapp/main-features/#supported-file-formats

## Contact

For more information or collaboration inquiries, please contact:

- Organization: Cardiff BabyLab, Cardiff University Centre for Human Developmental Science (CUCHDS)
- Address: 70 Park Place, Cardiff, CF10 3AT, UK
- Email: [babylab@cardiff.ac.uk](mailto:babylab@cardiff.ac.uk)
- Phone: 029 2251 4800
- Website: [cardiff-babylab.com](https://www.cardiff-babylab.com/)

We look forward to seeing how this toolbox can support and advance your research in developmental psychology!

## Funding

This work was supported by a [James S. McDonnell Foundation (JSMF) Opportunity Award](https://doi.org/10.37717/2022-3711) and a UKRI Future Leaders Fellowship (MR/X032922/1) awarded to HD.
