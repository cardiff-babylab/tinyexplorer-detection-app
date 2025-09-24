---
title: TinyExplorer FaceDetectionApp
description: Desktop face detection app for developmental research using YOLO and RetinaFace. Batch processing with CSV exports and visual outputs.
image: assets/images/dragon.png
---

# TinyExplorer FaceDetectionApp

<div align="center">
  <img src="assets/screenshots/app-main-interface.png" alt="TinyExplorer FaceDetectionApp Interface" />
  <br>
  <em>Main application interface showing file selection, model options, and confidence threshold controls</em>
</div>

## Overview
The TinyExplorer FaceDetectionApp is a user-friendly graphical interface designed specifically for developmental psychologists working with infants and young children. This toolbox integrates state-of-the-art open-source face recognition algorithms into an easy-to-use software package, streamlining the process of analyzing facial data in developmental research.

## Features
- Simple graphical user interface for easy operation
- Integration of cutting-edge face recognition models
- Batch processing capabilities for efficient analysis of large datasets
- Customizable confidence thresholds for detection accuracy

## Basic Usage

### Install a Prebuilt Release (recommended)
- Download the latest installer for your OS from the [Releases page](https://github.com/cardiff-babylab/tinyexplorer-facedetectionapp/releases).
- Run the installer and launch the app.

### Build Locally with npm (fallback)
If no release exists for your system or the installer doesn’t work, you can build locally.

- Install dependencies:
  ```bash
  npm install
  ```
- Start in development:
  ```bash
  npm run start
  ```
- Create a distributable build:
  ```bash
  npm run build
  ```

See also: [Supported File Formats](main-features.md#supported-file-formats) and [Getting Started](getting-started.md).

## Documentation Sections

- [Getting Started](getting-started.md)
- [Main Features](main-features.md)
- [Understanding Results](understanding-results.md)
- [Advanced Options](advanced-options.md)
- [Troubleshooting](troubleshooting.md)
- [Support and Updates](support.md)
- [About](about.md)

---

## Copyright & Attribution

**© Cardiff Babylab**

### Project Team

- **Concept and Project Management:** Teodor Nikolov & Hana D'Souza
- **Lead Development and Implementation:** Tamas Foldes
- **Code Contributions:** Ziye Zhang & Teodor Nikolov

## Funding

This work was supported by a [James S. McDonnell Foundation (JSMF) Opportunity Award](https://doi.org/10.37717/2022-3711) and a UKRI Future Leaders Fellowship (MR/X032922/1) awarded to HD.
