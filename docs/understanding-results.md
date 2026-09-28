# Understanding Results

=== "Face Detection"

    ## CSV Output

    ### results.csv

    #### Single File Mode (Image or Video)

    - **filename** – processed file or video frame name
    - **face_detected** – 1 if a face was found, 0 otherwise
    - **face_count** – number of faces in the image or frame
    - **face_X_x**, **face_X_y** – center coordinates of each face bounding box
    - **face_X_width**, **face_X_height** – dimensions of each face bounding box
    - **face_X_confidence** – confidence score for each detected face

    #### Folder Mode

    Same columns as single file mode but includes entries for every processed file or video frame in the folder.

    ### summary.csv

    #### Single File Mode

    - **path** – name of the processed file
    - **type** – `image` or `video`
    - **total_processed_frames** – number of frames processed (1 for images)
    - **total_duration** – video duration in seconds (N/A for images)
    - **processed_frames_with_faces** – frames where faces were detected
    - **face_percentage** – percentage of frames with faces
    - **model** – detection model used
    - **confidence_threshold** – threshold applied during detection

    #### Folder Mode

    Same columns as single file mode but provides two rows summarising:

    1. All images in the folder
    2. All videos in the folder

    Each row contains aggregate values for path, type, total processed frames,
    total duration, frames with faces, face percentage, model, and confidence threshold.

    ## Image Output

    The application saves visual outputs for all images and video frames with detected faces.

    ### Bounding Boxes

    - Green rectangles show each detected face and match the coordinates in `results.csv`.

    ### Confidence Scores

    - Each bounding box displays the detection confidence between 0 and 1.
    - Values correspond to `face_X_confidence` in `results.csv`.

    ### Output File Names

    - For images, the output file retains the original name.
    - For videos, each processed frame is saved separately using the format:
      `[video_name]_[frame_number]_[timestamp].jpg`

    These outputs make it easy to verify detection results and assess model performance.

=== "Hand Detection"

    "Under Construction"
    This module is still under active development.
    
    ## CSV Output

    ### detections.csv
    The 'detections.csv' file contains per detection information, where each record represents one hand detected

    - **dataset** – Name of processed dataset/Folder title
    - **filename** – Name of processed image/video frame
    - **hand_id** – hand identifier assigned to each detected hand within a frame (Starting at 0, i.e if 4 hands are present, labels will be 0,1,2, and 3
    - **hand_x1** – x-coordinate of the left edge of the bounding box
    - **hand_y1** – y-coordinate of the top edge of the bounding box
    - **hand_x2** – x-coordinate of the right edge of the bounding box
    - **hand_y2** – y-coordinate of the bottom edge of the bounding box
    - **Hand_confidence** – Confidence score of the detected hand
    - **State** – Predicted hand state ID (0 - No touch, 1 - Self touch, 2 - Other touch, 3 - Portable object touch, 4 - Furniture touch)
    - **Hand_side** – Predicted hand side ('Left' or 'Right')
    - **Owner_label** – Predicted owner label ('Own' or 'Other')
    - **frame_idx** – Video frame index, where available 
    - **state_raw** – raw state ID output
    - **state_label** – human-readable contact state label
    

    ### summary.csv
    The 'summary.csv' file contains a frame-level summary of the overall hand detections. Each record represents one processed image/video frame

    - **filename** – Name of processed image/video frames
    - **frame_idx** – Video frame index, where available 
    - **img_w** – image width in pixels
    - **img_h** – image height in pixels
    - **n_hands** – Total number of hands detected in the frame
    - **n_own** – Total number of own hands detected in the frame
    - **n_other** – Total number of other hands detected in the frame
    - **n_state0_none** – Total number of detected hands classified as having no contact
    - **n_state1_self** – Total number of detected hands classified as touching self (Body belonging to hand owner)
    - **n_state2_other** – Total number of detected hands classified as touching another person (not self)
    - **n_state3_portable** – Total number of detected hands classified as interacting with a portable object
    - **n_state3_furniture** – Total number of detected hands classified as interacting with furniture/fixed environmental surface i.e. Floor, door, sofa
    
    
   ### Visualisation outputs

   Under construction

   ## Baseline

   ## Tuned


=== "Automatic Speech Recognition"

    ## CSV Output

    ### detections.csv

    #### Single File Mode (Audio or Video)

    - **id** – sequential detection/segment ID
    - **frame_idx** – frame index (blank for audio-only files; used only when detections are tied to video frames)
    - **filename** – processed audio or video file name
    - **mode** – detection mode, e.g., `speech`
    - **start**, **end** – start and end time of the segment, in seconds
    - **label** – detected label (e.g., `speech`)
    - **confidence** – model confidence/log-probability score for the segment
    - **model** – transcription model used (e.g. `Whisper (OpenAI)`)
    - **model_size** – Whisper model size used (e.g. `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`), as chosen in the size dropdown
    - **text** – transcribed text for the segment
    - **language** – detected or specified language code (e.g. `en`)
    - **speaker** – speaker label, if speaker diarization was enabled (blank otherwise)

    #### Folder Mode

    Same columns as single file mode but includes entries for every processed audio/video file in the folder, distinguished by **filename**.

    ### detections_words.csv

    Word-level breakdown of each transcribed segment.

    - **filename** – processed audio or video file name
    - **word** – individual transcribed word
    - **start**, **end** – start and end time of the word, in seconds
    - **speaker** – speaker label, if diarization was enabled (blank otherwise)
    - **word_score** – model confidence score for the individual word
    - **model** – transcription model used (e.g. `Whisper (OpenAI)`)
    - **model_size** – Whisper model size used (e.g. `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`), as chosen in the size dropdown
    - **segment_start**, **segment_end** – start and end time of the parent segment the word belongs to
    - **segment_text** – full text of the parent segment, for cross-reference with `detections.csv`

    ### summary.csv

    #### Single File Mode

    - **path** – full path of the processed file
    - **type** – `audio` or `video`
    - **segments** – total number of speech segments detected
    - **duration** – total duration of the file, in seconds
    - **language** – detected or specified language code
    - **model** – transcription model used
    - **model_size** – Whisper model size used (e.g. `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`), as chosen in the size dropdown

    #### Folder Mode

    Same columns as single file mode but provides one row per processed file.
    
    ### Per-File Transcription Output

    In addition to the combined `detections.csv` / `detections_words.csv`, the application also saves an individual pair of CSVs for **each processed file**, named after the source file:

    - `[filename]_transcript.csv` – same columns as `detections.csv`, scoped to that file only
    - `[filename]_words.csv` – same columns as `detections_words.csv`, scoped to that file only
    - `[filename]_transcript.txt` – A human-readable transcript. The first line records the model and size (e.g. `# Model: Faster Whisper (small)`), followed by one line per detected segment, formatted as:
      
      [start-end] Segment text
      **start**, **end** – segment timestamps in seconds, matching `detections.csv` 
    
    !!! note
        In **Single File Mode**, these per-file CSVs are identical to `detections.csv` and `detections_words.csv`, since only one file is processed.
        In **Folder Mode**, they diverge: `detections.csv` / `detections_words.csv` aggregate rows across *all* files in the folder, while `[filename]_transcript.csv` / `[filename]_words.csv` contain only the rows for that specific file — useful for reviewing or sharing results on a per-recording basis without filtering the combined output.
