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

    ### results.csv

    _Documentation coming soon._


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

    #### Folder Mode

    Same columns as single file mode but provides one row per processed file.
    
    ## Per-File Transcription Output

    In addition to the combined `detections.csv` / `detections_words.csv`, the application also saves an individual pair of CSVs for **each processed file**, named after the source file:

    - `[filename]_transcript.csv` – same columns as `detections.csv`, scoped to that file only
    - `[filename]_words.csv` – same columns as `detections_words.csv`, scoped to that file only

    !!! note
        In **Single File Mode**, these per-file CSVs are identical to `detections.csv` and `detections_words.csv`, since only one file is processed.
        In **Folder Mode**, they diverge: `detections.csv` / `detections_words.csv` aggregate rows across *all* files in the folder, while `[filename]_transcript.csv` / `[filename]_words.csv` contain only the rows for that specific file — useful for reviewing or sharing results on a per-recording basis without filtering the combined output.

    ### [filename]_transcript.txt

    A human-readable transcript with one line per detected segment, formatted as:
    [start-end] Segment text.
    
    - **start**, **end** – segment timestamps in seconds, matching `detections.csv`
