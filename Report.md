# Project: Face Authentication Attendance System

## 1. Approach & Model
[cite_start]This system utilizes the **OpenCV LBPH (Local Binary Patterns Histograms)** Face Recognizer for real-time attendance tracking[cite: 20].
* **Library:** `opencv-contrib-python` (Lightweight, CPU-efficient).
* **Face Detection:** Haar Cascade Classifiers (`haarcascade_frontalface_default.xml`) are used for rapid face localization.
* [cite_start]**Feature Extraction:** The system uses LBPH to extract local features from facial images, creating a histogram that represents the face texture[cite: 20].
* **Comparison:** Recognition utilizes Chi-square distance comparison. [cite_start]A confidence score (0-100) determines if a face matches the stored training data[cite: 7].

## 2. Training Process
[cite_start]The system implements an on-the-fly training mechanism[cite: 19]:
* **Registration:** The user captures a reference image via the webcam (`r` key).
* **Processing:** The image is converted to grayscale, and the face region is cropped and assigned a unique ID.
* **Model Update:** The LBPH recognizer is immediately retrained on the new dataset, allowing for instant recognition without restarting the application.

## 3. Spoof Prevention
[cite_start]Basic spoof prevention [cite: 14] is handled through:
1.  **Confidence Thresholding:** The system rejects matches where the confidence score is above 80 (lower is better in LBPH), reducing false positives from low-quality inputs.
2.  **Face Structure Validation:** The Haar Cascade classifier requires specific facial features (eyes, nose bridge) to be present, offering basic resistance against simple photo attacks compared to raw pixel matching.

## 4. Accuracy & Limitations
* [cite_start]**Accuracy:** The system performs well in consistent lighting conditions with a frontal face view[cite: 21].
* **Limitations:**
    * [cite_start]**Lighting:** Haar Cascades are sensitive to uneven lighting or shadows[cite: 25].
    * [cite_start]**Rotation:** Significant head tilt or side profiles may not be detected[cite: 22].

## 5. Deliverables Checklist
* [cite_start][x] Working demo (Local Python Script)[cite: 16].
* [cite_start][x] Complete codebase (`attendance_system.py`)[cite: 16].
* [cite_start][x] Attendance Logs (`attendance_log.csv`)[cite: 9].