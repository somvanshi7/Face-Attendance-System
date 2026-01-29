# Face Authentication Attendance System

A real-time face authentication system built for the AI/ML Intern Assignment. This project uses computer vision to register users, mark attendance (punch-in/punch-out), and prevent basic spoofing attempts.

## Features
* **Real-time Face Detection:** Uses Haar Cascades for fast detection.
* **Face Recognition:** Implements LBPH (Local Binary Patterns Histograms) for training and recognition.
* **Attendance Logging:** Automatically logs "Punch-in" and "Punch-out" times to a CSV file.
* **Spoof Prevention:** Basic confidence thresholding to reject low-quality matches.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/somvanshi7/Face-Attendance-System.git](https://github.com/somvanshi7/Face-Attendance-System.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the application:
    ```bash
    python att.py
    ```
2.  **Controls:**
    * `r`: Register a new user (Look at the terminal to type your name).
    * `p`: Punch In / Punch Out (Must be recognized first).
    * `q`: Quit the application.

## Technologies Used
* Python 3.x
* OpenCV (`opencv-contrib-python`)
* Numpy
