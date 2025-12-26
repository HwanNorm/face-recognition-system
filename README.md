
# Face Recognition System

A real-time face recognition system built with DeepFace and OpenCV, using state-of-the-art deep learning models for accurate facial identification.

## Features

- **Real-time Face Recognition**: Live webcam-based face detection and recognition
- **Multiple AI Models**: Support for 9 different recognition models (Facenet512, VGG-Face, ArcFace, etc.)
- **Easy Dataset Creation**: Built-in tool to capture and register new faces
- **High Accuracy**: Uses Facenet512 model for industry-standard recognition
- **Interactive Interface**: User-friendly menu system for all operations

## Requirements

- Python 3.11 or 3.12 (Python 3.14 not supported due to TensorFlow compatibility)
- Webcam/Camera
- Windows/Linux/MacOS

## Installation

### Step 1: Install Python 3.12

Download and install Python 3.12 from [python.org](https://www.python.org/downloads/) or via Microsoft Store.

Verify installation:
```bash
py --list
```

### Step 2: Clone the Repository

```bash
# Clone this repository
git clone https://github.com/HwanNorm/face-recognition-system.git

# Navigate to the project directory
cd face-recognition-system
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment with Python 3.12
py -3.12 -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Windows CMD:
venv\Scripts\activate.bat

# On Linux/Mac:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python pandas deepface tf-keras
```

## Usage

### Run the Application

```bash
python main.py
```

### Menu Options

1. **Capture images for new person**
   - Register a new person in the system
   - Enter the person's name
   - Press SPACE to capture photos (recommended: 10+ images)
   - Press Q to quit early
   - Capture from different angles for better accuracy

2. **Verify dataset**
   - Check registered people and image counts
   - Verify dataset structure

3. **Start real-time recognition**
   - Launch webcam-based face recognition
   - Press Q to quit
   - Recognized faces show with green labels
   - Unknown faces show with red labels

4. **Test with image file**
   - Test recognition on a static image
   - Provide path to image file

5. **Change recognition model**
   - Switch between different AI models:
     - Facenet512 (default, best accuracy)
     - VGG-Face
     - ArcFace
     - OpenFace
     - DeepFace
     - And more...

6. **Exit**
   - Close the application

## Quick Start Guide

1. Run the application: `python main.py`
2. Choose option 1 to register yourself
3. Enter your name and capture 10 photos
4. Choose option 3 to start real-time recognition
5. Your face will be recognized with confidence percentage

## Project Structure

```
face-recognition-system/
│
├── main.py                 # Main application file
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── dataset/              # Face database (created automatically)
│   ├── person1/         # Images for person 1
│   ├── person2/         # Images for person 2
│   └── ...
├── models/               # Model cache (created automatically)
└── venv/                 # Virtual environment
```

## Technical Details

### Models Used

- Default Model: Facenet512 (128-dimensional embeddings)
- Detector: OpenCV Haar Cascades
- Framework: DeepFace + TensorFlow/Keras

### How It Works

1. **Face Detection**: OpenCV detects faces in real-time
2. **Face Embedding**: DeepFace converts faces to numerical vectors
3. **Face Matching**: Compares embeddings with database using cosine similarity
4. **Recognition**: Returns closest match with confidence score

### Performance

- Recognition interval: Every 30 frames (adjustable in code)
- Typical accuracy: 95%+ with good lighting and 10+ training images
- Real-time FPS: 20-30 depending on hardware

## Troubleshooting

### "ModuleNotFoundError: No module named 'cv2'"

Make sure you installed opencv-python (not cv2):

```bash
python -m pip install opencv-python
```

### "No suitable Python runtime found"

Install Python 3.12 or 3.11. Python 3.14 is not supported by TensorFlow yet.

### "Could not open camera"

- Ensure webcam is connected and not used by another application
- Check camera permissions in system settings
- Try changing camera index in code (line 35, 154): `cv2.VideoCapture(1)` instead of 0

### Low Recognition Accuracy

- Capture more training images (15-20 recommended)
- Ensure good lighting during capture and recognition
- Capture images from multiple angles
- Try different recognition models (option 5 in menu)

## Dependencies

- opencv-python: Computer vision and camera handling
- pandas: Data manipulation
- deepface: Face recognition framework
- tf-keras: Deep learning backend
- numpy: Numerical computations (installed with above packages)

## License

This project is for educational purposes.

## Acknowledgments

- DeepFace - Face recognition framework
- OpenCV - Computer vision library
- Facenet - Face recognition model

## Author

Ho Anh - Face Recognition Project

## Version

v1.0 - December 2024
```
