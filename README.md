# 🧘 AsanaVision-Guided-yoga-pose-Corrector

A real-time yoga pose detection application powered by MediaPipe and OpenCV. Get instant feedback on your yoga poses with joint-specific corrections and pose accuracy tracking.

---

## ✨ Features

- **Real-Time Pose Detection**: Detects your body position using your webcam in real-time
- **5 Pre-Built Yoga Poses**: 
  - Warrior II
  - T Pose
  - Tree Pose
  - Mountain Pose
  - Chair Pose
- **Live Accuracy Feedback**: Get instant feedback on how well you're performing each pose (0-100%)
- **Joint-Specific Instructions**: Detailed guidance on which joints need adjustment (e.g., "Bend arm", "Straighten leg")
- **Pose Hold Timer**: Track how long you successfully maintain a pose above the accuracy threshold
- **Session Statistics**: Monitor total session time, poses held count, and best accuracy achieved
- **Professional UI**: Modern visual interface with status badges, progress bars, and real-time feedback
- **Customizable Reference Angles**: Easily modify target angles for each pose

---

## 🔧 Requirements

- **Python Version**: 3.12 or lower (MediaPipe does NOT support Python 3.13+)
- **Webcam**: A working webcam/camera
- **OS**: Windows, macOS, or Linux

### Why Python 3.12?
MediaPipe currently has compatibility issues with Python 3.13+. If you're on Python 3.13 or higher, you'll need to download and use Python 3.12 or lower.

---

## 📋 Installation & Setup

### Step 1: Check Your Python Version

```bash
python --version
```

**If you have Python 3.13 or higher**, skip to **Step 2 (Python 3.13+ Users)**.
**If you have Python 3.12 or lower**, skip to **Step 3**.

---

### Step 2: Python 3.13+ Users - Download Lower Version

1. Go to [python.org](https://www.python.org/downloads/)
2. Download **Python 3.12.x** (latest 3.12 version)
3. During installation:
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install pip"
   - Click "Install Now"

4. Verify installation:
```bash
python --version
# Should show: Python 3.12.x
```

---

### Step 3: Create Virtual Environment

Open your terminal/command prompt and navigate to your project folder, then run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal line.

---

### Step 4: Install Dependencies

Make sure you're inside the virtual environment (you should see `(venv)` in your terminal).

```bash
pip install -r requirements.txt
```

This will install:
- **opencv-python** - For video capture and UI rendering
- **mediapipe** - For pose detection (works with Python 3.12)
- **numpy** - For mathematical computations

---

## 🚀 Running the App

Make sure your virtual environment is activated (you should see `(venv)` in your terminal).

```bash
python yoga_pose_detector.py
```

The app will launch with a pose selection menu.

---

## 🎮 Controls & Shortcuts

| Key | Action |
|-----|--------|
| **1-5** | Select a yoga pose (1=Warrior II, 2=T Pose, 3=Tree Pose, 4=Mountain Pose, 5=Chair Pose) |
| **SPACE** | Toggle between pose selection menu and active pose detection |
| **R** | Reset all session statistics (timer, pose count, best accuracy) |
| **F** | Toggle fullscreen mode |
| **Q** | Quit the application |

---

## 📊 Understanding the UI

### Top Bar
- **Current Pose**: Shows which pose you're practicing
- **Accuracy Meter**: Real-time percentage (0-100%) of how well you match the target pose
- **Hold Timer**: Counts how long you've maintained the pose above the 80% accuracy threshold

### Status Badge (Top Right)
- **PERFECT!** (Green) - Accuracy ≥ 80%
- **GOOD** (Blue) - Accuracy 60-79%
- **ADJUST** (Light Blue) - Accuracy < 60%

### Joint Instructions
Red circles appear around joints that need adjustment with direct instructions:
- "Bend arm", "Straighten arm"
- "Bend forward", "Straighten hip"
- "Bend knee", "Straighten leg"

### Bottom Stats Panel
- **Session**: Total time spent in the app
- **Poses Held**: Number of times you successfully held a pose
- **Best**: Your highest accuracy percentage achieved

---

## 🎯 How to Use

1. **Select a Pose**: When the app starts, press 1-5 to choose your yoga pose
2. **Position Yourself**: Stand in front of the camera and try to match the target pose
3. **Watch the Accuracy**: The meter at the top shows how close you are to the perfect pose
4. **Follow Instructions**: Red circles with instructions appear on joints that need adjustment
5. **Hold the Pose**: Maintain above 80% accuracy for at least 1 second to start the hold timer
6. **Switch Poses**: Press SPACE and select a different pose number to practice another pose

---

## 🐛 Troubleshooting

### "Error: Cannot access camera"
- Check if your webcam is connected
- Make sure no other app is using your camera
- Try restarting the app

### "ModuleNotFoundError: No module named 'mediapipe'"
- Make sure you're inside the virtual environment `(venv)`
- Re-run: `pip install -r requirements.txt`

### "Pose not being detected"
- Move into better lighting
- Make sure your full body is visible to the camera
- Try adjusting your camera angle

### Python 3.13+ Compatibility Error
- You MUST use Python 3.12 or lower
- Follow Step 2 above to download Python 3.12
- After installation, delete your current venv and create a new one with Python 3.12

### App is slow or laggy
- Check your camera resolution settings
- Move closer to the camera
- Close other resource-heavy applications
- Try lowering the resolution by modifying the `cv2.resize()` values in the code

---

## 📁 Project Structure

```
yoga-pose-detector/
├── yoga_pose_detector.py    # Main application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🔬 Technical Details

### Pose Detection
- Uses MediaPipe Pose solution for real-time body landmark detection
- Detects 33 key body points including shoulders, elbows, wrists, hips, knees, and ankles
- Calculates 8 key joint angles for pose accuracy evaluation

### Accuracy Calculation
- Compares user's joint angles with target pose angles
- Accuracy threshold for "holding" a pose: 80%
- Formula: `Accuracy = 1 - (average angle difference / target angle)`

### Reference Angles
Each pose has 8 target angles:
1. Right Elbow
2. Left Elbow
3. Right Shoulder
4. Left Shoulder
5. Right Hip
6. Left Hip
7. Right Knee
8. Left Knee

---

## 🎓 Yoga Poses Explained

### Warrior II (Virabhadrasana II)
- Strong warrior stance with arms extended horizontally
- Great for building balance and leg strength

### T Pose
- Stand straight with arms horizontally extended
- Perfect for shoulder alignment and posture

### Tree Pose (Vrksasana)
- Balance on one leg with foot on inner thigh
- Excellent for balance and stability training

### Mountain Pose (Tadasana)
- Stand tall with feet together, arms at sides
- Foundation for all standing poses

### Chair Pose (Utkatasana)
- Squat position with arms raised overhead
- Builds leg strength and core engagement

---

## 📝 Deactivating Virtual Environment

When you're done, you can exit the virtual environment:

```bash
deactivate
```

---

## 🤝 Tips for Best Results

- **Lighting**: Practice in a well-lit area
- **Space**: Ensure you have enough room to move freely
- **Camera Angle**: Position camera at chest height, about 6-8 feet away
- **Clothing**: Wear fitted clothing to help with pose detection
- **Patience**: The app learns better the more you practice!

---

## 📄 License

This project is open-source and available for personal use.

---

## 🙏 Enjoy Your Yoga Practice!

Press Q to quit anytime. Happy practicing! 🧘‍♀️🧘‍♂️
