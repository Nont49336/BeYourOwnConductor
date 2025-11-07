"""
YOLO COCO Pose Keypoint Reference
==================================

Visual representation of the 17 COCO keypoints used by YOLO pose models.

         0 (nose)
         |
    1   / \   2
  (L-eye)   (R-eye)
    |       |
    3       4
  (L-ear) (R-ear)


    5 ----------- 6
  (L-shoulder)  (R-shoulder)
    |             |
    7             8
  (L-elbow)    (R-elbow)
    |             |
    9 *         * 10  ← WRIST (PRIMARY TRACKING POINT)
  (L-wrist)   (R-wrist)


    11 --------- 12
  (L-hip)     (R-hip)
    |             |
    13           14
  (L-knee)    (R-knee)
    |             |
    15           16
  (L-ankle)   (R-ankle)


Keypoint Index Mapping:
-----------------------
0:  Nose
1:  Left Eye
2:  Right Eye
3:  Left Ear
4:  Right Ear
5:  Left Shoulder
6:  Right Shoulder
7:  Left Elbow
8:  Right Elbow
9:  Left Wrist   ← Used for left-hand conducting
10: Right Wrist  ← Used for right-hand conducting
11: Left Hip
12: Right Hip
13: Left Knee
14: Right Knee
15: Left Ankle
16: Right Ankle


Conducting Configuration:
-------------------------

Right-hand conducting (default):
    python demo_finger_conducting.py --use_yolo --primary_hand right
    → Tracks keypoint 10 (Right Wrist)

Left-hand conducting:
    python demo_finger_conducting.py --use_yolo --primary_hand left
    → Tracks keypoint 9 (Left Wrist)


Position Data Format:
---------------------
Each keypoint has 3 values: [x, y, confidence]

- x: Horizontal position (pixels)
- y: Vertical position (pixels)
- confidence: Detection confidence (0.0 - 1.0)

Example:
    keypoints[10] = [512, 384, 0.95]  # Right wrist at (512, 384) with 95% confidence


Conducting Position Extraction:
--------------------------------

# Get wrist keypoint
wrist_idx = 10  # Right wrist
wrist_data = keypoints[wrist_idx]  # [x, y, conf]

# Check confidence threshold
if wrist_data[2] > 0.3:  # Confidence > 30%
    wrist_position = (int(wrist_data[0]), int(wrist_data[1]))
    
    # Normalize for conducting analyzer (0-1 range)
    normalized = (wrist_data[0] / frame_width, wrist_data[1] / frame_height)
    
    # Pass to conducting analyzer
    conducting_frame, _ = analyzer.update_both_hands(
        primary_position=normalized,
        secondary_position=None,
        timestamp=time.time()
    )


Beat Detection:
---------------

Beats are detected at the LOWEST POINT of downward wrist motion:

    Position Y (screen coords, down = positive)
    
    Start ────○ (wrist high)
              │
              │ Moving DOWN
              │ (direction = DOWN, velocity < 0)
              ↓
    Ictus ────● (wrist lowest, velocity = 0)  ← BEAT DETECTED HERE!
              ↑
              │ Moving UP
              │ (direction = UP, velocity > 0)
    End ──────○ (wrist high)


The beat is triggered when:
1. Wrist was moving DOWN (negative velocity)
2. Wrist reaches local minimum Y position
3. Wrist starts moving UP (positive velocity)

This is the "ictus" point in conducting terminology.


Typical Conducting Motion:
---------------------------

4/4 Time Pattern (Downward-Left-Right-Up):

      1st beat (Down)
         ↓
         ●────→ 2nd beat (Left)
        ↗ ↘    
   4th ●   ↘
  (Up)      ↘
            ● 3rd beat (Right)

The system tracks only the VERTICAL (Y) component for beat detection.
Horizontal (X) movement is tracked but not used for beat timing.


Model Selection:
----------------

yolo11n-pose.pt:  Fastest, lowest accuracy
yolo11s-pose.pt:  Balanced (RECOMMENDED)
yolo11m-pose.pt:  More accurate, slower
yolo11l-pose.pt:  Highest accuracy, slowest

Performance comparison:
    Model     | FPS (CPU) | FPS (GPU) | Accuracy
    ----------|-----------|-----------|----------
    yolo11n   | ~20       | ~80       | Good
    yolo11s   | ~15       | ~60       | Better   ← Default
    yolo11m   | ~10       | ~40       | Best
    yolo11l   | ~5        | ~25       | Excellent


Optimal Setup:
--------------

Distance from camera:  1.5 - 2.5 meters
Lighting:              Front-lit, no backlighting
Visibility:            Full upper body in frame
Background:            Clean, uncluttered
Clothing:              Contrasting with background
Camera angle:          Eye level or slightly above
Resolution:            960x540 (default) or 1280x720


Troubleshooting:
----------------

Issue: Wrist not detected
Solution:
    - Ensure arms are visible (not behind body)
    - Move closer to camera
    - Improve lighting
    - Lower confidence threshold

Issue: Jittery tracking
Solution:
    - Increase velocity_smoothing parameter
    - Use larger model (better accuracy)
    - Improve lighting
    - Stabilize camera

Issue: Late beat detection
Solution:
    - Make more pronounced downward motions
    - Increase gesture amplitude
    - Check neutral_velocity_threshold setting

Issue: False beat detections
Solution:
    - Reduce hand motion between beats
    - Pause briefly at top of each beat
    - Increase velocity_smoothing
    - Adjust neutral_velocity_threshold


Further Reading:
----------------

- YOLO_TRACKING.md: Detailed integration guide
- YOLO_INTEGRATION_SUMMARY.md: Technical implementation details
- YOLO_QUICKSTART.py: Quick reference (run with: python YOLO_QUICKSTART.py)
- Ultralytics docs: https://docs.ultralytics.com/tasks/pose/
- COCO dataset: https://cocodataset.org/#keypoints-2020

"""

if __name__ == '__main__':
    print(__doc__)
