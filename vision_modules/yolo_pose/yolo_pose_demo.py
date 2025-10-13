import cv2
from ultralytics import YOLO

# Load YOLO11 pose model
model = YOLO('yolo11n-pose.pt')  # use yolo11s-pose.pt, yolo11m-pose.pt for better accuracy

# Open webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Run YOLO11 pose estimation
    results = model(frame, conf=0.3, verbose=False)
    
    # Draw results on frame
    annotated_frame = results[0].plot()
    
    # Mirror the frame horizontally
    annotated_frame = cv2.flip(annotated_frame, 1)
    
    # Display FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('YOLO11 Pose Tracking', annotated_frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()