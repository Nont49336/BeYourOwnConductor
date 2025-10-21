#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hand Gesture Recognition Demo Application

This demo uses the HandTracking class to perform real-time hand gesture recognition
from webcam input, replicating the functionality of the original app.py.
"""
import argparse
import copy

import cv2 as cv

from vision_modules.hand_tracking import HandTracking


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hand Gesture Recognition Demo')
    
    parser.add_argument("--device", type=int, default=0,
                       help='Camera device number (default: 0)')
    parser.add_argument("--width", type=int, default=960,
                       help='Camera capture width (default: 960)')
    parser.add_argument("--height", type=int, default=540,
                       help='Camera capture height (default: 540)')
    parser.add_argument('--use_static_image_mode', action='store_true',
                       help='Use static image mode for MediaPipe')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7,
                       help='Minimum confidence for hand detection (default: 0.7)')
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                       help='Minimum confidence for hand tracking (default: 0.5)')
    
    return parser.parse_args()


def draw_landmarks(image, landmark_point):
    """Draw hand landmarks and connections on the image."""
    if len(landmark_point) == 0:
        return image
    
    # Thumb
    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
            (255, 255, 255), 2)
    
    # Index finger
    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
            (255, 255, 255), 2)
    
    # Middle finger
    cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
            (255, 255, 255), 2)
    
    # Ring finger
    cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
            (255, 255, 255), 2)
    
    # Little finger
    cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
            (255, 255, 255), 2)
    
    # Palm
    cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
            (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
            (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
            (255, 255, 255), 2)
    
    # Draw keypoints
    for index, landmark in enumerate(landmark_point):
        if index in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index in [4, 8, 12, 16, 20]:  # Finger tips
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
    
    return image


def draw_bounding_rect(image, brect):
    """Draw bounding rectangle around hand."""
    if len(brect) > 0:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    """Draw hand information text on image."""
    if len(brect) == 0:
        return image
    
    # Draw black background for text
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                (0, 0, 0), -1)
    
    # Draw handedness and hand sign
    info_text = handedness
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    # Draw finger gesture
    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    
    return image


def draw_point_history(image, point_history):
    """Draw the history trail of finger movements."""
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                     (152, 251, 152), 2)
    return image


def draw_fps(image, fps):
    """Draw FPS counter on image."""
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
              1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
              1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image


def main():
    """Main application loop."""
    # Parse arguments
    args = get_args()
    
    # Initialize camera
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize hand tracking
    hand_tracker = HandTracking(
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        use_static_image_mode=args.use_static_image_mode,
        history_length=16
    )
    
    # FPS calculation
    import time
    from collections import deque
    fps_buffer = deque(maxlen=10)
    last_time = time.time()
    
    print("Hand Gesture Recognition Demo")
    print("Press ESC to exit")
    print("-" * 50)
    
    try:
        while True:
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            fps_buffer.append(fps)
            avg_fps = sum(fps_buffer) / len(fps_buffer)
            
            # Check for exit key
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror the frame
            frame = cv.flip(frame, 1)
            
            # Create display image
            display_image = copy.deepcopy(frame)
            
            # Process frame with hand tracking
            results = hand_tracker.process_frame(frame)
            
            # Draw visualizations if hand detected
            if results['hand_detected']:
                # Draw landmarks
                display_image = draw_landmarks(display_image, results['landmark_list'])
                
                # Draw bounding rectangle
                display_image = draw_bounding_rect(display_image, results['bounding_rect'])
                
                # Draw info text
                display_image = draw_info_text(
                    display_image,
                    results['bounding_rect'],
                    results['handedness'],
                    results['hand_sign_label'],
                    results['finger_gesture_label']
                )
            
            # Draw point history
            display_image = draw_point_history(display_image, results['point_history'])
            
            # Draw FPS
            display_image = draw_fps(display_image, round(avg_fps, 2))
            
            # Display the frame
            cv.imshow('Hand Gesture Recognition', display_image)
    
    finally:
        # Clean up
        cap.release()
        cv.destroyAllWindows()
        hand_tracker.close()
        print("\nApplication closed")


if __name__ == '__main__':
    main()
