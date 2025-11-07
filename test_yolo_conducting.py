#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for YOLO pose tracking integration.

Tests the YoloPoseTracking class and its integration with conducting analysis.
"""
import cv2
import time
from vision_modules.yolo_pose_tracking import YoloPoseTracking
from vision_modules.conducting_protocol import FingerConductingAnalyzer


def main():
    print("YOLO Pose Tracking Test")
    print("=" * 60)
    print("This test verifies YOLO pose tracking for conducting gestures.")
    print("Press 'q' to quit, 'h' to switch tracked wrist")
    print("-" * 60)
    
    # Initialize YOLO tracker
    tracker = YoloPoseTracking(
        model_path='yolo11s-pose.pt',
        use_right_hand=True,
        confidence_threshold=0.3
    )
    
    # Initialize conducting analyzer
    analyzer = FingerConductingAnalyzer(
        history_length=40,
        velocity_smoothing=4,
        neutral_velocity_threshold=0.24
    )
    analyzer.set_time_signature(4)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS tracking
    fps_buffer = []
    last_time = time.time()
    
    print("\nStarting webcam capture...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # Process with YOLO
            pose_result, _ = tracker.process_frame(frame)
            
            # Extract wrist position
            wrist_position_normalized = None
            if pose_result and pose_result.wrist_position:
                h, w = frame.shape[:2]
                x, y = pose_result.wrist_position
                wrist_position_normalized = (x / w, y / h)
            
            # Update conducting analyzer
            conducting_frame, _ = analyzer.update_both_hands(
                primary_position=wrist_position_normalized,
                secondary_position=None,
                timestamp=pose_result.timestamp if pose_result else None
            )
            
            # Draw pose visualization
            display_frame = tracker.get_annotated_frame(frame, pose_result)
            
            # Draw conducting info
            if conducting_frame:
                info_y = 30
                line_height = 35
                
                # Tempo
                if conducting_frame.tempo_estimate:
                    tempo_text = f"Tempo: {conducting_frame.tempo_estimate:.1f} BPM"
                    cv2.putText(display_frame, tempo_text, (10, info_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    info_y += line_height
                
                # Beat info
                if conducting_frame.beat_index:
                    beat_text = f"Beat: {conducting_frame.beat_index}/4"
                    cv2.putText(display_frame, beat_text, (10, info_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    info_y += line_height
                
                # Direction
                dir_text = f"Direction: {conducting_frame.direction}"
                cv2.putText(display_frame, dir_text, (10, info_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                info_y += line_height
                
                # Beat event
                if conducting_frame.beat_event:
                    cv2.putText(display_frame, "BEAT!", (w // 2 - 80, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            fps_buffer.append(fps)
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            avg_fps = sum(fps_buffer) / len(fps_buffer)
            
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display tracking mode
            mode_text = f"Tracking: {tracker.hand_label.upper()} wrist"
            cv2.putText(display_frame, mode_text, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('YOLO Conducting Test', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('h'):  # Switch wrist
                tracker.set_tracking_hand(not tracker.use_right_hand)
            elif key == ord('r'):  # Reset analyzer
                analyzer.reset()
                print("Conducting analyzer reset")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("\nTest completed")


if __name__ == '__main__':
    main()
