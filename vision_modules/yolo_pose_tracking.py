#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO Pose Tracking Module

Provides pose-based tracking using Ultralytics YOLO models as an alternative
to MediaPipe hand tracking. Uses wrist position for conducting gestures.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import time

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package not found. Install with: pip install ultralytics")


@dataclass
class YoloPoseResult:
    """
    Per-frame pose tracking information.
    
    This data structure encapsulates pose keypoint information from YOLO,
    providing similar interface to HandTrackingResult for compatibility.
    """
    
    # Detection status
    pose_detected: bool = True
    
    # Keypoint data (17 keypoints for COCO pose format)
    # Index mapping:
    # 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 
    # 7-8: elbows, 9-10: wrists, 11-12: hips, 13-14: knees, 15-16: ankles
    keypoints: np.ndarray = field(default_factory=lambda: np.zeros((17, 3)))  # x, y, confidence
    
    # Primary tracking point (wrist position)
    wrist_position: Optional[Tuple[int, int]] = None
    wrist_confidence: float = 0.0
    
    # Bounding box [x_min, y_min, x_max, y_max]
    bounding_box: List[int] = field(default_factory=list)
    
    # Detection confidence
    detection_confidence: float = 0.0
    
    # Timestamp
    timestamp: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'pose_detected': self.pose_detected,
            'keypoints': self.keypoints.tolist() if isinstance(self.keypoints, np.ndarray) else self.keypoints,
            'wrist_position': self.wrist_position,
            'wrist_confidence': self.wrist_confidence,
            'bounding_box': self.bounding_box,
            'detection_confidence': self.detection_confidence,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"YoloPoseResult("
            f"detected={self.pose_detected}, "
            f"wrist={self.wrist_position}, "
            f"conf={self.wrist_confidence:.2f})"
        )


class YoloPoseTracking:
    """
    YOLO-based pose tracking for conducting gestures.
    
    Uses Ultralytics YOLO pose estimation to track body keypoints,
    specifically the wrist position for conducting control.
    """
    
    # COCO pose keypoint indices
    KEYPOINT_NOSE = 0
    KEYPOINT_LEFT_EYE = 1
    KEYPOINT_RIGHT_EYE = 2
    KEYPOINT_LEFT_EAR = 3
    KEYPOINT_RIGHT_EAR = 4
    KEYPOINT_LEFT_SHOULDER = 5
    KEYPOINT_RIGHT_SHOULDER = 6
    KEYPOINT_LEFT_ELBOW = 7
    KEYPOINT_RIGHT_ELBOW = 8
    KEYPOINT_LEFT_WRIST = 9
    KEYPOINT_RIGHT_WRIST = 10
    KEYPOINT_LEFT_HIP = 11
    KEYPOINT_RIGHT_HIP = 12
    KEYPOINT_LEFT_KNEE = 13
    KEYPOINT_RIGHT_KNEE = 14
    KEYPOINT_LEFT_ANKLE = 15
    KEYPOINT_RIGHT_ANKLE = 16
    
    def __init__(
        self,
        model_path: str = 'yolo11s-pose.pt',
        use_right_hand: bool = True,
        confidence_threshold: float = 0.3,
        device: str = None
    ):
        """
        Initialize YOLO pose tracking.
        
        Args:
            model_path: Path to YOLO pose model weights (e.g., 'yolo11n-pose.pt', 'yolo11s-pose.pt')
            use_right_hand: If True, track right wrist; if False, track left wrist
            confidence_threshold: Minimum confidence for pose detection
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_path = model_path
        self.use_right_hand = use_right_hand
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO model
        print(f"Loading YOLO pose model: {model_path}")
        self.model = YOLO(model_path)
        
        # Set device if specified
        if device is not None:
            self.model.to(device)
        
        # Determine which wrist to track
        # NOTE: Since the camera is mirrored, right hand in real life appears on left side of image
        # So we swap: use_right_hand=True means track LEFT_WRIST keypoint (appears on right in mirror)
        self.wrist_keypoint_idx = self.KEYPOINT_LEFT_WRIST if use_right_hand else self.KEYPOINT_RIGHT_WRIST
        self.hand_label = "right" if use_right_hand else "left"
        
        print(f"YOLO pose tracking initialized (tracking {self.hand_label} wrist)")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[YoloPoseResult], Optional[YoloPoseResult]]:
        """
        Process a single frame and return pose information for both wrists.
        
        Args:
            frame: BGR image from OpenCV (numpy array)
            
        Returns:
            tuple: (primary_result, secondary_result)
                primary_result contains the primary wrist position (based on use_right_hand)
                secondary_result contains the secondary wrist position (opposite hand)
        """
        timestamp = time.time()
        
        # Run YOLO pose estimation
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Extract first detected person (if any)
        if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
            # No pose detected
            return None, None
        
        # Get the first person's keypoints
        first_person = results[0].keypoints.data[0]  # Shape: [17, 3] (x, y, conf)
        
        # Extract bounding box if available
        bounding_box = []
        if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            bounding_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        
        # Get detection confidence
        detection_conf = 0.0
        if results[0].boxes is not None and len(results[0].boxes.conf) > 0:
            detection_conf = float(results[0].boxes.conf[0].cpu().numpy())
        
        # Extract both wrist keypoints
        keypoints_np = first_person.cpu().numpy()
        
        # Primary wrist (based on use_right_hand setting)
        primary_wrist_data = keypoints_np[self.wrist_keypoint_idx]  # [x, y, conf]
        primary_wrist_position = None
        primary_wrist_confidence = float(primary_wrist_data[2])
        
        if primary_wrist_confidence > self.confidence_threshold:
            primary_wrist_position = (int(primary_wrist_data[0]), int(primary_wrist_data[1]))
        
        # Secondary wrist (opposite hand)
        secondary_wrist_idx = self.KEYPOINT_RIGHT_WRIST if self.use_right_hand else self.KEYPOINT_LEFT_WRIST
        secondary_wrist_data = keypoints_np[secondary_wrist_idx]  # [x, y, conf]
        secondary_wrist_position = None
        secondary_wrist_confidence = float(secondary_wrist_data[2])
        
        if secondary_wrist_confidence > self.confidence_threshold:
            secondary_wrist_position = (int(secondary_wrist_data[0]), int(secondary_wrist_data[1]))
        
        # Create primary result object
        primary_result = YoloPoseResult(
            pose_detected=True,
            keypoints=keypoints_np,
            wrist_position=primary_wrist_position,
            wrist_confidence=primary_wrist_confidence,
            bounding_box=bounding_box,
            detection_confidence=detection_conf,
            timestamp=timestamp
        )
        
        # Create secondary result object if secondary wrist is detected
        secondary_result = None
        if secondary_wrist_position is not None:
            secondary_result = YoloPoseResult(
                pose_detected=True,
                keypoints=keypoints_np,
                wrist_position=secondary_wrist_position,
                wrist_confidence=secondary_wrist_confidence,
                bounding_box=bounding_box,
                detection_confidence=detection_conf,
                timestamp=timestamp
            )
        
        return primary_result, secondary_result
    
    def get_annotated_frame(self, frame: np.ndarray, pose_result: Optional[YoloPoseResult]) -> np.ndarray:
        """
        Draw pose visualization on frame (minimal - just tracked wrist).
        
        Args:
            frame: Original frame
            pose_result: Pose detection result
            
        Returns:
            Annotated frame with wrist position highlighted
        """
        import cv2
        annotated = frame.copy()
        
        if pose_result is None or not pose_result.pose_detected:
            return annotated
        
        # Only highlight tracked wrist (no skeleton overlay)
        if pose_result.wrist_position is not None:
            # Draw crosshair at wrist position
            x, y = pose_result.wrist_position
            cv2.line(annotated, (x - 20, y), (x + 20, y), (0, 255, 0), 2)
            cv2.line(annotated, (x, y - 20), (x, y + 20), (0, 255, 0), 2)
            
            # Draw small circle
            cv2.circle(annotated, pose_result.wrist_position, 8, (0, 255, 0), 2)
        
        return annotated
    
    def set_tracking_hand(self, use_right_hand: bool):
        """
        Switch which wrist to track.
        
        Args:
            use_right_hand: If True, track right wrist; if False, track left wrist
        """
        self.use_right_hand = use_right_hand
        # NOTE: Since camera is mirrored, swap the keypoint indices
        self.wrist_keypoint_idx = self.KEYPOINT_LEFT_WRIST if use_right_hand else self.KEYPOINT_RIGHT_WRIST
        self.hand_label = "right" if use_right_hand else "left"
        print(f"Switched to tracking {self.hand_label} wrist")
    
    def close(self):
        """Clean up resources."""
        # YOLO model cleanup (if needed)
        pass
