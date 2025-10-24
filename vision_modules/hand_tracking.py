#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hand Tracking Module
A class-based hand gesture recognition system using MediaPipe and TensorFlow Lite.
"""
import csv
import copy
import itertools
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from enum import Enum
from pathlib import Path

import cv2 as cv
import numpy as np
import mediapipe as mp


class Handedness(Enum):
    """Enum representing hand type (Left or Right)."""
    LEFT = "Left"
    RIGHT = "Right"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(label: str):
        """Create Handedness enum from string."""
        if label.lower() == "left":
            return Handedness.LEFT
        elif label.lower() == "right":
            return Handedness.RIGHT
        else:
            raise ValueError(f"Unknown Handedness label: {label}")

@dataclass
class HandTrackingResult:
    """
    Per-frame hand tracking information for a single detected hand.
    
    This data structure encapsulates all hand tracking and gesture recognition
    information for a single hand in a frame.
    """
    
    # Detection status
    hand_detected: bool = True
    
    # Classification results
    hand_sign_id: int = -1
    hand_sign_label: str = ""
    finger_gesture_id: int = 0
    finger_gesture_label: str = ""
    
    # Hand information
    handedness: Handedness = Handedness.LEFT
    
    # Landmark data
    landmark_list: List[List[int]] = field(default_factory=list)
    bounding_rect: List[int] = field(default_factory=list)
    
    # History tracking
    point_history: List[List[int]] = field(default_factory=list)
    
    # Pre-processed data
    pre_processed_landmarks: List[float] = field(default_factory=list)
    pre_processed_point_history: List[float] = field(default_factory=list)

    # Timestamp
    timestamp: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        data = asdict(self)
        # Convert Handedness enum to string
        data['handedness'] = str(self.handedness)
        return data
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"HandTrackingResult("
            f"hand={self.handedness}, "
            f"detected={self.hand_detected}, "
            f"sign={self.hand_sign_label}(id={self.hand_sign_id}), "
            f"gesture={self.finger_gesture_label}(id={self.finger_gesture_id}), "
            f"landmarks={len(self.landmark_list)})"
        )


class HandTracking:
    """
    Hand gesture recognition class that processes video frames and returns gesture states.
    
    This class uses MediaPipe for hand detection and custom TFLite models for
    gesture classification.
    """
    
    def __init__(
        self,
        max_num_hands=1,
        primary_hand=None,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        use_static_image_mode=False,
        history_length=16,
        model_base_path=None
    ):
        """
        Initialize the HandTracking system.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            primary_hand: The primary hand to track (Handedness.LEFT or Handedness.RIGHT). If None, tracks only the first detected hand.
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            use_static_image_mode: Whether to treat each frame independently
            history_length: Length of point history for gesture classification
            model_base_path: Base path for model files (if None, uses default location)
        """
        assert max_num_hands < 2 or primary_hand is not None, "When tracking multiple hands, primary_hand must be specified."
        assert max_num_hands <= 2, "Currently supports tracking up to 2 hands only."
        self.primary_hand = primary_hand

        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        # Set model base path
        if model_base_path is None:
            # Default to the hand-gesture-recognition-mediapipe folder
            model_base_path = Path(__file__).parent / "hand-gesture-recognition-mediapipe"
        else:
            model_base_path = Path(model_base_path)
        
        # Load classification models
        self._load_models(model_base_path)
        
        # Load labels
        self._load_labels(model_base_path)
        
        # Initialize history tracking
        self.history_length = history_length
        self.max_num_hands = max_num_hands
        
        # Track history for each hand separately (indexed by Handedness enum)
        self.point_history = {
            Handedness.LEFT: deque(maxlen=history_length),
            Handedness.RIGHT: deque(maxlen=history_length)
        }
        self.finger_gesture_history = {
            Handedness.LEFT: deque(maxlen=history_length),
            Handedness.RIGHT: deque(maxlen=history_length)
        }
        
        # State tracking
        self._last_results = []
    
    def _load_models(self, model_base_path):
        """Load TFLite classification models."""
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(model_base_path))
        
        from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
        from model.point_history_classifier.point_history_classifier import PointHistoryClassifier
        
        keypoint_model_path = model_base_path / "model" / "keypoint_classifier" / "keypoint_classifier.tflite"
        point_history_model_path = model_base_path / "model" / "point_history_classifier" / "point_history_classifier.tflite"
        
        self.keypoint_classifier = KeyPointClassifier(
            model_path=str(keypoint_model_path)
        )
        self.point_history_classifier = PointHistoryClassifier(
            model_path=str(point_history_model_path)
        )
    
    def _load_labels(self, model_base_path):
        """Load classification labels from CSV files."""
        keypoint_label_path = model_base_path / "model" / "keypoint_classifier" / "keypoint_classifier_label.csv"
        point_history_label_path = model_base_path / "model" / "point_history_classifier" / "point_history_classifier_label.csv"
        
        with open(keypoint_label_path, encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        
        with open(point_history_label_path, encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]
    
    def process_frame(self, frame) -> Tuple[Optional[HandTrackingResult], Optional[HandTrackingResult]]:
        """
        Process a single frame and return hand gesture information.
        
        Args:
            frame: BGR image from OpenCV (numpy array)
            
        Returns:
            tuple: A tuple containing HandTrackingResult objects.
                The first element corresponds to the primary hand and the second to the secondary hand (if detected).
                If a hand is not detected, its corresponding element will be None.
        """
        # Convert BGR to RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        
        # Initialize outputs
        primary_result = None
        secondary_result = None
        detected_hands = set()
        
        # Process hand landmarks if detected
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                hand_label_str = handedness.classification[0].label
                hand_label = Handedness.from_str(hand_label_str)
                detected_hands.add(hand_label)

                # No primary hand specified, return first detected hand as primary
                if self.primary_hand is None:
                    primary_result = self._build_hand_result(frame, hand_landmarks, handedness)
                    break
                else:
                    hand_result = self._build_hand_result(frame, hand_landmarks, handedness)
                    if hand_label == self.primary_hand:
                        primary_result = hand_result
                    else:
                        secondary_result = hand_result

        # Update point history for hands that weren't detected
        for hand_label in [Handedness.LEFT, Handedness.RIGHT]:
            if hand_label not in detected_hands:
                self.point_history[hand_label].append([0, 0])

        # Store last results
        self._last_results = [primary_result, secondary_result]

        return primary_result, secondary_result
    
    def _build_hand_result(self, frame: np.ndarray, hand_landmarks, handedness) -> HandTrackingResult:
        hand_label_str = handedness.classification[0].label
        hand_label = Handedness.from_str(hand_label_str)
        
        # Calculate bounding box
        brect = self._calc_bounding_rect(frame, hand_landmarks)
        
        # Calculate landmark list
        landmark_list = self._calc_landmark_list(frame, hand_landmarks)
        
        # Pre-process landmarks and point history
        pre_processed_landmark_list = self._pre_process_landmark(landmark_list)
        pre_processed_point_history_list = self._pre_process_point_history(
            frame, self.point_history[hand_label]
        )
        
        # Hand sign classification
        hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
        
        # Update point history for this hand
        if hand_sign_id == 2:  # Point gesture
            self.point_history[hand_label].append(landmark_list[8])
        else:
            self.point_history[hand_label].append([0, 0])
        
        # Finger gesture classification
        finger_gesture_id = 0
        point_history_len = len(pre_processed_point_history_list)
        if point_history_len == (self.history_length * 2):
            finger_gesture_id = self.point_history_classifier(
                pre_processed_point_history_list
            )
        
        # Update finger gesture history for this hand
        self.finger_gesture_history[hand_label].append(finger_gesture_id)
        most_common_fg_id = Counter(self.finger_gesture_history[hand_label]).most_common()
        
        # Get most common finger gesture
        if most_common_fg_id:
            finger_gesture_id = most_common_fg_id[0][0]
        
        # Build result object for this hand
        return HandTrackingResult(
            hand_detected=True,
            hand_sign_id=hand_sign_id,
            hand_sign_label=self.keypoint_classifier_labels[hand_sign_id],
            finger_gesture_id=finger_gesture_id,
            finger_gesture_label=self.point_history_classifier_labels[finger_gesture_id],
            handedness=hand_label,
            landmark_list=landmark_list,
            bounding_rect=brect,
            point_history=list(self.point_history[hand_label]),
            pre_processed_landmarks=pre_processed_landmark_list,
            pre_processed_point_history=pre_processed_point_history_list
        )

    
    def _calc_bounding_rect(self, image, landmarks):
        """Calculate bounding rectangle for hand landmarks."""
        image_width, image_height = image.shape[1], image.shape[0]
        
        landmark_array = np.empty((0, 2), int)
        
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        
        x, y, w, h = cv.boundingRect(landmark_array)
        
        return [x, y, x + w, y + h]
    
    def _calc_landmark_list(self, image, landmarks):
        """Calculate landmark coordinates in image space."""
        image_width, image_height = image.shape[1], image.shape[0]
        
        landmark_point = []
        
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            
            landmark_point.append([landmark_x, landmark_y])
        
        return landmark_point
    
    def _pre_process_landmark(self, landmark_list):
        """Convert landmarks to relative and normalized coordinates."""
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        
        # Convert to one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list)
        )
        
        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))
        
        def normalize_(n):
            return n / max_value
        
        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        
        return temp_landmark_list
    
    def _pre_process_point_history(self, image, point_history):
        """Convert point history to relative and normalized coordinates."""
        image_width, image_height = image.shape[1], image.shape[0]
        
        temp_point_history = copy.deepcopy(point_history)
        
        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]
            
            temp_point_history[index][0] = (temp_point_history[index][0] -
                                            base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                            base_y) / image_height
        
        # Convert to one-dimensional list
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history)
        )
        
        return temp_point_history
    
    def set_primary_hand(self, primary_hand: Handedness):
        """Set the primary hand to track."""
        # This method can be expanded to adjust internal logic if needed
        self.primary_hand = primary_hand
    
    def reset_history(self):
        """Reset point and gesture history for all hands."""
        for hand_label in [Handedness.LEFT, Handedness.RIGHT]:
            self.point_history[hand_label].clear()
            self.finger_gesture_history[hand_label].clear()
    
    def get_last_results(self):
        """Get the last processing results."""
        return self._last_results
    
    def close(self):
        """Clean up resources."""
        self.hands.close()
