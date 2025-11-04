#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conducting Protocol Module

Defines the common data structures and protocol for conducting gesture analysis.
This protocol is used by both finger-based and full-body conducting modes to
communicate conducting information to the audio playback system.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List
from enum import Enum
import time
from collections import deque
import numpy as np


class Direction(Enum):
    """Enumeration of vertical motion directions."""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"
    
    def __str__(self):
        return self.value


class MotionPhase(Enum):
    """Enumeration of motion phases within a beat cycle."""
    PRE_BEAT = "pre_beat"      # Preparation/anticipation before beat
    ON_BEAT = "on_beat"        # The moment of the beat (ictus)
    POST_BEAT = "post_beat"    # Follow-through after beat
    TRANSITION = "transition"   # Moving between beats
    
    def __str__(self):
        return self.value


class BeatEvent(Enum):
    """Enumeration of beat event types."""
    BEAT = "beat"              # A beat was detected (ictus at lowest point)
    NONE = None
    
    def __str__(self):
        return self.value if self.value else "none"


@dataclass
class ConductingFrame:
    """
    Per-frame conducting information following the common protocol.
    
    This data structure encapsulates all conducting-related information
    for a single frame, providing a standardized interface between
    vision tracking and audio playback systems.
    """
    
    # Temporal information
    timestamp: float = field(default_factory=time.time)
    
    # Position information (normalized 0-1, relative to frame)
    position: Tuple[float, float] = (0.5, 0.5)
    
    # Motion information
    velocity: Tuple[float, float] = (0.0, 0.0)
    direction: Direction = Direction.NEUTRAL
    
    # Beat information
    motion_phase: MotionPhase = MotionPhase.TRANSITION
    beat_event: Optional[BeatEvent] = None
    beat_index: Optional[int] = None  # 1-based index within measure
    
    # Tempo information
    tempo_estimate: Optional[float] = None  # BPM
    
    # Volume information
    volume_estimate: Optional[float] = None  # Volume level (0.5-1.5 range)
    
    # Quality metrics
    gesture_energy: float = 0.0  # Magnitude of motion (0-1 normalized)
    
    # Optional metadata
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        data = asdict(self)
        # Convert enums to strings
        data['direction'] = str(self.direction)
        data['motion_phase'] = str(self.motion_phase)
        data['beat_event'] = str(self.beat_event) if self.beat_event else None
        return data
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ConductingFrame(t={self.timestamp:.3f}, "
            f"pos={self.position[0]:.2f},{self.position[1]:.2f}, "
            f"vel={self.velocity[0]:.2f},{self.velocity[1]:.2f}, "
            f"dir={self.direction}, phase={self.motion_phase}, "
            f"beat={self.beat_event}, tempo={self.tempo_estimate}, "
            f"volume={self.volume_estimate}, "
            f"energy={self.gesture_energy:.2f})"
        )


class ConductingAnalyzer:
    """
    Base analyzer for converting raw position data into conducting protocol frames.
    
    This class provides common functionality for analyzing conducting gestures,
    including velocity calculation, direction detection, and tempo estimation.
    Subclasses can extend this for specific conducting styles (finger vs full-body).
    """
    
    def __init__(
        self,
        history_length: int = 30,
        velocity_smoothing: int = 3,
        tempo_memory: int = 10,
        neutral_velocity_threshold: float = 0.3,  # Velocity below this may indicate neutral (if sustained)
        neutral_duration_threshold: float = 0.5,  # Time (seconds) below threshold to confirm neutral
        min_beat_interval: float = 0.2,  # Minimum 200ms between beats (300 BPM max)
        volume_smoothing: int = 10,  # Number of frames to smooth volume over
        min_volume: float = 0.5,  # Minimum volume level
        max_volume: float = 2.0,  # Maximum volume level
        volume_displacement_threshold: float = 0.02  # Minimum displacement to adjust volume
    ):
        """
        Initialize the conducting analyzer.
        
        Args:
            history_length: Number of frames to keep in position history
            velocity_smoothing: Number of frames to smooth velocity over
            tempo_memory: Number of beats to use for tempo estimation
            neutral_velocity_threshold: Velocity threshold for potential neutral motion (units/sec)
            neutral_duration_threshold: Time below velocity threshold to confirm neutral (seconds)
            min_beat_interval: Minimum time between beats in seconds
            volume_smoothing: Number of frames to smooth volume estimates over
            min_volume: Minimum volume level (default 0.5)
            max_volume: Maximum volume level (default 1.5)
            volume_velocity_threshold: # Minimum displacement to adjust volume
        """
        self.history_length = history_length
        self.velocity_smoothing = velocity_smoothing
        self.tempo_memory = tempo_memory
        self.neutral_velocity_threshold = neutral_velocity_threshold
        self.neutral_duration_threshold = neutral_duration_threshold
        self.min_beat_interval = min_beat_interval
        self.volume_smoothing = volume_smoothing
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.volume_displacement_threshold = volume_displacement_threshold
        
        # History buffers for primary hand
        self.position_history = deque(maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length)
        self.timestamp_history = deque(maxlen=history_length)
        
        # History buffers for secondary hand
        self.secondary_position_history = deque(maxlen=history_length)
        self.secondary_velocity_history = deque(maxlen=history_length)
        self.secondary_timestamp_history = deque(maxlen=history_length)
        
        # Volume estimation buffers
        self.volume_history = deque(maxlen=volume_smoothing)
        self.last_volume_estimate = None
        
        # Beat-based displacement tracking for volume
        # Track position at last beat and maximum displacement since then
        self.beat_position = None  # Position where last beat occurred
        self.max_displacement_since_beat = 0.0  # Maximum distance from beat position
        
        # Beat tracking
        self.beat_times = deque(maxlen=tempo_memory)
        self.last_beat_time = 0.0
        self.current_beat_index = 0
        self.beats_per_measure = 4  # Default to 4/4 time
        
        # State tracking for beat detection (primary hand)
        self.last_direction = Direction.NEUTRAL
        self.last_y_position = None
        self.current_phase = MotionPhase.TRANSITION
        
        # State tracking for secondary hand
        self.secondary_last_direction = Direction.NEUTRAL
        self.secondary_current_phase = MotionPhase.TRANSITION
        
        # Neutral detection tracking
        self.low_velocity_start_time = None  # When velocity first dropped below threshold
        self.secondary_low_velocity_start_time = None

    def update_position(
        self,
        position: Tuple[float, float],
        timestamp: Optional[float] = None
    ) -> ConductingFrame:
        """
        Update with new position data and return a conducting frame.
        This method is only used for processing a single hand.
        
        Args:
            position: (x, y) position in normalized coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            ConductingFrame with analyzed conducting information
        """
        if timestamp is None:
            timestamp = time.time()
        
        return self._update_primary_hand(position, timestamp)
    
    def update_both_hands(
        self,
        primary_position: Optional[Tuple[float, float]] = None,
        secondary_position: Optional[Tuple[float, float]] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[Optional[ConductingFrame], Optional[ConductingFrame]]:
        """
        Update with positions from both primary and secondary hands.
        
        Args:
            primary_position: (x, y) position of primary hand in normalized coordinates (0-1)
            secondary_position: (x, y) position of secondary hand in normalized coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Tuple of (primary_frame, secondary_frame) where each is a ConductingFrame or None
            if that hand is not detected. The primary hand drives beat detection.
        """
        if timestamp is None:
            timestamp = time.time()
        
        primary_frame = None
        secondary_frame = None
        
        # Process primary hand (this drives beat detection)
        if primary_position is not None:
            primary_frame = self._update_primary_hand(primary_position, timestamp)
        
        # Process secondary hand (observational only, no beat detection)
        if secondary_position is not None:
            secondary_frame = self._update_secondary_hand(secondary_position, timestamp)
        
        return primary_frame, secondary_frame
    
    def _update_primary_hand(
        self,
        position: Tuple[float, float],
        timestamp: float
    ) -> ConductingFrame:
        """
        Update primary hand position and generate conducting frame with beat detection.
        
        Args:
            position: (x, y) position in normalized coordinates (0-1)
            timestamp: Timestamp of the position
            
        Returns:
            ConductingFrame with analyzed conducting information
        """
        # Store in history
        self.position_history.append(position)
        self.timestamp_history.append(timestamp)
        
        # Calculate velocity
        velocity = self._calculate_velocity(
            self.position_history,
            self.timestamp_history
        )
        self.velocity_history.append(velocity)
        
        # Determine direction
        direction = self._calculate_direction(velocity, is_primary=True)
        
        # Calculate gesture energy
        gesture_energy = self._calculate_energy(velocity)
        
        # Detect beat events (only for primary hand)
        beat_event = self._detect_beat(
            position, velocity, direction, gesture_energy, timestamp
        )
        
        # Update beat index if beat detected
        beat_index = None
        if beat_event:
            self.current_beat_index = (self.current_beat_index % self.beats_per_measure) + 1
            beat_index = self.current_beat_index
            self.beat_times.append(timestamp)
            self.last_beat_time = timestamp
            
            # Reset displacement tracking - new beat position established
            self.beat_position = position
            self.max_displacement_since_beat = 0.0
        
        # Determine motion phase
        motion_phase = self._determine_phase(velocity, gesture_energy, timestamp)
        
        # Estimate tempo
        tempo_estimate = self._estimate_tempo()
        
        # Estimate volume based on maximum displacement from last beat
        volume_estimate = self._estimate_volume(position, self.beat_position)
        
        # Create and return conducting frame
        return ConductingFrame(
            timestamp=timestamp,
            position=position,
            velocity=velocity,
            direction=direction,
            motion_phase=motion_phase,
            beat_event=beat_event,
            beat_index=beat_index,
            tempo_estimate=tempo_estimate,
            volume_estimate=volume_estimate,
            gesture_energy=gesture_energy,
            metadata={'hand': 'primary'}
        )
    
    def _update_secondary_hand(
        self,
        position: Tuple[float, float],
        timestamp: float
    ) -> ConductingFrame:
        """
        Update secondary hand position and generate conducting frame (no beat detection).
        
        Args:
            position: (x, y) position in normalized coordinates (0-1)
            timestamp: Timestamp of the position
            
        Returns:
            ConductingFrame with motion analysis but no beat events
        """
        # Store in history
        self.secondary_position_history.append(position)
        self.secondary_timestamp_history.append(timestamp)
        
        # Calculate velocity
        velocity = self._calculate_velocity(
            self.secondary_position_history,
            self.secondary_timestamp_history
        )
        self.secondary_velocity_history.append(velocity)
        
        # Determine direction
        direction = self._calculate_direction(velocity, is_primary=False)
        
        # Calculate gesture energy
        gesture_energy = self._calculate_energy(velocity)
        
        # Create and return conducting frame (no beat_event)
        return ConductingFrame(
            timestamp=timestamp,
            position=position,
            velocity=velocity,
            direction=direction,
            gesture_energy=gesture_energy,
            metadata={'hand': 'secondary'}
        )
         
    def _calculate_velocity(
        self,
        position_history: deque,
        timestamp_history: deque
    ) -> Tuple[float, float]:
        """Calculate smoothed velocity from position history."""
        if len(position_history) < 2:
            return (0.0, 0.0)
        
        # Calculate instantaneous velocity
        positions = list(position_history)
        timestamps = list(timestamp_history)
        
        velocities = []
        for i in range(1, min(self.velocity_smoothing + 1, len(positions))):
            dt = timestamps[-1] - timestamps[-i-1]
            if dt > 0:
                dx = positions[-1][0] - positions[-i-1][0]
                dy = positions[-1][1] - positions[-i-1][1]
                velocities.append((dx / dt, dy / dt))
        
        if not velocities:
            return (0.0, 0.0)
        
        # Average velocities for smoothing
        vx = sum(v[0] for v in velocities) / len(velocities)
        vy = sum(v[1] for v in velocities) / len(velocities)
        
        return (vx, vy)
    
    def _calculate_direction(
        self,
        velocity: Tuple[float, float],
        is_primary: bool = True
    ) -> Direction:
        """
        Determine vertical direction from velocity vector.
        
        Neutral is detected only when velocity stays below threshold for a sustained duration,
        representing a true pause rather than just slow motion during direction changes.
        
        Args:
            velocity: (vx, vy) velocity vector
            is_primary: Whether this is for the primary hand (uses primary tracking state)
        """
        vx, vy = velocity
        current_time = time.time()
        
        # Select the appropriate tracking variables based on hand type
        if is_primary:
            low_velocity_start = self.low_velocity_start_time
        else:
            low_velocity_start = self.secondary_low_velocity_start_time
        
        # Check if velocity is below threshold
        if abs(vy) < self.neutral_velocity_threshold:
            # Start tracking low velocity period if not already tracking
            if low_velocity_start is None:
                if is_primary:
                    self.low_velocity_start_time = current_time
                else:
                    self.secondary_low_velocity_start_time = current_time
                low_velocity_start = current_time
            
            # Check if velocity has been low for long enough
            low_velocity_duration = current_time - low_velocity_start
            if low_velocity_duration >= self.neutral_duration_threshold:
                # Sustained low velocity = true pause/neutral
                return Direction.NEUTRAL
        else:
            # Velocity is above threshold, reset the timer
            if is_primary:
                self.low_velocity_start_time = None
            else:
                self.secondary_low_velocity_start_time = None
        
        # Determine vertical direction based on velocity
        # In image coordinates, positive y is DOWN
        if vy > 0:
            return Direction.DOWN
        else:
            return Direction.UP
    
    def _calculate_energy(self, velocity: Tuple[float, float]) -> float:
        """Calculate gesture energy (normalized magnitude of motion)."""
        vx, vy = velocity
        speed = np.sqrt(vx**2 + vy**2)
        
        # Normalize to 0-1 range (assuming max velocity of ~5 units/sec)
        energy = min(speed / 5.0, 1.0)
        return energy
    
    def _detect_beat(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        direction: Direction,
        energy: float,
        timestamp: float
    ) -> Optional[BeatEvent]:
        """
        Detect beat (ictus) at the lowest point of downward motion.
        
        A beat is detected when:
        1. We transition from downward to upward motion (direction reversal)
        2. Minimum time has elapsed since last beat
        
        The neutral detection (requiring sustained low velocity) prevents spurious beats
        from slow direction changes. Only true pauses are classified as NEUTRAL.
        
        Returns:
            BeatEvent.BEAT if beat detected, None otherwise
        """
        # Check minimum time interval
        time_since_last_beat = timestamp - self.last_beat_time
        if time_since_last_beat < self.min_beat_interval:
            return None
        
        # Beat detection: detect transition from DOWN to UP (lowest point)
        beat_event = None
        
        if (self.last_direction == Direction.DOWN and 
            direction == Direction.UP):
            
            # Beat detected at the reversal point (ictus/lowest point)
            beat_event = BeatEvent.BEAT
        
        self.last_direction = direction
        return beat_event
    
    def _determine_phase(
        self,
        velocity: Tuple[float, float],
        energy: float,
        timestamp: float
    ) -> MotionPhase:
        """Determine the current motion phase within beat cycle."""
        time_since_last_beat = timestamp - self.last_beat_time
        
        # If no recent beat, we're in transition
        if time_since_last_beat > 2.0:
            return MotionPhase.TRANSITION
        
        # Get average beat interval
        avg_beat_interval = self._get_average_beat_interval()
        if avg_beat_interval is None:
            return MotionPhase.TRANSITION
        
        # Determine phase based on position in beat cycle
        phase_position = time_since_last_beat / avg_beat_interval
        
        if phase_position < 0.1:
            return MotionPhase.ON_BEAT
        elif phase_position < 0.4:
            return MotionPhase.POST_BEAT
        elif phase_position < 0.8:
            return MotionPhase.TRANSITION
        else:
            return MotionPhase.PRE_BEAT
    
    def _estimate_tempo(self) -> Optional[float]:
        """Estimate tempo in BPM from recent beat intervals."""
        if len(self.beat_times) < 2:
            return None
        
        # Calculate average interval between beats
        avg_interval = self._get_average_beat_interval()
        if avg_interval is None or avg_interval == 0:
            return None
        
        # Convert to BPM
        bpm = 60.0 / avg_interval
        
        # Clamp to reasonable range
        bpm = max(30.0, min(bpm, 300.0))
        
        return round(bpm, 1)
    
    def _get_average_beat_interval(self) -> Optional[float]:
        """Calculate average interval between recent beats."""
        if len(self.beat_times) < 2:
            return None
        
        intervals = []
        beat_times = list(self.beat_times)
        for i in range(1, len(beat_times)):
            intervals.append(beat_times[i] - beat_times[i-1])
        
        if not intervals:
            return None
        
        return sum(intervals) / len(intervals)
    
    def _estimate_volume(
        self,
        position: Tuple[float, float],
        beat_position: Optional[Tuple[float, float]],
    ) -> tuple[Optional[float], float]:
        """
        Estimate volume level based on maximum displacement from last beat position.
        Measures how far the hand has moved since the last beat was detected.
        Only considers movement during PRE_BEAT and TRANSITION phases
        (excludes ON_BEAT and POST_BEAT phases).
        
        Args:
            position: (x, y) current position
            beat_position: (x, y) position where last beat occurred (or None if no beat yet)
            
        Returns:
            volume_estimate: Estimated volume level (float) or None
        """
        # If no beat position yet, can't calculate displacement
        if beat_position is None:
            return self.last_volume_estimate
        
        # Calculate current displacement from beat position
        dx = position[0] - beat_position[0]
        dy = position[1] - beat_position[1]
        current_displacement = np.sqrt(dx**2 + dy**2)
        
        # Update maximum displacement if current is larger
        max_displacement = max(self.max_displacement_since_beat, current_displacement)
        self.max_displacement_since_beat = max_displacement
        
        # Only adjust volume if there's significant displacement
        if max_displacement < self.volume_displacement_threshold:
            # Below threshold, maintain last volume estimate
            return self.last_volume_estimate
        
        # Normalize displacement (assuming typical max displacement is 0-0.3 of screen)
        # Larger displacement = louder volume
        print("Max Displacement:", max_displacement)
        normalized_displacement = min(max_displacement / 0.3, 1.0)  # Cap at 1.0
        
        # Scale to volume range [min_volume, max_volume]
        raw_volume = normalized_displacement * (self.max_volume - self.min_volume) + self.min_volume
        
        # Add to volume history for smoothing
        self.volume_history.append(raw_volume)
        
        # Calculate smoothed volume (average of recent estimates)
        if len(self.volume_history) > 0:
            smoothed_volume = sum(self.volume_history) / len(self.volume_history)
            self.last_volume_estimate = smoothed_volume
            return round(smoothed_volume, 3)
        
        return self.last_volume_estimate
    
    def set_time_signature(self, beats_per_measure: int):
        """
        Set the time signature (beats per measure).
        
        Args:
            beats_per_measure: Number of beats per measure (typically 2, 3, or 4)
        """
        if beats_per_measure < 1:
            raise ValueError(f"beats_per_measure must be at least 1, got {beats_per_measure}")
        
        self.beats_per_measure = beats_per_measure
        self.current_beat_index = 0
        
        # Clear beat history when changing time signature
        self.beat_times.clear()
    
    def get_pattern_info(self) -> dict:
        """
        Get information about the current time signature.
        
        Returns:
            dict: Contains time_signature and beats_per_measure
        """
        return {
            'time_signature': f"{self.beats_per_measure}/4",
            'beats_per_measure': self.beats_per_measure,
        }
    
    def reset(self):
        """Reset all tracking state."""
        # Primary hand state
        self.position_history.clear()
        self.velocity_history.clear()
        self.timestamp_history.clear()
        
        # Secondary hand state
        self.secondary_position_history.clear()
        self.secondary_velocity_history.clear()
        self.secondary_timestamp_history.clear()
        
        # Beat tracking
        self.beat_times.clear()
        self.last_beat_time = 0.0
        self.current_beat_index = 0
        
        # Volume tracking
        self.volume_history.clear()
        self.last_volume_estimate = None
        self.beat_position = None
        self.max_displacement_since_beat = 0.0
        
        # Direction tracking
        self.last_direction = Direction.NEUTRAL
        self.secondary_last_direction = Direction.NEUTRAL
        
        # Phase tracking
        self.current_phase = MotionPhase.TRANSITION
        self.secondary_current_phase = MotionPhase.TRANSITION
        
        # Neutral detection
        self.low_velocity_start_time = None
        self.secondary_low_velocity_start_time = None


class FingerConductingAnalyzer(ConductingAnalyzer):
    """
    Conducting analyzer specialized for finger-based conducting.
    
    Uses fingertip position tracking for fine-grained conducting gestures.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Finger conducting settings
        self.neutral_velocity_threshold = kwargs.get('neutral_velocity_threshold', 0.25)
        self.neutral_duration_threshold = kwargs.get('neutral_duration_threshold', 0.5)


class FullBodyConductingAnalyzer(ConductingAnalyzer):
    """
    Conducting analyzer specialized for full-body conducting.
    
    Uses arm/wrist position tracking for larger conducting gestures.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Full body conducting settings
        self.neutral_velocity_threshold = kwargs.get('neutral_velocity_threshold', 0.4)
        self.neutral_duration_threshold = kwargs.get('neutral_duration_threshold', 0.5)
        # May need more smoothing due to larger motions
        self.velocity_smoothing = kwargs.get('velocity_smoothing', 5)
