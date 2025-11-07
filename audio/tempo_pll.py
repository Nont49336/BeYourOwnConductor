"""
Tempo Phase-Locked Loop for stable conducting tempo tracking.

This module provides a PLL-based tempo tracker that filters noisy beat
detections from CV hand tracking into stable, musically responsive BPM values.
"""
import time
from typing import Optional


class TempoPhaseLockedLoop:
    """
    Phase-locked loop for stable tempo tracking with adaptive response.
    
    Predicts when the next beat should occur based on current tempo, then
    corrects the tempo using the phase error between predicted and detected beats.
    Uses adaptive correction strength based on error magnitude and gesture confidence.
    """
    
    def __init__(self, initial_bpm: float = 120.0):
        """
        Initialize the PLL with a starting tempo.
        
        Args:
            initial_bpm: Initial tempo in beats per minute
        """
        self.current_bpm = initial_bpm
        self.beat_period = 60.0 / initial_bpm
        self.last_beat_time = None
        self.predicted_next_beat = None
        
        # PLL parameters
        self.Kp = 0.3  # Proportional gain (how much to correct on each beat)
        self.Ki = 0.1  # Integral gain (accumulates error for sustained changes)
        self.integral_error = 0.0
        
        # Adaptive thresholds
        self.dead_zone = 0.030           # for ignoring jitter/noise
        self.gentle_zone = 0.100         # proportional correction
        self.large_error_threshold = 0.200  # rapid adjustment
        
        # Resync logic ( for sustatined errors )
        self.consecutive_large_errors = 0
        self.resync_threshold = 3  # reset after n consecutive large errors
        
        # Debug stats
        self.total_beats = 0
        self.ignored_beats = 0
        self.gentle_corrections = 0
        self.rapid_corrections = 0
        self.hard_resyncs = 0
    
    def initialize(self, bpm: float):
        """
        Set initial tempo after calibration phase.
        
        Args:
            bpm: Calibrated tempo in beats per minute
        """
        self.current_bpm = bpm
        self.beat_period = 60.0 / bpm
        self.last_beat_time = None
        self.integral_error = 0.0
        self.consecutive_large_errors = 0
        print(f"[PLL] Initialized at {bpm:.1f} BPM")
    
    def update(self, beat_time: float, confidence: float = 1.0) -> float:
        """
        Update tempo based on detected beat.
        
        Args:
            beat_time: Timestamp of detected beat (seconds, from time.time())
            confidence: Gesture confidence (0.0 to 1.0)
                       Higher confidence = stronger correction
                       Use gesture energy, hand velocity, or tracking quality
        
        Returns:
            Updated BPM value (filtered and stabilized)
        """
        self.total_beats += 1
        
        # First beat
        if self.last_beat_time is None:
            self.last_beat_time = beat_time
            self.predicted_next_beat = beat_time + self.beat_period
            return self.current_bpm
        
        phase_error = beat_time - self.predicted_next_beat
        
        # If within threshold, ignore
        if abs(phase_error) < self.dead_zone:
            self.predicted_next_beat += self.beat_period
            self.consecutive_large_errors = 0
            self.ignored_beats += 1
            return self.current_bpm
        
        # Else
        if abs(phase_error) > self.large_error_threshold:
            self.consecutive_large_errors += 1
            
            if self.consecutive_large_errors >= self.resync_threshold:
                # Measure actual period from last two beats
                measured_period = beat_time - self.last_beat_time
                self.current_bpm = 60.0 / measured_period
                self.beat_period = measured_period
                self.integral_error = 0.0
                self.consecutive_large_errors = 0
                self.last_beat_time = beat_time
                self.predicted_next_beat = beat_time + self.beat_period
                self.hard_resyncs += 1
                print(f"[PLL] Hard resync to {self.current_bpm:.1f} BPM (large sustained error)")
                return self.current_bpm
        else:
            self.consecutive_large_errors = 0
        
        # Adaptive correction based on error magnitude and confidence
        if abs(phase_error) < self.gentle_zone:
            # Gentle proportional correction for small errors
            correction_strength = self.Kp * confidence * 0.5
            self.gentle_corrections += 1
        else:
            # Stronger correction for larger errors (but not resync threshold)
            correction_strength = self.Kp * confidence
            self.rapid_corrections += 1
        
        # PI controller: Proportional + Integral terms
        # Convert phase error to period error (percentage change needed)
        period_error = phase_error / self.beat_period
        
        # Update integral term (accumulates sustained error)
        self.integral_error += period_error * self.Ki * confidence
        
        # Clamp integral term to prevent windup
        self.integral_error = max(-0.2, min(0.2, self.integral_error))
        
        # Calculate total correction
        correction = (period_error * correction_strength) + self.integral_error
        
        # Update beat period
        self.beat_period *= (1.0 - correction)
        
        # Convert to BPM and clamp to reasonable conducting range
        self.current_bpm = 60.0 / self.beat_period
        self.current_bpm = max(40.0, min(240.0, self.current_bpm))
        
        # Update state for next beat
        self.last_beat_time = beat_time
        self.predicted_next_beat = beat_time + self.beat_period
        
        return self.current_bpm
    
    def get_predicted_next_beat(self) -> Optional[float]:
        """
        Get the predicted time of the next beat.
        
        Returns:
            Predicted beat timestamp, or None if not initialized
        """
        return self.predicted_next_beat
    
    def get_phase_error_ms(self, current_time: float) -> Optional[float]:
        """
        Get current phase error in milliseconds.
        Positive = ahead of beat, Negative = behind beat
        
        Args:
            current_time: Current timestamp (from time.time())
        
        Returns:
            Phase error in milliseconds, or None if not initialized
        """
        if self.predicted_next_beat is None:
            return None
        return (current_time - self.predicted_next_beat) * 1000.0
    
    def get_statistics(self) -> dict:
        """
        Get PLL performance statistics for debugging/tuning.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_beats': self.total_beats,
            'ignored_beats': self.ignored_beats,
            'gentle_corrections': self.gentle_corrections,
            'rapid_corrections': self.rapid_corrections,
            'hard_resyncs': self.hard_resyncs,
            'ignore_rate': self.ignored_beats / max(1, self.total_beats),
            'current_bpm': self.current_bpm,
            'integral_error': self.integral_error,
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_beats = 0
        self.ignored_beats = 0
        self.gentle_corrections = 0
        self.rapid_corrections = 0
        self.hard_resyncs = 0
    
    def __repr__(self) -> str:
        return (f"TempoPhaseLockedLoop(bpm={self.current_bpm:.1f}, "
                f"beats={self.total_beats}, resyncs={self.hard_resyncs})")


# Example usage and testing
if __name__ == "__main__":
    import random
    
    print("=== Testing Tempo PLL ===\n")
    
    # Simulate a conductor at 120 BPM with realistic variations
    pll = TempoPhaseLockedLoop(initial_bpm=120.0)
    pll.initialize(120.0)
    
    base_bpm = 120.0
    beat_period = 60.0 / base_bpm  # 0.5 seconds per beat
    
    print("Test 1: Stable tempo with small jitter")
    current_time = time.time()
    for beat_num in range(20):
        # Add realistic jitter (±20ms)
        jitter = random.gauss(0, 0.020)
        current_time += beat_period + jitter
        
        # Simulate varying gesture confidence
        confidence = 0.8 + random.uniform(-0.1, 0.2)
        
        filtered_bpm = pll.update(current_time, confidence)
        print(f"Beat {beat_num+1:2d}: Raw={base_bpm:.1f} BPM, "
              f"Filtered={filtered_bpm:.1f} BPM, "
              f"Jitter={jitter*1000:+.0f}ms")
    
    print(f"\nStatistics: {pll.get_statistics()}\n")
    
    # Test 2: Gradual tempo change (accelerando)
    print("\nTest 2: Gradual accelerando (120 → 140 BPM)")
    pll.reset_statistics()
    pll.initialize(120.0)
    
    current_time = time.time()
    for beat_num in range(30):
        # Gradually speed up
        progress = beat_num / 30.0
        current_bpm = 120.0 + (140.0 - 120.0) * progress
        beat_period = 60.0 / current_bpm
        
        current_time += beat_period
        filtered_bpm = pll.update(current_time, confidence=0.9)
        
        if beat_num % 5 == 0:
            print(f"Beat {beat_num+1:2d}: Target={current_bpm:.1f} BPM, "
                  f"Filtered={filtered_bpm:.1f} BPM")
    
    print(f"\nStatistics: {pll.get_statistics()}\n")
    
    # Test 3: Sudden tempo change
    print("\nTest 3: Sudden tempo change (120 → 90 BPM)")
    pll.reset_statistics()
    pll.initialize(120.0)
    
    current_time = time.time()
    for beat_num in range(15):
        # Sudden change after beat 5
        current_bpm = 120.0 if beat_num < 5 else 90.0
        beat_period = 60.0 / current_bpm
        
        current_time += beat_period
        filtered_bpm = pll.update(current_time, confidence=0.95)
        
        print(f"Beat {beat_num+1:2d}: Target={current_bpm:.1f} BPM, "
              f"Filtered={filtered_bpm:.1f} BPM")
    
    print(f"\nStatistics: {pll.get_statistics()}")
    print("\n=== Testing Complete ===")