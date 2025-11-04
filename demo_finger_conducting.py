#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finger Conducting Demo

Demonstrates using the HandTracking class with the ConductingProtocol
to perform finger-based music conducting.
"""
import argparse
import copy
import cv2 as cv
from collections import deque
import os

from vision_modules.hand_tracking import Handedness, HandTracking
from vision_modules.conducting_protocol import (
    FingerConductingAnalyzer,
    ConductingFrame,
    Direction,
    MotionPhase,
    BeatEvent
)
from audio.midiplayer import DynamicMidiPlayer


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Finger Conducting Demo')
    
    parser.add_argument("--song_path", type=str, default='ode_to_joy.mid',
                        help='Path to the MIDI file to play (default: ode_to_joy.mid)')
    parser.add_argument("--songfont_path", type=str, default='FluidR3Mono_GM.sf3',
                        help='Path to the SoundFont file (default: FluidR3Mono_GM.sf3)')
    parser.add_argument("--device", type=int, default=0,
                       help='Camera device number (default: 0)')
    parser.add_argument("--width", type=int, default=960,
                       help='Camera capture width (default: 960)')
    parser.add_argument("--height", type=int, default=540,
                       help='Camera capture height (default: 540)')
    parser.add_argument("--time_signature", type=int, default=4,
                       help='Beats per measure (default: 4 for 4/4 time)')
    parser.add_argument("--primary_hand", type=str, default=Handedness.RIGHT.value,
                       help='Primary conducting hand: "right" or "left" (default: right)')
    parser.add_argument("--num_hands", type=int, default=2,
                        help='Maximum number of hands to track (default: 2)')
    
    return parser.parse_args()


def draw_point_history(image, point_history, beat_position=None):
    """Draw the trail of finger movement history with optional beat highlight."""
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            # Calculate radius based on age (newer points are larger)
            radius = 1 + int(index / 2)
            
            # Highlight the beat position with a different color
            if beat_position and abs(point[0] - beat_position[0]) < 5 and abs(point[1] - beat_position[1]) < 5:
                # Draw bright yellow circle for beat ictus point
                cv.circle(image, (point[0], point[1]), radius + 3, (0, 255, 255), -1)
                cv.circle(image, (point[0], point[1]), radius + 3, (0, 200, 255), 2)
            else:
                # Draw circles with green color that fades
                cv.circle(image, (point[0], point[1]), radius, (152, 251, 152), 2)
    return image


def draw_conducting_info(image, conducting_frame: ConductingFrame, beat_display_time: float = 0.0):
    """Draw conducting information on the image."""
    h, w = image.shape[:2]
    
    # Draw position indicator
    if conducting_frame.position:
        x = int(conducting_frame.position[0] * w)
        y = int(conducting_frame.position[1] * h)
        
        # Draw crosshair at conducting position
        color = (0, 255, 0) if conducting_frame.beat_event else (255, 255, 255)
        thickness = 3 if conducting_frame.beat_event else 1
        
        cv.line(image, (x - 20, y), (x + 20, y), color, thickness)
        cv.line(image, (x, y - 20), (x, y + 20), color, thickness)
        
        # Draw beat indicator circle
        if conducting_frame.beat_event:
            radius = 30  # Fixed radius for beat indicator
            cv.circle(image, (x, y), radius, (0, 255, 0), 2)
    
    # Draw info panel
    info_y = 30
    line_height = 35
    
    # Tempo
    if conducting_frame.tempo_estimate:
        tempo_text = f"Tempo: {conducting_frame.tempo_estimate:.1f} BPM"
        cv.putText(image, tempo_text, (10, info_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(image, tempo_text, (10, info_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
        info_y += line_height
    
    # Beat info
    if conducting_frame.beat_index:
        beat_text = f"Beat: {conducting_frame.beat_index}"
        cv.putText(image, beat_text, (10, info_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(image, beat_text, (10, info_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
        info_y += line_height
    
    # Direction (vertical only)
    dir_text = f"Direction: {conducting_frame.direction}"
    cv.putText(image, dir_text, (10, info_y),
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(image, dir_text, (10, info_y),
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv.LINE_AA)
    info_y += line_height
    
    # Motion phase
    phase_text = f"Phase: {conducting_frame.motion_phase}"
    cv.putText(image, phase_text, (10, info_y),
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(image, phase_text, (10, info_y),
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv.LINE_AA)
    info_y += line_height
    
    # Energy bar
    energy_text = f"Energy: {conducting_frame.gesture_energy:.2f}"
    bar_length = int(200 * conducting_frame.gesture_energy)
    cv.rectangle(image, (10, info_y - 20), (10 + bar_length, info_y - 5),
                (0, 255, 0), -1)
    cv.rectangle(image, (10, info_y - 20), (210, info_y - 5), (255, 255, 255), 1)
    cv.putText(image, energy_text, (220, info_y - 5),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    info_y += line_height
    
    # Beat event display - show for 0.3 seconds after beat
    import time
    if beat_display_time > 0 and (time.time() - beat_display_time) < 0.3:
        event_text = f"BEAT!"
        cv.putText(image, event_text, (w // 2 - 80, 100),
                  cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6, cv.LINE_AA)
        cv.putText(image, event_text, (w // 2 - 80, 100),
                  cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv.LINE_AA)
    
    return image


def draw_pattern_guide(image, conducting_analyzer):
    """Draw the time signature guide in the corner."""
    pattern_info = conducting_analyzer.get_pattern_info()
    
    # Position in top-right corner
    h, w = image.shape[:2]
    start_x = w - 220
    start_y = 30
    
    # Draw background
    cv.rectangle(image, (start_x - 10, start_y - 25), (w - 10, start_y + 95),
                (0, 0, 0), -1)
    cv.rectangle(image, (start_x - 10, start_y - 25), (w - 10, start_y + 95),
                (255, 255, 255), 1)

    # Draw time signature
    time_sig_text = f"Time: {pattern_info['time_signature']}"
    cv.putText(image, time_sig_text, (start_x, start_y),
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    
    # Draw current beat indicator
    beats_text = f"Beat: {conducting_analyzer.current_beat_index}/{conducting_analyzer.beats_per_measure}"
    color = (0, 255, 0) if conducting_analyzer.current_beat_index > 0 else (200, 200, 200)
    cv.putText(image, beats_text, (start_x, start_y + 35),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv.LINE_AA)
    
    # Draw beat detection explanation
    info_text = "Beat at lowest point"
    cv.putText(image, info_text, (start_x, start_y + 60),
              cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv.LINE_AA)
    
    down_up_text = "(DOWN -> UP transition)"
    cv.putText(image, down_up_text, (start_x, start_y + 80),
              cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv.LINE_AA)
    
    return image

def load_player_files(midi_path: str, soundfont_path: str, initial_bpm: int):
    """Load MIDI file and SoundFont into the DynamicMidiPlayer."""
    # Check if files exist
    if not os.path.exists(midi_path):
        print(f"Warning: MIDI file not found: {midi_path}")
        print(f"Please make sure '{midi_path}' is in the current directory.")
        print("Continuing without audio playback...")
        return None

    if not os.path.exists(soundfont_path):
        print(f"Warning: Soundfont not found: {soundfont_path}")
        print("Continuing without audio playback...")
        return None

    # Initialize MIDI player
    try:
        player = DynamicMidiPlayer(soundfont_path=soundfont_path, bpm=initial_bpm)
        success = player.load_file(midi_path)

        if not success:
            player.close()
            raise Exception("DynamicMidiPlayer failed to load MIDI file.")

        print(f"Loaded MIDI file: {midi_path}")
        print("Press SPACE to start/pause playback")
    except Exception as e:
        print(f"Error initializing MIDI player: {e}")
        print("Continuing without audio playback...")
        return None

    return player

def main():
    """Main application loop."""
    args = get_args()
    
    # Initialize camera
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize hand tracking
    primary_hand = Handedness.from_str(args.primary_hand) if args.num_hands > 1 else None
    hand_tracker = HandTracking(
        max_num_hands=args.num_hands,
        primary_hand=primary_hand,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        history_length=16
    )
    
    # Initialize conducting analyzer with improved neutral detection
    conducting_analyzer = FingerConductingAnalyzer(
        history_length=30,
        velocity_smoothing=3,
        tempo_memory=10,
        neutral_velocity_threshold=0.25,  # Higher threshold now that we require duration
        neutral_duration_threshold=0.5    # Must be below threshold for 0.5s to be neutral
    )
    conducting_analyzer.set_time_signature(args.time_signature)

    # Initialize MIDI player
    INITIAL_BPM = 120.0
    player = load_player_files(
        midi_path=args.song_path,
        soundfont_path=args.songfont_path,
        initial_bpm=INITIAL_BPM
    )
    
    # Initialize point history for trail visualization
    history_length = 32  # Length of the trail
    point_history = deque(maxlen=history_length)
    secondary_point_history = deque(maxlen=history_length)
    
    # Beat display tracking
    import time
    last_beat_time = 0.0
    beat_position = None  # Position where beat occurred
    
    # Get and display pattern information
    pattern_info = conducting_analyzer.get_pattern_info()
    
    print("Finger Conducting Demo")
    print(f"Time Signature: {pattern_info['time_signature']}")
    print("Beat Detection: Lowest point of downward motion (ictus)")
    print("Press ESC to exit")
    print("Press 'r' to reset conducting state")
    print("Press '2', '3', or '4' to change time signature")
    print("Press SPACE to Play/Pause music")
    print("-" * 50)

    try:
        while True:
            # Check for keys
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                if player is not None:
                    if not player.running:
                        player.start()
                        print("Playback started")
                    elif player.is_paused():
                        player.resume()
                        print("Playback resumed")
                    else:
                        player.pause()
                        print("Playback paused")
            elif key == ord('r'):  # Reset
                conducting_analyzer.reset()
                print("Conducting state reset")
            elif key == ord('2'):  # Switch to 2/4 time
                conducting_analyzer.set_time_signature(2)
                print("\nSwitched to 2/4 time")
            elif key == ord('3'):  # Switch to 3/4 time
                conducting_analyzer.set_time_signature(3)
                print("\nSwitched to 3/4 time")
            elif key == ord('4'):  # Switch to 4/4 time
                conducting_analyzer.set_time_signature(4)
                print("\nSwitched to 4/4 time")
            elif key == ord('h'):  # Switch primary hand
                new_primary_hand = Handedness.LEFT if hand_tracker.primary_hand == Handedness.RIGHT else Handedness.RIGHT
                hand_tracker.set_primary_hand(new_primary_hand)
                print(f"\nSwitched primary hand to: {hand_tracker.primary_hand.value}")
            
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
            primary_hand_results, secondary_hand_results = hand_tracker.process_frame(frame)
            
            # Extract positions for both hands
            primary_position = None
            secondary_position = None
            h, w = frame.shape[:2]
            
            if primary_hand_results is not None and primary_hand_results.hand_detected:
                landmark_list = primary_hand_results.landmark_list
                if len(landmark_list) > 8:
                    finger_tip = landmark_list[8]
                    # Add to point history for trail visualization
                    point_history.append(finger_tip)
                    # Normalize position to 0-1
                    primary_position = (finger_tip[0] / w, finger_tip[1] / h)
            else:
                # No primary hand detected, add empty point
                point_history.append([0, 0])
            
            if secondary_hand_results is not None and secondary_hand_results.hand_detected:
                landmark_list = secondary_hand_results.landmark_list
                if len(landmark_list) > 8:
                    finger_tip = landmark_list[8]
                    # Add to secondary point history for trail visualization
                    secondary_point_history.append(finger_tip)
                    # Normalize position to 0-1
                    secondary_position = (finger_tip[0] / w, finger_tip[1] / h)
            else:
                # No secondary hand detected, add empty point
                secondary_point_history.append([0, 0])
            
            # Update conducting analyzer with both hands
            conducting_frame, secondary_conducting_frame = conducting_analyzer.update_both_hands(
                primary_position=primary_position,
                secondary_position=secondary_position,
                timestamp=primary_hand_results.timestamp if primary_hand_results else None
            )
            
            # Track beat events (only from primary hand)
            if conducting_frame and conducting_frame.beat_event:
                last_beat_time = time.time()
                # Get beat position from primary hand for trail highlight
                if primary_hand_results and len(primary_hand_results.landmark_list) > 8:
                    beat_position = primary_hand_results.landmark_list[8]
                print(f"Beat {conducting_frame.beat_index}/{conducting_analyzer.beats_per_measure}: "
                      f"tempo {conducting_frame.tempo_estimate} BPM")
                
                # Play the next beat when a conducting beat is detected
                if player is not None and player.running:
                    if conducting_frame.tempo_estimate:
                        player.set_bpm(conducting_frame.tempo_estimate)
                    player.play_next_beat()  # Add this method call to play the next beat
            
            # Track sound effects (only from secondary hand)
            if secondary_conducting_frame:
                # Only adjust volume if secondary hand is moving (not neutral)
                if secondary_conducting_frame.direction != Direction.NEUTRAL:
                    # Adjust volume based on secondary hand position (higher = louder)
                    tracked_vol_position = 1.0 - secondary_conducting_frame.position[1]
                    vol_level = tracked_vol_position * (1.5 - 0.5) + 0.5  # Scale to 0.5 - 1.5

                    if player is not None:
                        if secondary_conducting_frame.position:
                            player.set_volume(vol_level)
            
            # Draw point history trail with beat highlight
            display_image = draw_point_history(display_image, point_history, beat_position)
            display_image = draw_point_history(display_image, secondary_point_history)
            
            # Draw conducting visualization
            if conducting_frame:
                display_image = draw_conducting_info(display_image, conducting_frame, last_beat_time)
            
            # Draw pattern guide
            display_image = draw_pattern_guide(display_image, conducting_analyzer)
            
            # Display the frame
            cv.imshow('Finger Conducting', display_image)
    
    finally:
        # Clean up
        if player is not None:
            if player.running:
                player.stop()
            player.close()
            print("MIDI player closed")
        cap.release()
        cv.destroyAllWindows()
        hand_tracker.close()
        print("\nApplication closed")


if __name__ == '__main__':
    main()
