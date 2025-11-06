#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finger Conducting Demo

Demonstrates using the HandTracking class with the ConductingProtocol
to perform finger-based music conducting.
"""
import argparse
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
import time
import numpy as np


def draw_text_with_background(image, text, pos, font_scale=0.8, text_color=(255, 255, 255),
                               bg_color=(0, 0, 0), bg_alpha=0.6, thickness=1, outline=True):
    """Draw text with a semi-transparent background for better readability."""
    font = cv.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)

    # Calculate background rectangle
    padding = 8
    x, y = pos
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding

    # Draw semi-transparent background
    overlay = image.copy()
    cv.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv.addWeighted(overlay, bg_alpha, image, 1 - bg_alpha, 0, image)

    # Draw text with optional outline
    if outline:
        cv.putText(image, text, pos, font, font_scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
    cv.putText(image, text, pos, font, font_scale, text_color, thickness, cv.LINE_AA)

    return image


def get_velocity_color(velocity, max_velocity=0.5):
    """Get color based on velocity magnitude (blue -> green -> yellow)."""
    if max_velocity == 0:
        return (255, 200, 150)  # Default light blue

    # Normalize velocity to 0-1
    normalized = min(velocity / max_velocity, 1.0)

    # Create gradient: Blue (slow) -> Green (medium) -> Yellow (fast)
    if normalized < 0.5:
        # Blue to Green
        ratio = normalized * 2
        b = int(255 * (1 - ratio))
        g = int(255 * ratio)
        r = 0
    else:
        # Green to Yellow
        ratio = (normalized - 0.5) * 2
        b = 0
        g = 255
        r = int(255 * ratio)

    return (b, g, r)


def calculate_tempo_stability(tempo_history, window=5):
    """Calculate tempo stability score based on recent tempo estimates."""
    if len(tempo_history) < 2:
        return 1.0  # Perfect stability if not enough data

    recent = list(tempo_history)[-window:]
    if len(recent) < 2:
        return 1.0

    # Calculate coefficient of variation (std/mean)
    mean_tempo = np.mean(recent)
    std_tempo = np.std(recent)

    if mean_tempo == 0:
        return 0.0

    cv = std_tempo / mean_tempo
    # Convert to stability score (0 = unstable, 1 = stable)
    # CV > 0.1 is considered unstable, CV < 0.02 is very stable
    stability = max(0.0, min(1.0, 1.0 - (cv / 0.1)))

    return stability


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


def draw_point_history(image, point_history, beat_position=None, beat_time=0.0,
                        velocities=None, hand_color=(255, 150, 100), is_primary=True):
    """Draw the trail of finger movement history with velocity-based colors and beat highlight."""
    current_time = time.time()
    beat_age = current_time - beat_time if beat_time > 0 else 999

    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            # Calculate radius based on age (newer points are larger)
            base_radius = 1 + int(index / 2)

            # Check if this is the beat ictus point
            is_beat_point = False
            if beat_position is not None and len(beat_position) >= 2:
                try:
                    is_beat_point = (abs(point[0] - beat_position[0]) < 5 and
                                   abs(point[1] - beat_position[1]) < 5)
                except (TypeError, IndexError):
                    is_beat_point = False

            if is_beat_point and beat_age < 0.5:
                # Pulsing bright magenta for beat ictus (fades over 0.5s)
                pulse_factor = 1.0 - (beat_age / 0.5)
                pulse_radius = int(base_radius + 8 * pulse_factor)
                alpha = int(255 * pulse_factor)

                # Draw pulsing magenta circle
                cv.circle(image, (point[0], point[1]), pulse_radius, (255, 0, 255), -1)
                cv.circle(image, (point[0], point[1]), pulse_radius + 2, (255, 100, 255), 2)
            else:
                # Use velocity-based color if available
                if velocities and index < len(velocities):
                    velocity = velocities[index]
                    color = get_velocity_color(velocity, max_velocity=0.8)
                else:
                    # Fallback to hand-specific color with fade
                    fade = index / len(point_history)
                    color = tuple(int(c * (0.5 + 0.5 * fade)) for c in hand_color)

                # Draw trail circle
                thickness = 2 if index > len(point_history) * 0.7 else 1
                cv.circle(image, (point[0], point[1]), base_radius, color, thickness)

    return image


def draw_conducting_info(image, conducting_frame: ConductingFrame, beat_display_time: float = 0.0,
                         tempo_stability: float = 1.0, hand_label: str = "PRIMARY",
                         hand_color=(100, 150, 255)):
    """Draw conducting information on the image with improved layout."""
    h, w = image.shape[:2]
    current_time = time.time()

    # Draw position indicator with hand label
    if conducting_frame.position:
        x = int(conducting_frame.position[0] * w)
        y = int(conducting_frame.position[1] * h)

        # Draw crosshair at conducting position
        color = hand_color if not conducting_frame.beat_event else (255, 0, 255)
        thickness = 3 if conducting_frame.beat_event else 2

        cv.line(image, (x - 20, y), (x + 20, y), color, thickness)
        cv.line(image, (x, y - 20), (x, y + 20), color, thickness)

        # Draw hand label above crosshair
        label_y = max(y - 35, 20)
        draw_text_with_background(image, hand_label, (x - 30, label_y),
                                 font_scale=0.5, text_color=hand_color,
                                 bg_alpha=0.7, thickness=1)

        # Draw beat indicator circle
        if conducting_frame.beat_event:
            radius = 30
            cv.circle(image, (x, y), radius, (255, 0, 255), 3)

    # Full-screen beat flash
    if beat_display_time > 0 and (current_time - beat_display_time) < 0.5:
        flash_age = current_time - beat_display_time
        flash_alpha = 0.3 * (1.0 - flash_age / 0.5)  # Fade from 0.3 to 0
        overlay = image.copy()
        cv.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
        cv.addWeighted(overlay, flash_alpha, image, 1 - flash_alpha, 0, image)

        # Draw border pulse
        border_thickness = int(10 * (1.0 - flash_age / 0.5))
        cv.rectangle(image, (0, 0), (w, h), (255, 0, 255), border_thickness)

        # Beat text display (increased to 0.5s)
        event_text = "BEAT!"
        text_size = cv.getTextSize(event_text, cv.FONT_HERSHEY_SIMPLEX, 2.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        draw_text_with_background(image, event_text, (text_x, 120),
                                 font_scale=2.5, text_color=(255, 0, 255),
                                 bg_alpha=0.8, thickness=3)

    # TOP-LEFT ZONE: Tempo and Beat info
    info_y = 30
    line_height = 35

    # Tempo with stability indicator
    if conducting_frame.tempo_estimate:
        tempo_text = f"Tempo: {conducting_frame.tempo_estimate:.1f} BPM"

        # Color-code based on stability
        if tempo_stability > 0.8:
            tempo_color = (0, 255, 0)  # Green = stable
        elif tempo_stability > 0.5:
            tempo_color = (0, 255, 255)  # Yellow = moderate
        else:
            tempo_color = (0, 100, 255)  # Red = unstable

        draw_text_with_background(image, tempo_text, (15, info_y),
                                 font_scale=0.9, text_color=tempo_color,
                                 bg_alpha=0.7)
        info_y += line_height

    # Beat info
    if conducting_frame.beat_index:
        beat_text = f"Beat: {conducting_frame.beat_index}"
        draw_text_with_background(image, beat_text, (15, info_y),
                                 font_scale=0.9, text_color=(255, 255, 255),
                                 bg_alpha=0.7)
        info_y += line_height

    # BOTTOM-LEFT ZONE: Motion details
    bottom_y = h - 120

    # Direction
    dir_str = conducting_frame.direction.name if conducting_frame.direction else "NEUTRAL"
    dir_text = f"Direction: {dir_str}"
    draw_text_with_background(image, dir_text, (15, bottom_y),
                             font_scale=0.7, text_color=(200, 200, 200),
                             bg_alpha=0.6)
    bottom_y += 30

    # Motion phase
    phase_str = conducting_frame.motion_phase.name if conducting_frame.motion_phase else "NEUTRAL"
    phase_text = f"Phase: {phase_str}"
    draw_text_with_background(image, phase_text, (15, bottom_y),
                             font_scale=0.7, text_color=(200, 200, 200),
                             bg_alpha=0.6)
    bottom_y += 30

    # Energy bar with background
    energy_value = conducting_frame.gesture_energy if conducting_frame.gesture_energy is not None else 0.0
    energy_text = f"Energy: {energy_value:.2f}"
    bar_length = int(180 * min(energy_value, 1.0))

    # Draw energy bar background
    overlay = image.copy()
    cv.rectangle(overlay, (10, bottom_y - 25), (220, bottom_y + 5), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Draw energy bar
    if bar_length > 0:
        energy_color = (0, int(255 * energy_value), 0)
        cv.rectangle(image, (15, bottom_y - 20), (15 + bar_length, bottom_y), energy_color, -1)
    cv.rectangle(image, (15, bottom_y - 20), (195, bottom_y), (255, 255, 255), 1)

    cv.putText(image, energy_text, (15, bottom_y - 25),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_pattern_guide(image, conducting_analyzer):
    """Draw the time signature guide in the top-right corner."""
    pattern_info = conducting_analyzer.get_pattern_info()

    # Position in top-right corner
    h, w = image.shape[:2]
    start_x = w - 230
    start_y = 35

    # Draw semi-transparent background
    overlay = image.copy()
    cv.rectangle(overlay, (start_x - 15, start_y - 30), (w - 10, start_y + 100),
                (20, 20, 40), -1)
    cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv.rectangle(image, (start_x - 15, start_y - 30), (w - 10, start_y + 100),
                (100, 100, 150), 2)

    # Draw time signature
    time_sig_text = f"Time: {pattern_info['time_signature']}"
    draw_text_with_background(image, time_sig_text, (start_x, start_y),
                             font_scale=0.8, text_color=(255, 255, 150),
                             bg_alpha=0, thickness=2, outline=False)

    # Draw current beat indicator with visual beat boxes
    current_beat = conducting_analyzer.current_beat_index
    total_beats = conducting_analyzer.beats_per_measure

    # Draw beat boxes
    box_y = start_y + 20
    box_size = 25
    box_spacing = 30
    start_box_x = start_x + 10

    for i in range(1, total_beats + 1):
        box_x = start_box_x + (i - 1) * box_spacing
        if i == current_beat:
            # Current beat - bright and filled
            cv.rectangle(image, (box_x, box_y), (box_x + box_size, box_y + box_size),
                        (255, 0, 255), -1)
            cv.rectangle(image, (box_x, box_y), (box_x + box_size, box_y + box_size),
                        (255, 150, 255), 2)
        else:
            # Other beats - outline only
            cv.rectangle(image, (box_x, box_y), (box_x + box_size, box_y + box_size),
                        (150, 150, 150), 2)

        # Beat number
        num_text = str(i)
        text_size = cv.getTextSize(num_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = box_x + (box_size - text_size[0]) // 2
        text_y = box_y + box_size // 2 + text_size[1] // 2
        cv.putText(image, num_text, (text_x, text_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Draw beat detection explanation
    info_y = box_y + box_size + 25
    info_text = "Beat at lowest point"
    cv.putText(image, info_text, (start_x, info_y),
              cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 200), 1, cv.LINE_AA)

    down_up_text = "(DOWN \u2192 UP)"
    cv.putText(image, down_up_text, (start_x, info_y + 18),
              cv.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv.LINE_AA)

    return image


def draw_track_selection_overlay(image, player, secondary_hand_pos=None, hovered_track_idx=None):
    """
    Draw a translucent overlay showing all tracks for volume control.
    Appears when secondary hand is in "Pointer" mode.
    
    Args:
        image: The image to draw on
        player: The DynamicMidiPlayer instance
        secondary_hand_pos: Tuple (x, y) of secondary hand position in pixels, or None
        hovered_track_idx: Index of the track currently being hovered
    
    Returns:
        Updated image
    """
    if player is None:
        return image
    
    h, w = image.shape[:2]
    _, track_count = player.get_tracks_with_notes()

    if track_count == 0:
        return image
    
    # Create a translucent overlay
    overlay = image.copy()
    alpha = 0.5  # Transparency level (reduced for better hand visibility)
    
    # Draw semi-transparent background covering the whole screen
    cv.rectangle(overlay, (0, 0), (w, h), (40, 40, 40), -1)
    
    # Calculate column dimensions
    padding = 20
    column_width = (w - padding * (track_count + 1)) // track_count
    
    # Draw title at the top
    title_text = "Track Volume Control - Pointer Mode Active"
    title_size = cv.getTextSize(title_text, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    title_x = (w - title_size[0]) // 2
    cv.putText(overlay, title_text, (title_x, 50),
              cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    
    # Draw instruction
    instruction_text = "Point at a track to adjust its volume"
    instruction_size = cv.getTextSize(instruction_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    instruction_x = (w - instruction_size[0]) // 2
    cv.putText(overlay, instruction_text, (instruction_x, 85),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)
    
    # Start position for track columns
    column_start_y = 120
    column_height = h - column_start_y - 50
    
    # Draw each track column
    for i in range(track_count):
        track_info = player.get_track_info(i)
        if track_info is None:
            continue
        
        # Calculate column position
        column_x = padding + i * (column_width + padding)
        
        # Determine column color based on hover state
        if hovered_track_idx == i:
            column_color = (100, 150, 255)  # Bright blue when hovered
            border_color = (150, 200, 255)
            border_thickness = 3
        else:
            column_color = (70, 70, 70)  # Dark gray
            border_color = (120, 120, 120)
            border_thickness = 2
        
        # Draw column background
        cv.rectangle(overlay, 
                    (column_x, column_start_y),
                    (column_x + column_width, column_start_y + column_height),
                    column_color, -1)
        cv.rectangle(overlay,
                    (column_x, column_start_y),
                    (column_x + column_width, column_start_y + column_height),
                    border_color, border_thickness)
        
        # Draw track name (wrapped if too long)
        track_name = track_info['name']
        max_chars_per_line = max(1, column_width // 10)  # Rough estimate
        
        # Split long names into multiple lines
        name_lines = []
        if len(track_name) > max_chars_per_line:
            words = track_name.split()
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= max_chars_per_line:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        name_lines.append(current_line)
                    current_line = word
            if current_line:
                name_lines.append(current_line)
        else:
            name_lines = [track_name]
        
        # Limit to 3 lines maximum
        name_lines = name_lines[:3]
        
        # Draw track name (centered)
        text_y = column_start_y + 40
        for line in name_lines:
            text_size = cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = column_x + (column_width - text_size[0]) // 2
            cv.putText(overlay, line, (text_x, text_y),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            text_y += 25
        
        # Draw volume bar (vertical) - Range: 50% to 150%
        bar_width = 50
        bar_height = column_height - 150
        bar_x = column_x + (column_width - bar_width) // 2
        bar_y = column_start_y + 120
        
        # Draw bar background
        cv.rectangle(overlay,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (30, 30, 30), -1)
        cv.rectangle(overlay,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (150, 150, 150), 2)
        
        # Draw reference lines
        # 100% line (middle)
        mid_y = bar_y + bar_height // 2
        cv.line(overlay, (bar_x, mid_y), (bar_x + bar_width, mid_y),
                (100, 100, 100), 1)
        
        # Draw volume level mapped to 50%-150% range
        track_volume = track_info['volume']
        # Map volume from [0.5, 1.5] to [0, 1] for bar display
        normalized_volume = (track_volume - 0.5) / (1.5 - 0.5)
        normalized_volume = max(0.0, min(1.0, normalized_volume))
        filled_height = int(bar_height * normalized_volume)
        
        # Choose color based on volume level
        if track_volume > 1.0:
            bar_color = (0, 200, 255)  # Orange for above 100%
        elif track_volume == 1.0:
            bar_color = (0, 255, 0)  # Green for 100%
        else:
            bar_color = (100, 200, 100)  # Light green for below 100%
        
        if filled_height > 0:
            cv.rectangle(overlay,
                        (bar_x + 2, bar_y + bar_height - filled_height),
                        (bar_x + bar_width - 2, bar_y + bar_height - 2),
                        bar_color, -1)
        
        # Draw volume percentage
        vol_text = f"{int(track_volume * 100)}%"
        vol_text_size = cv.getTextSize(vol_text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        vol_text_x = column_x + (column_width - vol_text_size[0]) // 2
        vol_text_y = bar_y + bar_height + 35
        
        # Draw text with shadow for better visibility
        cv.putText(overlay, vol_text, (vol_text_x + 2, vol_text_y + 2),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(overlay, vol_text, (vol_text_x, vol_text_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        
        # Draw min/max labels
        min_label = "50%"
        max_label = "150%"
        label_size = 0.35
        
        # Min label (bottom)
        cv.putText(overlay, min_label, (bar_x - 5, bar_y + bar_height + 15),
                  cv.FONT_HERSHEY_SIMPLEX, label_size, (150, 150, 150), 1, cv.LINE_AA)
        
        # Max label (top)
        cv.putText(overlay, max_label, (bar_x - 5, bar_y - 5),
                  cv.FONT_HERSHEY_SIMPLEX, label_size, (150, 150, 150), 1, cv.LINE_AA)
        
        # 100% label (middle reference line)
        cv.putText(overlay, "100%", (bar_x + bar_width + 5, mid_y + 4),
                  cv.FONT_HERSHEY_SIMPLEX, label_size, (150, 150, 150), 1, cv.LINE_AA)
    
    # Draw secondary hand indicator if present
    if secondary_hand_pos is not None:
        cv.circle(overlay, secondary_hand_pos, 12, (255, 100, 255), -1)
        cv.circle(overlay, secondary_hand_pos, 15, (255, 150, 255), 2)
    
    # Blend the overlay with the original image
    cv.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    return image


def get_hovered_track_in_overlay(player, secondary_hand_pos, image_shape):
    """
    Determine which track column is being hovered in the overlay.
    
    Args:
        player: The DynamicMidiPlayer instance
        secondary_hand_pos: Tuple (x, y) of secondary hand position in pixels
        image_shape: Tuple (height, width, channels) of the image
    
    Returns:
        Track index being hovered, or None if not hovering over any column
    """
    if player is None or secondary_hand_pos is None:
        return None
    
    h, w = image_shape[:2]
    hand_x, hand_y = secondary_hand_pos
    
    # Check if hand is in the track column area
    column_start_y = 120
    column_height = h - column_start_y - 50
    
    if hand_y < column_start_y or hand_y > column_start_y + column_height:
        return None
    
    # Calculate column dimensions
    _, track_count = player.get_tracks_with_notes()
    padding = 20
    column_width = (w - padding * (track_count + 1)) // track_count
    
    # Check each track column
    for i in range(track_count):
        column_x = padding + i * (column_width + padding)
        if column_x <= hand_x <= column_x + column_width:
            return i
    
    return None


def draw_volume_indicator(image, volume: float, player_running: bool = False):
    """Draw global volume level indicator in top-center."""
    h, w = image.shape[:2]

    # Position at top-center
    bar_width = 150
    bar_height = 20
    x = (w - bar_width) // 2
    y = 15

    # Draw background
    overlay = image.copy()
    cv.rectangle(overlay, (x - 10, y - 5), (x + bar_width + 10, y + bar_height + 5),
                (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Draw volume bar background
    cv.rectangle(image, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
    cv.rectangle(image, (x, y), (x + bar_width, y + bar_height), (150, 150, 150), 1)

    # Draw filled portion (volume ranges from 0.5 to 1.5, map to bar)
    normalized_vol = (volume - 0.5) / (1.5 - 0.5)
    filled_width = int(bar_width * normalized_vol)

    # Color based on volume level
    if volume > 1.2:
        vol_color = (0, 150, 255)  # Orange for high
    elif volume > 0.8:
        vol_color = (0, 255, 0)  # Green for normal
    else:
        vol_color = (150, 150, 0)  # Cyan for low

    if filled_width > 0:
        cv.rectangle(image, (x, y), (x + filled_width, y + bar_height), vol_color, -1)

    # Draw center line (100% reference)
    center_x = x + bar_width // 2
    cv.line(image, (center_x, y), (center_x, y + bar_height), (200, 200, 200), 1)

    # Draw volume text
    vol_text = f"Vol: {int(volume * 100)}%"
    text_size = cv.getTextSize(vol_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = x + (bar_width - text_size[0]) // 2
    text_y = y - 8
    cv.putText(image, vol_text, (text_x, text_y),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    # Show playback status
    if player_running:
        status_indicator = "\u25B6"  # Play symbol
        cv.putText(image, status_indicator, (x - 25, y + 16),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)

    return image


def draw_control_hints(image, show_minimal: bool = True):
    """Draw control hints at the bottom of the screen."""
    h, w = image.shape[:2]

    if show_minimal:
        # Minimal hints - just essential controls
        hints = "SPACE: Play/Pause  |  ESC: Exit  |  H: Help"
        y = h - 20

        # Draw background
        text_size = cv.getTextSize(hints, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x = (w - text_size[0]) // 2

        overlay = image.copy()
        cv.rectangle(overlay, (x - 10, y - 20), (x + text_size[0] + 10, y + 5),
                    (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv.putText(image, hints, (x, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)
    else:
        # Full help overlay
        help_lines = [
            "KEYBOARD CONTROLS",
            "",
            "SPACE   - Play/Pause music",
            "ESC     - Exit application",
            "R       - Reset conducting state",
            "2/3/4   - Change time signature",
            "H       - Switch primary hand",
            "D       - Toggle debug info",
            "",
            "HAND GESTURES",
            "Primary Hand:   Control tempo & beats",
            "Secondary Hand: Adjust volume (open)",
            "                Track control (pointer)",
        ]

        # Draw semi-transparent background
        overlay = image.copy()
        panel_w = 500
        panel_h = 380
        panel_x = (w - panel_w) // 2
        panel_y = (h - panel_h) // 2

        cv.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                    (20, 20, 40), -1)
        cv.addWeighted(overlay, 0.9, image, 0.1, 0, image)
        cv.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                    (100, 150, 255), 3)

        # Draw title
        title = "HELP - Press H to close"
        title_size = cv.getTextSize(title, cv.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        title_x = panel_x + (panel_w - title_size[0]) // 2
        cv.putText(image, title, (title_x, panel_y + 40),
                  cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 150), 2, cv.LINE_AA)

        # Draw help text
        text_y = panel_y + 80
        line_height = 25

        for line in help_lines:
            if line == "" or line.isupper():
                # Section header or blank line
                if line.isupper():
                    cv.putText(image, line, (panel_x + 30, text_y),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 2, cv.LINE_AA)
                text_y += line_height
            else:
                # Regular line
                cv.putText(image, line, (panel_x + 40, text_y),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv.LINE_AA)
                text_y += line_height

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

    # Mouse click handling state
    mouse_click_track = None

    def mouse_callback(event, x, y, flags, param):
        """Handle mouse clicks for track selection."""
        nonlocal mouse_click_track
        if event == cv.EVENT_LBUTTONDOWN:
            # Store click position to be processed in main loop
            mouse_click_track = (x, y)

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
    last_beat_time = 0.0
    beat_position = None  # Position where beat occurred

    # Track volume control state
    last_hovered_track = None
    last_pointer_y_position = None  # Track last Y position for delta calculation

    # Tempo stability tracking
    tempo_history = deque(maxlen=10)

    # FPS and timing
    frame_times = deque(maxlen=30)
    last_frame_time = time.time()
    target_fps = 30
    frame_time = 1.0 / target_fps

    # UI state
    debug_mode = False
    show_help = False
    current_volume = 1.0
    
    # Get and display pattern information
    pattern_info = conducting_analyzer.get_pattern_info()

    print("\n" + "=" * 70)
    print("  FINGER CONDUCTING DEMO")
    print("=" * 70)
    print(f"\nTime Signature: {pattern_info['time_signature']}")
    print(f"MIDI File: {args.song_path}")
    print(f"Initial BPM: {INITIAL_BPM}")
    print("\n" + "-" * 70)
    print("CONTROLS:")
    print("  SPACE   - Play/Pause music")
    print("  ESC     - Exit")
    print("  H       - Toggle help overlay / Switch primary hand")
    print("  D       - Toggle debug mode")
    print("  R       - Reset conducting state")
    print("  2/3/4   - Change time signature")
    print("\nCONDUCTING:")
    print("  Primary Hand   - Controls tempo and beats (shown in BLUE)")
    print("  Secondary Hand - Global volume (open) or track control (pointer)")
    print("                   Shown in ORANGE")
    print("\nPress H in the application for detailed help.")
    print("=" * 70 + "\n")

    # Set up window and mouse callback
    window_name = 'Finger Conducting'
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            frame_start = time.time()
            # Check for keys
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                if player is not None:
                    if not player.running:
                        player.start()
                        print("▶ Playback started")
                    elif player.is_paused():
                        player.resume()
                        print("▶ Playback resumed")
                    else:
                        player.pause()
                        print("⏸ Playback paused")
            elif key == ord('r'):  # Reset
                conducting_analyzer.reset()
                tempo_history.clear()
                print("🔄 Conducting state reset")
            elif key == ord('2'):  # Switch to 2/4 time
                conducting_analyzer.set_time_signature(2)
                print("♪ Switched to 2/4 time")
            elif key == ord('3'):  # Switch to 3/4 time
                conducting_analyzer.set_time_signature(3)
                print("♪ Switched to 3/4 time")
            elif key == ord('4'):  # Switch to 4/4 time
                conducting_analyzer.set_time_signature(4)
                print("♪ Switched to 4/4 time")
            elif key == ord('h') or key == ord('H'):  # Toggle help or switch hand
                if show_help:
                    show_help = False
                else:
                    show_help = True
            elif key == ord('d') or key == ord('D'):  # Toggle debug mode
                debug_mode = not debug_mode
                print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('p'):  # Alternative: switch primary hand with 'p'
                new_primary_hand = Handedness.LEFT if hand_tracker.primary_hand == Handedness.RIGHT else Handedness.RIGHT
                hand_tracker.set_primary_hand(new_primary_hand)
                print(f"👋 Switched primary hand to: {hand_tracker.primary_hand.value}")
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Mirror the frame (MUST be done before hand tracking)
            frame = cv.flip(frame, 1)
            # Use frame directly for display (no deep copy needed)
            display_image = frame

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
                    try:
                        beat_position = primary_hand_results.landmark_list[8]
                    except (IndexError, TypeError):
                        beat_position = None

                # Track tempo for stability calculation
                if conducting_frame.tempo_estimate and conducting_frame.tempo_estimate > 0:
                    tempo_history.append(conducting_frame.tempo_estimate)

                tempo_str = f"{conducting_frame.tempo_estimate:.1f}" if conducting_frame.tempo_estimate else "N/A"
                print(f"🎵 Beat {conducting_frame.beat_index}/{conducting_analyzer.beats_per_measure}: "
                      f"tempo {tempo_str} BPM")

                # Play the next beat when a conducting beat is detected
                if player is not None and player.running:
                    try:
                        if conducting_frame.tempo_estimate and conducting_frame.tempo_estimate > 0:
                            player.set_bpm(conducting_frame.tempo_estimate)
                        player.play_next_beat()
                    except Exception as e:
                        print(f"⚠️  Audio playback error: {e}")
            
            # Track volume control with secondary hand
            is_pointer_mode = False
            hovered_track_idx = None
            secondary_hand_pixel_pos = None
            
            if secondary_hand_results is not None and secondary_hand_results.hand_detected:
                # Check if secondary hand is in "Pointer" mode (gesture ID 2)
                is_pointer_mode = (secondary_hand_results.hand_sign_id == 2)
                
                # Get pixel position for UI interaction
                if len(secondary_hand_results.landmark_list) > 8:
                    finger_tip = secondary_hand_results.landmark_list[8]
                    secondary_hand_pixel_pos = (finger_tip[0], finger_tip[1])
            
            if secondary_conducting_frame:
                if player is not None:
                    if is_pointer_mode:
                        # Pointer mode: Adjust individual track volume based on hover
                        hovered_track_idx = get_hovered_track_in_overlay(player, secondary_hand_pixel_pos, frame.shape)
                        
                        current_y_position = secondary_conducting_frame.position[1]
                        
                        if hovered_track_idx is not None:
                            # Check if we just entered a new track
                            if last_hovered_track != hovered_track_idx:
                                # Just entered this track: Record initial position and current volume
                                # DO NOT change the volume yet!
                                track_info = player.get_track_info(hovered_track_idx)
                                last_pointer_y_position = current_y_position
                                last_hovered_track = hovered_track_idx
                                # Store the initial volume when we entered (will be used as baseline)
                                # Note: We're not changing volume here, just recording the state
                            else:
                                # Still in same track: Apply relative changes from entry point
                                if last_pointer_y_position is not None:
                                    # Calculate change in Y position from when we entered
                                    delta_y = last_pointer_y_position - current_y_position
                                    
                                    # Only apply if there's significant movement
                                    if abs(delta_y) > 0.005:  # Threshold to avoid jitter
                                        # Get current track volume
                                        track_info = player.get_track_info(hovered_track_idx)
                                        current_volume = track_info['volume']
                                        
                                        # Apply delta change (delta_y is inverted because lower y = higher in image coords)
                                        # Positive delta_y means pointer moved up (decrease y) → increase volume
                                        # Scale delta to reasonable volume change
                                        volume_change = delta_y * 2.0
                                        new_volume = current_volume + volume_change
                                        
                                        # Clamp to valid range [0.5, 1.5]
                                        new_volume = max(0.5, min(1.5, new_volume))
                                        
                                        # Apply new volume
                                        player.set_track_volume(hovered_track_idx, new_volume)
                                        
                                        # Update the recorded position to current position
                                        # This makes the next delta relative to this new position
                                        last_pointer_y_position = current_y_position
                        else:
                            # Not hovering over any track - clear the record
                            last_hovered_track = None
                            last_pointer_y_position = None
                    else:
                        # Normal mode: Adjust all tracks (global volume)
                        # Reset pointer mode tracking when exiting pointer mode
                        last_hovered_track = None
                        last_pointer_y_position = None

                        if secondary_conducting_frame.direction != Direction.NEUTRAL:
                            # Calculate volume level from hand position
                            tracked_vol_position = 1.0 - secondary_conducting_frame.position[1]
                            vol_level = tracked_vol_position * (1.5 - 0.5) + 0.5  # Scale to 0.5 - 1.5
                            player.set_volume(vol_level)
                            current_volume = vol_level

            # Handle mouse clicks for track selection
            if mouse_click_track is not None and is_pointer_mode:
                clicked_track = get_hovered_track_in_overlay(player, mouse_click_track, display_image.shape)
                if clicked_track is not None:
                    print(f"🖱️  Clicked track {clicked_track}")
                mouse_click_track = None

            # Calculate tempo stability for visualization
            tempo_stability = calculate_tempo_stability(tempo_history)

            # Define hand colors (BGR format)
            primary_color = (255, 150, 100)  # Blue (high B, medium G, low R)
            secondary_color = (0, 150, 255)  # Orange (low B, medium G, high R)

            # Draw point history trail with beat highlight
            display_image = draw_point_history(display_image, point_history,
                                               beat_position=beat_position,
                                               beat_time=last_beat_time,
                                               hand_color=primary_color,
                                               is_primary=True)
            display_image = draw_point_history(display_image, secondary_point_history,
                                               beat_position=None,
                                               beat_time=0.0,
                                               hand_color=secondary_color,
                                               is_primary=False)

            # Draw conducting visualization with tempo stability
            if conducting_frame:
                display_image = draw_conducting_info(display_image, conducting_frame, last_beat_time,
                                                    tempo_stability, "PRIMARY", primary_color)

            # Draw secondary hand indicator if present
            if secondary_conducting_frame and secondary_conducting_frame.position:
                h, w = display_image.shape[:2]
                x = int(secondary_conducting_frame.position[0] * w)
                y = int(secondary_conducting_frame.position[1] * h)

                # Draw crosshair for secondary hand
                cv.line(display_image, (x - 15, y), (x + 15, y), secondary_color, 2)
                cv.line(display_image, (x, y - 15), (x, y + 15), secondary_color, 2)

                # Draw label
                label_y = max(y - 30, 20)
                draw_text_with_background(display_image, "SECONDARY", (x - 35, label_y),
                                         font_scale=0.5, text_color=secondary_color,
                                         bg_alpha=0.7, thickness=1)

            # Draw pattern guide
            display_image = draw_pattern_guide(display_image, conducting_analyzer)

            # Draw volume indicator (always visible)
            if player is not None:
                display_image = draw_volume_indicator(display_image, current_volume,
                                                     player.running and not player.is_paused())

            # Draw track selection overlay if in pointer mode
            if is_pointer_mode:
                display_image = draw_track_selection_overlay(display_image, player, secondary_hand_pixel_pos, hovered_track_idx)

            # Draw control hints or help overlay
            if show_help:
                display_image = draw_control_hints(display_image, show_minimal=False)
            else:
                display_image = draw_control_hints(display_image, show_minimal=True)

            # Draw debug info if enabled
            if debug_mode:
                # Calculate FPS
                frame_times.append(time.time())
                if len(frame_times) > 1:
                    fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                else:
                    fps = 0

                debug_y = 150
                debug_text = [
                    f"FPS: {fps:.1f}",
                    f"Tempo Stability: {tempo_stability:.2f}",
                    f"Frame Times: {len(frame_times)}",
                ]

                for text in debug_text:
                    draw_text_with_background(display_image, text, (15, debug_y),
                                             font_scale=0.6, text_color=(255, 255, 0),
                                             bg_alpha=0.7)
                    debug_y += 25

            # Frame rate limiting
            frame_elapsed = time.time() - frame_start
            sleep_time = max(0, frame_time - frame_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Display the frame
            cv.imshow(window_name, display_image)
    
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
