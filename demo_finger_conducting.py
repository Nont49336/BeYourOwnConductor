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
    parser.add_argument("--velocity_smoothing", type=int, default=4,
                        help='Smoothing factor for velocity calculation (default: 4)')
    parser.add_argument("--neutral_velocity_threshold", type=float, default=0.24,
                        help='Neutral velocity threshold for gesture volume control (default: 0.3)')
    parser.add_argument("--history_length", type=int, default=40,
                        help='Length of position history for conducting analysis (default: 40)')
    
    return parser.parse_args()


def draw_point_history(image, conducting_frame: ConductingFrame, beat_history: list, velocity_smoothing: int, history_length: int):
    """Draw the trail of finger position history with beat highlight.
    
    Args:
        image: The image to draw on
        conducting_frame: The conducting frame with position history
        beat_history: List tracking which positions in history were beats (same length as position_history)
    """
    if not conducting_frame or not conducting_frame.metadata:
        return image
    
    # only draw if position history is full
    if len(conducting_frame.metadata.get('position_history', [])) < history_length:
        return image

    # Get position history from metadata
    position_history = conducting_frame.metadata.get('position_history', [])
    
    if len(position_history) < 2:
        return image
    
    h, w = image.shape[:2]
    
    # Draw position history as circles
    for index, pos in enumerate(position_history):
        if pos[0] != 0 and pos[1] != 0:
            # De-normalize to pixel coordinates
            x = int(pos[0] * w)
            y = int(pos[1] * h)
            
            # Calculate radius based on age (newer points are larger)
            radius = 1 + int(index / 2)
            
            # Check if this position was a beat event
            # offset by velocity smoothing to align with history
            offset_index = min(len(beat_history) - 1, index + velocity_smoothing)
            is_beat = offset_index < len(beat_history) and beat_history[offset_index]
            
            if is_beat:
                # Draw bright yellow circle for beat ictus point (persists and fades)
                cv.circle(image, (x, y), radius + 3, (0, 255, 255), -1)
                cv.circle(image, (x, y), radius + 3, (0, 200, 255), 2)
            else:
                # Draw circles with green color that fades
                cv.circle(image, (x, y), radius, (152, 251, 152), 2)
    return image


def draw_smoothed_position_trail(image, conducting_frame: ConductingFrame):
    """Draw smoothed position history as connected lines."""
    if not conducting_frame or not conducting_frame.metadata:
        return image
    
    position_history = conducting_frame.metadata.get('position_history', [])
    if len(position_history) < 2:
        return image
    
    h, w = image.shape[:2]
    
    # Convert normalized positions to pixel coordinates and draw lines
    points = []
    for pos in position_history:
        if pos[0] != 0 and pos[1] != 0:
            x = int(pos[0] * w)
            y = int(pos[1] * h)
            points.append((x, y))
    
    # Draw connected lines for smoothed trail (use blue to distinguish from green circles)
    for i in range(1, len(points)):
        # Calculate alpha based on age (older = more transparent)
        alpha = i / len(points)
        thickness = 2 if i >= len(points) - 5 else 1  # Thicker for recent points
        color = (255, int(100 + 155 * alpha), 0)  # Blue to cyan gradient
        cv.line(image, points[i-1], points[i], color, thickness, cv.LINE_AA)
    
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


def draw_velocity_graph(image, conducting_frame: ConductingFrame, neutral_velocity_threshold: float = 0.25):
    """Draw velocity history graph in the top-right corner."""
    if not conducting_frame or not conducting_frame.metadata:
        return image
    
    velocity_history = conducting_frame.metadata.get('velocity_history', [])
    if len(velocity_history) < 2:
        return image
    
    h, w = image.shape[:2]
    
    # Graph dimensions and position (top-right corner)
    graph_width = 200
    graph_height = 80
    graph_x = w - graph_width - 10
    graph_y = 30
    padding = 5
    
    # Draw background
    cv.rectangle(image, (graph_x - padding, graph_y - padding), 
                (graph_x + graph_width + padding, graph_y + graph_height + padding),
                (0, 0, 0), -1)
    cv.rectangle(image, (graph_x - padding, graph_y - padding), 
                (graph_x + graph_width + padding, graph_y + graph_height + padding),
                (255, 255, 255), 1)
    
    # Draw title
    cv.putText(image, "Velocity (Y)", (graph_x, graph_y - 10),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    
    # Extract vertical velocities (vy) with negative values
    velocities = [-v[1] for v in velocity_history]  # Negative because down is positive in screen coordinates
    
    if len(velocities) == 0:
        return image
    
    # Find min/max for symmetric scaling around zero
    max_abs_velocity = max(abs(v) for v in velocities) if velocities else 1.0
    max_abs_velocity = max(max_abs_velocity, 0.5)  # Ensure a minimum scale
    
    # Draw reference lines
    center_y = graph_y + graph_height // 2  # Zero line at center
    
    # Zero line (horizontal center)
    cv.line(image, (graph_x, center_y), (graph_x + graph_width, center_y),
           (150, 150, 150), 1, cv.LINE_AA)
    
    # Positive neutral threshold line
    pos_threshold_y = center_y - int((neutral_velocity_threshold / max_abs_velocity) * (graph_height // 2))
    if 0 <= pos_threshold_y - graph_y <= graph_height:
        cv.line(image, (graph_x, pos_threshold_y), (graph_x + graph_width, pos_threshold_y),
               (100, 100, 100), 1, cv.LINE_AA)
    
    # Negative neutral threshold line
    neg_threshold_y = center_y + int((neutral_velocity_threshold / max_abs_velocity) * (graph_height // 2))
    if 0 <= neg_threshold_y - graph_y <= graph_height:
        cv.line(image, (graph_x, neg_threshold_y), (graph_x + graph_width, neg_threshold_y),
               (100, 100, 100), 1, cv.LINE_AA)
    
    # Draw velocity graph
    points = []
    for i, vel in enumerate(velocities):
        x = graph_x + int((i / (len(velocities) - 1)) * graph_width)
        # Scale velocity to graph, centered at zero
        y = center_y - int((vel / max_abs_velocity) * (graph_height // 2))
        # Clamp to graph bounds
        y = max(graph_y, min(graph_y + graph_height, y))
        points.append((x, y))
    
    # Draw lines connecting points
    for i in range(1, len(points)):
        # Color gradient from older (darker) to newer (brighter)
        alpha = i / len(points)
        color = (0, int(100 + 155 * alpha), int(100 + 155 * alpha))
        cv.line(image, points[i-1], points[i], color, 2, cv.LINE_AA)
    
    # Draw current velocity value
    current_velocity = velocities[-1]
    vel_text = f"{current_velocity:.3f}"
    cv.putText(image, vel_text, (graph_x + graph_width - 60, graph_y + graph_height + 18),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    
    return image


def draw_pattern_guide(image, conducting_analyzer):
    """Draw the time signature guide in the corner."""
    pattern_info = conducting_analyzer.get_pattern_info()
    
    # Position in top-right corner (moved down to make room for velocity graph)
    h, w = image.shape[:2]
    start_x = w - 220
    start_y = 150  # Moved down from 30 to make room for velocity graph
    
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

def draw_track_selection_overlay(image, player, secondary_hand_pos=None, hovered_track_idx=None, primary_hand=Handedness.RIGHT):
    """
    Draw a translucent overlay showing all tracks for volume control.
    Appears when secondary hand is in "Pointer" mode.
    
    Args:
        image: The image to draw on
        player: The DynamicMidiPlayer instance
        secondary_hand_pos: Tuple (x, y) of secondary hand position in pixels, or None
        hovered_track_idx: Index of the track currently being hovered
        primary_hand: Handedness enum indicating which hand is primary (default: RIGHT)
    
    Returns:
        Updated image
    """
    if player is None:
        return image

    h, w = image.shape[:2]
    tracks_w_notes, track_count = player.get_tracks_with_notes()

    if track_count == 0:
        return image

    # Calculate UI bounds (60% of screen opposite to primary hand)
    ui_width = int(w * 0.6)
    if primary_hand == Handedness.RIGHT:
        # Primary hand is right, so UI spans left 60%
        ui_start_x = 0
        ui_end_x = ui_width
    else:
        # Primary hand is left, so UI spans right 60%
        ui_start_x = w - ui_width
        ui_end_x = w

    # Create a translucent overlay
    overlay = image.copy()
    alpha = 0.4  # Transparency level (0.0 = fully transparent, 1.0 = opaque)

    # Draw semi-transparent background covering only the UI area
    cv.rectangle(overlay, (ui_start_x, 0), (ui_end_x, h), (40, 40, 40), -1)

    # Calculate column dimensions within the UI width
    padding = 20
    column_width = (ui_width - padding * (track_count + 1)) // track_count

    # Draw title at the top (centered within UI area)
    title_text = "Track Volume Control"
    title_size = cv.getTextSize(title_text, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    title_x = ui_start_x + (ui_width - title_size[0]) // 2
    cv.putText(overlay, title_text, (title_x, 50),
              cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    # Draw instruction (centered within UI area)
    instruction_text = "Point at a track to adjust its volume"
    instruction_size = cv.getTextSize(instruction_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    instruction_x = ui_start_x + (ui_width - instruction_size[0]) // 2
    cv.putText(overlay, instruction_text, (instruction_x, 85),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)

    # Start position for track columns
    column_start_y = 120
    column_height = h - column_start_y - 50

    # Draw each track column
    for col_idx, track_idx in enumerate(tracks_w_notes):
        track_info = player.get_track_info(track_idx)
        if track_info is None:
            continue

        # Calculate column position within UI area (use col_idx for positioning, not track_idx)
        column_x = ui_start_x + padding + col_idx * (column_width + padding)

        # Determine column color based on hover state
        if hovered_track_idx == track_idx:
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
        track_name = track_info['label']
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

def get_hovered_track_in_overlay(player, secondary_hand_pos, image_shape, primary_hand):
    """
    Determine which track column is being hovered in the overlay.
    
    Args:
        player: The DynamicMidiPlayer instance
        secondary_hand_pos: Tuple (x, y) of secondary hand position in pixels
        image_shape: Tuple (height, width, channels) of the image
        primary_hand: Handedness enum indicating which hand is primary
    
    Returns:
        Track index being hovered, or None if not hovering over any column
    """
    if player is None or secondary_hand_pos is None:
        return None

    h, w = image_shape[:2]
    hand_x, hand_y = secondary_hand_pos

    # Calculate UI bounds (60% of screen opposite to primary hand)
    ui_width = int(w * 0.6)
    if primary_hand == Handedness.RIGHT:
        # Primary hand is right, so UI spans left 60%
        ui_start_x = 0
        ui_end_x = ui_width
    else:
        # Primary hand is left, so UI spans right 60%
        ui_start_x = w - ui_width
        ui_end_x = w

    # Check if hand is in the UI area horizontally
    if hand_x < ui_start_x or hand_x > ui_end_x:
        return None

    # Check if hand is in the track column area
    column_start_y = 120
    column_height = h - column_start_y - 50

    if hand_y < column_start_y or hand_y > column_start_y + column_height:
        return None

    # Calculate column dimensions within the UI width
    tracks_w_notes, track_count = player.get_tracks_with_notes()
    padding = 20
    column_width = (ui_width - padding * (track_count + 1)) // track_count

    # Check each track column
    for col_idx, track_idx in enumerate(tracks_w_notes):
        column_x = ui_start_x + padding + col_idx * (column_width + padding)
        if column_x <= hand_x <= column_x + column_width:
            return track_idx

    return None

def draw_countdown_overlay(image, countdown_beats_detected, countdown_required, calibrated_bpm=None):
    """Draw countdown overlay showing beat calibration progress."""
    h, w = image.shape[:2]
    
    # Smaller overlay in top-right corner (below velocity graph and above pattern guide)
    overlay_width = 250
    overlay_height = 140
    overlay_x = w - overlay_width - 20
    overlay_y = 120  # Position below velocity graph
    padding = 10
    
    # Create semi-transparent overlay
    overlay = image.copy()
    cv.rectangle(overlay, (overlay_x - padding, overlay_y - padding), 
                (overlay_x + overlay_width + padding, overlay_y + overlay_height + padding),
                (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.5, image, 0.5, 0, image)  # More transparent (50% instead of 80%)
    
    # Draw subtle border
    cv.rectangle(image, (overlay_x - padding, overlay_y - padding), 
                (overlay_x + overlay_width + padding, overlay_y + overlay_height + padding),
                (0, 200, 200), 1)
    
    # Title - smaller font
    title_text = "Calibrating"
    cv.putText(image, title_text, (overlay_x, overlay_y + 20),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1, cv.LINE_AA)
    
    # Beat counter - smaller and more compact
    counter_text = f"{countdown_beats_detected} / {countdown_required}"
    cv.putText(image, counter_text, (overlay_x, overlay_y + 60),
              cv.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2, cv.LINE_AA)
    
    # Show calibrated BPM if available - smaller font
    if calibrated_bpm is not None:
        bpm_text = f"{calibrated_bpm:.1f} BPM"
        cv.putText(image, bpm_text, (overlay_x, overlay_y + 90),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
    
    # Draw beat indicators (smaller circles)
    indicator_y = overlay_y + overlay_height - 20
    total_width = countdown_required * 30
    start_x = overlay_x + (overlay_width - total_width) // 2
    
    for i in range(countdown_required):
        x = start_x + i * 30 + 15
        if i < countdown_beats_detected:
            # Filled circle for detected beat - smaller
            cv.circle(image, (x, indicator_y), 8, (0, 255, 0), -1)
            cv.circle(image, (x, indicator_y), 8, (0, 200, 0), 1)
        else:
            # Empty circle for pending beat - smaller
            cv.circle(image, (x, indicator_y), 8, (80, 80, 80), 1)
    
    return image

def draw_controls_overlay(image, waiting_for_conducting=False):
    """Draw controls overlay at the bottom of the screen."""
    h, w = image.shape[:2]
    
    # Create semi-transparent overlay background at the bottom
    overlay = image.copy()
    overlay_height = 180
    overlay_y_start = h - overlay_height
    cv.rectangle(overlay, (0, overlay_y_start), (w, h), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Instructions at top of overlay
    if waiting_for_conducting:
        instruction_text = "Ready! Start conducting to begin music..."
        instruction_color = (0, 255, 255)  # Yellow to indicate waiting
    else:
        instruction_text = "Press SPACE to start countdown and calibrate tempo!"
        instruction_color = (0, 255, 0)
    
    instruction_size = cv.getTextSize(instruction_text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    instruction_x = (w - instruction_size[0]) // 2
    cv.putText(image, instruction_text, (instruction_x, overlay_y_start + 30),
              cv.FONT_HERSHEY_SIMPLEX, 0.7, instruction_color, 2, cv.LINE_AA)
    
    # Title
    title_text = "CONTROLS"
    title_size = cv.getTextSize(title_text, cv.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    title_x = (w - title_size[0]) // 2
    cv.putText(image, title_text, (title_x, overlay_y_start + 65),
              cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv.LINE_AA)
    
    # Controls list
    controls = [
        "SPACE - Start Countdown/Pause",
        "2/3/4 - Change Time Signature",
        "G - Toggle Pattern Guide",
        "H - Switch Primary Hand",
        "R - Reset Conducting State",
        "ESC - Exit"
    ]
    
    # Draw controls in two columns
    start_y = overlay_y_start + 100
    line_height = 25
    col1_x = 50
    col2_x = w // 2 + 50
    
    for i, control in enumerate(controls):
        if i < 3:
            x = col1_x
            y = start_y + i * line_height
        else:
            x = col2_x
            y = start_y + (i - 3) * line_height
        
        cv.putText(image, control, (x, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)
    
    return image

def draw_conducting_path(image, time_signature: int, primary_hand: Handedness):
    """Draw conducting path based on time signature."""
    overlay = cv.imread(f'resources/{time_signature}_time_signature_guide.png', cv.IMREAD_UNCHANGED)
    if overlay is None:
        return image

    h, w = image.shape[:2]
    overlay_height = int(h * 0.7) # 70% of window height
    overlay_width = int(overlay.shape[1] * (overlay_height / overlay.shape[0]))
    overlay = cv.resize(overlay, (overlay_width, overlay_height))

    if primary_hand == Handedness.LEFT:
        # Flip overlay for left hand
        overlay = cv.flip(overlay, 1)

    # Position of overlay
    y1, y2 = (h - overlay_height) // 2, (h + overlay_height) // 2
    if primary_hand == Handedness.RIGHT:
        x1, x2 = w - overlay_width - 50, w - 50
    else:
        x1, x2 = 50, 50 + overlay_width
    
    # Extract BGR and alpha channels
    overlay_bgr = overlay[:, :, :3]
    overlay_alpha = overlay[:, :, 3:4] / 255.0  # Normalize to 0-1
    
    # Get the region of interest from the background
    background_region = image[y1:y2, x1:x2, :3]
    
    # Blend using alpha channel
    blended = overlay_bgr * overlay_alpha + background_region * (1 - overlay_alpha)
    
    # Place the blended result back onto the image
    image[y1:y2, x1:x2, :3] = blended.astype('uint8')

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
    # try:
    player = DynamicMidiPlayer(soundfont_path=soundfont_path, bpm=initial_bpm)
    success = player.load_file(midi_path)

    if not success:
        player.close()
        raise Exception("DynamicMidiPlayer failed to load MIDI file.")

    print(f"Loaded MIDI file: {midi_path}")
    print("Press SPACE to start/pause playback")
    # except Exception as e:
    #     print(f"Error initializing MIDI player: {e}")
    #     print("Continuing without audio playback...")
    #     return None

    return player

def main():
    """Main application loop."""
    args = get_args()
    
    # Initialize camera
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize hand tracking
    hand_tracker = HandTracking(
        max_num_hands=args.num_hands,
        primary_hand=Handedness.from_str(args.primary_hand),
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        history_length=16
    )
    
    # Initialize conducting analyzer with improved neutral detection
    conducting_analyzer = FingerConductingAnalyzer(
        history_length=args.history_length,
        velocity_smoothing=args.velocity_smoothing,
        tempo_memory=2,
        neutral_velocity_threshold=args.neutral_velocity_threshold,
        neutral_duration_threshold=0.5,   # Must be below threshold for 0.5s to be neutral
        volume_smoothing=10,              # Smooth volume over frames to prevent rapid changes
        min_volume=0.5,                   # Minimum volume level
        max_volume=2.0,                   # Maximum volume level
        volume_displacement_threshold=0.02,
    )
    conducting_analyzer.set_time_signature(args.time_signature)

    # Initialize MIDI player
    INITIAL_BPM = 120.0
    player = load_player_files(
        midi_path=args.song_path,
        soundfont_path=args.songfont_path,
        initial_bpm=INITIAL_BPM
    )
    
    # Beat display tracking
    import time
    last_beat_time = 0.0
    
    # Beat history tracking for visualization (tracks which positions were beats)
    from collections import deque
    primary_beat_history = deque(maxlen=40)  # Match history_length from analyzer
    secondary_beat_history = deque(maxlen=40)

    # Track volume control state
    last_hovered_track = None
    last_pointer_y_position = None  # Track last Y position for delta calculation
    
    # Get and display pattern information
    pattern_info = conducting_analyzer.get_pattern_info()
    guide_enabled = False
    
    # Waiting for first conducting gesture to start playback
    waiting_for_conducting = False
    
    # Beat countdown state
    countdown_active = False
    countdown_beats_detected = 0
    countdown_required = conducting_analyzer.beats_per_measure
    countdown_beat_times = []
    calibrated_bpm = None
    
    print("Finger Conducting Demo")
    print(f"Time Signature: {pattern_info['time_signature']}")
    print("Beat Detection: Lowest point of downward motion (ictus)")
    print("Press ESC to exit")
    print("Press 'r' to reset conducting state")
    print("Press 'h' to switch primary hand")
    print("Press '2', '3', or '4' to change time signature")
    print(f"Press SPACE to start countdown ({countdown_required} beats to calibrate tempo)")
    print("\nConducting:")
    print("  Primary Hand   - Controls tempo and beats")
    print("  Secondary Hand - Normal: Adjust global volume (all tracks)")
    print("                   Pointer Gesture: Show track overlay + adjust individual tracks")
    print("                   Move hand UP/DOWN to increase/decrease volume")
    print("-" * 70)

    try:
        while True:
            # Check for keys
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                if player is not None:
                    if not player.running:
                        # Arm playback - start countdown for tempo calibration
                        waiting_for_conducting = True
                        countdown_active = True
                        countdown_beats_detected = 0
                        countdown_required = conducting_analyzer.beats_per_measure
                        countdown_beat_times = []
                        calibrated_bpm = None
                        print(f"Countdown started - conduct {countdown_required} beats to calibrate tempo")
                    elif player.is_paused():
                        player.resume()
                        print("Playback resumed")
                    else:
                        player.pause()
                        print("Playback paused")
            elif key == ord('r'):  # Reset
                conducting_analyzer.reset()
                countdown_active = False
                waiting_for_conducting = False
                print("Conducting state reset")
            elif key == ord('2'):  # Switch to 2/4 time
                conducting_analyzer.set_time_signature(2)
                countdown_required = 2
                print("\nSwitched to 2/4 time")
            elif key == ord('3'):  # Switch to 3/4 time
                conducting_analyzer.set_time_signature(3)
                countdown_required = 3
                print("\nSwitched to 3/4 time")
            elif key == ord('4'):  # Switch to 4/4 time
                conducting_analyzer.set_time_signature(4)
                countdown_required = 4
                print("\nSwitched to 4/4 time")
            elif key == ord('g'): # Toggle guide
                guide_enabled = not guide_enabled
                print(f"\nPattern guide {'enabled' if guide_enabled else 'disabled'}")
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
            secondary_hand_pixel_pos = None
            is_pointer_mode = False
            h, w = frame.shape[:2]
            
            if primary_hand_results is not None and primary_hand_results.hand_detected:
                landmark_list = primary_hand_results.landmark_list
                if len(landmark_list) > 8:
                    finger_tip = landmark_list[8]
                    # Normalize position to 0-1
                    primary_position = (finger_tip[0] / w, finger_tip[1] / h)
            
            if secondary_hand_results is not None and secondary_hand_results.hand_detected:
                landmark_list = secondary_hand_results.landmark_list
                if len(landmark_list) > 8:
                    finger_tip = landmark_list[8]
                    # Normalize position to 0-1
                    secondary_position = (finger_tip[0] / w, finger_tip[1] / h)
                    # Get pixel position for UI interaction
                    secondary_hand_pixel_pos = (finger_tip[0], finger_tip[1])

                # Check if secondary hand is in "Pointer" mode (gesture ID 2)
                is_pointer_mode = (secondary_hand_results.hand_sign_id == 2)
            
            # Update conducting analyzer with both hands
            conducting_frame, secondary_conducting_frame = conducting_analyzer.update_both_hands(
                primary_position=primary_position,
                secondary_position=secondary_position,
                timestamp=primary_hand_results.timestamp if primary_hand_results else None
            )
            
            # Track beat events for visualization (primary hand)
            if conducting_frame:
                primary_beat_history.append(conducting_frame.beat_event == BeatEvent.BEAT)
            else:
                primary_beat_history.append(False)
            
            # Track beat events for visualization (secondary hand)
            if secondary_conducting_frame:
                secondary_beat_history.append(secondary_conducting_frame.beat_event == BeatEvent.BEAT)
            else:
                secondary_beat_history.append(False)
            
            # Handle countdown for tempo calibration
            if countdown_active and conducting_frame:
                # Check for beat events during countdown
                if conducting_frame.beat_event:
                    countdown_beats_detected += 1
                    countdown_beat_times.append(time.time())
                    print(f"Countdown beat {countdown_beats_detected}/{countdown_required}")
                    
                    # Calculate calibrated BPM from countdown beats
                    if len(countdown_beat_times) >= 2:
                        intervals = []
                        for i in range(1, len(countdown_beat_times)):
                            intervals.append(countdown_beat_times[i] - countdown_beat_times[i-1])
                        avg_interval = sum(intervals) / len(intervals)
                        calibrated_bpm = 60.0 / avg_interval
                    
                    # Check if countdown complete
                    if countdown_beats_detected >= countdown_required:
                        countdown_active = False
                        # Start playback with calibrated tempo
                        if player is not None and calibrated_bpm is not None:
                            player.set_bpm(calibrated_bpm)
                            player.start()
                            waiting_for_conducting = False
                            print(f"Playback started with calibrated tempo: {calibrated_bpm:.1f} BPM")
                        else:
                            # Fallback: just start playback
                            if player is not None:
                                player.start()
                                waiting_for_conducting = False
                                print("Playback started")
            
            # Legacy: Check if waiting for conducting gesture to start playback (no countdown)
            elif waiting_for_conducting and not countdown_active and conducting_frame:
                if conducting_frame.direction != Direction.NEUTRAL:
                    # First non-neutral conducting gesture detected, start playback
                    if player is not None:
                        player.start()
                        waiting_for_conducting = False
                        print("Playback started (conducting gesture detected)")
            
            # Track beat events (only from primary hand)
            if conducting_frame and conducting_frame.beat_event:
                last_beat_time = time.time()
                print(f"Beat {conducting_frame.beat_index}/{conducting_analyzer.beats_per_measure}: "
                      f"tempo {conducting_frame.tempo_estimate} BPM")
                
                # Play the next beat when a conducting beat is detected (only if not neutral)
                if player is not None and player.running and conducting_frame.tempo_estimate:
                    if conducting_frame.direction != Direction.NEUTRAL:
                            player.set_bpm(conducting_frame.tempo_estimate)
                    player.play_next_beat()  # Add this method call to play the next beat
            
            # Pause/resume music based on conducting state
            if player is not None and player.running:
                if conducting_frame:
                    if conducting_frame.direction == Direction.NEUTRAL:
                        # Pause music when conductor stops moving
                        if not player.is_paused():
                            player.pause()
                            print("Music paused (neutral state)")
                    else:
                        # Resume music when conductor starts moving
                        if player.is_paused():
                            player.resume()
                            print("Music resumed")
            
            # Adjust volume
            # Based on secondary hand's vertical position
            # Track sound effects (only from secondary hand)
            hovered_track_idx = None
            if secondary_conducting_frame and player is not None:
                if player is not None:
                    if is_pointer_mode:
                        # Pointer mode: Adjust individual track volume based on hover
                        hovered_track_idx = get_hovered_track_in_overlay(player, secondary_hand_pixel_pos, frame.shape, hand_tracker.primary_hand)

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

            # Draw point history trail with beat highlight (green circles for smoothed positions, yellow for beats)
            if conducting_frame:
                display_image = draw_point_history(display_image, conducting_frame, list(primary_beat_history), args.velocity_smoothing, args.history_length)
            if secondary_conducting_frame:
                display_image = draw_point_history(display_image, secondary_conducting_frame, list(secondary_beat_history), args.velocity_smoothing, args.history_length)

            # Draw smoothed position trails (blue lines) over raw detection (green circles)
            if conducting_frame:
                display_image = draw_smoothed_position_trail(display_image, conducting_frame)
            if secondary_conducting_frame:
                display_image = draw_smoothed_position_trail(display_image, secondary_conducting_frame)
            
            # Draw conducting visualization
            if conducting_frame:
                display_image = draw_conducting_info(display_image, conducting_frame, last_beat_time)
            
            # Draw velocity graph
            if conducting_frame:
                display_image = draw_velocity_graph(display_image, conducting_frame, args.neutral_velocity_threshold)
            
            # Draw pattern guide
            display_image = draw_pattern_guide(display_image, conducting_analyzer)
            if guide_enabled:
                display_image = draw_conducting_path(display_image, conducting_analyzer.beats_per_measure, hand_tracker.primary_hand)
            
            # Draw countdown overlay if active
            if countdown_active:
                display_image = draw_countdown_overlay(display_image, countdown_beats_detected, 
                                                      countdown_required, calibrated_bpm)
            
            # Draw controls overlay if music hasn't started yet or waiting for conducting (but not during countdown)
            elif not player.running or waiting_for_conducting:
                display_image = draw_controls_overlay(display_image, waiting_for_conducting)
            
            # Draw track selection overlay if in pointer mode
            if secondary_conducting_frame and is_pointer_mode:
                display_image = draw_track_selection_overlay(display_image, player, secondary_hand_pixel_pos, hovered_track_idx, hand_tracker.primary_hand)
            
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
