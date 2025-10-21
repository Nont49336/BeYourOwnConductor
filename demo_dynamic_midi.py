#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic MIDI Player Demo

Plays a MIDI file with live keyboard control for tempo adjustment.
Uses DynamicMidiPlayer with FluidSynth for real-time BPM changes.
"""

import time
import os
import keyboard  # pip install keyboard
from audio.midiplayer import DynamicMidiPlayer


def print_controls():
    print("\n" + "=" * 60)
    print("🎵 DYNAMIC MIDI PLAYER DEMO 🎵")
    print("=" * 60)
    print("Controls:")
    print("  [SPACE]    ▶️  Start / Resume playback")
    print("  [P]        ⏸️  Pause playback")
    print("  [S]        ⏹️  Stop playback")
    print("  [↑]        ⬆️  Increase tempo (+5 BPM)")
    print("  [↓]        ⬇️  Decrease tempo (-5 BPM)")
    print("  [→]        ⏩  Speed up (+10 BPM)")
    print("  [←]        ⏪  Slow down (-10 BPM)")
    print("  [R]        🔄  Reset to original tempo")
    print("  [Q]        ❌  Quit")
    print("=" * 60 + "\n")


def main():
    # Configuration
    MIDI_FILE = "ode_to_joy.mid"
    # MIDI_FILE = "HotelCalifornia.mid"
    SOUNDFONT_PATH = "FluidR3Mono_GM.sf3"
    INITIAL_BPM = 120.0
    
    print_controls()
    
    # Check if files exist
    if not os.path.exists(MIDI_FILE):
        print(f"❌ Error: MIDI file not found: {MIDI_FILE}")
        print(f"   Please make sure '{MIDI_FILE}' is in the current directory.")
        return
    
    if not os.path.exists(SOUNDFONT_PATH):
        print(f"❌ Error: Soundfont not found: {SOUNDFONT_PATH}")
        print(f"   Please download it with:")
        print(f"   curl -L -O https://github.com/musescore/MuseScore/raw/2.3.2/share/sound/FluidR3Mono_GM.sf3")
        return
    
    try:
        # Initialize player
        player = DynamicMidiPlayer(soundfont_path=SOUNDFONT_PATH, bpm=INITIAL_BPM)
        
        # Load MIDI file
        if not player.load_file(MIDI_FILE):
            print("❌ Failed to load MIDI file.")
            player.close()
            return
        
        original_bpm = INITIAL_BPM
        print(f"\n🎼 Ready to play: {MIDI_FILE}")
        print(f"🎵 Initial tempo: {INITIAL_BPM:.1f} BPM")
        print(f"⌨️  Press SPACE to start...\n")
        
        is_running = True
        last_bpm_display = time.time()
        song_ended_notified = False
        
        while is_running:
            current_time = time.time()
            
            # Check if song has ended
            if player.running == False and not song_ended_notified:
                print("\n🎵 Song finished! Press SPACE to restart or Q to quit.")
                song_ended_notified = True
            
            # Start / Resume
            if keyboard.is_pressed("space"):
                if not player.running:
                    player.start()
                    song_ended_notified = False  # Reset notification flag
                elif player.is_paused():
                    player.resume()
                time.sleep(0.3)  # Debounce
            
            # Pause
            elif keyboard.is_pressed("p"):
                if player.running and not player.is_paused():
                    player.pause()
                time.sleep(0.3)
            
            # Stop
            elif keyboard.is_pressed("s"):
                if player.running:
                    player.stop()
                time.sleep(0.3)
            
            # Tempo controls - fine adjustment (+/- 5 BPM)
            elif keyboard.is_pressed("up"):
                new_bpm = player.get_bpm() + 5
                player.set_bpm(new_bpm)
                print(f"🎵 Tempo: {new_bpm:.1f} BPM")
                time.sleep(0.2)
            
            elif keyboard.is_pressed("down"):
                new_bpm = player.get_bpm() - 5
                player.set_bpm(new_bpm)
                print(f"🎵 Tempo: {new_bpm:.1f} BPM")
                time.sleep(0.2)
            
            # Tempo controls - coarse adjustment (+/- 10 BPM)
            elif keyboard.is_pressed("right"):
                new_bpm = player.get_bpm() + 10
                player.set_bpm(new_bpm)
                print(f"🎵 Tempo: {new_bpm:.1f} BPM ⏩")
                time.sleep(0.2)
            
            elif keyboard.is_pressed("left"):
                new_bpm = player.get_bpm() - 10
                player.set_bpm(new_bpm)
                print(f"🎵 Tempo: {new_bpm:.1f} BPM ⏪")
                time.sleep(0.2)
            
            # Reset to original tempo
            elif keyboard.is_pressed("r"):
                player.set_bpm(original_bpm)
                print(f"🔄 Reset tempo to: {original_bpm:.1f} BPM")
                time.sleep(0.3)
            
            # Quit
            elif keyboard.is_pressed("q"):
                print("\n👋 Exiting demo...")
                is_running = False
            
            # Periodic status display (every 2 seconds)
            if current_time - last_bpm_display > 2.0 and player.running:
                status = "⏸️  PAUSED" if player.is_paused() else "▶️  PLAYING"
                print(f"{status} | Tempo: {player.get_bpm():.1f} BPM")
                last_bpm_display = current_time
            
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
        
        # Cleanup
        if player.running:
            player.stop()
        player.close()
        print("✅ Demo closed successfully.")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
