import mido
import fluidsynth
import threading
import time
import os
import platform


class DynamicMidiPlayer:
    """
    A MIDI player with dynamic BPM control using FluidSynth.
    Allows real-time adjustment of playback tempo and beat-by-beat playback.
    """

    def __init__(self, soundfont_path: str, bpm: float = 120.0, volume: float = 1.0):
        """
        Initialize the MIDI player with FluidSynth.
        
        Args:
            soundfont_path: Path to a .sf2 or .sf3 soundfont file
            bpm: Initial tempo in beats per minute (default: 120)
            volume: Initial volume level (0.0 to 2.0, default: 1.0)
        """
        self.midi = None
        self.midi_path = None
        self.original_tempo = None  # microseconds per beat from MIDI file
        self.current_bpm = bpm
        self.volume = max(0.0, min(2.0, volume))  # Clamp volume between 0.0 and 2.0
        self.running = False
        self.paused = False
        
        # Beat and note organization
        self.current_beat_index = 0  # Track which beat we're on
        self.beats_per_measure = 4  # Default 4/4 time
        self.notes_queue = []  # List of (beat_time, note_events) tuples
        self.active_notes = set()  # Currently playing notes
        
        # Multiple track handling
        self.tracks = []  # List of track information: {'name': str, 'events': list, 'channel': int}
        self.track_muted = []  # List of boolean values for each track
        self.track_volumes = []  # List of volume multipliers for each track
        self.track_solo = []  # List of boolean values for solo mode
        
        os_name = platform.system()
        if os_name == "Windows":
            driver = "dsound"
        elif os_name == "Darwin":
            driver = "coreaudio"
        else:
            print(f"Incompatible OS detected: {os_name}")
        self.fs = fluidsynth.Synth()
        self.fs.start(driver=driver)
        
        if not os.path.isfile(soundfont_path):
            raise FileNotFoundError(f"Soundfont not found: {soundfont_path}")
        
        try:
            self.sfid = self.fs.sfload(soundfont_path)
        except Exception as e:
            print("Error: not a valid Soundfont file")
            print("curl -L -O https://github.com/musescore/MuseScore/raw/2.3.2/share/sound/FluidR3Mono_GM.sf3")
        
        self.sfid = self.fs.sfload(soundfont_path)
        
        # Select General MIDI program for all channels
        for channel in range(16):
            self.fs.program_select(channel, self.sfid, 0, 0)
        
        print(f"[MIDI] ✓ FluidSynth initialized with soundfont: {os.path.basename(soundfont_path)}")

    def _organize_notes_by_beats(self):
        """Organize MIDI messages into beats based on timing, preserving track information."""
        if not self.midi:
            return

        # Reset queues
        self.notes_queue = []
        self.tracks = []
        self.track_muted = []
        self.track_volumes = []
        self.track_solo = []
        
        # Calculate microseconds per beat from BPM
        microseconds_per_beat = mido.bpm2tempo(self.current_bpm)
        print("⚠️⚠️⚠️   the value is still brute forcing    ⚠️⚠️⚠️")
        seconds_per_beat = microseconds_per_beat / 2000000.0
        
        # Process each track separately
        active_track_idx = 0  # Index for tracks with notes
        for original_track_idx, track in enumerate(self.midi.tracks):
            track_name = track.name if hasattr(track, 'name') and track.name else f"Track {original_track_idx}"
            track_events_by_beat = {}  # Dictionary mapping beat_index -> list of events
            current_time = 0
            has_notes = False  # Flag to check if track has any note events
            
            # Determine the channel used by this track (default to original_track_idx % 16)
            track_channel = None
            for msg in track:
                if hasattr(msg, 'channel'):
                    track_channel = msg.channel
                    break
            if track_channel is None:
                track_channel = original_track_idx % 16
            
            # Process messages in this track
            for msg in track:
                current_time += msg.time
                beat_index = int(current_time / seconds_per_beat)
                
                # Check if this track has any note events
                if msg.type in ['note_on', 'note_off']:
                    has_notes = True
                
                # Add message to the appropriate beat (use active_track_idx for events)
                if msg.type in ['note_on', 'note_off', 'program_change', 'control_change', 'pitchwheel']:
                    if beat_index not in track_events_by_beat:
                        track_events_by_beat[beat_index] = []
                    track_events_by_beat[beat_index].append((current_time, msg, active_track_idx))
            
            # Skip tracks with no note events (metadata-only tracks)
            if not has_notes:
                print(f"[MIDI] Skipping track {original_track_idx} '{track_name}' (no note events)")
                continue
            
            # Store track information
            self.tracks.append({
                'name': track_name,
                'events_by_beat': track_events_by_beat,
                'channel': track_channel,
                'original_index': original_track_idx
            })
            self.track_muted.append(False)
            self.track_volumes.append(1.0)
            self.track_solo.append(False)
            active_track_idx += 1
        
        # Merge all tracks into a unified beat queue
        all_beat_indices = set()
        for track in self.tracks:
            all_beat_indices.update(track['events_by_beat'].keys())
        
        # Create a merged beat queue
        for beat_idx in sorted(all_beat_indices):
            beat_events = []
            for track_idx, track in enumerate(self.tracks):
                if beat_idx in track['events_by_beat']:
                    for event in track['events_by_beat'][beat_idx]:
                        beat_events.append(event)  # (time, msg, track_idx)
            
            # Sort events within the beat by time
            beat_events.sort(key=lambda x: x[0])
            beat_time = beat_idx * seconds_per_beat
            self.notes_queue.append((beat_time, beat_events))
        
        print(f"[MIDI] Organized {len(self.notes_queue)} beats across {len(self.tracks)} tracks")
        for i, track in enumerate(self.tracks):
            print(f"  Track {i}: {track['name']} (Channel {track['channel']})")

    def load_file(self, path: str) -> bool:
        """Load a MIDI file into the player. Returns True if successful."""
        if not os.path.isfile(path):
            print(f"[Error] File not found: {path}")
            return False

        try:
            midi = mido.MidiFile(path)
            self.midi = midi
            self.midi_path = path
            
            # Extract original tempo from MIDI file
            self.original_tempo = 500000  # Default: 120 BPM (500000 microseconds per beat)
            for track in midi.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        self.original_tempo = msg.tempo
                        original_bpm = mido.tempo2bpm(msg.tempo)
                        print(f"[MIDI] Original tempo: {original_bpm:.1f} BPM")
                        break
                if self.original_tempo != 500000:
                    break
            
            # Organize notes by beats
            self._organize_notes_by_beats()

            print(f"[MIDI] Loaded file: {os.path.basename(path)} "
                  f"({len(midi.tracks)} tracks, {midi.length:.2f}s)")
            return True
        except Exception as e:
            print(f"[Error] Failed to load MIDI file: {e}")
            self.midi = None
            self.midi_path = None
            return False

    def is_file_loaded(self) -> bool:
        """Check whether a MIDI file is loaded."""
        return self.midi is not None

    def set_bpm(self, bpm: float):
        """
        Set the playback tempo in beats per minute.
        Changes take effect immediately (on the next MIDI message).
        
        Args:
            bpm: Tempo in beats per minute (typical range: 40-240)
        """
        self.current_bpm = max(20.0, min(400.0, bpm))
        print(f"[Tempo] BPM set to: {self.current_bpm:.1f}")

    def get_bpm(self) -> float:
        """Get the current playback tempo in beats per minute."""
        return self.current_bpm

    def set_volume(self, volume: float):
        """
        Set the playback volume level.
        Changes take effect immediately (on the next MIDI note).
        
        Args:
            volume: Volume level (0.0 = silent, 1.0 = full volume, 2.0 = twice full volume)
        """
        self.volume = max(0.0, min(2.0, volume))
        print(f"[Volume] Volume set to: {self.volume:.2f} ({int(self.volume * 100)}%)")

    def get_volume(self) -> float:
        """Get the current volume level (0.0 to 2.0)."""
        return self.volume

    def _calculate_time_scaling(self) -> float:
        """
        Calculate how much to scale the original MIDI timing.
        Returns the factor to multiply sleep times by.
        """
        # Original BPM from MIDI file
        original_bpm = mido.tempo2bpm(self.original_tempo)
        
        # Time scaling factor: original_bpm / current_bpm
        # If current BPM is higher, we sleep less (play faster)
        # If current BPM is lower, we sleep more (play slower)
        return original_bpm / self.current_bpm

    def _play_beat_events(self, beat_events):
        """Play all notes in the given beat events, respecting track mute/solo/volume settings."""
        # Check if any track is in solo mode
        any_solo = any(self.track_solo)
        
        # Collect all note_on events to play simultaneously
        notes_to_play = []
        other_events = []
        
        # First pass: organize events by type
        for event in beat_events:
            current_time, msg, track_idx = event
            
            # Check if this track should be played
            if track_idx >= len(self.tracks):
                continue
            
            # Skip if track is muted
            if self.track_muted[track_idx]:
                continue
            
            # If any track is solo, only play solo tracks
            if any_solo and not self.track_solo[track_idx]:
                continue
            
            # Apply track-specific volume
            track_volume = self.track_volumes[track_idx]
            
            if msg.type == 'note_on' and msg.velocity > 0:
                # Apply both global volume and track volume
                combined_volume = self.volume * track_volume
                adjusted_velocity = min(127, int(msg.velocity * combined_volume))
                notes_to_play.append((msg.channel, msg.note, adjusted_velocity))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Handle note off
                if (msg.channel, msg.note) in self.active_notes:
                    other_events.append(('note_off', msg.channel, msg.note))
            elif msg.type == 'program_change':
                other_events.append(('program_change', msg.channel, msg.program))
            elif msg.type == 'control_change':
                other_events.append(('control_change', msg.channel, msg.control, msg.value))
            elif msg.type == 'pitchwheel':
                other_events.append(('pitchwheel', msg.channel, msg.pitch))
        
        # Process note_off events first
        for event in other_events:
            if event[0] == 'note_off':
                channel, note = event[1], event[2]
                self.fs.noteoff(channel, note)
                if (channel, note) in self.active_notes:
                    self.active_notes.remove((channel, note))
        
        # Then play all new notes simultaneously
        for channel, note, velocity in notes_to_play:
            self.fs.noteon(channel, note, velocity)
            self.active_notes.add((channel, note))
        
        # Finally, process other events (program changes, etc.)
        for event in other_events:
            if event[0] == 'program_change':
                self.fs.program_change(event[1], event[2])
            elif event[0] == 'control_change':
                self.fs.cc(event[1], event[2], event[3])
            elif event[0] == 'pitchwheel':
                self.fs.pitch_bend(event[1], event[2])

    def play_next_beat(self):
        """
        Play the next beat in the sequence.
        Returns True if a beat was played, False if we've reached the end.
        """
        if not self.running or self.paused:
            return False

        if self.current_beat_index >= len(self.notes_queue):
            print("[MIDI] End of piece reached")
            self.stop()
            return False

        # Play the current beat
        _, beat_events = self.notes_queue[self.current_beat_index]
        self._play_beat_events(beat_events)
        
        # Increment beat counter
        self.current_beat_index += 1
        return True

    def _all_notes_off(self):
        """Send all notes off and all sound off messages to all channels."""
        for channel in range(16):
            self.fs.cc(channel, 123, 0)  # All notes off
            self.fs.cc(channel, 120, 0)  # All sound off
        self.active_notes.clear()

    def start(self):
        """Prepare for playback but wait for beat-by-beat triggers."""
        if not self.is_file_loaded():
            print("[Error] No MIDI file loaded. Load a file before playing.")
            return

        if self.running:
            print("[MIDI] Already playing.")
            return

        self.running = True
        self.paused = False
        self.current_beat_index = 0  # Reset to beginning
        print(f"[MIDI] ▶ Ready to play at {self.current_bpm:.1f} BPM: {os.path.basename(self.midi_path)}")

    def pause(self):
        """Pause playback. Call resume() to continue."""
        if not self.running:
            print("[MIDI] Not playing.")
            return
        
        if self.paused:
            print("[MIDI] Already paused.")
            return
        
        self.paused = True
        self._all_notes_off()  # Silence any currently playing notes
        print("[MIDI] ⏸ Playback paused.")

    def resume(self):
        """Resume playback from where it was paused."""
        if not self.running:
            print("[MIDI] Not playing.")
            return
        
        if not self.paused:
            print("[MIDI] Not paused.")
            return
        
        self.paused = False
        print(f"[MIDI] ▶ Ready to resume at {self.current_bpm:.1f} BPM.")

    def is_paused(self) -> bool:
        """Check if playback is currently paused."""
        return self.paused

    def stop(self):
        """Stop playback."""
        if not self.running:
            return
        
        self.running = False
        self.paused = False
        self._all_notes_off()
        self.current_beat_index = 0
        print("[MIDI] ■ Playback stopped.")

    # Track-specific controls
    
    def get_track_count(self) -> int:
        """Get the number of tracks in the loaded MIDI file."""
        return len(self.tracks)
    
    def get_track_info(self, track_idx: int) -> dict:
        """Get information about a specific track."""
        if 0 <= track_idx < len(self.tracks):
            return {
                'name': self.tracks[track_idx]['name'],
                'channel': self.tracks[track_idx]['channel'],
                'muted': self.track_muted[track_idx],
                'volume': self.track_volumes[track_idx],
                'solo': self.track_solo[track_idx]
            }
        return None
    
    def list_tracks(self):
        """Print information about all tracks."""
        print(f"\n[MIDI] Tracks in {os.path.basename(self.midi_path)}:")
        for i in range(len(self.tracks)):
            info = self.get_track_info(i)
            status = []
            if info['muted']:
                status.append("MUTED")
            if info['solo']:
                status.append("SOLO")
            status_str = f" [{', '.join(status)}]" if status else ""
            print(f"  {i}: {info['name']} (Ch.{info['channel']}, Vol:{info['volume']:.1f}){status_str}")
    
    def mute_track(self, track_idx: int, muted: bool = True):
        """Mute or unmute a specific track."""
        if 0 <= track_idx < len(self.tracks):
            self.track_muted[track_idx] = muted
            status = "muted" if muted else "unmuted"
            print(f"[MIDI] Track {track_idx} ({self.tracks[track_idx]['name']}) {status}")
        else:
            print(f"[Error] Invalid track index: {track_idx}")
    
    def solo_track(self, track_idx: int, solo: bool = True):
        """Solo or unsolo a specific track."""
        if 0 <= track_idx < len(self.tracks):
            self.track_solo[track_idx] = solo
            status = "soloed" if solo else "unsoloed"
            print(f"[MIDI] Track {track_idx} ({self.tracks[track_idx]['name']}) {status}")
        else:
            print(f"[Error] Invalid track index: {track_idx}")
    
    def set_track_volume(self, track_idx: int, volume: float):
        """Set the volume for a specific track (0.0 to 2.0)."""
        if 0 <= track_idx < len(self.tracks):
            self.track_volumes[track_idx] = max(0.0, min(2.0, volume))
            print(f"[MIDI] Track {track_idx} ({self.tracks[track_idx]['name']}) volume set to {volume:.2f}")
        else:
            print(f"[Error] Invalid track index: {track_idx}")
    
    def reset_all_tracks(self):
        """Reset all track settings (unmute all, unsolo all, volume to 1.0)."""
        for i in range(len(self.tracks)):
            self.track_muted[i] = False
            self.track_solo[i] = False
            self.track_volumes[i] = 1.0
        print("[MIDI] All track settings reset")

    def close(self):
        """Close FluidSynth and clean up resources."""
        if self.running:
            self.stop()
        self.fs.delete()
        print("[MIDI] FluidSynth closed.")


# Example usage
if __name__ == "__main__":
    # Update this path to your soundfont location
    # SOUNDFONT_PATH = os.path.expanduser("~/FluidR3Mono_GM.sf3")
    SOUNDFONT_PATH = os.path.expanduser("../FluidR3Mono_GM.sf3")
    
    try:
        # Initialize at 120 BPM with full volume (1.0)
        player = DynamicMidiPlayer(soundfont_path=SOUNDFONT_PATH, bpm=120, volume=1.0)
        
        # Load a MIDI file
        if player.load_file("../ouverture.mid"):
            # Display track information
            player.list_tracks()
            
            player.start()
            
            # Play a few beats with all tracks
            print("\nPlaying with all tracks...")
            for _ in range(50):
                player.play_next_beat()
                time.sleep(0.5)
            
            # # Mute track 0 and play
            # if player.get_track_count() > 0:
            #     print("\nMuting track 0...")
            #     player.mute_track(0)
            #     for _ in range(4):
            #         player.play_next_beat()
            #         time.sleep(0.5)
            
            # # Solo track 1 if it exists
            # if player.get_track_count() > 1:
            #     print("\nSoloing track 1...")
            #     player.reset_all_tracks()
            #     player.solo_track(1)
            #     for _ in range(4):
            #         player.play_next_beat()
            #         time.sleep(0.5)
            
            # # Reset and adjust track volumes
            # if player.get_track_count() > 1:
            #     print("\nAdjusting track volumes...")
            #     player.reset_all_tracks()
            #     player.set_track_volume(0, 0.3)  # Track 0 at 30%
            #     player.set_track_volume(1, 1.5)  # Track 1 at 150%
            #     for _ in range(4):
            #         player.play_next_beat()
            #         time.sleep(0.5)
            
            # # Reset and play normally
            # print("\nResetting all tracks and playing normally...")
            # player.reset_all_tracks()
            # for _ in range(8):
            #     player.play_next_beat()
            #     time.sleep(0.5)
            
            # player.stop()
        
        player.close()
        
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        print("Please download a soundfont first:")
        print("  curl -L -O https://github.com/musescore/MuseScore/raw/2.3.2/share/sound/FluidR3Mono_GM.sf3")