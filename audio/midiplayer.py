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
        self.channel_queues = {}  # Dictionary mapping channel -> list of beats with events
        self.max_beats = 0  # Total number of beats in the song
        self.active_notes = set()  # Currently playing notes
        self.beat_timer = None  # Timer for stopping notes after beat duration
        
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
        """Organize MIDI messages into beats based on tick timing, with separate queues per channel."""
        if not self.midi:
            return

        # Reset queues and track data
        self.channel_queues = {}  # Dictionary: channel -> {beat_index -> [events]}
        self.max_beats = 0
        self.tracks = []
        self.track_muted = []
        self.track_volumes = []
        self.track_solo = []
        
        # Get ticks per beat from MIDI file
        ticks_per_beat = self.midi.ticks_per_beat
        print(f"[MIDI] Organizing beats: {self.current_bpm:.1f} BPM, {ticks_per_beat} ticks/beat")
        
        # Process each track separately
        for track_idx, track in enumerate(self.midi.tracks):
            track_name = track.name if hasattr(track, 'name') and track.name else f"Track {track_idx}"
            track_events_by_beat = {}  # Dictionary mapping beat_index -> list of events
            current_time_ticks = 0  # Accumulated time in MIDI ticks
            
            # Determine the channel used by this track (default to track_idx % 16)
            track_channel = None
            for msg in track:
                if hasattr(msg, 'channel'):
                    track_channel = msg.channel
                    break
            if track_channel is None:
                track_channel = track_idx % 16
            
            # Process messages in this track
            for msg in track:
                current_time_ticks += msg.time
                # Calculate beat index based on ticks
                beat_index = current_time_ticks // ticks_per_beat  # Integer division
                
                # Update max beats
                if beat_index > self.max_beats:
                    self.max_beats = beat_index
                
                # Add message to the appropriate beat (use track_idx for events)
                if msg.type in ['note_on', 'note_off', 'program_change', 'control_change', 'pitchwheel']:
                    if beat_index not in track_events_by_beat:
                        track_events_by_beat[beat_index] = []
                    # Store: (tick_time, message, track_idx)
                    track_events_by_beat[beat_index].append((current_time_ticks, msg, track_idx))
                    
                    # Also add to channel queue
                    if track_channel not in self.channel_queues:
                        self.channel_queues[track_channel] = {}
                    if beat_index not in self.channel_queues[track_channel]:
                        self.channel_queues[track_channel][beat_index] = []
                    self.channel_queues[track_channel][beat_index].append((current_time_ticks, msg, track_idx))
            
            # Store track information for all tracks (including metadata-only tracks)
            self.tracks.append({
                'name': track_name,
                'events_by_beat': track_events_by_beat,
                'channel': track_channel,
                'original_index': track_idx
            })
            self.track_muted.append(False)
            self.track_volumes.append(1.0)
            self.track_solo.append(False)
        
        # Sort events within each beat for each channel by tick time
        for channel in self.channel_queues:
            for beat_idx in self.channel_queues[channel]:
                self.channel_queues[channel][beat_idx].sort(key=lambda x: x[0])
        
        print(f"[MIDI] Organized {self.max_beats + 1} beats across {len(self.channel_queues)} channels")
        print(f"[MIDI] Channels in use: {sorted(self.channel_queues.keys())}")
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
        """Play events in the beat asynchronously, respecting their timing and track mute/solo/volume settings."""
        if not beat_events:
            return
        
        # Check if any track is in solo mode
        any_solo = any(self.track_solo)
        
        # Get ticks per beat from MIDI file
        ticks_per_beat = self.midi.ticks_per_beat
        
        # Calculate seconds per tick based on current BPM
        microseconds_per_beat = mido.bpm2tempo(self.current_bpm)
        seconds_per_beat = microseconds_per_beat / 1000000.0
        seconds_per_tick = seconds_per_beat / ticks_per_beat
        
        # Group events by their tick time
        events_by_tick = {}
        for event in beat_events:
            tick_time, msg, track_idx = event
            
            # Check if this track should be played
            if track_idx >= len(self.tracks):
                continue
            
            # Skip if track is muted
            if self.track_muted[track_idx]:
                continue
            
            # If any track is solo, only play solo tracks
            if any_solo and not self.track_solo[track_idx]:
                continue
            
            if tick_time not in events_by_tick:
                events_by_tick[tick_time] = []
            events_by_tick[tick_time].append((msg, track_idx))
        
        # Get the first tick time as reference (start of beat)
        if not events_by_tick:
            return
        
        sorted_ticks = sorted(events_by_tick.keys())
        first_tick = sorted_ticks[0]
        
        # Function to play events at a specific time offset
        def play_events_at_tick(tick_time, events):
            for msg, track_idx in events:
                track_volume = self.track_volumes[track_idx]
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Apply both global volume and track volume
                    combined_volume = self.volume * track_volume
                    adjusted_velocity = min(127, int(msg.velocity * combined_volume))
                    self.fs.noteon(msg.channel, msg.note, adjusted_velocity)
                    self.active_notes.add((msg.channel, msg.note))
                    
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Handle note off
                    self.fs.noteoff(msg.channel, msg.note)
                    if (msg.channel, msg.note) in self.active_notes:
                        self.active_notes.remove((msg.channel, msg.note))
                        
                elif msg.type == 'program_change':
                    self.fs.program_change(msg.channel, msg.program)
                    
                elif msg.type == 'control_change':
                    self.fs.cc(msg.channel, msg.control, msg.value)
                    
                elif msg.type == 'pitchwheel':
                    self.fs.pitch_bend(msg.channel, msg.pitch)
        
        # Schedule events at each tick time using timers (non-blocking)
        for tick_time in sorted_ticks:
            # Calculate delay from the start of the beat
            delay = (tick_time - first_tick) * seconds_per_tick
            
            events = events_by_tick[tick_time]
            
            if delay == 0:
                # Play immediately
                play_events_at_tick(tick_time, events)
            else:
                # Schedule to play after delay
                timer = threading.Timer(delay, play_events_at_tick, args=(tick_time, events))
                timer.start()

    def play_next_beat(self):
        """
        Play the next beat in the sequence asynchronously.
        Plays events from all channels simultaneously for the current beat.
        Returns True if a beat was played, False if we've reached the end.
        The beat will play for its full duration based on BPM without blocking.
        """
        if not self.running or self.paused:
            return False

        if self.current_beat_index > self.max_beats:
            print("[MIDI] End of piece reached")
            self.stop()
            return False

        # Cancel any existing beat timer
        if self.beat_timer is not None:
            self.beat_timer.cancel()
            self.beat_timer = None

        # Collect events from all channels for the current beat
        beat_events = []
        for channel in self.channel_queues:
            if self.current_beat_index in self.channel_queues[channel]:
                beat_events.extend(self.channel_queues[channel][self.current_beat_index])
        
        # Sort all events by tick time
        beat_events.sort(key=lambda x: x[0])
        
        # Play the events if there are any
        if beat_events:
            self._play_beat_events(beat_events)
        
        # Calculate beat duration in seconds based on current BPM
        beat_duration = 60.0 / self.current_bpm
        
        # Schedule notes to be silenced after the beat duration (asynchronously)
        self.beat_timer = threading.Timer(beat_duration, self._all_notes_off)
        self.beat_timer.start()
        
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
        
        # Cancel any pending beat timer
        if self.beat_timer is not None:
            self.beat_timer.cancel()
            self.beat_timer = None
        
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
        
        # Cancel any pending beat timer
        if self.beat_timer is not None:
            self.beat_timer.cancel()
            self.beat_timer = None
        
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
        
        # Cancel any pending beat timer
        if self.beat_timer is not None:
            self.beat_timer.cancel()
            self.beat_timer = None
        
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