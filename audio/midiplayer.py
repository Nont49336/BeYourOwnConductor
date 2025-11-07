import mido
import fluidsynth
import threading
from queue import Queue, Empty
import time
import os
import platform


# General MIDI instrument names (Program numbers 0-127)
GM_INSTRUMENTS = [
    # Piano (0-7)
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    # Chromatic Percussion (8-15)
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    # Organ (16-23)
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    # Guitar (24-31)
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
    "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
    # Bass (32-39)
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    # Strings (40-47)
    "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    # Ensemble (48-55)
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Choir", "Orchestra Hit",
    # Brass (56-63)
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    # Reed (64-71)
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet",
    # Pipe (72-79)
    "Piccolo", "Flute", "Recorder", "Pan Flute",
    "Blown bottle", "Shakuhachi", "Whistle", "Ocarina",
    # Synth Lead (80-87)
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
    # Synth Pad (88-95)
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    # Synth Effects (96-103)
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    # Ethnic (104-111)
    "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bag pipe", "Fiddle", "Shanai",
    # Percussive (112-119)
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    # Sound Effects (120-127)
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot"
]


class DynamicMidiPlayer:
    """
    A MIDI player with dynamic BPM control using FluidSynth.
    Allows real-time adjustment of playback tempo in beats per minute.
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
        self.last_bpm = bpm
        self.volume = max(0.0, min(2.0, volume))  # Clamp volume between 0.0 and 2.0
        self.target_volume = self.volume  # Target volume for fading
        self.fade_active = False  # Whether a fade is in progress
        self.running = False
        self.paused = False
        self.thread = None
        self.fade_thread = None  # Thread for volume fading
        
        # Multiple track handling
        self.tracks = []  # List of track information: {'name': str, 'events': list, 'channel': int}
        self.track_volumes = []  # List of volume multipliers for each track
        self.beat_chunks = {}
        self.current_beat_index = 0
        self.beat_queue = Queue()
        self.beat_thread = None
        self.beat_thread_running = False
        
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

    def _organize_tracks(self):
        """Organize MIDI tracks and extract track information including instruments."""
        if not self.midi:
            return

        # Reset track data
        self.tracks = []
        self.track_volumes = []
        
        # First pass: collect track information
        track_data = []
        for track_idx, track in enumerate(self.midi.tracks):
            track_name = track.name if hasattr(track, 'name') and track.name else f"Track {track_idx}"
            
            # Determine the channel used by this track (default to track_idx % 16)
            track_channel = None
            track_program = None  # MIDI program number (instrument)
            
            for msg in track:
                if hasattr(msg, 'channel') and track_channel is None:
                    track_channel = msg.channel
                
                # Extract program change (instrument) information
                if msg.type == 'program_change':
                    track_program = msg.program
                    
                # Stop searching once we have both channel and program
                if track_channel is not None and track_program is not None:
                    break
            
            if track_channel is None:
                track_channel = track_idx % 16
            
            # Get instrument name from program number
            # Special case: Channel 10 (index 9) is always drums in General MIDI
            instrument_name = None
            if track_channel == 9:  # Channel 10 in 1-indexed (9 in 0-indexed)
                instrument_name = "Drum Kit"
            elif track_program is not None and 0 <= track_program < len(GM_INSTRUMENTS):
                instrument_name = GM_INSTRUMENTS[track_program]
            
            track_data.append({
                'name': track_name,
                'channel': track_channel,
                'program': track_program,
                'instrument': instrument_name,
                'original_index': track_idx
            })
        
        # Second pass: count instrument occurrences and assign numbers to duplicates
        instrument_counts = {}
        instrument_indices = {}
        
        for data in track_data:
            instrument_name = data['instrument']
            if instrument_name:
                if instrument_name not in instrument_counts:
                    instrument_counts[instrument_name] = 0
                    instrument_indices[instrument_name] = []
                instrument_counts[instrument_name] += 1
                instrument_indices[instrument_name].append(len(self.tracks))
        
        # Third pass: create labels with unique numbering
        for data in track_data:
            instrument_name = data['instrument']
            
            # Create a descriptive label for the track
            if instrument_name:
                # If there are multiple tracks with the same instrument, add a number
                if instrument_counts[instrument_name] > 1:
                    # Find which occurrence this is (1-indexed)
                    occurrence = instrument_indices[instrument_name].index(len(self.tracks)) + 1
                    track_label = f"{instrument_name} {occurrence}"
                else:
                    track_label = instrument_name
            else:
                track_label = data['name']
            
            # Store track information
            self.tracks.append({
                'name': data['name'],
                'label': track_label,
                'channel': data['channel'],
                'program': data['program'],
                'instrument': data['instrument'],
                'original_index': data['original_index']
            })
            self.track_volumes.append(1.0)
        
        print(f"[MIDI] Organized {len(self.tracks)} tracks")
        for i, track in enumerate(self.tracks):
            instrument_info = f" - {track['instrument']}" if track['instrument'] else ""
            print(f"  Track {i}: {track['label']} (Ch.{track['channel']}){instrument_info}")

    def chop_midi_into_beats(self):
        current_tempo = self.original_tempo #might be good to use adaptive tempo to adjsut the 
        current_ticks=0
        self.beat_chunks = {} 
        print("[DEBUG] Starting to chop MIDI into beats")
        print("[DEBUG] MIDI file details:", self.midi)
        # Helper function to convert ticks → beats
        def ticks_to_beats(ticks):
            return ticks / (self.midi.ticks_per_beat)
        
        # def time_to_ticks(microsecond):
            # mido.tick2second
            # return mido.second2tick(microsecond)

        # Scan all messages
        for msg in self.midi:
            time = mido.second2tick(msg.time,self.midi.ticks_per_beat,current_tempo)
            current_ticks += time # accumulate delta times
            # remap_tick = mido.second2tick(msg.time,midi.ticks_per_beat,self.original_tempo)
            # # Update tempo or time signature if they appear
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo  # microseconds per beat
            # elif msg.type == 'time_signature':
            #     beats_per_bar = msg.numerator
            # Compute current beat number (integer index)
            beat_index = int(ticks_to_beats(current_ticks))
            
            # Add message to that beat’s list
            if beat_index not in self.beat_chunks:
                self.beat_chunks[beat_index] = []
            self.beat_chunks[beat_index].append(msg)

        # self.beat_chunks = sorted(self.beat_chunks.keys())

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
            self._organize_tracks()
            print("debugging")
            self.chop_midi_into_beats()
            # print(self.beat_chunks)
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
        self.target_volume = max(0.0, min(2.0, volume))
        # Update immediately if no fade is active
        if not self.fade_active:
            self.volume = self.target_volume
        print(f"[Volume] Volume set to: {self.target_volume:.2f} ({int(self.target_volume * 100)}%)")

    def get_volume(self) -> float:
        """Get the current volume level (0.0 to 2.0)."""
        return self.volume
    
    def _calculate_time_scaling(self) -> float:
        """
        Calculate how much to scale the original MIDI timing.
        Returns the factor to multiply sleep times by.
        """
        threshold = 0.2
        # Original BPM from MIDI file
        # original_bpm = mido.tempo2bpm(self.original_tempo)
        # current_bpm = self.current_bpm
        # Time scaling factor: original_bpm / current_bpm
        # If current BPM is higher, we sleep less (play faster)
        # If current BPM is lower, we sleep more (play slower)
        # add threshold for robustness
        scale = self.last_bpm / self.current_bpm
        
        if scale >= threshold:
            self.last_bpm = self.current_bpm * threshold
            return threshold
        self.last_bpm = self.current_bpm
        return scale

    def play_next_beat(self):
        """Queue the next beat to be played."""
        if not self.running or self.paused:
            return False

        if self.current_beat_index >= len(self.beat_chunks.keys()):
            print("[MIDI] End of piece reached")
            self.stop()
            return False
                # Queue the beat to be played
        self.beat_queue.put(True)
        return True

    def _get_track_for_channel(self, channel: int) -> int:
        """Find the track index that uses this channel (returns first match or -1 if none)."""
        for i, track in enumerate(self.tracks):
            if track['channel'] == channel:
                return i
        return -1

    def _get_track_volume(self, track_idx: int) -> float:
        """Get the volume multiplier for a track."""
        if track_idx < 0 or track_idx >= len(self.tracks):
            return 1.0  # Unknown track, use full volume
        return self.track_volumes[track_idx]


    
    def _playback_loop(self):
        """Internal playback thread."""
        if not self.is_file_loaded():
            print("[Error] No MIDI file loaded. Cannot start playback.")
            return

        for msg in self.midi.play():
            if not self.running:
                break
            
            # Check if paused
            while self.paused and self.running:
                time.sleep(0.000001)  # Small sleep to avoid busy waiting
            
            if not self.running:
                break
            
            # Determine which track this message belongs to (if it has a channel)
            track_idx = -1
            if hasattr(msg, 'channel'):
                track_idx = self._get_track_for_channel(msg.channel)
            
            # Get track volume multiplier
            track_volume = self._get_track_volume(track_idx)
            
            # Send MIDI messages to FluidSynth
            if msg.type == 'note_on':
                # Apply both global volume and track volume to velocity
                combined_volume = self.volume * track_volume
                adjusted_velocity = min(127, int(msg.velocity * combined_volume))
                self.fs.noteon(msg.channel, msg.note, adjusted_velocity)
            elif msg.type == 'note_off':
                self.fs.noteoff(msg.channel, msg.note)
            elif msg.type == 'program_change':
                self.fs.program_change(msg.channel, msg.program)
            elif msg.type == 'control_change':
                self.fs.cc(msg.channel, msg.control, msg.value)
            elif msg.type == 'pitchwheel':
                self.fs.pitch_bend(msg.channel, msg.pitch)
            
            # Apply BPM scaling to timing
            # time_scale = self._calculate_time_scaling()
            time.sleep(msg.time * time_scale)
        
        self._all_notes_off()
        self.running = False  # Mark as finished
        print("[MIDI] Playback finished.")

    def _beat_player_thread(self):
        """Thread that handles playing beats from the queue."""
        while self.beat_thread_running:
            try:
                # Wait for next beat request with a timeout
                beat_requested = self.beat_queue.get(timeout=0.1)
                if beat_requested and self.running and not self.paused:
                    if self.current_beat_index >= len(self.beat_chunks.keys()):
                        print("[MIDI] End of piece reached")
                        self.stop()
                        continue
                    time_scale = self._calculate_time_scaling()
                    # Play the current beat
                    for msg in self.beat_chunks[self.current_beat_index]:
                        if not self.running or self.paused:
                            break
                    
                        time.sleep(msg.time*time_scale)
                        # time.sleep(msg.time)
                        if msg.type == 'note_on':
                            adjusted_velocity = min(int(msg.velocity * self.volume), 127)
                            self.fs.noteon(msg.channel, msg.note, adjusted_velocity)
                        elif msg.type == 'note_off':
                            self.fs.noteoff(msg.channel, msg.note)
                        elif msg.type == 'program_change':
                            self.fs.program_change(msg.channel, msg.program)
                        elif msg.type == 'control_change':
                            self.fs.cc(msg.channel, msg.control, msg.value)
                        elif msg.type == 'pitchwheel':
                            self.fs.pitch_bend(msg.channel, msg.pitch)
                    
                    # Increment beat counter
                    self.current_beat_index += 1

            except Empty:
                continue  # No beat requested, continue waiting

    def _all_notes_off(self):
        """Send all notes off and all sound off messages to all channels."""
        for channel in range(16):
            self.fs.cc(channel, 123, 0)  # All notes off
            self.fs.cc(channel, 120, 0)  # All sound off

    def _fade_volume(self, target_volume: float, duration: float = 0.3):
        """
        Smoothly fade volume from current level to target level over duration.
        Runs in a background thread to avoid blocking.
        
        Args:
            target_volume: Target volume level (0.0 to 2.0)
            duration: Fade duration in seconds (default: 0.3)
        """
        def fade_worker():
            self.fade_active = True
            start_volume = self.volume
            start_time = time.time()
            
            while time.time() - start_time < duration and self.running:
                elapsed = time.time() - start_time
                progress = elapsed / duration
                
                # Linear interpolation
                self.volume = start_volume + (target_volume - start_volume) * progress
                time.sleep(0.01)  # Update every 10ms for smooth fade
            
            # Ensure we reach the exact target volume
            self.volume = target_volume
            self.fade_active = False
        
        # Stop any existing fade
        if self.fade_thread and self.fade_thread.is_alive():
            self.fade_active = False
            self.fade_thread.join(timeout=0.1)
        
        # Start new fade in background thread
        self.fade_thread = threading.Thread(target=fade_worker, daemon=True)
        self.fade_thread.start()

    def start(self):
        """Start playback system."""
        if not self.is_file_loaded():
            print("[Error] No MIDI file loaded. Load a file before playing.")
            return

        if self.running:
            print("[MIDI] Already playing.")
            return

        self.running = True
        self.paused = False
        self.current_beat_index = 0
        
        # Start the beat player thread
        self.beat_thread_running = True
        self.beat_thread = threading.Thread(target=self._beat_player_thread, daemon=True)
        self.beat_thread.start()
        
        print(f"[MIDI] ▶ Ready to play at {self.current_bpm:.1f} BPM: {os.path.basename(self.midi_path)}")

    def pause(self):
        """Pause playback with a smooth fade-down effect. Call resume() to continue."""
        if not self.running:
            print("[MIDI] Not playing.")
            return
        
        if self.paused:
            print("[MIDI] Already paused.")
            return
        
        # Fade volume down to 0 over 1 second (non-blocking)
        self._fade_volume(0.0, duration=1.0)
        
        # Set paused flag immediately - fade happens in background
        self.paused = True
        
        print("[MIDI] ⏸ Playback pausing (with fade).")

    def resume(self):
        """Resume playback with a smooth fade-in effect."""
        if not self.running:
            print("[MIDI] Not playing.")
            return
        
        if not self.paused:
            print("[MIDI] Not paused.")
            return
        
        # Unpause immediately so music resumes
        self.paused = False
        
        print(f"[MIDI] ▶ Playback resuming at {self.current_bpm:.1f} BPM (with fade).")
        
        # Fade volume up from current (likely 0) to target over 1 second (non-blocking)
        self._fade_volume(self.target_volume, duration=1.0)

    def is_paused(self) -> bool:
        """Check if playback is currently paused."""
        return self.paused

    def stop(self):
        """Stop playback gracefully."""
        if not self.running:
            return
        
        self.running = False
        self.paused = False
        self.beat_thread_running = False
        self._all_notes_off()
        
        # Clear the beat queue
        while not self.beat_queue.empty():
            self.beat_queue.get()
        
        if self.beat_thread:
            self.beat_thread.join()
            self.beat_thread = None
            
        print("[MIDI] ■ Playback stopped.")

    # Track-specific controls
    
    def get_track_info(self, track_idx: int) -> dict:
        """
        Get information about a specific track.
        
        Args:
            track_idx: Index of the track (0-based)
        
        Returns:
            Dictionary with track information:
            - name: Track name
            - label: Track label (name + instrument if available)
            - channel: MIDI channel used by the track
            - program: MIDI program number (0-127, or None if not set)
            - instrument: Instrument name (or None if not identified)
            - volume: Track volume multiplier (0.0 to 2.0)
            
            Returns None if track index is invalid.
        """
        if 0 <= track_idx < len(self.tracks):
            return {
                'name': self.tracks[track_idx]['name'],
                'label': self.tracks[track_idx]['label'],
                'channel': self.tracks[track_idx]['channel'],
                'program': self.tracks[track_idx]['program'],
                'instrument': self.tracks[track_idx]['instrument'],
                'volume': self.track_volumes[track_idx]
            }
        return None
    
    def set_track_volume(self, track_idx: int, volume: float):
        """
        Set the volume for a specific track.
        
        Args:
            track_idx: Index of the track (0-based)
            volume: Volume level (0.0 = silent, 1.0 = full volume, 2.0 = twice full volume)
        """
        if 0 <= track_idx < len(self.tracks):
            self.track_volumes[track_idx] = max(0.0, min(2.0, volume))
            print(f"[MIDI] Track {track_idx} ({self.tracks[track_idx]['label']}) volume set to {volume:.2f}")
        else:
            print(f"[Error] Invalid track index: {track_idx}")
    
    def get_tracks_with_notes(self):
        """
        Get a list of track indices that contain note events.
        
        Returns:
            Tuple of (list of track indices with notes, count of tracks with notes)
        """
        tracks_with_notes = []
        
        for i in range(len(self.tracks)):
            # Check if track has any note events by examining the MIDI file
            has_notes = False
            if self.midi and i < len(self.midi.tracks):
                track = self.midi.tracks[i]
                for msg in track:
                    if msg.type in ['note_on', 'note_off']:
                        has_notes = True
                        break
            
            if has_notes:
                tracks_with_notes.append(i)
        
        return tracks_with_notes, len(tracks_with_notes)
    
    def get_track_count(self) -> int:
        """Get the number of tracks in the loaded MIDI file."""
        return len(self.tracks)
    
    def list_tracks(self):
        """Print information about all tracks."""
        if not self.midi_path:
            print("[MIDI] No file loaded")
            return
            
        print(f"\n[MIDI] Tracks in {os.path.basename(self.midi_path)}:")
        for i in range(len(self.tracks)):
            info = self.get_track_info(i)
            instrument_str = f" - {info['instrument']}" if info['instrument'] else ""
            print(f"  {i}: {info['name']} (Ch.{info['channel']}, Vol:{info['volume']:.1f}){instrument_str}")
    
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
        if player.load_file("../ode_to_joy.mid"):
            # Display track information
            player.list_tracks()
            
            # Get tracks with notes
            tracks_with_notes, count = player.get_tracks_with_notes()
            print(f"\n[MIDI] Found {count} tracks with notes: {tracks_with_notes}")
            
            player.start()
            
            # Play for 3 seconds at 120 BPM, full volume
            time.sleep(3)
            
            # Demonstrate track volume control
            if player.get_track_count() > 0:
                print("\n[Demo] Reducing volume of first track...")
                player.set_track_volume(0, 0.3)
                time.sleep(3)
                
                # Get track info
                info = player.get_track_info(0)
                print(f"[Demo] Track 0 info: {info}")
            
            # Reduce global volume to 50%
            player.set_volume(0.5)
            time.sleep(3)
            
            # Pause playback
            player.pause()
            print("Paused for 2 seconds...")
            time.sleep(2)
            
            # Resume playback at lower volume
            player.resume()
            time.sleep(2)
            
            # Increase volume to 80%
            player.set_volume(0.8)
            time.sleep(3)
            
            # Speed up to 160 BPM
            player.set_bpm(160)
            time.sleep(3)
            
            # Very loud (200% volume)
            player.set_volume(2.0)
            time.sleep(2)
            
            # Pause again
            player.pause()
            time.sleep(2)
            player.resume()
            
            # Slow down to 80 BPM and full volume
            player.set_bpm(80)
            player.set_volume(1.0)
            time.sleep(3)
            
            player.stop()
        
        player.close()
        
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        print("Please download a soundfont first:")
        print("  curl -L -O https://github.com/musescore/MuseScore/raw/2.3.2/share/sound/FluidR3Mono_GM.sf3")