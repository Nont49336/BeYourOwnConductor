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
        """Organize MIDI messages into beats based on timing."""
        if not self.midi:
            return

        # Reset queues
        self.notes_queue = []
        current_time = 0
        current_beat_time = 0
        beat_events = []
        
        # Calculate microseconds per beat from BPM
        microseconds_per_beat = mido.bpm2tempo(self.current_bpm)
        print("⚠️⚠️⚠️   the value is still brute forcing    ⚠️⚠️⚠️")
        seconds_per_beat = microseconds_per_beat / 2000000.0
        
        # Process all MIDI messages
        for msg in self.midi:
            current_time += msg.time
            
            # If we've moved to a new beat
            if current_time >= current_beat_time + seconds_per_beat:
                if beat_events:
                    self.notes_queue.append((current_beat_time, beat_events))
                current_beat_time = current_time
                beat_events = []
            
            # Add message to current beat's events
            if msg.type in ['note_on', 'note_off', 'program_change', 'control_change', 'pitchwheel']:
                beat_events.append(msg)
        
        # Add any remaining events
        if beat_events:
            self.notes_queue.append((current_beat_time, beat_events))
        
        print(f"[MIDI] Organized {len(self.notes_queue)} beats of music")

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
        """Play all notes in the given beat events."""
        # Stop any currently playing notes
        self._all_notes_off()
        
        # Small gap before starting new beat
        time.sleep(0.02)
        
        # Play all notes in this beat simultaneously
        for msg in beat_events:
            if msg.type == 'note_on':
                adjusted_velocity = min(127, int(msg.velocity * self.volume))
                self.fs.noteon(msg.channel, msg.note, adjusted_velocity)
                self.active_notes.add((msg.channel, msg.note))
            elif msg.type == 'note_off':
                if (msg.channel, msg.note) in self.active_notes:
                    self.fs.noteoff(msg.channel, msg.note)
                    self.active_notes.remove((msg.channel, msg.note))
            elif msg.type == 'program_change':
                self.fs.program_change(msg.channel, msg.program)
            elif msg.type == 'control_change':
                self.fs.cc(msg.channel, msg.control, msg.value)
            elif msg.type == 'pitchwheel':
                self.fs.pitch_bend(msg.channel, msg.pitch)

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
            player.start()
            
            # Play for 3 seconds at 120 BPM, full volume
            time.sleep(3)
            
            # Reduce volume to 50%
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