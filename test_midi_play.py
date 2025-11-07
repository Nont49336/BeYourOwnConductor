import mido
import fluidsynth
import threading
import time
import os
import platform


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
        self.volume = max(0.0, min(2.0, volume))  # Clamp volume between 0.0 and 2.0
        self.running = False
        self.paused = False
        self.thread = None
        self.beat_chunks = {}
        
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

    def chop_midi_into_beats(self,midi):
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
        for msg in midi:
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

    def load_file(self, path: str) -> bool:
        """Load a MIDI file into the player. Returns True if successful."""
        if not os.path.isfile(path):
            print(f"[Error] File not found: {path}")
            return False

        # try:
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
        self.chop_midi_into_beats(self.midi)
        print(self.beat_chunks)
        print(f"[MIDI] Loaded file: {os.path.basename(path)} "
                f"({len(midi.tracks)} tracks, {midi.length:.2f}s)")
        return True
        # except Exception as e:
        #     print(f"[Error] Failed to load MIDI file: {e}")
        #     self.midi = None
        #     self.midi_path = None
            # return False

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

    # def _calculate_time_scaling(self) -> float:
    #     """
    #     Calculate how much to scale the original MIDI timing.
    #     Returns the factor to multiply sleep times by.
    #     """
    #     # Original BPM from MIDI file
    #     original_bpm = mido.tempo2bpm(self.original_tempo)
        
    #     # Time scaling factor: original_bpm / current_bpm
    #     # If current BPM is higher, we sleep less (play faster)
    #     # If current BPM is lower, we sleep more (play slower)
    #     return original_bpm / self.current_bpm
    def _playback_loop(self):
        """Internal playback thread."""
        if not self.is_file_loaded():
            print("[Error] No MIDI file loaded. Cannot start playback.")
            return

        if not self.beat_chunks:
            print("[Error] No beat chunks available. Run chop_midi_into_beats first.")
            return

        beat_indices = sorted(self.beat_chunks.keys())  # Get sorted beat numbers
        seconds_per_beat = 60.0 / self.current_bpm  # Calculate time per beat
        print(f"[DEBUG] Found {len(beat_indices)} beats to play")
        print(f"[DEBUG] Seconds per beat: {seconds_per_beat}")

        for beat_index in beat_indices:
            print(f"[DEBUG] Playing beat {beat_index} with {len(self.beat_chunks[beat_index])} messages")
            if not self.running:
                break

            # Check if paused
            while self.paused and self.running:
                time.sleep(0.01)  # Small sleep to avoid busy waiting
            
            if not self.running:
                break
            # self.midi.play()
            # Play all messages in this beat chunk
            for msg in self.beat_chunks[beat_index]:
                # Send MIDI messages to FluidSynth
                time.sleep(msg.time)
                if msg.type == 'note_on':
                    # Apply volume scaling to velocity, clamp to MIDI max (127)
                    adjusted_velocity = 127
                    self.fs.noteon(msg.channel, msg.note, adjusted_velocity)
                elif msg.type == 'note_off':
                    self.fs.noteoff(msg.channel, msg.note)
                elif msg.type == 'program_change':
                    self.fs.program_change(msg.channel, msg.program)
                elif msg.type == 'control_change':
                    self.fs.cc(msg.channel, msg.control, msg.value)
                elif msg.type == 'pitchwheel':
                    self.fs.pitch_bend(msg.channel, msg.pitch)
            
            # Wait one beat duration before next chunk
            # time.sleep(seconds_per_beat)
            
        self._all_notes_off()
        self.running = False  # Mark as finished
        print("[MIDI] Playback finished.")

    def _all_notes_off(self):
        """Send all notes off and all sound off messages to all channels."""
        for channel in range(16):
            self.fs.cc(channel, 123, 0)  # All notes off
            self.fs.cc(channel, 120, 0)  # All sound off

    def start(self):
        """Start playback in a background thread."""
        if not self.is_file_loaded():
            print("[Error] No MIDI file loaded. Load a file before playing.")
            return

        if self.running:
            print("[MIDI] Already playing.")
            return

        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.thread.start()
        print(f"[MIDI] ▶ Playback started at {self.current_bpm:.1f} BPM: {os.path.basename(self.midi_path)}")

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
        print(f"[MIDI] ▶ Playback resumed at {self.current_bpm:.1f} BPM.")

    def is_paused(self) -> bool:
        """Check if playback is currently paused."""
        return self.paused

    def stop(self):
        """Stop playback gracefully."""
        if not self.running:
            return
        
        self.running = False
        self.paused = False
        self._all_notes_off()
        
        if self.thread:
            self.thread.join()
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
    SOUNDFONT_PATH = os.path.expanduser("FluidR3Mono_GM.sf3")
    
    try:
        # Initialize at 120 BPM with full volume (1.0)
        player = DynamicMidiPlayer(soundfont_path=SOUNDFONT_PATH, bpm=120, volume=1.0)
        
        # Load a MIDI file
        if player.load_file("ode_to_joy.mid"):
            midi = player.midi
            player.start()

            while player.running:
                time.sleep(0.001)

            player.stop()
        
        player.close()
        
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        print("Please download a soundfont first:")
        print("  curl -L -O https://github.com/musescore/MuseScore/raw/2.3.2/share/sound/FluidR3Mono_GM.sf3")