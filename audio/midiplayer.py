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
        self.target_volume = self.volume  # Target volume for fading
        self.fade_active = False  # Whether a fade is in progress
        self.running = False
        self.paused = False
        self.thread = None
        self.fade_thread = None  # Thread for volume fading
        
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
        # Original BPM from MIDI file
        original_bpm = mido.tempo2bpm(self.original_tempo)
        
        # Time scaling factor: original_bpm / current_bpm
        # If current BPM is higher, we sleep less (play faster)
        # If current BPM is lower, we sleep more (play slower)
        return original_bpm / self.current_bpm

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
                time.sleep(0.01)  # Small sleep to avoid busy waiting
            
            if not self.running:
                break
            
            # Send MIDI messages to FluidSynth
            if msg.type == 'note_on':
                # Apply volume scaling to velocity, clamp to MIDI max (127)
                adjusted_velocity = min(127, int(msg.velocity * self.volume))
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
            time_scale = self._calculate_time_scaling()
            time.sleep(msg.time * time_scale)
        
        self._all_notes_off()
        self.running = False  # Mark as finished
        print("[MIDI] Playback finished.")

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