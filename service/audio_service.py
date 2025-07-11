import glob
import os
import shutil

import numpy as np
import scipy
import sounddevice as sd
import librosa
import librosa.display
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter
import math
import queue
import wave
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from domain.audio_device_manager import AudioDeviceManager
from domain.recording import Recording
from domain.audio_processor import AudioProcessor


class AudioService:
    def __init__(self, config):
        self.device_manager = AudioDeviceManager()
        self.undo_stack = []
        self.config = config
        self.recording = None
        self.recording_thread = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.stop_audio = False
        self.state_history = []
        self.current_state_index = -1
        self.progress_callback = None
        self.duration_callback = None  # Callback pentru actualizarea duratei
        self.processing_chunk_size = 1024 * 1024  # 1MB chunks by default
        self.cache = []
        self.processor = AudioProcessor()  # Inițializăm procesorul audio
        self.is_playing = False  # Adăugăm flag-ul pentru starea de redare
        self.is_loading = False  # Adăugăm flag-ul pentru starea de încărcare
        self.is_saving = False  # Adăugăm flag-ul pentru starea de salvare

        # Constante pentru cache
        self.MAX_CACHE_SIZE = 1024 * 1024 * 1024  # 1GB în bytes
        self.MAX_CACHE_ENTRIES = 50  # Numărul maxim de intrări în cache
        self.current_cache_size = 0  # Dimensiunea curentă a cache-ului în bytes

        # Constante pentru undo stack
        self.MAX_UNDO_STACK_SIZE = 20  # Numărul maxim de stări pentru undo
        self.current_undo_size = 0  # Dimensiunea curentă a stivei de undo în bytes

    def set_progress_callback(self, callback):
        """
        Setează funcția de callback pentru actualizarea progresului.
        :param callback: Funcția care va fi apelată cu procentul de progres (0-100)
        """
        self.progress_callback = callback

    def set_duration_callback(self, callback):
        """
        Setează funcția de callback pentru actualizarea duratei.
        :param callback: Funcția care va fi apelată cu durata în secunde
        """
        self.duration_callback = callback

    def _update_duration(self):
        """
        Actualizează durata în interfață dacă există un callback setat.
        """
        if self.duration_callback and self.recording is not None:
            duration = len(self.recording.data) / self.recording.sample_rate
            self.duration_callback(duration)

    def set_processing_config(self, chunk_size=None, num_threads=None):
        """
        Configurează parametrii procesării paralelă.
        :param chunk_size: Dimensiunea chunk-ului în bytes
        :param num_threads: Numărul de threaduri pentru procesare
        """
        if chunk_size is not None:
            self.processing_chunk_size = chunk_size

        if num_threads is not None:
            # Oprim procesorul vechi
            self.processor.shutdown()
            # Creăm unul nou cu numărul specificat de threaduri
            self.processor = AudioProcessor(max_workers=num_threads)

    def get_processing_config(self):
        """
        Returnează configurația curentă a procesării.
        :return: Tuple (chunk_size, num_threads)
        """
        return self.processing_chunk_size, self.processor.max_workers

    def record(self, duration_seconds):
        print("Începere înregistrare...")
        self.is_recording = True
        if self.progress_callback:
            self.progress_callback(0)

        def recording_task():
            try:
                data = sd.rec(
                    int(duration_seconds * self.config.sample_rate),
                    samplerate=self.config.sample_rate,
                    channels=1,
                    dtype='float64',
                    device=self.config.input_device,
                    blocksize=self.config.buffer_size
                )
                # În loc de sd.wait(), folosește un loop care verifică flag-ul
                total_samples = int(duration_seconds * self.config.sample_rate)
                while self.is_recording and (sd.get_stream().active or sd.get_stream().time < duration_seconds):
                    sd.sleep(100)
                sd.stop()
                if self.is_recording:
                    self.recording = Recording(data.flatten(), self.config.sample_rate)
                    self.cache_state("recording")
                    print("Înregistrare finalizată.")
                else:
                    print("Înregistrare oprită manual.")
            except Exception as e:
                print(f"Eroare la înregistrare: {e}")
            finally:
                self.is_recording = False
                if self.progress_callback:
                    self.progress_callback(100)

        self.recording_thread = threading.Thread(target=recording_task)
        self.recording_thread.start()

    def stop_recording(self):
        """
        Oprește înregistrarea curentă.
        """
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)  # Așteptăm maxim 2 secunde
            print("Înregistrare oprită.")
            return True
        return False

    def _calculate_cache_entry_size(self, entry):
        """
        Calculează dimensiunea unei intrări din cache în bytes.
        """
        if entry is None or "data" not in entry:
            return 0
        return entry["data"].nbytes

    def _cleanup_cache(self):
        """
        Curăță cache-ul când se atinge limita de dimensiune sau număr de intrări.
        """
        while (self.current_cache_size > self.MAX_CACHE_SIZE or
               len(self.cache) > self.MAX_CACHE_ENTRIES):
            if not self.cache:
                break
            # Eliminăm cea mai veche intrare
            removed_entry = self.cache.pop(0)
            self.current_cache_size -= self._calculate_cache_entry_size(removed_entry)

    def cache_state(self, effect_name, params=None):
        """
        Salvează starea curentă a înregistrării în cache, cu numele efectului și parametrii.
        Verifică și menține limitele de dimensiune și număr de intrări.
        """
        if self.recording is not None:
            new_entry = {
                "data": np.copy(self.recording.data),
                "sample_rate": self.recording.sample_rate,
                "effect": effect_name,
                "params": params if params else {}
            }

            # Calculăm dimensiunea noii intrări
            entry_size = self._calculate_cache_entry_size(new_entry)

            # Verificăm dacă avem suficient spațiu
            if entry_size > self.MAX_CACHE_SIZE:
                print("Avertisment: Înregistrarea este prea mare pentru cache.")
                return

            # Adăugăm noua intrare
            self.cache.append(new_entry)
            self.current_cache_size += entry_size

            # Curățăm cache-ul dacă este necesar
            self._cleanup_cache()

    def get_cache_info(self):
        """
        Returnează informații despre starea curentă a cache-ului.
        """
        return {
            "current_size": self.current_cache_size,
            "max_size": self.MAX_CACHE_SIZE,
            "entries": len(self.cache),
            "max_entries": self.MAX_CACHE_ENTRIES
        }

    def clear_cache(self):
        """
        Curăță complet cache-ul.
        """
        self.cache.clear()
        self.current_cache_size = 0

    def load_from_cache(self, index):
        """
        Încarcă o stare din cache după index.
        """
        if 0 <= index < len(self.cache):
            cached = self.cache[index]
            self.recording = Recording(np.copy(cached["data"]), cached["sample_rate"])
            return True
        return False

    def get_cache_history(self):
        """
        Returnează o listă cu descrierea pașilor din cache.
        """
        return [
            f"{i}: {entry['effect']} {entry['params']}" for i, entry in enumerate(self.cache)
        ]

    def export_onsets(self, filepath):
        """
        Detectează onsets și exportă timpii (în secunde) într-un fișier CSV/TXT.
        """
        if self.recording is None:
            print("Nu există înregistrare pentru export onsets.")
            return False

        y = self.recording.data
        sr = self.recording.sample_rate
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        with open(filepath, "w") as f:
            f.write("onset_time_sec\n")
            for t in onset_times:
                f.write(f"{t:.6f}\n")
        print(f"Onsets exportate în {filepath}")
        return True

    def save_recording(self, filename):
        """
        Salvează înregistrarea într-un thread separat.
        :param filename: Numele fișierului unde se va salva înregistrarea
        :return: True dacă salvarea a început cu succes, False altfel
        """
        if not self.recording:
            print("Nu există înregistrare de salvat.")
            return False

        # Verificăm dacă directorul există
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                print(f"Nu s-a putut crea directorul {directory}: {e}")
                return False

        # Verificăm dacă avem permisiuni de scriere
        if os.path.exists(filename):
            if not os.access(filename, os.W_OK):
                print(f"Nu există permisiuni de scriere pentru {filename}")
                return False
        else:
            if not os.access(directory or '.', os.W_OK):
                print(f"Nu există permisiuni de scriere în directorul {directory or '.'}")
                return False

        def save_task():
            try:
                self.is_saving = True
                print(f"Salvare înregistrare: {len(self.recording.data) / self.recording.sample_rate:.2f} secunde")

                # Verificăm dacă avem suficient spațiu pe disc
                required_space = len(self.recording.data) * 2  # 2 bytes per sample
                if os.path.exists(filename):
                    required_space -= os.path.getsize(filename)

                if os.path.exists(filename):
                    free_space = shutil.disk_usage(os.path.dirname(filename)).free
                else:
                    free_space = shutil.disk_usage('.').free

                if free_space < required_space:
                    print(
                        f"Nu există suficient spațiu pe disc. Necesar: {required_space / 1024 / 1024:.1f}MB, Disponibil: {free_space / 1024 / 1024:.1f}MB")
                    self.is_saving = False
                    return False

                # Convertim la int16 și salvăm întregul semnal
                audio_data = np.int16(self.recording.data * 32767)

                # Salvăm într-un fișier temporar mai întâi
                temp_filename = filename + '.tmp'
                wav.write(temp_filename, self.recording.sample_rate, audio_data)

                # Dacă salvarea a reușit, redenumim fișierul temporar
                if os.path.exists(filename):
                    os.remove(filename)
                os.rename(temp_filename, filename)

                print(f"Înregistrare salvată în {filename}")
                return True

            except MemoryError:
                print("Nu există suficientă memorie pentru a salva înregistrarea.")
                return False
            except Exception as e:
                print(f"Eroare la salvarea înregistrării: {e}")
                # Încercăm să curățăm fișierul temporar dacă există
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
                return False
            finally:
                self.is_saving = False

        # Pornim salvarea într-un thread separat
        save_thread = threading.Thread(target=save_task)
        save_thread.start()
        return True

    def play(self):
        """
        Redă audio-ul într-un thread separat, procesând datele pe chunks.
        """
        if self.recording is None:
            print("Nu există înregistrare pentru redare.")
            return

        def playback_task():
            try:
                self.is_playing = True
                # Verificăm și afișăm informații despre semnal
                signal_length = len(self.recording.data)
                duration = signal_length / self.recording.sample_rate
                print(f"Redare audio:")
                print(f"- Lungime semnal: {signal_length} eșantioane")
                print(f"- Rata de eșantionare: {self.recording.sample_rate} Hz")
                print(f"- Durată: {duration:.2f} secunde")

                # Ne asigurăm că semnalul este în formatul corect
                if self.recording.data.dtype != np.float32:
                    self.recording.data = self.recording.data.astype(np.float32)

                # Definim dimensiunea chunk-ului pentru redare (50ms)
                chunk_size = int(self.recording.sample_rate * 0.05)

                # Creăm un stream pentru redare
                with sd.OutputStream(
                        samplerate=self.recording.sample_rate,
                        channels=1,
                        dtype=np.float32,
                        device=self.config.output_device,
                        blocksize=chunk_size
                ) as stream:
                    # Procesăm semnalul pe chunks
                    for i in range(0, signal_length, chunk_size):
                        if not self.is_playing:  # Verificăm dacă trebuie să oprim redarea
                            break

                        # Extragem chunk-ul curent
                        end_idx = min(i + chunk_size, signal_length)
                        chunk = self.recording.data[i:end_idx]

                        # Redăm chunk-ul
                        stream.write(chunk)

                        # Actualizăm progresul dacă există callback
                        if self.progress_callback:
                            progress = (i / signal_length) * 100
                            self.progress_callback(progress)

                print("Redare finalizată")
            except Exception as e:
                print(f"Eroare la redare: {e}")
            finally:
                self.is_playing = False
                if self.progress_callback:
                    self.progress_callback(100)

        # Oprim orice redare anterioară
        self.stop_playback()

        # Pornim redarea într-un thread separat
        self.audio_thread = threading.Thread(target=playback_task)
        self.audio_thread.start()

    def stop_playback(self):
        """
        Oprește redarea curentă.
        """
        if self.is_playing:
            self.is_playing = False
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1.0)  # Așteptăm maxim 1 secundă
            print("Redare oprită.")
            return True
        return False

    def _calculate_undo_entry_size(self, recording):
        """
        Calculează dimensiunea unei intrări din stiva de undo în bytes.
        """
        if recording is None or not hasattr(recording, 'data'):
            return 0
        return recording.data.nbytes

    def save_state(self):
        """
        Salvează starea curentă a înregistrării pe stivă.
        Verifică și menține limita de dimensiune a stivei de undo.
        """
        if self.recording is not None:
            new_state = Recording(np.copy(self.recording.data), self.recording.sample_rate)
            entry_size = self._calculate_undo_entry_size(new_state)

            # Verificăm dacă avem suficient spațiu
            if entry_size > self.MAX_CACHE_SIZE:
                print("Avertisment: Înregistrarea este prea mare pentru stiva de undo.")
                return

            # Adăugăm noua stare
            self.undo_stack.append(new_state)
            self.current_undo_size += entry_size

            # Curățăm stiva dacă depășim limita de intrări
            while len(self.undo_stack) > self.MAX_UNDO_STACK_SIZE:
                removed_state = self.undo_stack.pop(0)
                self.current_undo_size -= self._calculate_undo_entry_size(removed_state)

    def get_undo_stack_info(self):
        """
        Returnează informații despre starea curentă a stivei de undo.
        """
        return {
            "current_size": self.current_undo_size,
            "max_size": self.MAX_CACHE_SIZE,
            "entries": len(self.undo_stack),
            "max_entries": self.MAX_UNDO_STACK_SIZE
        }

    def undo(self):
        """
        Revine la ultima stare salvată a înregistrării.
        """
        if self.undo_stack:
            self.recording = self.undo_stack.pop()
            # Actualizăm durata în interfață după undo
            self._update_duration()
            return True
        print("Nu există stări anterioare pentru Undo.")
        return False

    def pitch_shift(self, up=True):
        if self.recording is None:
            print("Nu există înregistrare pentru pitch shift.")
            return None

        # Validăm factorul de pitch
        if not isinstance(self.config.pitch_factor, (int, float)) or self.config.pitch_factor <= 0:
            print("Factorul de pitch trebuie să fie un număr pozitiv.")
            return None

        # Limităm factorul de pitch la valori rezonabile (0.5 - 2.0)
        MIN_PITCH_FACTOR = 0.5
        MAX_PITCH_FACTOR = 2.0
        if not (MIN_PITCH_FACTOR <= self.config.pitch_factor <= MAX_PITCH_FACTOR):
            print(f"Factorul de pitch trebuie să fie între {MIN_PITCH_FACTOR} și {MAX_PITCH_FACTOR}.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        n_steps = 12 * np.log2(self.config.pitch_factor) if up else -12 * np.log2(self.config.pitch_factor)

        def process_pitch_chunk(chunk, sr, n_steps):
            return librosa.effects.pitch_shift(chunk, sr=sr, n_steps=n_steps)

        # Procesăm pe chunks mai mici
        chunk_size = min(1024 * 1024, len(self.recording.data))  # 1MB sau mai puțin
        processed_data = self.process_in_chunks(
            process_pitch_chunk,
            chunk_size=chunk_size,
            n_steps=n_steps
        )

        if processed_data is not None:
            # Normalizăm rezultatul final
            max_val = np.max(np.abs(processed_data))
            if max_val > 0:
                processed_data = processed_data / max_val * 0.95  # Lăsăm un pic de headroom

            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("pitch_shift", {"up": up, "n_steps": n_steps})
            self._update_duration()

        if self.progress_callback:
            self.progress_callback(100)

        return self.recording

    def apply_reverb(self, decay=0.5, delay=0.02, ir_duration=0.5):
        """
        Aplică efectul de reverb pe înregistrare.
        :param decay: Factorul de descreștere (0.1-0.9)
        :param delay: Întârzierea inițială în secunde (0.001-0.1)
        :param ir_duration: Durata răspunsului la impuls în secunde (0.1-2.0)
        """
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea reverbului.")
            return None

        # Validăm parametrii
        if not isinstance(decay, (int, float)) or not isinstance(delay, (int, float)) or not isinstance(ir_duration, (int, float)):
            print("Toți parametrii trebuie să fie numere.")
            return None

        # Limităm parametrii la valori rezonabile
        if not (0.1 <= decay <= 0.9):
            print("Factorul de descreștere trebuie să fie între 0.1 și 0.9.")
            return None

        if not (0.001 <= delay <= 0.1):
            print("Întârzierea inițială trebuie să fie între 0.001 și 0.1 secunde.")
            return None

        if not (0.1 <= ir_duration <= 2.0):
            print("Durata răspunsului la impuls trebuie să fie între 0.1 și 2.0 secunde.")
            return None

        # Verificăm dacă delay-ul nu depășește durata înregistrării
        recording_duration = len(self.recording.data) / self.recording.sample_rate
        if delay >= recording_duration:
            print("Întârzierea inițială nu poate fi mai mare sau egală cu durata înregistrării.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        def process_reverb_chunk(chunk, sr, decay=0.5, delay=0.02, ir_duration=0.5):
            ir_length = int(ir_duration * sr)
            t = np.linspace(0, ir_duration, ir_length)
            ir = np.zeros(ir_length)
            start_idx = int(delay * sr)
            if start_idx < ir_length:
                ir[start_idx] = 1.0
            ir[start_idx:] += decay ** (t[start_idx:] / ir_duration)

            reverb_signal = np.convolve(chunk, ir, mode='same')
            max_val = np.max(np.abs(reverb_signal))
            if max_val > 0:
                reverb_signal = reverb_signal / max_val
            return reverb_signal

        processed_data = self.process_in_chunks(
            process_reverb_chunk,
            decay=decay,
            delay=delay,
            ir_duration=ir_duration
        )

        if processed_data is not None:
            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("Reverb", {"decay": decay, "delay": delay, "ir_duration": ir_duration})
            self._update_duration()

        if self.progress_callback:
            self.progress_callback(100)

        return self.recording

    def apply_echo(self, decay=0.5, delay=0.2, repeats=5):
        """
        Aplică efectul de echo pe înregistrare.
        :param decay: Factorul de descreștere (0.1-0.9)
        :param delay: Întârzierea între repetiții în secunde (0.01-1.0)
        :param repeats: Numărul de repetiții (1-10)
        """
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea efectului de echo.")
            return None

        # Validăm parametrii
        if not isinstance(decay, (int, float)) or not isinstance(delay, (int, float)) or not isinstance(repeats, int):
            print("Toți parametrii trebuie să fie numere.")
            return None

        # Limităm parametrii la valori rezonabile
        if not (0.1 <= decay <= 0.9):
            print("Factorul de descreștere trebuie să fie între 0.1 și 0.9.")
            return None

        if not (0.01 <= delay <= 1.0):
            print("Întârzierea trebuie să fie între 0.01 și 1.0 secunde.")
            return None

        if not (1 <= repeats <= 10):
            print("Numărul de repetiții trebuie să fie între 1 și 10.")
            return None

        # Verificăm dacă delay-ul nu depășește durata înregistrării
        recording_duration = len(self.recording.data) / self.recording.sample_rate
        if delay >= recording_duration:
            print("Întârzierea nu poate fi mai mare sau egală cu durata înregistrării.")
            return None

        # Verificăm dacă efectul total nu ar depăși o durată maximă rezonabilă
        max_echo_duration = recording_duration * 2  # Maximum dublarea duratei
        if delay * repeats > max_echo_duration:
            print(f"Combinația de delay și repeats ar crea un efect prea lung (maxim {max_echo_duration:.1f} secunde).")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        def process_echo_chunk(chunk, sr, decay=0.5, delay=0.2):
            delay_samples = int(delay * sr)
            b = [1] + [0] * delay_samples + [decay]
            a = [1]
            echo_signal = lfilter(b, a, chunk)
            max_val = np.max(np.abs(echo_signal))
            if max_val > 0:
                echo_signal = echo_signal / max_val
            return echo_signal

        processed_data = self.process_in_chunks(
            process_echo_chunk,
            decay=decay,
            delay=delay
        )

        if processed_data is not None:
            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("Echo", {"decay": decay, "delay": delay, "repeats": repeats})
            self._update_duration()

        if self.progress_callback:
            self.progress_callback(100)

        return self.recording

    def apply_equalizer(self, bands):
        if self.recording is None:
            print("Nu există înregistrare pentru egalizator.")
            return None

        # Validăm benzile de frecvență
        if not isinstance(bands, (list, tuple, np.ndarray)) or len(bands) != 10:
            print("Trebuie să furnizați exact 10 benzi de frecvență.")
            return None

        # Validăm valorile pentru fiecare bandă
        MIN_GAIN = -12.0  # -12 dB
        MAX_GAIN = 12.0  # +12 dB
        for i, gain in enumerate(bands):
            if not isinstance(gain, (int, float)):
                print(f"Valoarea pentru banda {i + 1} trebuie să fie un număr.")
                return None
            if not (MIN_GAIN <= gain <= MAX_GAIN):
                print(f"Valoarea pentru banda {i + 1} trebuie să fie între {MIN_GAIN} și {MAX_GAIN} dB.")
                return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        def process_equalizer_chunk(chunk, sr, bands):
            nyquist = sr / 2
            freqs = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
            normalized_freqs = [f / nyquist for f in freqs]

            y_eq = np.copy(chunk)
            for freq, gain_db in zip(normalized_freqs, bands):
                gain = 10 ** (gain_db / 20)
                b, a = scipy.signal.butter(2, [freq * 0.7, freq * 1.3], btype='band')
                filtered = scipy.signal.filtfilt(b, a, chunk)
                y_eq += (filtered * (gain - 1))

            y_eq = np.clip(y_eq, -1, 1)
            y_eq = y_eq / np.max(np.abs(y_eq))
            return y_eq

        processed_data = self.process_in_chunks(
            process_equalizer_chunk,
            bands=bands
        )

        if processed_data is not None:
            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("Equalizer", {"bands": bands})
            self._update_duration()

        if self.progress_callback:
            self.progress_callback(100)

        return self.recording

    def apply_reverb_with_ir(self, ir_name="bathroom.wav"):
        """
        Aplică efectul de reverb folosind un fișier IR.
        :param ir_name: Numele fișierului IR
        :return: Recording sau None în caz de eroare
        """
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea reverbului.")
            return None

        # Obținem calea către fișierul IR
        ir_path = self.device_manager.get_ir_path(ir_name)
        if not ir_path.exists():
            print(f"Fișierul IR {ir_name} nu există în directorul {self.device_manager.ir_base_path}")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        try:
            # Încarcă și pregătește IR-ul
            ir, ir_sr = librosa.load(str(ir_path), sr=None)
            if ir_sr != self.recording.sample_rate:
                ir = librosa.resample(ir, orig_sr=ir_sr, target_sr=self.recording.sample_rate)
            ir = ir / np.max(np.abs(ir))

            processed_data = self.process_in_chunks(
                self.process_reverb_ir_chunk,
                ir=ir
            )

            # Normalizăm rezultatul final
            max_val = np.max(np.abs(processed_data))
            if max_val > 0:
                processed_data = processed_data / max_val * 0.95  # Lăsăm un pic de headroom

            if processed_data is not None:
                self.recording = Recording(processed_data, self.recording.sample_rate)
                self.cache_state("ReverbIR", {"ir_name": ir_name})
                self._update_duration()

            if self.progress_callback:
                self.progress_callback(100)

            return self.recording

        except Exception as e:
            print(f"Eroare la aplicarea reverbului cu IR: {e}")
            return None

    def calculate_spectral_features(self):
        if self.recording is None:
            print("Nu există înregistrare pentru calculul caracteristicilor spectrale.")
            return None

        def analysis_task():
            try:
                y = self.recording.data
                sr = self.recording.sample_rate

                # Verificăm dacă semnalul este suficient de lung
                if len(y) < 2048:  # Minim 2048 eșantioane pentru analiză
                    print("Semnalul este prea scurt pentru analiza spectrală.")
                    return None

                # Calcularea caracteristicilor spectrale
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)

                # Convertim la float pentru a evita probleme de formatare
                return {
                    "Spectral Centroid": float(spectral_centroid),
                    "Spectral Bandwidth": float(spectral_bandwidth),
                    "Spectral Rolloff": float(spectral_rolloff),
                    "Spectral Contrast": [float(v) for v in spectral_contrast],
                }
            except Exception as e:
                print(f"Eroare la calculul caracteristicilor spectrale: {e}")
                return None

        # Rulăm analiza într-un thread separat
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(analysis_task)
            return future.result()

    def analyze_pitch_and_tuning(self):
        if self.recording is None:
            print("Nu există înregistrare pentru analiza pitch-ului.")
            return None

        def analysis_task():
            try:
                y = self.recording.data
                sr = self.recording.sample_rate

                # Verificăm dacă semnalul este suficient de lung
                if len(y) < 2048:  # Minim 2048 eșantioane pentru analiză
                    print("Semnalul este prea scurt pentru analiza pitch-ului.")
                    return None

                # Pitch tracking folosind piptrack
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_fundamental = np.max(pitches[magnitudes > np.median(magnitudes)])

                # Estimarea tuning-ului
                tuning = librosa.estimate_tuning(y=y, sr=sr)

                return {
                    "Pitch Fundamental": float(pitch_fundamental),
                    "Tuning Adjustment": float(tuning),
                }
            except Exception as e:
                print(f"Eroare la analiza pitch-ului și tuning-ului: {e}")
                return None

        # Rulăm analiza într-un thread separat
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(analysis_task)
            return future.result()

    def estimate_bpm(self):
        if self.recording is None:
            return None

        # Verificăm dacă avem deja BPM-ul calculat pentru această înregistrare
        if hasattr(self.recording, 'cached_bpm'):
            return self.recording.cached_bpm

        def analysis_task():
            try:
                y = self.recording.data
                if len(y) < 2048:
                    print("Semnalul este prea scurt pentru estimarea BPM.")
                    return None

                # Ne asigurăm că semnalul este mono
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)

                # Pentru fișiere mari, folosim o metodă mai robustă
                if len(y) > 1000000:  # Pentru fișiere mai mari de ~23 secunde la 44.1kHz
                    # Folosim o fereastră de 30 secunde din mijlocul fișierului
                    window_size = int(30 * self.recording.sample_rate)
                    start = len(y) // 2 - window_size // 2
                    end = start + window_size
                    y = y[start:end]

                # Ajustăm n_fft în funcție de lungimea semnalului
                n_fft = min(2048, len(y))
                hop_length = n_fft // 4

                # Folosim beat_track pentru consistență cu interfața
                tempo, _ = librosa.beat.beat_track(y=y, sr=self.recording.sample_rate,
                                                   hop_length=hop_length, sparse=False)

                # Convertim la float pentru a evita probleme de formatare
                tempo = float(tempo)
                print(f"BPM estimat: {tempo:.2f}")

                # Salvăm BPM-ul în cache
                self.recording.cached_bpm = tempo
                return tempo
            except Exception as e:
                print(f"Eroare la estimarea BPM: {e}")
                return None

        # Rulăm analiza într-un thread separat
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(analysis_task)
            return future.result()

    def generate_spectrogram(self):
        if self.recording is None:
            return None

        def generate_task():
            try:
                y = self.recording.data
                if len(y) < 2048:
                    print("Semnalul este prea scurt pentru generarea spectrogramei.")
                    return None

                # Ne asigurăm că semnalul este mono
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)

                # Ajustăm n_fft în funcție de lungimea semnalului
                n_fft = min(2048, len(y))
                hop_length = n_fft // 4

                # Calculăm spectrograma
                D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

                return {
                    'data': S_db,
                    'x_axis': 'time',
                    'y_axis': 'log',
                    'title': 'Spectrogramă',
                    'colorbar': True,
                    'colorbar_format': '%+2.f dB'
                }
            except Exception as e:
                print(f"Eroare la generarea spectrogramei: {e}")
                return None

        # Rulăm generarea într-un thread separat
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_task)
            return future.result()

    def generate_chroma(self):
        if self.recording is None:
            return None

        def generate_task():
            try:
                y = self.recording.data
                if len(y) < 2048:
                    print("Semnalul este prea scurt pentru generarea chroma.")
                    return None

                # Ne asigurăm că semnalul este mono
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)

                # Ajustăm n_fft în funcție de lungimea semnalului
                n_fft = min(2048, len(y))
                hop_length = n_fft // 4

                # Calculăm chroma
                chroma = librosa.feature.chroma_stft(y=y, sr=self.config.sample_rate,
                                                     n_fft=n_fft, hop_length=hop_length)

                return {
                    'data': chroma,
                    'x_axis': 'time',
                    'y_axis': 'chroma',
                    'title': 'Chroma',
                    'colorbar': True
                }
            except Exception as e:
                print(f"Eroare la generarea chroma: {e}")
                return None

        # Rulăm generarea într-un thread separat
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_task)
            return future.result()

    def generate_mel_spectrogram(self):
        """
        Generează datele pentru Mel Spectrogram.
        """
        if self.recording is None:
            print("Nu există înregistrare pentru Mel Spectrogram.")
            return None, None

        y = self.recording.data
        sr = self.recording.sample_rate

        # Calcularea Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db, sr

    def generate_mfcc(self):
        """
        Generează datele pentru MFCC (Mel Frequency Cepstral Coefficients).
        """
        if self.recording is None:
            print("Nu există înregistrare pentru MFCC.")
            return None, None

        y = self.recording.data
        sr = self.recording.sample_rate

        # Calcularea MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc, sr

    def generate_cqt(self):
        """
        Generează datele pentru Constant-Q Transform (CQT).
        """
        if self.recording is None:
            print("Nu există înregistrare pentru Constant-Q Transform (CQT).")
            return None, None

        y = self.recording.data
        sr = self.recording.sample_rate

        # Calcularea transformării Constant-Q
        cqt = librosa.cqt(y=y, sr=sr)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        return cqt_db, sr

    def process_compressor_chunk(self, chunk, sr, threshold_db=-30.0, ratio=8.0):
        """
        Procesează un chunk pentru efectul de compresor.
        """
        threshold = 10 ** (threshold_db / 20.0)
        out = np.copy(chunk)

        for i in range(len(chunk)):
            if abs(chunk[i]) > threshold:
                sign = np.sign(chunk[i])
                out[i] = sign * (threshold + (abs(chunk[i]) - threshold) / ratio)

        return out

    def apply_simple_compressor(self, threshold_db=-10.0, ratio=2.0, normalize=False):
        if self.recording is None:
            print("Nu există înregistrare pentru compresor.")
            return None

        # Validăm threshold-ul
        MIN_THRESHOLD = -60.0  # -60 dB
        MAX_THRESHOLD = 0.0  # 0 dB
        if not isinstance(threshold_db, (int, float)):
            print("Threshold-ul trebuie să fie un număr.")
            return None
        if not (MIN_THRESHOLD <= threshold_db <= MAX_THRESHOLD):
            print(f"Threshold-ul trebuie să fie între {MIN_THRESHOLD} și {MAX_THRESHOLD} dB.")
            return None

        # Validăm ratio-ul
        MIN_RATIO = 1.0  # 1:1 (no compression)
        MAX_RATIO = 20.0  # 20:1 (heavy compression)
        if not isinstance(ratio, (int, float)):
            print("Ratio-ul trebuie să fie un număr.")
            return None
        if not (MIN_RATIO <= ratio <= MAX_RATIO):
            print(f"Ratio-ul trebuie să fie între {MIN_RATIO} și {MAX_RATIO}.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        processed_data = self.process_in_chunks(
            self.process_compressor_chunk,
            threshold_db=threshold_db,
            ratio=ratio
        )

        if processed_data is not None:
            if normalize:
                max_val = np.max(np.abs(processed_data))
                if max_val > 0:
                    processed_data = processed_data / max_val
            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("Compressor", {"threshold": threshold_db, "ratio": ratio, "normalize": normalize})

        if self.progress_callback:
            self.progress_callback(100)
        self._update_duration()
        return self.recording

    def apply_lowpass_filter(self, cutoff_hz=1000.0, order=5):
        if self.recording is None:
            print("Nu există înregistrare pentru filtrare.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        def process_lpf_chunk(chunk, sr, cutoff_hz=1000.0, order=5):
            nyq = 0.5 * sr
            normal_cutoff = cutoff_hz / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return lfilter(b, a, chunk)

        processed_data = self.process_in_chunks(
            process_lpf_chunk,
            cutoff_hz=cutoff_hz,
            order=order
        )

        if processed_data is not None:
            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("LPF", {"cutoff": cutoff_hz, "order": order})

        if self.progress_callback:
            self.progress_callback(100)
        self._update_duration()
        return self.recording

    def apply_highpass_filter(self, cutoff_hz=1000.0, order=5):
        if self.recording is None:
            print("Nu există înregistrare pentru filtrare.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        def process_hpf_chunk(chunk, sr, cutoff_hz=1000.0, order=5):
            nyq = 0.5 * sr
            normal_cutoff = cutoff_hz / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return lfilter(b, a, chunk)

        processed_data = self.process_in_chunks(
            process_hpf_chunk,
            cutoff_hz=cutoff_hz,
            order=order
        )

        if processed_data is not None:
            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("HPF", {"cutoff": cutoff_hz, "order": order})

        if self.progress_callback:
            self.progress_callback(100)
        self._update_duration()
        return self.recording

    def apply_bandpass_filter(self, lowcut_hz=300.0, highcut_hz=3000.0, order=5):
        if self.recording is None:
            print("Nu există înregistrare pentru filtrare.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        def process_bpf_chunk(chunk, sr, lowcut_hz=300.0, highcut_hz=3000.0, order=5):
            nyq = 0.5 * sr
            low = lowcut_hz / nyq
            high = highcut_hz / nyq
            b, a = butter(order, [low, high], btype='band', analog=False)
            return lfilter(b, a, chunk)

        processed_data = self.process_in_chunks(
            process_bpf_chunk,
            lowcut_hz=lowcut_hz,
            highcut_hz=highcut_hz,
            order=order
        )

        if processed_data is not None:
            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("BPF", {"lowcut": lowcut_hz, "highcut": highcut_hz, "order": order})

        if self.progress_callback:
            self.progress_callback(100)
        self._update_duration()
        return self.recording

    def process_in_chunks(self, process_func, chunk_size=None, **kwargs):
        """
        Procesează audio-ul în chunks pentru a reduce consumul de memorie.
        :param process_func: Funcția de procesare pentru fiecare chunk
        :param chunk_size: Dimensiunea chunk-ului în eșantioane
        :param kwargs: Argumente adiționale pentru funcția de procesare
        :return: Audio-ul procesat
        """
        if self.recording is None:
            return None

        data = self.recording.data
        sr = self.recording.sample_rate

        # Folosim chunk_size din parametri sau din instanță
        chunk_size = chunk_size or self.processing_chunk_size
        total_chunks = (len(data) + chunk_size - 1) // chunk_size

        # Procesăm în paralel
        processed_data = self.processor.process_chunks_parallel(
            data,
            chunk_size,
            process_func,
            progress_callback=self.progress_callback,
            sr=sr,
            **kwargs
        )

        # Actualizăm progresul
        if self.progress_callback:
            self.progress_callback(100)

        return processed_data

    def process_reverb_chunk(self, chunk, sr, decay=0.5, delay=0.02, ir_duration=0.5):
        """
        Procesează un chunk pentru efectul de reverb.
        """
        ir_length = int(ir_duration * sr)
        t = np.linspace(0, ir_duration, ir_length)
        ir = np.zeros(ir_length)
        start_idx = int(delay * sr)
        if start_idx < ir_length:
            ir[start_idx] = 1.0
        ir[start_idx:] += decay ** (t[start_idx:] / ir_duration)

        # Calculăm convoluția
        reverb_signal = np.convolve(chunk, ir, mode='same')  # Folosim mode='same' pentru a păstra lungimea originală

        # Normalizăm semnalul pentru a evita clipping-ul
        max_val = np.max(np.abs(reverb_signal))
        if max_val > 0:
            reverb_signal = reverb_signal / max_val

        return reverb_signal

    def process_echo_chunk(self, chunk, sr, decay=0.5, delay=0.2):
        """
        Procesează un chunk pentru efectul de echo.
        """
        delay_samples = int(delay * sr)
        b = [1] + [0] * delay_samples + [decay]
        a = [1]
        echo_signal = lfilter(b, a, chunk)

        max_val = np.max(np.abs(echo_signal))
        if max_val > 0:
            echo_signal = echo_signal / max_val

        return echo_signal

    def process_equalizer_chunk(self, chunk, sr, bands):
        """
        Procesează un chunk pentru efectul de egalizator.
        """
        nyquist = sr / 2
        freqs = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        normalized_freqs = [f / nyquist for f in freqs]

        y_eq = np.copy(chunk)
        for freq, gain_db in zip(normalized_freqs, bands):
            gain = 10 ** (gain_db / 20)
            b, a = scipy.signal.butter(2, [freq * 0.7, freq * 1.3], btype='band')
            filtered = scipy.signal.filtfilt(b, a, chunk)
            y_eq += (filtered * (gain - 1))

        y_eq = np.clip(y_eq, -1, 1)
        y_eq = y_eq / np.max(np.abs(y_eq))

        return y_eq

    def apply_distortion(self, drive=1.0, tone=0.5, mix=1.0):
        """
        Aplică efectul de distorsiune pe înregistrare.
        :param drive: Factorul de distorsiune (1-10)
        :param tone: Controlul tonului (0-1), unde 0 accentuează frecvențele joase și 1 accentuează frecvențele înalte
        :param mix: Amestecul între semnalul original și cel distorsionat (0-1)
        """
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea distorsiunii.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        def process_distortion_chunk(chunk, sr, drive=1.0, tone=0.5, mix=1.0):
            # Amplificăm semnalul inițial
            amplified = chunk * (drive * 10)  # Mărim factorul de drive pentru mai multă distorsiune

            # Aplicăm hard clipping pentru distorsiune agresivă
            threshold = 0.1  # Scădem threshold-ul pentru mai multă distorsiune
            distorted = np.clip(amplified, -threshold, threshold)

            # Aplicăm soft clipping pentru a adăuga armonici
            distorted = np.tanh(distorted * 3)  # Mărim factorul pentru mai multe armonici

            # Aplicăm controlul tonului
            nyquist = sr / 2

            if tone < 0.5:
                # Pentru frecvențe joase
                cutoff = 300 + (tone * 1200)  # Cutoff între 300Hz și 1500Hz
                b, a = butter(2, cutoff / nyquist, btype='low')
                distorted = lfilter(b, a, distorted)
            else:
                # Pentru frecvențe înalte
                cutoff = 1000 + ((tone - 0.5) * 10000)  # Cutoff între 1000Hz și 11000Hz
                b, a = butter(2, cutoff / nyquist, btype='high')
                distorted = lfilter(b, a, distorted)

            # Amestecăm semnalul original cu cel distorsionat
            output = (1 - mix) * chunk + mix * distorted

            # Normalizăm pentru a evita clipping-ul
            max_val = np.max(np.abs(output))
            if max_val > 0:
                output = output / max_val * 0.95  # Lăsăm un pic de headroom

            return output

        processed_data = self.process_in_chunks(
            process_distortion_chunk,
            drive=drive,
            tone=tone,
            mix=mix
        )

        if processed_data is not None:
            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("Distortion", {"drive": drive, "tone": tone, "mix": mix})

        if self.progress_callback:
            self.progress_callback(100)

        self._update_duration()
        return self.recording

    def apply_reverb_chunked(self, decay=0.5, delay=0.02, ir_duration=0.5):
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea reverbului.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        original_length = len(self.recording.data)
        sr = self.recording.sample_rate

        # Procesăm pe bucăți
        processed_data = self.process_in_chunks(
            self.process_reverb_chunk,
            decay=decay,
            delay=delay,
            ir_duration=ir_duration
        )

        # Limităm lungimea la original + 25%
        max_length = int(original_length * 1.25)
        if len(processed_data) > max_length:
            processed_data = processed_data[:max_length]
            # Fade-out pe ultimele 100ms
            fade_length = int(0.1 * sr)
            fade_out = np.linspace(1, 0, fade_length)
            processed_data[-fade_length:] *= fade_out

        # Normalizăm
        max_val = np.max(np.abs(processed_data))
        if max_val > 0:
            processed_data = processed_data / max_val * 0.95

        self.recording = Recording(processed_data, sr)
        self.cache_state("ReverbChunked", {"decay": decay, "delay": delay, "ir_duration": ir_duration})
        self._update_duration()
        if self.progress_callback:
            self.progress_callback(100)
        return self.recording

    def process_reverb_ir_chunk(self, chunk, sr, ir):
        from scipy.signal import fftconvolve
        # Convoluție pe chunk cu IR
        reverb_signal = fftconvolve(chunk, ir, mode='full')
        # Normalizare locală pentru chunk
        max_val = np.max(np.abs(reverb_signal))
        if max_val > 0:
            reverb_signal = reverb_signal / max_val
        return reverb_signal

    def apply_time_stretch_bpm(self, target_bpm):
        """
        Aplică time stretching pe înregistrare pentru a atinge BPM-ul țintă.
        :param target_bpm: BPM-ul dorit (trebuie să fie între 20 și 300)
        """
        if self.recording is None:
            print("Nu există înregistrare pentru time-stretching.")
            return None

        # Validăm BPM-ul țintă
        if not isinstance(target_bpm, (int, float)) or target_bpm <= 0:
            print("BPM-ul țintă trebuie să fie un număr pozitiv.")
            return None

        # Limităm BPM-ul la valori rezonabile
        MIN_BPM = 20
        MAX_BPM = 300
        if target_bpm < MIN_BPM or target_bpm > MAX_BPM:
            print(f"BPM-ul țintă trebuie să fie între {MIN_BPM} și {MAX_BPM}.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        try:
            y = self.recording.data
            sr = self.recording.sample_rate

            # Ne asigurăm că semnalul este mono
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)

            # Verificăm dacă semnalul este suficient de lung
            if len(y) < 2048:
                print("Semnalul este prea scurt pentru time-stretching.")
                return None

            # Folosim BPM-ul original salvat sau îl estimăm
            if not hasattr(self, 'original_bpm'):
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                self.original_bpm = tempo
                print(f"BPM original estimat: {self.original_bpm:.2f}")

            original_bpm = self.original_bpm
            print("DEBUG BPM:", original_bpm, type(original_bpm))

            if original_bpm is None or original_bpm <= 0:
                print("Nu s-a putut estima BPM-ul original.")
                return None

            # Calculăm rata de stretching
            stretch_rate = target_bpm / original_bpm
            print(f"Original BPM: {original_bpm:.2f}, Țintă BPM: {target_bpm:.2f}, Stretch Rate: {stretch_rate:.3f}")

            # Limităm rata de stretching la valori rezonabile
            MIN_STRETCH = 0.25  # 4x mai lent
            MAX_STRETCH = 4.0  # 4x mai rapid
            if stretch_rate < MIN_STRETCH or stretch_rate > MAX_STRETCH:
                print(
                    f"Rata de stretching ({stretch_rate:.2f}) este în afara limitelor permise ({MIN_STRETCH}-{MAX_STRETCH}).")
                return None

            # Aplicăm time stretching
            y_stretched = librosa.effects.time_stretch(y=y, rate=stretch_rate)

            # Normalizăm semnalul
            max_val = np.max(np.abs(y_stretched))
            if max_val > 0:
                y_stretched = y_stretched / max_val

            # Creăm noua înregistrare
            self.recording = Recording(y_stretched, sr)

            # Forțăm BPM-ul țintă în cache
            self.recording.cached_bpm = target_bpm
            self.recording.original_bpm = target_bpm

            self.cache_state("TimeStretch", {"target_bpm": target_bpm, "stretch_rate": stretch_rate})
            self._update_duration()

            if self.progress_callback:
                self.progress_callback(100)

            print("Time-stretching aplicat cu succes.")
            return self.recording

        except Exception as e:
            print(f"Eroare la aplicarea time-stretching: {e}")
            return None

        except Exception as e:
            print(f"Eroare la aplicarea time-stretching: {e}")
            return None

    def load_audio(self, path: str) -> bool:
        """
        Încarcă un fișier audio în format WAV folosind procesare pe chunks.
        :param path: Calea către fișierul audio
        :return: True dacă încărcarea a reușit, False altfel
        """
        # Verificăm compatibilitatea fișierului
        is_compatible, error_message = self.device_manager.check_file_compatibility(path)
        if not is_compatible:
            print(error_message)
            return False

        try:
            with wave.open(path, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()

                # Mărim dimensiunea chunk-ului la 4MB pentru fișiere mari
                chunk_size = min(4 * 1024 * 1024, n_frames * n_channels * sample_width)
                chunk_frames = chunk_size // (n_channels * sample_width)

                # Inițializăm array-ul pentru datele audio
                audio_data = np.zeros(n_frames, dtype=np.float32)

                # Procesăm fișierul în chunks
                total_chunks = (n_frames + chunk_frames - 1) // chunk_frames
                bytes_read = 0
                total_bytes = n_frames * n_channels * sample_width

                for i in range(total_chunks):
                    # Citim chunk-ul curent
                    frames = wav_file.readframes(chunk_frames)
                    if not frames:
                        break

                    # Actualizăm progresul bazat pe bytes citiți
                    bytes_read += len(frames)
                    if self.progress_callback:
                        progress = (bytes_read / total_bytes) * 100
                        self.progress_callback(progress)

                    # Convertim la numpy array
                    chunk_data = np.frombuffer(frames, dtype=np.int16)

                    # Dacă e stereo, convertim la mono prin mediere
                    if n_channels == 2:
                        chunk_data = chunk_data.reshape(-1, 2).mean(axis=1)

                    # Normalizăm la float32 între -1 și 1
                    chunk_data = chunk_data.astype(np.float32) / 32768.0

                    # Calculăm poziția de start și end pentru acest chunk
                    start_idx = i * chunk_frames
                    end_idx = min(start_idx + len(chunk_data), n_frames)

                    # Salvăm chunk-ul în array-ul principal
                    audio_data[start_idx:end_idx] = chunk_data[:end_idx - start_idx]

                # Creăm înregistrarea
                self.recording = Recording(audio_data, frame_rate)
                # Adăugăm înregistrarea în cache
                self.cache_state("load_audio", {"file_path": path})
                self._update_duration()
                return True

        except Exception as e:
            print(f"Eroare la încărcarea fișierului: {e}")
            return False

    def cleanup(self):
        """
        Curăță resursele folosite de serviciu.
        """
        try:
            # Oprim procesorul audio
            if hasattr(self, 'processor'):
                self.processor.shutdown()

            # Oprim orice înregistrare sau redare în curs
            if self.is_recording:
                self.stop_recording()
            if self.is_playing:
                self.stop_playback()

            # Eliberăm memoria
            self.undo_stack.clear()
            self.cache.clear()
            self.recording = None

            # Curățăm orice fișiere temporare rămase
            temp_files = glob.glob('*.tmp')
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass

        except Exception as e:
            print(f"Eroare la curățarea resurselor: {e}")
            # Continuăm cu curățarea chiar dacă apare o eroare
            try:
                self.undo_stack.clear()
                self.cache.clear()
                self.recording = None
            except:
                pass
