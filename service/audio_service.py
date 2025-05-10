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

from domain.recording import Recording
from domain.audio_processor import AudioProcessor


class AudioService:
    def __init__(self, config):
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
        data = sd.rec(
            int(duration_seconds * self.config.sample_rate),
            samplerate=self.config.sample_rate,
            channels=1,
            dtype='float64',
            device=self.config.input_device,
            blocksize=self.config.buffer_size
        )
        sd.wait()
        self.recording = Recording(data.flatten(), self.config.sample_rate)
        self.cache_state("recording")
        print("Înregistrare finalizată.")

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

    def cache_state(self, effect_name, params=None):
        """
        Salvează starea curentă a înregistrării în cache, cu numele efectului și parametrii.
        """
        if self.recording is not None:
            self.cache.append({
                "data": np.copy(self.recording.data),
                "sample_rate": self.recording.sample_rate,
                "effect": effect_name,
                "params": params if params else {}
            })

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
        """
        if not self.recording:
            print("Nu există înregistrare de salvat.")
            return

        def save_task():
            try:
                self.is_saving = True
                print(f"Salvare înregistrare: {len(self.recording.data) / self.recording.sample_rate:.2f} secunde")
                # Convertim la int16 și salvăm întregul semnal
                audio_data = np.int16(self.recording.data * 32767)
                wav.write(filename, self.recording.sample_rate, audio_data)
                print(f"Înregistrare salvată în {filename}")
            except Exception as e:
                print(f"Eroare la salvarea înregistrării: {e}")
            finally:
                self.is_saving = False

        save_thread = threading.Thread(target=save_task)
        save_thread.start()

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

    def save_state(self):
        """
        Salvează starea curentă a înregistrării pe stivă.
        """
        if self.recording is not None:
            self.undo_stack.append(Recording(np.copy(self.recording.data), self.recording.sample_rate))

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
        self.save_state()  # Salvează starea înainte de aplicarea efectului
        if self.recording:
            n_steps = 12 * np.log2(self.config.pitch_factor) if up else -12 * np.log2(self.config.pitch_factor)
            shifted = librosa.effects.pitch_shift(self.recording.data, sr=self.recording.sample_rate, n_steps=n_steps)
            self.cache_state("pitch_shift", {})
            self._update_duration()
            return Recording(shifted, self.recording.sample_rate)
        else:
            print("Nu există înregistrare pentru pitch shift.")
            return None

    def apply_reverb(self, decay=0.5, delay=0.02, ir_duration=0.5):
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea reverbului.")
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
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea efectului de echo.")
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

    def apply_reverb_with_ir(self, ir_path="C:/Faculta/an_3/Licenta/Licenta_tkinter/IR/bathroom.wav"):
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea reverbului.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        # Încarcă și pregătește IR-ul
        ir, ir_sr = librosa.load(ir_path, sr=None)
        if ir_sr != self.recording.sample_rate:
            ir = librosa.resample(ir, orig_sr=ir_sr, target_sr=self.recording.sample_rate)
        ir = ir / np.max(np.abs(ir))

        def process_reverb_ir_chunk(chunk, sr, ir):
            from scipy.signal import fftconvolve
            # Folosim mode='same' pentru a păstra lungimea originală
            reverb_signal = fftconvolve(chunk, ir, mode='same')
            # Normalizare locală pentru chunk
            max_val = np.max(np.abs(reverb_signal))
            if max_val > 0:
                reverb_signal = reverb_signal / max_val
            return reverb_signal

        # Procesăm pe bucăți mai mici pentru a evita probleme de memorie
        chunk_size = min(1024 * 1024, len(self.recording.data))  # 1MB sau mai puțin
        processed_data = self.process_in_chunks(
            process_reverb_ir_chunk,
            chunk_size=chunk_size,
            ir=ir
        )

        if processed_data is not None:
            # Asigurăm-ne că lungimea este aceeași cu originalul
            if len(processed_data) != len(self.recording.data):
                processed_data = processed_data[:len(self.recording.data)]

            # Normalizăm rezultatul final
            max_val = np.max(np.abs(processed_data))
            if max_val > 0:
                processed_data = processed_data / max_val * 0.95

            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("ReverbIR", {"ir_path": ir_path})
            self._update_duration()

        if self.progress_callback:
            self.progress_callback(100)
        self._update_duration()
        return self.recording

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

                print(f"BPM estimat: {tempo:.2f}")
                return float(tempo)
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

    def apply_reverb_with_ir_chunked(self, ir_path="C:/Faculta/an_3/Licenta/Licenta_tkinter/IR/bathroom.wav"):
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea reverbului.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        data = self.recording.data
        sr = self.recording.sample_rate

        # Încarcă și pregătește IR-ul
        ir, ir_sr = librosa.load(ir_path, sr=None)
        if ir_sr != sr:
            ir = librosa.resample(ir, orig_sr=ir_sr, target_sr=sr)
        ir = ir / np.max(np.abs(ir))

        # Procesăm pe bucăți
        processed_data = self.process_in_chunks(
            self.process_reverb_ir_chunk,
            ir=ir
        )

        # Limităm lungimea la original + 25%
        max_length = int(len(data) * 1.25)
        if len(processed_data) > max_length:
            processed_data = processed_data[:max_length]
            # Fade-out pe ultimele 100ms
            fade_length = int(0.1 * sr)
            fade_out = np.linspace(1, 0, fade_length)
            processed_data[-fade_length:] *= fade_out

        # Normalizăm rezultatul final
        max_val = np.max(np.abs(processed_data))
        if max_val > 0:
            processed_data = processed_data / max_val * 0.95

        self.recording = Recording(processed_data, sr)
        self.cache_state("ReverbIRChunked", {"ir_path": ir_path})
        self._update_duration()
        if self.progress_callback:
            self.progress_callback(100)
        return self.recording

    def apply_time_stretch_bpm(self, target_bpm):
        """
        Aplică time stretching pe înregistrare pentru a atinge BPM-ul țintă.
        :param target_bpm: BPM-ul dorit
        """
        if self.recording is None:
            print("Nu există înregistrare pentru time-stretching.")
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
                # Estimăm BPM-ul original folosind o metodă mai robustă
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
                if isinstance(tempo, np.ndarray):
                    tempo = float(tempo.item())
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

            # Aplicăm time stretching
            y_stretched = librosa.effects.time_stretch(y=y, rate=stretch_rate)

            # Normalizăm semnalul
            max_val = np.max(np.abs(y_stretched))
            if max_val > 0:
                y_stretched = y_stretched / max_val

            # Creăm noua înregistrare
            self.recording = Recording(y_stretched, sr)
            self.cache_state("TimeStretch", {"target_bpm": target_bpm, "stretch_rate": stretch_rate})
            self._update_duration()

            # Actualizăm BPM-ul original pentru următoarea operație
            self.original_bpm = target_bpm

            if self.progress_callback:
                self.progress_callback(100)

            print("Time-stretching aplicat cu succes.")
            return self.recording

        except Exception as e:
            print(f"Eroare la aplicarea time-stretching: {e}")
            return None

    def load_audio(self, path: str) -> bool:
        """
        Încarcă un fișier audio în format WAV folosind procesare pe chunks.
        """
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
                self._update_duration()
                return True

        except Exception as e:
            print(f"Eroare la încărcarea fișierului: {e}")
            return False


    def cleanup(self):
        """
        Curăță resursele folosite de serviciu.
        """
        # Oprim procesorul audio
        if hasattr(self, 'processor'):
            self.processor.shutdown()

        # Eliberăm memoria
        self.undo_stack.clear()
        self.cache.clear()
        self.recording = None






