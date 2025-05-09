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

from domain.recording import Recording


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
        if self.recording:
            print(f"Salvare înregistrare: {len(self.recording.data) / self.recording.sample_rate:.2f} secunde")
            # Convertim la int16 și salvăm întregul semnal
            audio_data = np.int16(self.recording.data * 32767)
            wav.write(filename, self.recording.sample_rate, audio_data)
            print(f"Înregistrare salvată în {filename}")
        else:
            print("Nu există înregistrare de salvat.")

    def play(self):
        if self.recording:
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

            # Redăm semnalul
            sd.play(
                self.recording.data,
                self.recording.sample_rate,
                device=self.config.output_device,
                blocksize=self.config.buffer_size
            )

            # Așteptăm până când redarea este completă
            sd.wait()

            print("Redare finalizată")
        else:
            print("Nu există înregistrare pentru redare.")

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

        # Procesăm întregul semnal ca un singur chunk
        y = self.recording.data
        sr = self.recording.sample_rate

        print(f"Procesare reverb:")
        print(f"- Lungime semnal original: {len(y)} eșantioane ({len(y) / sr:.2f} secunde)")

        # Generăm IR-ul
        ir_length = int(ir_duration * sr)
        t = np.linspace(0, ir_duration, ir_length)
        ir = np.zeros(ir_length)
        start_idx = int(delay * sr)
        if start_idx < ir_length:
            ir[start_idx] = 1.0
        ir[start_idx:] += decay ** (t[start_idx:] / ir_duration)

        print(f"- Lungime IR: {len(ir)} eșantioane ({len(ir) / sr:.2f} secunde)")

        # Aplicăm convoluția pe întregul semnal
        from scipy.signal import fftconvolve
        reverb_signal = fftconvolve(y, ir, mode='full')

        print(f"- Lungime semnal cu reverb: {len(reverb_signal)} eșantioane ({len(reverb_signal) / sr:.2f} secunde)")

        # Normalizăm semnalul
        max_val = np.max(np.abs(reverb_signal))
        if max_val > 0:
            reverb_signal = reverb_signal / max_val * 0.95

        # Creăm o nouă înregistrare cu semnalul complet
        self.recording = Recording(reverb_signal, sr)
        self.cache_state("Reverb", {"decay": decay, "delay": delay, "ir_duration": ir_duration})

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

        sr = self.recording.sample_rate
        data = self.recording.data
        delay_samples = int(delay * sr)
        output = np.copy(data)

        for i in range(1, repeats + 1):
            start = delay_samples * i
            if start < len(data):
                output[start:] += data[:-start] * (decay ** i)
            if self.progress_callback:
                self.progress_callback(int(80 * i / repeats))

        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95

        if self.progress_callback:
            self.progress_callback(90)

        self.recording = Recording(output, sr)

        if self.progress_callback:
            self.progress_callback(100)

        return self.recording

    def apply_distortion(self, drive=1.0, tone=0.5, mix=0.5):
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea distorsiunii.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        y = self.recording.data
        sr = self.recording.sample_rate

        y = y / np.max(np.abs(y))
        if self.progress_callback:
            self.progress_callback(30)

        y_driven = y * drive
        y_distorted = np.tanh(y_driven)

        if self.progress_callback:
            self.progress_callback(60)

        if tone < 1.0:
            cutoff = int(tone * sr / 2)
            b, a = scipy.signal.butter(4, cutoff / (sr / 2), btype='low')
            y_distorted = scipy.signal.filtfilt(b, a, y_distorted)

        if self.progress_callback:
            self.progress_callback(80)

        y_mixed = (1 - mix) * y + mix * y_distorted
        y_mixed = y_mixed / np.max(np.abs(y_mixed))

        self.recording = Recording(y_mixed, sr)

        if self.progress_callback:
            self.progress_callback(100)

        return self.recording

    def apply_equalizer(self, bands):
        if self.recording is None:
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        y = self.recording.data
        sr = self.recording.sample_rate

        nyquist = sr / 2
        freqs = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        normalized_freqs = [f / nyquist for f in freqs]

        y_eq = np.copy(y)
        for i, (freq, gain_db) in enumerate(zip(normalized_freqs, bands)):
            gain = 10 ** (gain_db / 20)
            b, a = scipy.signal.butter(2, [freq * 0.7, freq * 1.3], btype='band')
            filtered = scipy.signal.filtfilt(b, a, y)
            y_eq += (filtered * (gain - 1))
            if self.progress_callback:
                self.progress_callback(int(80 * (i + 1) / len(bands)))

        y_eq = np.clip(y_eq, -1, 1)
        y_eq = y_eq / np.max(np.abs(y_eq))

        if self.progress_callback:
            self.progress_callback(90)

        self.recording = Recording(y_eq, sr)

        if self.progress_callback:
            self.progress_callback(100)

        return self.recording

    def calculate_spectral_features(self):
        if self.recording is None:
            print("Nu există înregistrare pentru calculul caracteristicilor spectrale.")
            return None

        try:
            y = self.recording.data
            sr = self.recording.sample_rate

            # Calcularea caracteristicilor spectrale
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)

            # Returnăm rezultatele într-un dicționar
            return {
                "Spectral Centroid": spectral_centroid,
                "Spectral Bandwidth": spectral_bandwidth,
                "Spectral Rolloff": spectral_rolloff,
                "Spectral Contrast": spectral_contrast.tolist(),  # Convertim la listă pentru ușurință
            }
        except Exception as e:
            print(f"Eroare la calculul caracteristicilor spectrale: {e}")
            return None

    def analyze_pitch_and_tuning(self):
        """
        Detectează pitch-ul fundamental și estimează tuning-ul curent.
        :return: Dicționar cu pitch-ul detectat și ajustarea tuning-ului.
        """
        if self.recording is None:
            print("Nu există înregistrare pentru analiza pitch-ului.")
            return None

        try:
            y = self.recording.data
            sr = self.recording.sample_rate

            # Pitch tracking folosind piptrack
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_fundamental = np.max(pitches[magnitudes > np.median(magnitudes)])  # Selectăm pitch-ul dominant

            # Estimarea tuning-ului
            tuning = librosa.estimate_tuning(y=y, sr=sr)

            return {
                "Pitch Fundamental": pitch_fundamental,
                "Tuning Adjustment": tuning,
            }
        except Exception as e:
            print(f"Eroare la analiza pitch-ului și tuning-ului: {e}")
            return None

    def generate_spectrogram(self):
        if self.recording:
            y = self.recording
            fig, ax = plt.subplots()
            D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
            librosa.display.specshow(D, ax=ax, x_axis='time', y_axis='log')
            ax.set_title("Spectrogramă")
            plt.show()

    def generate_chroma(self):
        if self.recording:
            y = self.recording
            chroma = librosa.feature.chroma_stft(y=y, sr=self.config.sample_rate)
            fig, ax = plt.subplots()
            librosa.display.specshow(chroma, ax=ax, x_axis='time', y_axis='chroma')
            ax.set_title("Chroma")
            plt.show()

    def estimate_bpm(self):
        if self.recording is None:
            return None
        tempo, _ = librosa.beat.beat_track(y=self.recording.data, sr=self.config.sample_rate)
        return float(tempo)

    def apply_time_stretch_bpm(self, target_bpm):
        if self.recording is None:
            print("Nu există înregistrare pentru time-stretching.")
            return
        self.save_state()
        y = self.recording.data
        sr = self.recording.sample_rate
        original_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Folosim BPM-ul țintă anterior ca BPM original dacă există
        if hasattr(self, 'last_target_bpm'):
            original_bpm = self.last_target_bpm
        else:
            original_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(original_bpm, np.ndarray):
                original_bpm = float(original_bpm.item())

        print("DEBUG BPM:", original_bpm, type(original_bpm))

        if original_bpm is None or original_bpm <= 0:
            print("Nu s-a putut estima BPM-ul original.")
            return

        stretch_rate = target_bpm / original_bpm
        print(f"Original BPM: {original_bpm:.2f}, Țintă BPM: {target_bpm:.2f}, Stretch Rate: {stretch_rate:.3f}")

        try:
            y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)
            max_val = np.max(np.abs(y_stretched))
            if max_val > 0:
                y_stretched = y_stretched / max_val
            self.recording = Recording(y_stretched, sr)
            self.cache_state("Stretch Rate", stretch_rate)
            # Salvăm BPM-ul țintă curent pentru următorul stretch
            self.last_target_bpm = target_bpm
            print("Time-stretching aplicat cu succes.")
        except Exception as e:
            print(f"Eroare la aplicarea time-stretching: {e}")

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
                return True

        except Exception as e:
            print(f"Eroare la încărcarea fișierului: {e}")
            return False

    def detect_onsets(self):
        if self.recording is None:
            return []

        onset_frames = librosa.onset.onset_detect(y=self.recording, sr=self.config.sample_rate)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.config.sample_rate)
        return onset_times

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

        return self.recording

    def apply_lowpass_filter(self, cutoff_hz=1000.0, order=5):
        """
        Aplică un filtru trece-jos (LPF) pe semnalul curent.
        :param cutoff_hz: Frecvența de tăiere (Hz)
        :param order: Ordinul filtrului
        """
        if self.recording is None:
            print("Nu există înregistrare pentru filtrare.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        y = self.recording.data
        sr = self.recording.sample_rate

        nyq = 0.5 * sr
        normal_cutoff = cutoff_hz / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        if self.progress_callback:
            self.progress_callback(40)

        filtered = lfilter(b, a, y)

        if self.progress_callback:
            self.progress_callback(80)

        self.recording = Recording(filtered, sr)

        if self.progress_callback:
            self.progress_callback(100)

        return self.recording

    def apply_highpass_filter(self, cutoff_hz=1000.0, order=5):
        """
        Aplică un filtru trece-sus (HPF) pe semnalul curent.
        :param cutoff_hz: Frecvența de tăiere (Hz)
        :param order: Ordinul filtrului
        """
        if self.recording is None:
            print("Nu există înregistrare pentru filtrare.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        y = self.recording.data
        sr = self.recording.sample_rate

        nyq = 0.5 * sr
        normal_cutoff = cutoff_hz / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)

        if self.progress_callback:
            self.progress_callback(40)

        filtered = lfilter(b, a, y)

        if self.progress_callback:
            self.progress_callback(80)

        self.recording = Recording(filtered, sr)

        if self.progress_callback:
            self.progress_callback(100)

        return self.recording

    def apply_bandpass_filter(self, lowcut_hz=300.0, highcut_hz=3000.0, order=5):
        """
        Aplică un filtru trece-bandă (BPF) pe semnalul curent.
        :param lowcut_hz: Frecvența de tăiere inferioară (Hz)
        :param highcut_hz: Frecvența de tăiere superioară (Hz)
        :param order: Ordinul filtrului
        """
        if self.recording is None:
            print("Nu există înregistrare pentru filtrare.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        y = self.recording.data
        sr = self.recording.sample_rate

        nyq = 0.5 * sr
        low = lowcut_hz / nyq
        high = highcut_hz / nyq
        b, a = butter(order, [low, high], btype='band', analog=False)

        if self.progress_callback:
            self.progress_callback(40)

        filtered = lfilter(b, a, y)

        if self.progress_callback:
            self.progress_callback(80)

        self.recording = Recording(filtered, sr)

        if self.progress_callback:
            self.progress_callback(100)

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
        processed_chunks = []

        # Folosim chunk_size din parametri sau din instanță
        chunk_size = chunk_size or self.processing_chunk_size
        total_chunks = (len(data) + chunk_size - 1) // chunk_size

        # Procesăm în chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            processed_chunk = process_func(chunk, sr, **kwargs)
            processed_chunks.append(processed_chunk)

            # Actualizăm progresul
            if self.progress_callback:
                progress = (i // chunk_size + 1) / total_chunks * 100
                self.progress_callback(progress)

        # Combinăm rezultatele
        return np.concatenate(processed_chunks)

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

    def apply_echo(self, decay=0.5, delay=0.2, repeats=5):
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea efectului de echo.")
            return None

        self.save_state()
        processed_data = self.process_in_chunks(
            self.process_echo_chunk,
            decay=decay,
            delay=delay
        )

        if processed_data is not None:
            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("Echo", {"decay": decay, "delay": delay, "repeats": repeats})
        return self.recording

    def apply_equalizer(self, bands):
        if self.recording is None:
            return None

        self.save_state()
        processed_data = self.process_in_chunks(
            self.process_equalizer_chunk,
            bands=bands
        )

        if processed_data is not None:
            self.recording = Recording(processed_data, self.recording.sample_rate)
            self.cache_state("Equalizer", {"bands": bands})
        return self.recording

    def set_processing_chunk_size(self, size_bytes):
        """
        Setează dimensiunea chunk-ului pentru procesare.
        :param size_bytes: Dimensiunea în bytes
        """
        self.processing_chunk_size = size_bytes

    def apply_reverb_with_ir(self, ir_path="C:\Faculta/an_3/Licenta/Licenta_tkinter/IR/bathroom.wav"):
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea reverbului.")
            return None

        self.save_state()
        if self.progress_callback:
            self.progress_callback(0)

        data = self.recording.data
        sr = self.recording.sample_rate

        print(f"Procesare reverb cu IR:")
        print(f"- Lungime semnal original: {len(data)} eșantioane ({len(data) / sr:.2f} secunde)")

        # Încarcă IR-ul
        if self.progress_callback:
            self.progress_callback(20)

        ir, ir_sr = librosa.load(ir_path, sr=None)
        if ir_sr != sr:
            ir = librosa.resample(ir, orig_sr=ir_sr, target_sr=sr)
        ir = ir / np.max(np.abs(ir))

        print(f"- Lungime IR: {len(ir)} eșantioane ({len(ir) / sr:.2f} secunde)")

        if self.progress_callback:
            self.progress_callback(40)

        from scipy.signal import fftconvolve
        # Folosim mode='full' pentru a păstra întreaga durată a semnalului convoluat
        reverb_signal = fftconvolve(data, ir, mode='full')

        # Calculăm lungimea maximă dorită (original + 25%)
        max_length = int(len(data) * 1.25)
        if len(reverb_signal) > max_length:
            # Tăiem semnalul la lungimea maximă dorită
            reverb_signal = reverb_signal[:max_length]
            # Aplicăm un fade out pentru a evita tăierea bruscă
            fade_length = int(0.1 * sr)  # 100ms fade out
            fade_out = np.linspace(1, 0, fade_length)
            reverb_signal[-fade_length:] *= fade_out

        print(f"- Lungime semnal cu reverb: {len(reverb_signal)} eșantioane ({len(reverb_signal) / sr:.2f} secunde)")

        if self.progress_callback:
            self.progress_callback(80)

        # Normalizăm semnalul pentru a evita clipping-ul
        max_val = np.max(np.abs(reverb_signal))
        if max_val > 0:
            reverb_signal = reverb_signal / max_val * 0.95

        # Creăm o nouă înregistrare cu semnalul complet
        self.recording = Recording(reverb_signal, sr)

        # Actualizăm durata în interfață
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






