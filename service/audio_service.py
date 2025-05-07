import numpy as np
import scipy
import sounddevice as sd
import librosa
import librosa.display
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter
import math

from domain.recording import Recording


class AudioService:
    def __init__(self, config):
        self.undo_stack = []
        self.config = config
        self.recording = None
        self.cache = []

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
            wav.write(filename, self.recording.sample_rate, np.int16(self.recording.data * 32767))
            print(f"Înregistrare salvată în {filename}")
        else:
            print("Nu există înregistrare de salvat.")

    def play(self):
        if self.recording:
            sd.play(
                self.recording.data,
                self.recording.sample_rate,
                device=self.config.output_device,
                blocksize=self.config.buffer_size
            )
            sd.wait()
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

    def apply_reverb_with_ir(self, ir_path="C:\Faculta/an_3/Licenta/Licenta_tkinter/IR/bathroom.wav"):
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea reverbului.")
            return None

        self.save_state()
        data = self.recording.data
        sr = self.recording.sample_rate

        # Încarcă IR-ul
        import librosa
        ir, ir_sr = librosa.load(ir_path, sr=None)
        if ir_sr != sr:
            ir = librosa.resample(ir, orig_sr=ir_sr, target_sr=sr)
        ir = ir / np.max(np.abs(ir))

        from scipy.signal import fftconvolve
        convolved = fftconvolve(data, ir, mode='full')
        reverb_signal = convolved[:len(data)]

        max_val = np.max(np.abs(reverb_signal))
        if max_val > 0:
            reverb_signal = reverb_signal / max_val * 0.95

        self.recording = Recording(reverb_signal, sr)
        self.cache_state("Reverb (IR)", {"ir_file": ir_path})
        return self.recording

    def apply_echo(self, decay=0.5, delay=0.2, repeats=5):
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea efectului de echo.")
            return None

        self.save_state()
        sr = self.recording.sample_rate
        data = self.recording.data
        delay_samples = int(delay * sr)
        output = np.copy(data)

        for i in range(1, repeats + 1):
            start = delay_samples * i
            if start < len(data):
                output[start:] += data[:-start] * (decay ** i)

        # Normalizează pentru a evita clipping-ul
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95

        self.recording = Recording(output, sr)
        self.cache_state("Echo", {"decay": decay, "delay": delay, "repeats": repeats})
        return self.recording

    def apply_distortion(self, drive=1.0, tone=0.5, mix=0.5):
        """
        Aplică efect de distorsiune cu parametri ajustabili.
        :param drive: Nivelul de distorsiune (1.0 - 10.0)
        :param tone: Controlul tonului (0.0 - 1.0)
        :param mix: Amestecul între semnalul original și cel distorsionat (0.0 - 1.0)
        """
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea distorsiunii.")
            return None

        self.save_state()
        y = self.recording.data
        sr = self.recording.sample_rate

        # Normalizăm semnalul
        y = y / np.max(np.abs(y))

        # Aplicăm drive
        y_driven = y * drive

        # Aplicăm distorsiune (soft clipping)
        y_distorted = np.tanh(y_driven)

        # Aplicăm control de ton (filtru trece-jos)
        if tone < 1.0:
            cutoff = int(tone * sr / 2)
            b, a = scipy.signal.butter(4, cutoff / (sr / 2), btype='low')
            y_distorted = scipy.signal.filtfilt(b, a, y_distorted)

        # Amestecăm semnalul original cu cel distorsionat
        y_mixed = (1 - mix) * y + mix * y_distorted

        # Normalizăm rezultatul final
        y_mixed = y_mixed / np.max(np.abs(y_mixed))

        self.recording = Recording(y_mixed, sr)
        self.cache_state("Distortion", {"drive": drive, "tone": tone, "mix": mix})
        return self.recording

    def apply_equalizer(self, bands):
        """
        Aplică un egalizator cu 10 benzi.
        :param bands: Lista de 10 valori de gain în dB pentru fiecare bandă
        """
        if self.recording is None:
            return None

        print("\n=== Equalizer Debug Info ===")
        print(f"Sample Rate: {self.recording.sample_rate} Hz")
        print(f"Signal Length: {len(self.recording.data)} samples")

        y = self.recording.data
        sr = self.recording.sample_rate

        # Normalizăm frecvențele în raport cu frecvența Nyquist
        nyquist = sr / 2
        freqs = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        normalized_freqs = [f / nyquist for f in freqs]

        print("\nFrecvențe și gain-uri:")
        for freq, norm_freq, gain_db in zip(freqs, normalized_freqs, bands):
            print(f"Frecvență: {freq} Hz, Normalizată: {norm_freq:.4f}, Gain: {gain_db:.1f} dB")

        # Aplicăm fiecare bandă
        y_eq = np.copy(y)
        for i, (freq, gain_db) in enumerate(zip(normalized_freqs, bands)):
            # Convertim gain-ul din dB în factor de multiplicare
            gain = 10 ** (gain_db / 20)
            print(f"\nProcesare bandă {i + 1}:")
            print(f"Frecvență normalizată: {freq:.4f}")
            print(f"Gain (dB): {gain_db:.1f}")
            print(f"Factor multiplicare: {gain:.4f}")

            # Creăm un filtru trece-bandă pentru fiecare frecvență
            b, a = scipy.signal.butter(2, [freq * 0.7, freq * 1.3], btype='band')
            filtered = scipy.signal.filtfilt(b, a, y)

            # Aplicăm gain-ul și adăugăm la semnalul rezultat
            y_eq += (filtered * (gain - 1))

        # Normalizăm rezultatul
        y_eq = np.clip(y_eq, -1, 1)
        y_eq = y_eq / np.max(np.abs(y_eq))

        print("\nStatistici semnal final:")
        print(f"Min: {np.min(y_eq):.4f}")
        print(f"Max: {np.max(y_eq):.4f}")
        print(f"RMS: {np.sqrt(np.mean(y_eq ** 2)):.4f}")
        print("========================\n")

        # Salvăm starea anterioară
        self.save_state()

        # Actualizăm înregistrarea
        self.recording.data = y_eq
        self.cache_state("Equalizer", {"bands": bands})
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

    def load_audio(self, path):
        self.recording, self.config.sample_rate = librosa.load(path, sr=None)

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

    def apply_simple_compressor(self, threshold_db=-30.0, ratio=8.0, normalize=False):
        """
        Compresor audio simplu, fără attack/release.
        :param threshold_db: Pragul de la care începe compresia (în dB, ex: -10.0)
        :param ratio: Raportul de compresie (ex: 4.0)
        :param normalize: Normalizează semnalul după compresie (True/False)
        """
        if self.recording is None:
            print("Nu există înregistrare pentru compresor.")
            return None, None

        y = self.recording.data
        sr = self.recording.sample_rate

        # Pragul în amplitudine
        threshold = 10 ** (threshold_db / 20.0)

        out = np.copy(y)
        for i in range(len(y)):
            if abs(y[i]) > threshold:
                # Aplica compresia doar peste prag
                sign = np.sign(y[i])
                out[i] = sign * (threshold + (abs(y[i]) - threshold) / ratio)
            # altfel, semnalul rămâne neschimbat

        if normalize:
            max_val = np.max(np.abs(out))
            if max_val > 0:
                out = out / max_val

        self.recording = Recording(out, sr)
        self.cache_state("Compressor", {"treshold": threshold_db, "ratio": ratio, "normalize": normalize})
        return self.recording, out

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
        y = self.recording.data
        sr = self.recording.sample_rate

        nyq = 0.5 * sr
        normal_cutoff = cutoff_hz / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered = lfilter(b, a, y)
        self.recording = Recording(filtered, sr)
        self.cache_state("Filter", {"cutoff_hz": cutoff_hz, "order": order})
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
        y = self.recording.data
        sr = self.recording.sample_rate

        nyq = 0.5 * sr
        normal_cutoff = cutoff_hz / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered = lfilter(b, a, y)
        self.recording = Recording(filtered, sr)
        self.cache_state("Filter", {"cutoff_hz": cutoff_hz, "order": order})
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
        y = self.recording.data
        sr = self.recording.sample_rate

        nyq = 0.5 * sr
        low = lowcut_hz / nyq
        high = highcut_hz / nyq
        b, a = butter(order, [low, high], btype='band', analog=False)
        filtered = lfilter(b, a, y)
        self.recording = Recording(filtered, sr)
        self.cache_state("Filter", {"lowcut": low, "highcut": high, "order": order})
        return self.recording




