import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from domain.recording import Recording


class AudioService:
    def __init__(self, config):
        self.undo_stack = []
        self.config = config
        self.recording = None

    def record(self, duration_seconds):
        print("Începere înregistrare...")
        data = sd.rec(int(duration_seconds * self.config.sample_rate), samplerate=self.config.sample_rate, channels=1,
                      dtype='float64')
        sd.wait()
        self.recording = Recording(data.flatten(), self.config.sample_rate)
        print("Înregistrare finalizată.")


    def save_recording(self, filename):
        if self.recording:
            wav.write(filename, self.recording.sample_rate, np.int16(self.recording.data * 32767))
            print(f"Înregistrare salvată în {filename}")
        else:
            print("Nu există înregistrare de salvat.")

    def play(self):
        if self.recording:
            sd.play(self.recording.data, self.recording.sample_rate)
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
            return Recording(shifted, self.recording.sample_rate)
        else:
            print("Nu există înregistrare pentru pitch shift.")
            return None

    def apply_reverb(self, decay=0.5, delay=0.02, ir_duration=0.5):
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea reverbului.")
            return None

        self.save_state()  # Salvează starea înainte de aplicarea efectului
        sr = self.recording.sample_rate
        data = self.recording.data

        ir_length = int(ir_duration * sr)
        t = np.linspace(0, ir_duration, ir_length)
        ir = np.zeros(ir_length)
        start_idx = int(delay * sr)
        if start_idx < ir_length:
            ir[start_idx] = 1.0
        ir[start_idx:] += decay ** (t[start_idx:] / ir_duration)

        convolved = np.convolve(data, ir, mode='full')
        reverb_signal = convolved[:len(data)]

        max_val = np.max(np.abs(reverb_signal))
        if max_val > 0:
            reverb_signal = reverb_signal / max_val

        self.recording = Recording(reverb_signal, sr)
        return self.recording

    def apply_echo(self, decay=0.5, delay=0.2):
        """
        Aplică efectul de echo folosind scipy.signal.lfilter.
        :param decay: Factor de atenuare pentru semnalul întârziat (0-1).
        :param delay: Întârzierea semnalului în secunde.
        """
        if self.recording is None:
            print("Nu există înregistrare pentru aplicarea efectului de echo.")
            return None

        self.save_state()  # Salvează starea înainte de aplicarea efectului
        sr = self.recording.sample_rate
        data = self.recording.data

        # Calcularea numărului de eșantioane pentru delay
        delay_samples = int(delay * sr)

        # Coeficienții filtrului
        b = [1] + [0] * delay_samples + [decay]  # Numerator
        a = [1]  # Denominator

        # Aplicarea filtrului pentru a genera efectul de echo
        echo_signal = lfilter(b, a, data)

        # Normalizare (pentru a preveni clipping-ul)
        max_val = np.max(np.abs(echo_signal))
        if max_val > 0:
            echo_signal = echo_signal / max_val

        # Actualizarea înregistrării
        self.recording = Recording(echo_signal, sr)
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
        """
        Aplică time-stretching pe baza BPM-ului țintă introdus.
        :param target_bpm: BPM‑ul la care vrem să ajungem.
        """
        if self.recording is None:
            print("Nu există înregistrare pentru time-stretching.")
            return

        # Estimarea BPM-ului original
        original_bpm = self.estimate_bpm()
        if original_bpm is None or original_bpm <= 0:
            print("Nu s-a putut estima BPM-ul original.")
            return

        # Calcularea rate-ului de time-stretch
        stretch_rate = target_bpm / original_bpm
        print(f"Original BPM: {original_bpm:.2f}, Țintă BPM: {target_bpm:.2f}, Stretch Rate: {stretch_rate:.3f}")

        try:
            # Pas 1: Obține spectrograma
            stft = librosa.stft(self.recording.data)

            # Pas 2: Aplică time-stretching pe spectrogramă
            stretched_stft = librosa.effects.time_stretch(stft, stretch_rate)

            # Pas 3: Reconstruiește semnalul audio în domeniul temporal
            y_stretched = librosa.istft(stretched_stft)

            # Normalizare opțională
            max_val = np.max(np.abs(y_stretched))
            if max_val > 0:
                y_stretched = y_stretched / max_val

            # Actualizarea înregistrării
            self.recording = Recording(y_stretched, self.config.sample_rate)
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



