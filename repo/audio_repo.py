# /audio_app/repository/audio_repository.py

import scipy.io.wavfile as wav
import numpy as np
from domain.recording import Recording

class AudioRepository:
    @staticmethod
    def load(filename):
        sr, data = wav.read(filename)
        data = data.astype(np.float64) / 32768.0
        return Recording(data.flatten(), sr)

    @staticmethod
    def save(recording, filename):
        wav.write(filename, recording.sample_rate, np.int16(recording.data * 32767))
