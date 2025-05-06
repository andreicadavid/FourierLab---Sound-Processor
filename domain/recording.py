# /audio_app/domain/recording.py

class Recording:
    def __init__(self, data, sample_rate):
        """
        Reprezintă o înregistrare audio.

        :param data: numpy array cu semnalul audio
        :param sample_rate: rata de eșantionare (Hz)
        """
        self.data = data
        self.sample_rate = sample_rate
