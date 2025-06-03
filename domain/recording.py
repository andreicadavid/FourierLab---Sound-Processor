# /audio_app/domain/recording.py

class Recording:
    def __init__(self, data, sample_rate):
        """
        Clasa Recording surprinde caracteristicile de bază care
        constituie reprezentarea unei înregistrări audio.

        :param data: un vector de tip numpy array care stochează valorile numerice
        discretizate ale semnalului audio digitalizat
        :param sample_rate: rata de eșantionare (Hz)
        """
        self.data = data
        self.sample_rate = sample_rate
