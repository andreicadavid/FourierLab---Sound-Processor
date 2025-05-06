class Config:
    def __init__(self, sample_rate=44100, pitch_factor=2.0, save_directory="C:/Faculta/an_3/Licenta/ProiectLicena/recordings", bpm_current=120.0, bpm_target=120.0):
        """
        Configurație generală pentru aplicație.

        :param sample_rate: rata de eșantionare (Hz)
        :param pitch_factor: factor de modificare a pitch-ului (ex: 2.0 = o octavă mai sus)
        :param save_directory: calea implicită pentru salvarea fișierelor
        :param bpm_current: BPM-ul curent al înregistrării
        :param bpm_target: BPM-ul țintă pentru time-stretching
        """
        self.sample_rate = sample_rate
        self.pitch_factor = pitch_factor
        self.save_directory = save_directory
        self.bpm_current = bpm_current  # BPM-ul curent
        self.bpm_target = bpm_target  # BPM-ul țintă
