class Config:
    def __init__(self, sample_rate=44100, pitch_factor=2.0, save_directory="C:/Faculta/an_3/Licenta/ProiectLicena/recordings", bpm_current=120.0, bpm_target=120.0):
        self.sample_rate = 44100
        self.save_directory = ""
        self.pitch_factor = 2.0
        self.input_device = None
        self.output_device = None
        self.buffer_size = 1024
