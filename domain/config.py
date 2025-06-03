class Config:
    def __init__(self, sample_rate=44100, pitch_factor=2.0, save_directory="C:/Faculta/an_3/Licenta/ProiectLicena/recordings", bpm_current=120.0, bpm_target=120.0):
        '''
        Clasa Config instanțiază si centralizează toți parametrii de configurare utilizați în proiect.
        :param sample_rate: rata de eșantionare în Hz (implicit 44100 Hz), stocată ca un int.
        :param pitch_factor:indicele de modificare a tonalității, stocat ca un float.
        :param save_directory: reține printr-un string, calea unde sunt salvate înregistrările procesate.
        :param bpm_current și bpm_target: valori care reprezintă tempo-ul curent al înregistrării
        și tempo-ul dorit dupa ajustările utilizatorului  în interfață.
        :param: input_device, output_device și buffer_size: parametrii specifici pentru coordonarea dispozitivelor
        de captare și redare, puse la dispoziție de placa de sunet internă a dispozitivului de rulare.
        '''
        self.sample_rate = 44100
        self.save_directory = ""
        self.pitch_factor = 2.0
        self.input_device = None
        self.output_device = None
        self.buffer_size = 1024
