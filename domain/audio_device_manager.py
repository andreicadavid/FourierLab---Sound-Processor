import os
import wave
import platform
from pathlib import Path
import sounddevice as sd


class AudioDeviceManager:
    """
    Gestionează dispozitivele audio și compatibilitatea lor.
    """

    def __init__(self):
        self.system = platform.system()
        self.supported_formats = ['.wav', '.mp3', '.ogg', '.flac']
        self.ir_paths = {
            'Windows': 'IR',
            'Linux': 'IR',
            'Darwin': 'IR'  # macOS
        }
        self._init_ir_paths()

    def _init_ir_paths(self):
        """
        Inițializează căile către fișierele IR în funcție de sistemul de operare.
        """
        base_dir = Path(__file__).parent.parent
        self.ir_base_path = base_dir / self.ir_paths[self.system]

        # Creăm directorul IR dacă nu există
        if not self.ir_base_path.exists():
            self.ir_base_path.mkdir(parents=True)
            print(f"Directorul IR creat la: {self.ir_base_path}")

    def get_ir_path(self, ir_name):
        """
        Returnează calea completă către un fișier IR.
        :param ir_name: Numele fișierului IR (ex: 'bathroom.wav')
        :return: Path object pentru fișierul IR
        """
        return self.ir_base_path / ir_name

    def get_available_ir_files(self):
        """
        Returnează lista fișierelor IR disponibile.
        :return: Lista de fișiere IR
        """
        if not self.ir_base_path.exists():
            return []
        return [f.name for f in self.ir_base_path.glob('*.wav')]

    def check_audio_devices(self):
        """
        Verifică disponibilitatea și compatibilitatea dispozitivelor audio.
        :return: Tuple (input_devices, output_devices, error_message)
        """
        try:
            devices = sd.query_devices()
            input_devices = []
            output_devices = []

            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device['name']))
                if device['max_output_channels'] > 0:
                    output_devices.append((i, device['name']))

            if not input_devices:
                return None, None, "Nu s-au găsit dispozitive de intrare audio."
            if not output_devices:
                return None, None, "Nu s-au găsit dispozitive de ieșire audio."

            return input_devices, output_devices, None

        except Exception as e:
            return None, None, f"Eroare la verificarea dispozitivelor audio: {str(e)}"

    def check_file_compatibility(self, file_path):
        """
        Verifică compatibilitatea unui fișier audio.
        :param file_path: Calea către fișierul audio
        :return: Tuple (is_compatible, error_message)
        """
        try:
            # Verificăm extensia
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.supported_formats:
                return False, f"Formatul {ext} nu este suportat. Formate acceptate: {', '.join(self.supported_formats)}"

            # Verificăm dacă fișierul există
            if not os.path.exists(file_path):
                return False, "Fișierul nu există."

            # Verificăm dacă fișierul poate fi citit
            if not os.access(file_path, os.R_OK):
                return False, "Nu există permisiuni de citire pentru fișier."

            # Verificăm dimensiunea fișierului (max 1GB)
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024 * 1024:  # 1GB
                return False, "Fișierul este prea mare (maxim 1GB permis)."

            # Pentru fișiere WAV, verificăm formatul
            if ext == '.wav':
                try:
                    with wave.open(file_path, 'rb') as wav_file:
                        channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        frame_rate = wav_file.getframerate()

                        if channels not in [1, 2]:
                            return False, f"Numărul de canale ({channels}) nu este suportat. Se acceptă doar mono sau stereo."

                        if sample_width not in [1, 2, 4]:
                            return False, f"Lățimea de eșantionare ({sample_width}) nu este suportată."

                        if frame_rate not in [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]:
                            return False, f"Rata de eșantionare ({frame_rate}) nu este suportată."
                except wave.Error:
                    return False, "Fișierul WAV este corupt sau invalid."

            return True, None

        except Exception as e:
            return False, f"Eroare la verificarea fișierului: {str(e)}"