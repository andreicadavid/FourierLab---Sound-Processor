import unittest
from unittest.mock import patch, MagicMock
import os
import wave
from pathlib import Path
from domain.audio_device_manager import AudioDeviceManager

class TestAudioDeviceManagerComplete(unittest.TestCase):

    def setUp(self):
        self.manager = AudioDeviceManager()

    def test_get_ir_path(self):
        path = self.manager.get_ir_path("bathroom.wav")
        self.assertTrue(str(path).endswith(os.path.join("IR", "bathroom.wav")))

    def test_get_available_ir_files(self):
        files = self.manager.get_available_ir_files()
        self.assertIsInstance(files, list)

    @patch('sounddevice.query_devices')
    def test_check_audio_devices_success(self, mock_query_devices):
        mock_query_devices.return_value = [
            {'max_input_channels': 2, 'max_output_channels': 2, 'name': 'Input Device'},
            {'max_input_channels': 0, 'max_output_channels': 2, 'name': 'Output Device'},
        ]
        inputs, outputs, error = self.manager.check_audio_devices()
        self.assertIsInstance(inputs, list)
        self.assertIsInstance(outputs, list)
        self.assertIsNone(error)
        self.assertTrue(len(inputs) > 0)
        self.assertTrue(len(outputs) > 0)

    @patch('sounddevice.query_devices', side_effect=Exception("Simulated error"))
    def test_check_audio_devices_exception(self, mock_query_devices):
        inputs, outputs, error = self.manager.check_audio_devices()
        self.assertIsNone(inputs)
        self.assertIsNone(outputs)
        self.assertIn("Simulated error", error)

    def test_check_file_compatibility_valid_wav(self):
        # Creăm un fișier WAV temporar pentru test
        path = Path("test_valid.wav")
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b'\x00' * 100)
        try:
            is_compatible, error = self.manager.check_file_compatibility(str(path))
            self.assertTrue(is_compatible)
            self.assertIsNone(error)
        finally:
            path.unlink()  # Ștergem fișierul după test

    def test_check_file_compatibility_invalid_extension(self):
        is_compatible, error = self.manager.check_file_compatibility("file.unsupported")
        self.assertFalse(is_compatible)
        self.assertIn("nu este suportat", error.lower())

    def test_check_file_compatibility_nonexistent_file(self):
        is_compatible, error = self.manager.check_file_compatibility("nonexistent.wav")
        self.assertFalse(is_compatible)
        self.assertIn("nu există", error.lower())

    @patch('os.access', return_value=False)
    def test_check_file_compatibility_no_read_permission(self, mock_access):
        # Pentru a evita dependențe, patchăm os.path.exists să returneze True
        with patch('os.path.exists', return_value=True):
            is_compatible, error = self.manager.check_file_compatibility("somefile.wav")
            self.assertFalse(is_compatible)
            self.assertIn("permisiuni de citire", error.lower())

    @patch('os.path.getsize', return_value=2 * 1024 * 1024 * 1024)  # 2GB
    def test_check_file_compatibility_file_too_large(self, mock_getsize):
        with patch('os.path.exists', return_value=True), patch('os.access', return_value=True):
            is_compatible, error = self.manager.check_file_compatibility("largefile.wav")
            self.assertFalse(is_compatible)
            self.assertIn("prea mare", error.lower())

    @patch('wave.open')
    def test_check_file_compatibility_invalid_wav(self, mock_wave_open):
        # Simulăm o eroare la deschiderea fișierului wav
        mock_wave_open.side_effect = wave.Error("File corrupted")
        with patch('os.path.exists', return_value=True), patch('os.access', return_value=True), patch('os.path.getsize', return_value=1000):
            is_compatible, error = self.manager.check_file_compatibility("corrupted.wav")
            self.assertFalse(is_compatible)
            self.assertIn("corupt", error.lower())

    @patch('os.path.splitext', side_effect=Exception("Simulated splitext error"))
    def test_check_file_compatibility_exception(self, mock_splitext):
        is_compatible, error = self.manager.check_file_compatibility("file.wav")
        self.assertFalse(is_compatible)
        self.assertIn("Simulated splitext error", error)

if __name__ == '__main__':
    unittest.main()
