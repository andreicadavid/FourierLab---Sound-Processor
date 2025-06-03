import unittest
import numpy as np
import os
from tempfile import NamedTemporaryFile
from repo.audio_repo import AudioRepository
from domain.recording import Recording

class TestAudioRepository(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 44100
        self.data = np.random.rand(self.sample_rate).astype(np.float64) * 2 - 1
        self.recording = Recording(self.data, self.sample_rate)

    def test_save_and_load_recording(self):
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_filename = tmp_file.name

        try:
            # Salvăm înregistrarea
            AudioRepository.save(self.recording, temp_filename)
            self.assertTrue(os.path.exists(temp_filename))

            # Încărcăm înregistrarea
            loaded = AudioRepository.load(temp_filename)
            self.assertIsInstance(loaded, Recording)
            self.assertEqual(loaded.sample_rate, self.sample_rate)
            self.assertEqual(len(loaded.data), len(self.data))
            np.testing.assert_almost_equal(loaded.data, self.data, decimal=1)

        finally:
            os.remove(temp_filename)

if __name__ == '__main__':
    unittest.main()
