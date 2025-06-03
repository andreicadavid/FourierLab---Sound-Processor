import unittest
import numpy as np
from domain.config import Config
from domain.recording import Recording

class TestEntities(unittest.TestCase):

    def test_config_defaults(self):
        config = Config()
        self.assertEqual(config.sample_rate, 44100)
        self.assertEqual(config.pitch_factor, 2.0)
        self.assertEqual(config.save_directory, "")
        self.assertEqual(config.buffer_size, 1024)
        self.assertIsNone(config.input_device)
        self.assertIsNone(config.output_device)

    def test_config_update(self):
        config = Config()
        config.sample_rate = 48000
        config.pitch_factor = 1.5
        config.save_directory = "test"
        config.input_device = "Mic"
        config.output_device = "Speakers"
        config.buffer_size = 512

        self.assertEqual(config.sample_rate, 48000)
        self.assertEqual(config.pitch_factor, 1.5)
        self.assertEqual(config.save_directory, "test")
        self.assertEqual(config.input_device, "Mic")
        self.assertEqual(config.output_device, "Speakers")
        self.assertEqual(config.buffer_size, 512)

    def test_recording_initialization(self):
        data = np.ones(22050)
        sample_rate = 22050
        rec = Recording(data=data, sample_rate=sample_rate)

        self.assertEqual(rec.sample_rate, sample_rate)
        self.assertIs(rec.data, data)
        self.assertEqual(len(rec.data), 22050)
        self.assertTrue(np.all(rec.data == 1.0))

if __name__ == '__main__':
    unittest.main()
