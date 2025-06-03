import unittest
import numpy as np
from domain.audio_processor import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor(max_workers=2)

    def tearDown(self):
        self.processor.shutdown()

    def test_process_chunks_identity(self):
        data = np.random.rand(44100)

        def identity(chunk):
            return chunk

        result = self.processor.process_chunks_parallel(data, chunk_size=5000, process_func=identity)
        self.assertEqual(len(result), len(data))
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_process_chunks_gain(self):
        data = np.ones(10000)

        def apply_gain(chunk, gain=2.0):
            return chunk * gain

        result = self.processor.process_chunks_parallel(data, chunk_size=1000, process_func=apply_gain, gain=2.0)
        self.assertTrue(np.allclose(result, 2.0))

    def test_process_with_callback(self):
        data = np.ones(15000)
        progress = []

        def dummy(chunk):
            return chunk

        def callback(percent):
            progress.append(percent)

        self.processor.process_chunks_parallel(data, 3000, dummy, progress_callback=callback)
        self.assertTrue(progress[-1] == 100)
        self.assertGreater(len(progress), 1)

if __name__ == '__main__':
    unittest.main()
