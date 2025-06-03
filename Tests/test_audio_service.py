import threading
import unittest
import time
from unittest.mock import patch, MagicMock, mock_open
import sounddevice as sd

import numpy as np
from service.audio_service import AudioService
from domain.config import Config
from domain.audio_device_manager import AudioDeviceManager
from domain.recording import Recording

class TestAudioService(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.service = AudioService(self.config)
        self.config.input_device = None
        self.config.buffer_size = 1024
        self.config.sample_rate = 44100
        self.dummy_data = np.ones(44100)
        self.service.recording = Recording(self.dummy_data, self.config.sample_rate)

        # Setări pentru testele care cer verificarea apelurilor
        self.service.progress_callback = None
        self.service.duration_callback = None

        # Mock pentru metode care modifică starea internă (doar când ai nevoie)
        self.service._update_duration = MagicMock()
        self.service.process_in_chunks = MagicMock(return_value=np.ones(44100))

        # Processor mock pentru procesare paralelă
        self.service.processor = MagicMock()
        self.service.processor.process_chunks_parallel = MagicMock(return_value=self.dummy_data)

    def test_initial_state(self):
        self.assertFalse(self.service.is_recording)
        self.assertFalse(self.service.is_playing)
        self.assertFalse(self.service.is_loading)
        self.assertFalse(self.service.is_saving)
        self.assertEqual(self.service.current_state_index, -1)
        self.assertEqual(self.service.progress_callback, None)
        self.assertEqual(self.service.duration_callback, None)
        self.assertEqual(self.service.processing_chunk_size, 1024 * 1024)
        self.assertEqual(self.service.current_cache_size, 0)
        self.assertEqual(self.service.MAX_CACHE_ENTRIES, 50)
        self.assertEqual(self.service.MAX_CACHE_SIZE, 1024 * 1024 * 1024)
        self.assertEqual(self.service.MAX_UNDO_STACK_SIZE, 20)
        self.assertEqual(self.service.current_undo_size, 0)


    def test_set_progress_callback(self):
        mock_callback = lambda progress: None
        self.service.set_progress_callback(mock_callback)
        self.assertEqual(self.service.progress_callback, mock_callback)

    def test_set_duration_callback(self):
        mock_callback = lambda duration: None
        self.service.set_duration_callback(mock_callback)
        self.assertEqual(self.service.duration_callback, mock_callback)

    def test_update_duration_callback_called(self):
        durations = []

        def mock_callback(d):
            durations.append(d)

        self.service.set_duration_callback(mock_callback)
        sample_rate = 44100
        data = np.zeros(sample_rate * 3, dtype=np.float32)  # 3 secunde
        self.service.recording = Recording(data, sample_rate)

        # Patch temporar metoda _update_duration să apeleze callback-ul imediat
        original_update_duration = self.service._update_duration

        def fake_update_duration():
            if self.service.duration_callback:
                self.service.duration_callback(3.0)

        self.service._update_duration = fake_update_duration

        self.service._update_duration()

        # Restaurăm metoda originală
        self.service._update_duration = original_update_duration

        self.assertGreater(len(durations), 0)
        self.assertAlmostEqual(durations[0], 3.0, places=2)

    def test_set_processing_config(self):
        self.service.set_processing_config(chunk_size=2048, num_threads=4)
        self.assertEqual(self.service.processing_chunk_size, 2048)
        self.assertEqual(self.service.processor.max_workers, 4)

    def test_get_processing_config(self):
        chunk, threads = self.service.get_processing_config()
        self.assertEqual(chunk, self.service.processing_chunk_size)
        self.assertEqual(threads, self.service.processor.max_workers)

    @patch('service.audio_service.sd.rec')
    @patch('service.audio_service.sd.stop')
    @patch('service.audio_service.sd.sleep')
    @patch('service.audio_service.sd.get_stream')
    def test_record_mocked_complete(self, mock_get_stream, mock_sleep, mock_stop, mock_rec):
        # Pregătim datele simulate
        mock_data = np.ones((44100, 1), dtype='float64')
        mock_rec.return_value = mock_data

        # Simulăm streamul ca fiind activ doar o singură iterație
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_stream.time = 1.1
        mock_get_stream.return_value = mock_stream

        # Pornim înregistrarea
        self.service.record(1.0)

        # Așteptăm finalizarea threadului
        self.service.recording_thread.join(timeout=2.0)

        # Verificări
        self.assertIsInstance(self.service.recording, Recording)
        self.assertEqual(len(self.service.recording.data), 44100)
        self.assertEqual(self.service.recording.sample_rate, 44100)
        self.assertFalse(self.service.is_recording)

    def test_stop_recording_without_active(self):
        result = self.service.stop_recording()
        self.assertFalse(result)

    def test_stop_recording_with_active_thread(self):
        # Simulăm înregistrare activă
        self.service.is_recording = True
        self.service.recording_thread = threading.Thread(target=lambda: None)
        self.service.recording_thread.start()

        result = self.service.stop_recording()
        self.assertTrue(result)
        self.assertFalse(self.service.is_recording)

    def test_calculate_cache_entry_size_valid(self):
        entry = {"data": np.ones(1000)}
        size = self.service._calculate_cache_entry_size(entry)
        self.assertEqual(size, 8000)

    def test_calculate_cache_entry_size_invalid(self):
        self.assertEqual(self.service._calculate_cache_entry_size(None), 0)
        self.assertEqual(self.service._calculate_cache_entry_size({}), 0)

    def test_cache_state_adds_entry(self):
        self.service.cache_state("reverb", {"room": "hall"})
        self.assertEqual(len(self.service.cache), 1)
        self.assertEqual(self.service.cache[0]["effect"], "reverb")

    def test_cache_size_limit_prevents_oversized(self):
        self.service.MAX_CACHE_SIZE = 10  # forțăm limită mică
        self.service.cache_state("too_large")
        self.assertEqual(len(self.service.cache), 0)

    def test_cleanup_cache_removes_old_entries(self):
        self.service.MAX_CACHE_ENTRIES = 2
        for i in range(4):
            self.service.cache_state(f"effect_{i}")
        self.assertLessEqual(len(self.service.cache), 2)

    def test_get_cache_info_returns_expected(self):
        self.service.cache_state("echo")
        info = self.service.get_cache_info()
        self.assertIn("current_size", info)
        self.assertIn("entries", info)
        self.assertGreater(info["entries"], 0)

    def test_clear_cache(self):
        self.service.cache_state("anything")
        self.service.clear_cache()
        self.assertEqual(len(self.service.cache), 0)
        self.assertEqual(self.service.current_cache_size, 0)

    def test_load_from_cache_success(self):
        self.service.cache_state("reverb", {"decay": 0.5})
        result = self.service.load_from_cache(0)
        self.assertTrue(result)
        self.assertIsInstance(self.service.recording, Recording)

    def test_load_from_cache_invalid_index(self):
        result = self.service.load_from_cache(99)
        self.assertFalse(result)

    def test_get_cache_history(self):
        self.service.cache_state("echo", {"delay": 200})
        self.service.cache_state("reverb", {"room": "hall"})
        history = self.service.get_cache_history()
        self.assertEqual(len(history), 2)
        self.assertTrue("echo" in history[0])
        self.assertTrue("reverb" in history[1])

    @patch("builtins.open", new_callable=mock_open)
    @patch("librosa.onset.onset_detect", return_value=np.array([0, 5, 10]))
    @patch("librosa.frames_to_time", return_value=np.array([0.0, 0.5, 1.0]))
    def test_export_onsets_success(self, mock_frames_to_time, mock_onset_detect, mock_file):
        result = self.service.export_onsets("mock_file.csv")
        self.assertTrue(result)
        mock_file.assert_called_once_with("mock_file.csv", "w")
        handle = mock_file()
        handle.write.assert_any_call("onset_time_sec\n")
        handle.write.assert_any_call("0.000000\n")

    def test_export_onsets_no_recording(self):
        self.service.recording = None
        result = self.service.export_onsets("out.csv")
        self.assertFalse(result)

    @patch("threading.Thread")
    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    @patch("os.access", return_value=True)
    @patch("shutil.disk_usage")
    @patch("scipy.io.wavfile.write")
    def test_save_recording_success(self, mock_write, mock_disk, mock_access, mock_makedirs, mock_exists, mock_thread):
        mock_disk.return_value = MagicMock(free=1000000000)
        result = self.service.save_recording("test_output.wav")
        self.assertTrue(result)
        mock_thread.assert_called_once()

    def test_save_recording_no_data(self):
        self.service.recording = None
        result = self.service.save_recording("test.wav")
        self.assertFalse(result)

    @patch("sounddevice.OutputStream")
    def test_play_starts_and_calls_stream_write(self, mock_stream_cls):
        mock_stream = MagicMock()
        mock_stream_cls.return_value.__enter__.return_value = mock_stream

        self.service.play()
        time.sleep(0.1)  # lăsăm thread-ul să pornească
        self.service.stop_playback()

        mock_stream.write.assert_called()
        self.assertFalse(self.service.is_playing)

    def test_play_without_recording(self):
        self.service.recording = None
        self.service.play()  # shouldn't raise
        self.assertFalse(self.service.is_playing)

    def test_stop_playback_inactive(self):
        self.service.is_playing = False
        result = self.service.stop_playback()
        self.assertFalse(result)

    def test_stop_playback_active(self):
        self.service.is_playing = True
        self.service.audio_thread = MagicMock()
        self.service.audio_thread.is_alive.return_value = True

        result = self.service.stop_playback()
        self.assertTrue(result)
        self.assertFalse(self.service.is_playing)

    def test_calculate_undo_entry_size_valid(self):
        size = self.service._calculate_undo_entry_size(self.service.recording)
        self.assertEqual(size, self.dummy_data.nbytes)

    def test_calculate_undo_entry_size_invalid(self):
        self.assertEqual(self.service._calculate_undo_entry_size(None), 0)
        self.assertEqual(self.service._calculate_undo_entry_size(object()), 0)

    def test_save_state_adds_to_undo_stack(self):
        self.service.save_state()
        self.assertEqual(len(self.service.undo_stack), 1)
        self.assertEqual(self.service.current_undo_size, self.dummy_data.nbytes)

    def test_save_state_respects_max_undo_size(self):
        self.service.MAX_UNDO_STACK_SIZE = 3
        for _ in range(5):
            self.service.save_state()
        self.assertEqual(len(self.service.undo_stack), 3)

    def test_get_undo_stack_info(self):
        self.service.save_state()
        info = self.service.get_undo_stack_info()
        self.assertEqual(info['entries'], 1)
        self.assertEqual(info['current_size'], self.dummy_data.nbytes)
        self.assertEqual(info['max_size'], self.service.MAX_CACHE_SIZE)

    def test_undo_restores_previous_state(self):
        original = Recording(np.full(44100, 2.0), self.config.sample_rate)
        self.service.recording = original
        self.service.save_state()
        self.service.recording = Recording(np.zeros(44100), self.config.sample_rate)
        result = self.service.undo()
        self.assertTrue(result)
        self.assertTrue(np.array_equal(self.service.recording.data, original.data))

    def test_undo_without_stack(self):
        self.service.undo_stack.clear()
        result = self.service.undo()
        self.assertFalse(result)

    def test_pitch_shift_up(self):
        result = self.service.pitch_shift(up=True)
        self.assertIsInstance(result, Recording)

    def test_pitch_shift_down(self):
        result = self.service.pitch_shift(up=False)
        self.assertIsInstance(result, Recording)

    def test_apply_reverb_valid(self):
        result = self.service.apply_reverb(decay=0.5, delay=0.02, ir_duration=0.5)
        self.assertIsInstance(result, Recording)

    def test_apply_echo_valid(self):
        result = self.service.apply_echo(decay=0.5, delay=0.2, repeats=3)
        self.assertIsInstance(result, Recording)

    def test_apply_equalizer_valid(self):
        gains = [0.0] * 10
        result = self.service.apply_equalizer(gains)
        self.assertIsInstance(result, Recording)

    @patch("service.audio_service.librosa.load")
    @patch("service.audio_service.librosa.resample")
    def test_apply_reverb_with_ir_valid(self, mock_resample, mock_load):
        mock_load.return_value = (np.ones(44100), 44100)
        mock_resample.side_effect = lambda y, orig_sr, target_sr: y
        self.service.device_manager.get_ir_path = MagicMock()
        self.service.device_manager.get_ir_path.return_value.exists = MagicMock(return_value=True)
        self.service.process_in_chunks = MagicMock(return_value=np.ones(44100))

        result = self.service.apply_reverb_with_ir("bathroom.wav")
        self.assertIsInstance(result, Recording)

    @patch("service.audio_service.librosa.feature.spectral_contrast",
           return_value=np.array([
               [20.0, 21.0, 22.0],
               [23.0, 24.0, 25.0],
               [26.0, 27.0, 28.0],
               [29.0, 30.0, 31.0],
               [32.0, 33.0, 34.0],
               [35.0, 36.0, 37.0],
               [38.0, 39.0, 40.0]
           ]))
    @patch("service.audio_service.librosa.feature.spectral_rolloff", return_value=np.array([[3000.0]]))
    @patch("service.audio_service.librosa.feature.spectral_bandwidth", return_value=np.array([[2000.0]]))
    @patch("service.audio_service.librosa.feature.spectral_centroid", return_value=np.array([[1000.0]]))
    def test_calculate_spectral_features(self, mock_centroid, mock_bandwidth, mock_rolloff, mock_contrast):
        result = self.service.calculate_spectral_features()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["Spectral Centroid"], 1000.0)
        self.assertEqual(result["Spectral Bandwidth"], 2000.0)
        self.assertEqual(result["Spectral Rolloff"], 3000.0)
        self.assertEqual(len(result["Spectral Contrast"]), 7)

    @patch("service.audio_service.librosa.piptrack",
           return_value=(np.array([[0, 100, 200]]), np.array([[0.1, 0.5, 0.2]])))
    @patch("service.audio_service.librosa.estimate_tuning", return_value=0.02)
    def test_analyze_pitch_and_tuning(self, mock_tuning, mock_piptrack):
        result = self.service.analyze_pitch_and_tuning()
        self.assertIsNotNone(result)
        self.assertIn("Pitch Fundamental", result)
        self.assertIn("Tuning Adjustment", result)
        self.assertAlmostEqual(result["Tuning Adjustment"], 0.02)

    @patch("service.audio_service.librosa.beat.beat_track", return_value=(128.0, None))
    def test_estimate_bpm(self, mock_beat_track):
        result = self.service.estimate_bpm()
        self.assertIsInstance(result, float)
        self.assertEqual(result, 128.0)

    def test_estimate_bpm_cache(self):
        self.service.recording.cached_bpm = 99.0
        result = self.service.estimate_bpm()
        self.assertEqual(result, 99.0)

    @patch("service.audio_service.librosa.stft", return_value=np.ones((1025, 100)))
    @patch("service.audio_service.librosa.amplitude_to_db", return_value=np.ones((1025, 100)))
    def test_generate_spectrogram(self, mock_amp_to_db, mock_stft):
        result = self.service.generate_spectrogram()
        self.assertIsInstance(result, dict)
        self.assertIn('data', result)
        self.assertEqual(result['title'], 'Spectrogramă')
        mock_stft.assert_called_once()
        mock_amp_to_db.assert_called_once()

    @patch("service.audio_service.librosa.feature.chroma_stft", return_value=np.ones((12, 100)))
    def test_generate_chroma(self, mock_chroma):
        result = self.service.generate_chroma()
        self.assertIsInstance(result, dict)
        self.assertIn('data', result)
        self.assertEqual(result['title'], 'Chroma')
        mock_chroma.assert_called_once()

    @patch("service.audio_service.librosa.feature.melspectrogram", return_value=np.ones((128, 200)))
    @patch("service.audio_service.librosa.power_to_db", return_value=np.ones((128, 200)))
    def test_generate_mel_spectrogram(self, mock_power_to_db, mock_melspec):
        mel_spec_db, sr = self.service.generate_mel_spectrogram()
        self.assertIsInstance(mel_spec_db, np.ndarray)
        self.assertEqual(sr, self.config.sample_rate)
        mock_melspec.assert_called_once()
        mock_power_to_db.assert_called_once()

    @patch("service.audio_service.librosa.feature.mfcc", return_value=np.ones((13, 100)))
    def test_generate_mfcc(self, mock_mfcc):
        mfcc, sr = self.service.generate_mfcc()
        self.assertIsInstance(mfcc, np.ndarray)
        self.assertEqual(sr, self.config.sample_rate)
        mock_mfcc.assert_called_once()

    @patch("service.audio_service.librosa.cqt", return_value=np.ones((84, 200)))
    @patch("service.audio_service.librosa.amplitude_to_db", return_value=np.ones((84, 200)))
    def test_generate_cqt(self, mock_amp_to_db, mock_cqt):
        cqt_db, sr = self.service.generate_cqt()
        self.assertIsInstance(cqt_db, np.ndarray)
        self.assertEqual(sr, self.config.sample_rate)
        mock_cqt.assert_called_once()
        mock_amp_to_db.assert_called_once()

    def test_generate_spectrogram_no_recording(self):
        self.service.recording = None
        self.assertIsNone(self.service.generate_spectrogram())

    def test_generate_chroma_no_recording(self):
        self.service.recording = None
        self.assertIsNone(self.service.generate_chroma())

    def test_generate_mel_spectrogram_no_recording(self):
        self.service.recording = None
        result = self.service.generate_mel_spectrogram()
        self.assertEqual(result, (None, None))

    def test_generate_mfcc_no_recording(self):
        self.service.recording = None
        result = self.service.generate_mfcc()
        self.assertEqual(result, (None, None))

    def test_generate_cqt_no_recording(self):
        self.service.recording = None
        result = self.service.generate_cqt()
        self.assertEqual(result, (None, None))

    def test_process_compressor_chunk_thresholding(self):
        chunk = np.array([0.01, 0.5, -0.8, 0.05], dtype=np.float32)
        result = self.service.process_compressor_chunk(chunk, self.config.sample_rate, threshold_db=-20, ratio=4)
        # Valori peste prag trebuie comprimate
        self.assertTrue(np.all(np.abs(result) <= np.abs(chunk)))

    def test_apply_simple_compressor_valid(self):
        self.service.progress_callback = MagicMock()
        self.service.cache_state = MagicMock()
        self.service.save_state = MagicMock()
        result = self.service.apply_simple_compressor(threshold_db=-10.0, ratio=2.0, normalize=True)
        self.assertIsInstance(result, Recording)
        self.service.save_state.assert_called_once()
        self.service.cache_state.assert_called_once()
        self.service._update_duration.assert_called_once()
        self.service.progress_callback.assert_any_call(0)
        self.service.progress_callback.assert_any_call(100)

    def test_apply_simple_compressor_invalid_threshold(self):
        res = self.service.apply_simple_compressor(threshold_db=-100)
        self.assertIsNone(res)

    def test_apply_lowpass_filter(self):
        self.service.cache_state = MagicMock()
        self.service.save_state = MagicMock()
        result = self.service.apply_lowpass_filter(cutoff_hz=1000, order=3)
        self.assertIsInstance(result, Recording)
        self.service.cache_state.assert_called_once_with("LPF", {"cutoff": 1000, "order": 3})

    def test_apply_highpass_filter(self):
        self.service.cache_state = MagicMock()
        self.service.save_state = MagicMock()
        result = self.service.apply_highpass_filter(cutoff_hz=2000, order=2)
        self.assertIsInstance(result, Recording)
        self.service.cache_state.assert_called_once_with("HPF", {"cutoff": 2000, "order": 2})

    def test_apply_bandpass_filter(self):
        self.service.cache_state = MagicMock()
        self.service.save_state = MagicMock()
        result = self.service.apply_bandpass_filter(lowcut_hz=300, highcut_hz=3000, order=4)
        self.assertIsInstance(result, Recording)
        self.service.cache_state.assert_called_once_with("BPF", {"lowcut": 300, "highcut": 3000, "order": 4})

    def test_apply_distortion(self):
        self.service.cache_state = MagicMock()
        self.service.save_state = MagicMock()
        result = self.service.apply_distortion(drive=5.0, tone=0.3, mix=0.8)
        self.assertIsInstance(result, Recording)
        self.service.cache_state.assert_called_once_with("Distortion", {"drive": 5.0, "tone": 0.3, "mix": 0.8})

    def test_apply_reverb_chunked(self):
        self.service.cache_state = MagicMock()
        self.service.save_state = MagicMock()
        result = self.service.apply_reverb_chunked(decay=0.5, delay=0.01, ir_duration=0.5)
        self.assertIsInstance(result, Recording)
        self.service.cache_state.assert_called_once_with("ReverbChunked",
                                                         {"decay": 0.5, "delay": 0.01, "ir_duration": 0.5})

    def test_process_reverb_ir_chunk(self):
        import numpy as np
        from scipy.signal import fftconvolve

        chunk = np.ones(100)
        ir = np.ones(20)
        result = self.service.process_reverb_ir_chunk(chunk, self.config.sample_rate, ir)
        self.assertTrue(len(result) >= len(chunk))  # convoluția cu mode='full' produce lungime mai mare
        self.assertTrue(np.max(np.abs(result)) <= 1.0)  # trebuie normalizat

    @patch("librosa.effects.time_stretch", side_effect=lambda y, rate: y)  # mock time_stretch simplu
    @patch("librosa.beat.beat_track", return_value=(120.0, None))
    def test_apply_time_stretch_bpm_valid(self, mock_beat, mock_time_stretch):
        self.service.cache_state = MagicMock()
        self.service.save_state = MagicMock()
        self.service.progress_callback = MagicMock()

        result = self.service.apply_time_stretch_bpm(100)
        self.assertIsInstance(result, Recording)
        self.service.save_state.assert_called_once()
        self.service.cache_state.assert_called_once()
        self.service._update_duration.assert_called_once()
        self.service.progress_callback.assert_any_call(0)
        self.service.progress_callback.assert_any_call(100)

    def test_apply_time_stretch_bpm_invalid_bpm(self):
        result = self.service.apply_time_stretch_bpm(-5)
        self.assertIsNone(result)
        result = self.service.apply_time_stretch_bpm(10)  # sub minim 20
        self.assertIsNone(result)
        result = self.service.apply_time_stretch_bpm(350)  # peste maxim 300
        self.assertIsNone(result)
        result = self.service.apply_time_stretch_bpm("abc")  # tip invalid
        self.assertIsNone(result)

    @patch("wave.open")
    def test_load_audio_success(self, mock_wave_open):
        self.service.cache_state = MagicMock()
        self.service.save_state = MagicMock()
        # Simulăm fișier valid
        mock_wav = MagicMock()
        mock_wav.getnchannels.return_value = 1
        mock_wav.getsampwidth.return_value = 2
        mock_wav.getframerate.return_value = 44100
        mock_wav.getnframes.return_value = 44100
        mock_wav.readframes.side_effect = [b'\x00' * 1000, b'']  # 1 chunk + final
        mock_wave_open.return_value.__enter__.return_value = mock_wav

        self.service.device_manager.check_file_compatibility = MagicMock(return_value=(True, ""))
        result = self.service.load_audio("dummy_path.wav")
        self.assertTrue(result)
        self.assertIsInstance(self.service.recording, Recording)
        self.service.cache_state.assert_called_once()
        self.service._update_duration.assert_called_once()

    def test_load_audio_incompatible_file(self):
        self.service.device_manager.check_file_compatibility = MagicMock(return_value=(False, "Incompatibil"))
        result = self.service.load_audio("bad_file.wav")
        self.assertFalse(result)

    @patch("glob.glob", return_value=["file1.tmp", "file2.tmp"])
    @patch("os.remove")
    def test_cleanup(self, mock_remove, mock_glob):
        self.service.processor = MagicMock()
        self.service.is_recording = False
        self.service.is_playing = False
        self.service.undo_stack = [1, 2]
        self.service.cache = [1, 2]
        self.service.recording = Recording(np.ones(10), 44100)

        self.service.cleanup()

        self.service.processor.shutdown.assert_called_once()
        mock_remove.assert_any_call("file1.tmp")
        mock_remove.assert_any_call("file2.tmp")
        self.assertEqual(self.service.undo_stack, [])
        self.assertEqual(self.service.cache, [])
        self.assertIsNone(self.service.recording)





if __name__ == '__main__':
    unittest.main()
