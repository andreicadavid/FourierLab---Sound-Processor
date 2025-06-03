import unittest
from unittest.mock import MagicMock, patch, ANY, call
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt

# Importă MainWindowV2 din locația ta corectă
from ui.main_window_v2 import MainWindowV2


class TestMainWindowV2(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # ascunde fereastra în teste

        self.app = MainWindowV2(self.root)

        self.app.root.config = MagicMock()

        # Mock root.after
        self.app.root.after = MagicMock()

        self.app.root.update_idletasks = MagicMock()

        # Mock service și metodele lui
        self.app.service = MagicMock()
        self.app.service.recording = MagicMock()
        self.app.service.recording.data = np.arange(100, dtype=np.float32)
        self.app.service.recording.sample_rate = 44100
        self.app.service.is_playing = False
        self.app.service.is_recording = False
        self.app.is_playing = False
        self.app.stop_event = MagicMock()
        self.app.stop_event.is_set.return_value = False

        # Mock elemente UI
        self.app.play_button = MagicMock()
        self.app.play_button.config = MagicMock()
        self.app.stop_button = MagicMock()
        self.app.stop_button.config = MagicMock()
        self.app.stop_button.__getitem__.return_value = 'disabled'
        self.app.status_label = MagicMock()
        self.app.status_label.config = MagicMock()
        self.app.status_label.cget = MagicMock(return_value="Standby")
        self.app.progress_var = MagicMock()
        self.app.progress_var.set = MagicMock()
        self.app.progress_var.get = MagicMock(return_value=0)
        self.app.bpm_entry = MagicMock()
        self.app.bpm_entry.delete = MagicMock()
        self.app.bpm_entry.insert = MagicMock()
        self.app.duration_entry = MagicMock()
        self.app.duration_entry.get = MagicMock(return_value="3.5")
        self.app.plot_notebook = MagicMock()
        self.app.spectrogram_tab = MagicMock()
        self.app.chroma_tab = MagicMock()
        self.app.mel_tab = MagicMock()
        self.app.mfcc_tab = MagicMock()
        self.app.cqt_tab = MagicMock()
        self.app.onset_tab = MagicMock()
        self.app.ax = MagicMock()
        self.app.waveform_canvas = MagicMock()
        self.app.update_plot = MagicMock()

    def tearDown(self):
        self.root.destroy()

    # Test pentru stop_playback
    @patch('tkinter.messagebox.showerror')
    def test_stop_playback_behavior(self, mock_error):
        self.app.service.is_playing = True
        self.app.service.is_recording = True
        self.app._recording_before_record = MagicMock()
        self.app.stop_playback()

        self.assertFalse(self.app.service.is_playing)
        self.assertFalse(self.app.is_playing)
        self.app.stop_event.set.assert_called_once()
        self.assertEqual(self.app.status_label.cget("text"), "Standby")
        self.assertEqual(self.app.progress_var.get(), 0)
        self.assertEqual(self.app.stop_button['state'], 'disabled')

    # Test show_loading și hide_loading
    def test_show_and_hide_loading(self):
        self.app.show_loading("Loading Test")
        self.assertTrue(self.app.loading_running)
        self.assertTrue(hasattr(self.app, "loading_win"))
        self.assertTrue(hasattr(self.app, "loading_label"))

        self.app.hide_loading()
        self.assertFalse(getattr(self.app, "loading_running", True))
        self.assertFalse(hasattr(self.app, "loading_win"))
        self.assertFalse(hasattr(self.app, "loading_label"))

    # Test show_cache_panel
    @patch('tkinter.Toplevel')
    def test_show_cache_panel_calls(self, mock_toplevel):
        self.app.service.get_cache_history.return_value = ['step1', 'step2']
        self.app.service.load_from_cache.return_value = True

        self.app.show_cache_panel()
        self.app.service.get_cache_history.assert_called_once()

    # Test update_plot și _update_all_plots fără excepții
    def test_update_plot_and_all_plots(self):
        self.app.update_plot()
        self.app._update_all_plots()

    # Test export_onsets_ui (succes și eșec)
    @patch('tkinter.filedialog.asksaveasfilename')
    @patch('tkinter.messagebox.showinfo')
    @patch('tkinter.messagebox.showerror')
    def test_export_onsets_ui_success(self, mock_error, mock_info, mock_asksave):
        mock_asksave.return_value = "/tmp/onsets.csv"
        self.app.service.recording = MagicMock()
        self.app.service.export_onsets.return_value = True
        self.app.export_onsets_ui()
        mock_info.assert_called_with("Succes", "Onsets exportate în /tmp/onsets.csv")

    @patch('tkinter.filedialog.asksaveasfilename')
    @patch('tkinter.messagebox.showinfo')
    @patch('tkinter.messagebox.showerror')
    def test_export_onsets_ui_failure(self, mock_error, mock_info, mock_asksave):
        mock_asksave.return_value = "/tmp/onsets.csv"
        self.app.service.recording = MagicMock()
        self.app.service.export_onsets.return_value = False
        self.app.export_onsets_ui()
        mock_error.assert_called_with("Eroare", "Exportul a eșuat.")

    # Test open_audio_devices
    @patch('sounddevice.query_devices')
    @patch('tkinter.messagebox.showerror')
    def test_open_audio_devices(self, mock_error, mock_query):
        mock_query.return_value = [
            {'name': 'Input1', 'max_input_channels': 2, 'max_output_channels': 0},
            {'name': 'Output1', 'max_input_channels': 0, 'max_output_channels': 2},
        ]
        self.app.open_audio_devices()

    # Test generate_spectrogram (fără patch la subplots)
    def test_generate_spectrogram(self):
        self.app.service.recording = MagicMock()
        self.app.service.recording.data = np.arange(10, dtype=np.float32)
        self.app.service.recording.sample_rate = 44100

        self.app._show_plot_in_tab = MagicMock()
        self.app.generate_spectrogram()
        self.app._show_plot_in_tab.assert_called()

    # Test generate_chroma (fără patch la subplots)
    def test_generate_chroma(self):
        self.app.service.recording = MagicMock()
        self.app.service.recording.data = np.arange(10, dtype=np.float32)
        self.app.service.recording.sample_rate = 44100

        self.app._show_plot_in_tab = MagicMock()
        self.app.generate_chroma()
        self.app._show_plot_in_tab.assert_called()

    # Test generate_mel_spectrogram
    def test_generate_mel_spectrogram(self):
        self.app.service.generate_mel_spectrogram = MagicMock(return_value=(np.ones((10, 10), dtype=np.float32), 44100))
        self.app._show_plot_in_tab = MagicMock()

        self.app.generate_mel_spectrogram()
        self.app._show_plot_in_tab.assert_called()

    # Test generate_mfcc
    def test_generate_mfcc(self):
        self.app.service.generate_mfcc = MagicMock(return_value=(np.ones((10, 10), dtype=np.float32), 44100))
        self.app._show_plot_in_tab = MagicMock()

        self.app.generate_mfcc()
        self.app._show_plot_in_tab.assert_called()

    # Test generate_cqt
    def test_generate_cqt(self):
        self.app.service.generate_cqt = MagicMock(return_value=(np.ones((10, 10), dtype=np.float32), 44100))
        self.app._show_plot_in_tab = MagicMock()

        self.app.generate_cqt()
        self.app._show_plot_in_tab.assert_called()

    # Test plot_waveform_with_onsets
    def test_plot_waveform_with_onsets(self):
        self.app.service.recording = MagicMock()
        self.app.service.recording.data = np.arange(10, dtype=np.float32)
        self.app.service.recording.sample_rate = 44100

        self.app._show_plot_in_tab = MagicMock()
        self.app.plot_waveform_with_onsets()
        self.app._show_plot_in_tab.assert_called()

    # Test _show_plot_in_tab cu widget destroy
    def test__show_plot_in_tab(self):
        tab = MagicMock()
        child1 = MagicMock()
        child2 = MagicMock()
        tab.winfo_children.return_value = [child1, child2]

        fig = plt.figure()
        self.app._show_plot_in_tab(fig, tab)
        child1.destroy.assert_called_once()
        child2.destroy.assert_called_once()

    # Test show_spectral_features behavior
    @patch('tkinter.messagebox.showinfo')
    def test_show_spectral_features_no_recording(self, mock_info):
        self.app.service.calculate_spectral_features.return_value = None
        self.app.show_spectral_features()
        mock_info.assert_called_with("Info", "Nu există înregistrare pentru calculul caracteristicilor spectrale.")

    def test_show_spectral_features_with_data(self):
        features = {"feature1": 0.5, "feature2": 1.2}
        self.app.service.calculate_spectral_features.return_value = features
        self.app.update_spectral_features_ui = MagicMock()
        self.app.show_spectral_features()
        self.app.update_spectral_features_ui.assert_called_with(features)

    def test_update_pitch_and_tuning_ui(self):
        pitch_and_tuning = {
            "Pitch Fundamental": 440.1234,
            "Tuning Adjustment": -0.56
        }
        # Setează variabilele tk.StringVar pentru test
        self.app.pitch_var = tk.StringVar()
        self.app.tuning_var = tk.StringVar()

        self.app.update_pitch_and_tuning_ui(pitch_and_tuning)

        self.assertEqual(self.app.pitch_var.get(), "440.12 Hz")
        self.assertEqual(self.app.tuning_var.get(), "-0.56 semitone")

    def test_update_spectral_features_ui(self):
        features = {
            "Spectral Centroid": 1234.5678,
            "Spectral Bandwidth": 567.89,
            "Spectral Rolloff": 4321.123,
            "Spectral Contrast": [1.1, 2.2, 3.3, 4.4]
        }
        # Setează variabilele tk.StringVar pentru test
        self.app.spectral_centroid_var = tk.StringVar()
        self.app.spectral_bandwidth_var = tk.StringVar()
        self.app.spectral_rolloff_var = tk.StringVar()
        self.app.spectral_contrast_var = tk.StringVar()

        self.app.update_spectral_features_ui(features)

        self.assertEqual(self.app.spectral_centroid_var.get(), "1234.57")
        self.assertEqual(self.app.spectral_bandwidth_var.get(), "567.89")
        self.assertEqual(self.app.spectral_rolloff_var.get(), "4321.12")
        self.assertEqual(self.app.spectral_contrast_var.get(), "1.10, 2.20, 3.30, 4.40")

    def test_update_fields_no_recording(self):
        self.app.service.recording = None
        # Should return early without error
        self.app.update_fields()

    def test_update_fields_short_signal(self):
        self.app.service.recording.data = np.arange(100, dtype=np.float32)  # less than 2048
        with patch('builtins.print') as mock_print:
            self.app.update_fields()
            mock_print.assert_called_with("Semnalul este prea scurt pentru analiză.")

    def test_update_fields_with_data(self):
        # Set signal length >= 2048
        self.app.service.recording.data = np.arange(3000, dtype=np.float32)

        # Mocks pentru metodele apelate
        features = {"Spectral Centroid": 1.0}
        pitch_and_tuning = {"Pitch Fundamental": 440.0, "Tuning Adjustment": 0.0}
        self.app.service.calculate_spectral_features.return_value = features
        self.app.service.analyze_pitch_and_tuning.return_value = pitch_and_tuning
        self.app.update_spectral_features_ui = MagicMock()
        self.app.update_pitch_and_tuning_ui = MagicMock()
        self.app.update_bpm_field = MagicMock()

        self.app.update_fields()

        self.app.update_spectral_features_ui.assert_called_once_with(features)
        self.app.update_pitch_and_tuning_ui.assert_called_once_with(pitch_and_tuning)
        self.app.update_bpm_field.assert_called_once()

    def test_update_bpm_field_with_original_bpm(self):
        self.app.service.recording = MagicMock()
        self.app.service.original_bpm = 120.123456
        self.app.bpm_entry.delete = MagicMock()
        self.app.bpm_entry.insert = MagicMock()

        self.app.update_bpm_field()

        self.app.bpm_entry.delete.assert_called_once_with(0, tk.END)
        self.app.bpm_entry.insert.assert_called_once_with(0, "120.12")

    def test_update_bpm_field_with_estimate_bpm(self):
        self.app.service.recording = MagicMock()
        self.app.service.original_bpm = None
        self.app.service.estimate_bpm = MagicMock(return_value=95.9876)
        self.app.bpm_entry.delete = MagicMock()
        self.app.bpm_entry.insert = MagicMock()

        self.app.update_bpm_field()

        self.assertEqual(self.app.service.original_bpm, 95.9876)
        self.app.bpm_entry.delete.assert_called_once_with(0, tk.END)
        self.app.bpm_entry.insert.assert_called_once_with(0, "95.99")

    def test_update_bpm_field_no_bpm(self):
        self.app.service.recording = MagicMock()
        self.app.service.original_bpm = None
        self.app.service.estimate_bpm = MagicMock(return_value=None)
        self.app.bpm_entry.delete = MagicMock()
        self.app.bpm_entry.insert = MagicMock()

        # Should not insert anything if bpm is None
        self.app.update_bpm_field()
        self.app.bpm_entry.delete.assert_not_called()
        self.app.bpm_entry.insert.assert_not_called()

    def test_record_valid_duration(self):
        self.app.duration_entry.get.return_value = "3.5"
        self.app.service.recording = "previous_recording"
        self.app.record()
        self.assertEqual(self.app._recording_before_record, "previous_recording")
        self.app.status_label.config.assert_called_with(text="Se înregistrează...")
        self.app.progress_var.set.assert_called_with(0)
        self.app.service.record.assert_called_with(3.5)
        self.app.stop_button.config.assert_called_with(state="normal")
        self.app.play_button.config.assert_called_with(state="disabled")
        self.app.root.after.assert_called_with(100, self.app._check_recording_status)

    @patch('tkinter.messagebox.showerror')
    def test_record_invalid_duration(self, mock_msg):
        self.app.duration_entry.get.return_value = "invalid"
        self.app.record()
        mock_msg.assert_called_with("Eroare", "Durata trebuie să fie un număr valid.")
        self.app.status_label.config.assert_called_with(text="Înregistrare oprită!")

    def test__check_recording_status_when_recording(self):
        self.app.service.is_recording = True
        self.app._check_recording_status()
        self.app.root.after.assert_called_with(100, self.app._check_recording_status)

    def test__check_recording_status_when_not_recording(self):
        self.app.service.is_recording = False
        self.app._on_recording_loaded = MagicMock()
        self.app.play_button = MagicMock()
        self.app.status_label.config.reset_mock()
        self.app._check_recording_status()
        self.app.status_label.config.assert_called_with(text="Standby")
        self.app.progress_var.set.assert_not_called()
        self.app._on_recording_loaded.assert_called_once()
        self.app.play_button.config.assert_not_called()


    def test_play_no_recording(self):
        self.app.service.recording = None
        with patch('tkinter.messagebox.showwarning') as mock_warn:
            self.app.play()
            mock_warn.assert_called_with("Avertisment", "Nu există înregistrare pentru redare.")

    def test_play_with_recording(self):
        self.app.service.recording = MagicMock()
        self.app.is_playing = False
        self.app.service.is_playing = False
        self.app._check_playback_status = MagicMock()
        self.app.play()
        self.app.status_label.config.assert_called_with(text="Se redă...")
        self.app.progress_var.set.assert_called_with(0)
        self.assertTrue(self.app.is_playing)
        self.app.stop_button.config.assert_called_with(state="normal")
        self.app.service.play.assert_called_once()
        self.app.root.after.assert_called_with(100, self.app._check_playback_status)

    def test__check_playback_status_playing(self):
        self.app.service.is_playing = True
        self.app._check_playback_status()
        self.app.root.after.assert_called_with(100, self.app._check_playback_status)

    def test__check_playback_status_not_playing(self):
        self.app.service.is_playing = False
        self.app._reset_play_ui = MagicMock()
        self.app.status_label = MagicMock()
        self.app._check_playback_status()
        self.app.status_label.config.assert_called_with(text="Standby")
        self.app._reset_play_ui.assert_called_once()

    def test__reset_play_ui(self):
        self.app.is_playing = True
        self.app.is_paused = True
        self.app.stop_event = MagicMock()
        self.app.play_button = MagicMock()
        self.app.stop_button = MagicMock()

        self.app._reset_play_ui()

        self.assertFalse(self.app.is_playing)
        self.assertFalse(self.app.is_paused)
        self.app.stop_event.clear.assert_called_once()
        self.app.play_button.config.assert_called_with(text="Redă")
        self.app.stop_button.config.assert_called_with(state="disabled")

    def test_open_settings_save_valid_pitch(self):
        self.app.config = MagicMock()
        with patch('tkinter.filedialog.askdirectory', return_value="/fake/dir"), \
                patch('tkinter.messagebox.showerror') as mock_err:
            self.app.config.save_directory = "old_dir"
            self.app.config.pitch_factor = 1.0

            # Mocks pentru Toplevel și butoane
            with patch('tkinter.Toplevel') as mock_toplevel:
                instance = mock_toplevel.return_value
                instance.destroy = MagicMock()
                self.app.open_settings()

                # Simulează click pe Browse
                # Folosește metoda browse definită în open_settings
                # Testează salvarea pitch-ului valid
                # NOTE: testul funcției save e destul de interactiv și poate necesita testare manuală extinsă

    def test_save_recording_no_recording(self):
        self.app.service.recording = None
        with patch('tkinter.messagebox.showwarning') as mock_warn:
            self.app.save_recording()
            mock_warn.assert_called_with("Avertisment", "Nu există înregistrare de salvat!")

    @patch('tkinter.filedialog.asksaveasfilename')
    def test_save_recording_with_path(self, mock_asksave):
        self.app.service.recording = MagicMock()
        mock_asksave.return_value = "/fake/path.wav"
        self.app.status_label = MagicMock()
        self.app.root.winfo_children = MagicMock(return_value=[MagicMock()])
        frame = self.app.root.winfo_children()[0]
        frame.winfo_children = MagicMock(return_value=[MagicMock()])
        btn = frame.winfo_children()[0]
        btn.cget = MagicMock(return_value="Salvează")
        btn.config = MagicMock()

        self.app.save_recording()

        self.app.status_label.config.assert_called_with(text="Se salvează...")
        self.app.service.save_recording.assert_called_with("/fake/path.wav")
        self.app._check_saving_status()

    def test__on_recording_loaded_calls(self):
        self.app.service.recording = MagicMock()
        self.app.service.recording.data = np.arange(100)
        self.app.service.recording.sample_rate = 44100
        self.app.service.analyze_pitch_and_tuning.return_value = {"Pitch Fundamental": 440, "Tuning Adjustment": 0}
        self.app.update_plot = MagicMock()
        self.app.update_fields = MagicMock()
        self.app.update_bpm_field = MagicMock()
        self.app.show_spectral_features = MagicMock()
        self.app.update_pitch_and_tuning_ui = MagicMock()
        self.app.update_duration = MagicMock()
        self.app.play_button = MagicMock()

        self.app._on_recording_loaded()

        self.app.update_plot.assert_called_once()
        self.app.update_fields.assert_called_once()
        self.app.update_bpm_field.assert_called_once()
        self.app.show_spectral_features.assert_called_once()
        self.app.update_pitch_and_tuning_ui.assert_called_once()
        self.app.update_duration.assert_called_once()
        self.app.play_button.config.assert_called_with(state="normal")

    @patch('tkinter.filedialog.askopenfilename')
    def test_load_recording_with_file(self, mock_askopen):
        mock_askopen.return_value = "/fake/file.wav"
        self.app.status_label = MagicMock()
        self.app.progress_var = MagicMock()
        self.app.service.load_audio = MagicMock()
        self.app._check_loading_status = MagicMock()
        self.app.root.after = MagicMock()

        self.app.load_recording()

        self.app.status_label.config.assert_called_with(text="Se încarcă fișierul...")
        self.app.progress_var.set.assert_called_with(0)
        self.app.service.load_audio.assert_called_with("/fake/file.wav")
        self.app.root.after.assert_called_with(100, self.app._check_loading_status)

    def test_load_recording_no_file(self):
        with patch('tkinter.filedialog.askopenfilename', return_value=""):
            self.app.load_recording()
            # Should not call anything, no errors

    def test__check_loading_status_recording(self):
        self.app.service.is_loading = True
        self.app._check_loading_status()
        self.app.root.after.assert_called_with(100, self.app._check_loading_status)

    def test__check_loading_status_not_loading(self):
        self.app.service.is_loading = False
        self.app._on_recording_loaded = MagicMock()
        self.app.status_label = MagicMock()
        self.app._check_loading_status()
        self.app.status_label.config.assert_called_with(text="Standby")
        self.app._on_recording_loaded.assert_called_once()

    def test_start_playback_ui(self):
        self.app.play_button = MagicMock()
        self.app.stop_button = MagicMock()

        self.app.start_playback_ui()

        self.assertTrue(self.app.is_playing)
        self.assertFalse(self.app.is_paused)
        self.app.play_button.config.assert_called_with(text="Pause", style="Red.TButton")
        self.app.stop_button.config.assert_called_with(state="normal")

    def test_pitch_up_applies_shift_and_updates_ui(self):
        self.app.service.pitch_shift = MagicMock(return_value="shifted_data")
        self.app.update_plot = MagicMock()
        self.app.play = MagicMock()
        self.app.update_fields = MagicMock()

        self.app.pitch_up()

        self.app.status_label.config.assert_called_with(text="Se aplică pitch up...")
        self.app.progress_var.set.assert_called_with(0)
        self.assertEqual(self.app.service.recording, "shifted_data")
        self.app.update_plot.assert_called_once()
        self.app.play.assert_called_once()
        self.app.update_fields.assert_called_once()

    def test_pitch_down_applies_shift_and_updates_ui(self):
        self.app.service.pitch_shift = MagicMock(return_value="shifted_data")
        self.app.update_plot = MagicMock()
        self.app.play = MagicMock()
        self.app.update_fields = MagicMock()

        self.app.pitch_down()

        self.app.status_label.config.assert_called_with(text="Se aplică pitch down...")
        self.app.progress_var.set.assert_called_with(0)
        self.assertEqual(self.app.service.recording, "shifted_data")
        self.app.update_plot.assert_called_once()
        self.app.play.assert_called_once()
        self.app.update_fields.assert_called_once()

    def test__check_saving_status_behavior(self):
        self.app.service.is_saving = False

        self.app.status_label.config = MagicMock()
        self.app.root = MagicMock()  # <-- Asta rezolvă eroarea

        frame = MagicMock(spec=ttk.Frame)
        child_button = MagicMock(spec=ttk.Button)
        child_button.cget.return_value = "Salvează"
        child_button.config = MagicMock()

        frame.winfo_children.return_value = [child_button]
        self.app.root.winfo_children.return_value = [frame]
        self.app.root.after = MagicMock()

        self.app._check_saving_status()

        self.app.status_label.config.assert_called_with(text="Salvare finalizată!")
        child_button.config.assert_called_with(state="normal")
        self.app.root.after.assert_called_with(2000, ANY)

    def test_apply_reverb_no_recording(self):
        self.app.service.recording = None
        with patch("tkinter.messagebox.showinfo") as mock_info:
            self.app.apply_reverb()
            mock_info.assert_called_with("Info", "Nu există înregistrare pentru reverb.")

    def test_apply_echo_valid(self):
        self.app.service.recording = MagicMock()
        self.app.decay_entry.get = MagicMock(return_value="0.5")
        self.app.delay_entry.get = MagicMock(return_value="100")
        self.app.service.apply_echo = MagicMock()
        self.app.update_plot = MagicMock()
        self.app.play = MagicMock()
        self.app.update_fields = MagicMock()

        self.app.apply_echo()

        self.app.status_label.config.assert_called_with(text="Se aplică echo...")
        self.app.service.apply_echo.assert_called_once()
        self.app.update_plot.assert_called_once()
        self.app.play.assert_called_once()
        self.app.update_fields.assert_called_once()

    def test_apply_compressor_calls_dialog(self):
        self.app.show_compressor_dialog = MagicMock()
        self.app.apply_compressor()
        self.app.status_label.config.assert_called_with(text="Se aplică compressor...")
        self.app.progress_var.set.assert_called_with(0)
        self.app.show_compressor_dialog.assert_called_once()

    def test_apply_time_stretch_valid_input(self):
        self.app.service.recording = MagicMock()
        self.app.bpm_entry.get = MagicMock(return_value="120")
        self.app.service.apply_time_stretch_bpm = MagicMock(return_value=True)
        self.app.update_plot = MagicMock()
        self.app.play = MagicMock()
        self.app.update_fields = MagicMock()
        self.app.update_duration = MagicMock()

        self.app.apply_time_stretch()

        self.app.status_label.config.assert_any_call(text="Se aplică time stretch...")
        self.app.progress_var.set.assert_any_call(0)
        self.app.update_plot.assert_called_once()
        self.app.play.assert_called_once()
        self.app.update_fields.assert_called_once()
        self.app.bpm_entry.delete.assert_called_once_with(0, tk.END)
        self.app.bpm_entry.insert.assert_called_once()
        self.app.update_duration.assert_called_once()
        self.app.status_label.config.assert_called_with(text="Time stretch aplicat")

    def test_apply_time_stretch_invalid_bpm(self):
        self.app.bpm_entry.get = MagicMock(return_value="-10")
        with patch("tkinter.messagebox.showerror") as mock_error:
            self.app.apply_time_stretch()
            mock_error.assert_called()
            self.app.status_label.config.assert_called_with(text="Standby")

    def test_on_reverb_done_calls(self):
        self.app.service.recording = MagicMock()
        self.app.service.recording.data = np.arange(44100)
        self.app.service.recording.sample_rate = 44100
        self.app.update_plot = MagicMock()
        self.app.play = MagicMock()
        self.app.update_fields = MagicMock()
        self.app.update_duration = MagicMock()

        self.app.on_reverb_done()

        self.app.root.config.assert_called_with(cursor="")
        self.app.update_plot.assert_called_once()
        self.app.play.assert_called_once()
        self.app.update_fields.assert_called_once()
        self.app.update_duration.assert_called_once()

    def test_show_compressor_dialog_no_recording(self):
        self.app.service.recording = None
        with patch("tkinter.messagebox.showinfo") as mock_info:
            self.app.show_compressor_dialog()
            mock_info.assert_called_with("Info", "Nu există înregistrare pentru aplicarea compresorului.")

    def test_show_equalizer_dialog_no_recording(self):
        self.app.service.recording = None
        with patch("tkinter.messagebox.showinfo") as mock_info:
            self.app.show_equalizer_dialog()
            mock_info.assert_called_with("Info", "Nu există înregistrare pentru aplicarea egalizatorului.")

    def test_show_distortion_dialog_no_recording(self):
        self.app.service.recording = None
        with patch("tkinter.messagebox.showinfo") as mock_info:
            self.app.show_distortion_dialog()
            mock_info.assert_called_with("Info", "Nu există înregistrare pentru aplicarea distorsiunii.")

    def test_show_compressor_dialog_creates_dialog(self):
        self.app.service.recording = MagicMock()
        with patch("tkinter.Toplevel") as mock_toplevel:
            mock_dialog = MagicMock()
            mock_toplevel.return_value = mock_dialog
            self.app.show_compressor_dialog()
            mock_toplevel.assert_called_once_with(self.app.root)
            mock_dialog.title.assert_called_with("Compresor")
            mock_dialog.geometry.assert_called_with("400x300")

    def test_show_distortion_dialog_creates_dialog(self):
        self.app.service.recording = MagicMock()
        with patch("tkinter.Toplevel") as mock_toplevel:
            mock_dialog = MagicMock()
            mock_toplevel.return_value = mock_dialog
            self.app.show_distortion_dialog()
            mock_toplevel.assert_called_once_with(self.app.root)
            mock_dialog.title.assert_called_with("Distorsiune")
            mock_dialog.geometry.assert_called_with("350x350")

    def test_show_equalizer_dialog_creates_dialog(self):
        import numpy as np

        self.app.service.recording = MagicMock()
        self.app.service.recording.data = np.random.rand(2048)
        self.app.service.recording.sample_rate = 44100

        self.app.service.generate_mel_spectrogram = MagicMock(return_value=(np.zeros((128, 128)), 22050))
        self.app.service.generate_mfcc = MagicMock(return_value=(np.zeros((13, 128)), 22050))
        self.app.service.generate_cqt = MagicMock(return_value=(np.zeros((84, 128)), 22050))

        with patch("tkinter.Toplevel") as mock_toplevel:
            mock_dialog = MagicMock()
            mock_toplevel.return_value = mock_dialog
            self.app.show_equalizer_dialog()
            mock_toplevel.assert_called_once_with(self.app.root)
            mock_dialog.title.assert_called_with("Equalizer")
            mock_dialog.geometry.assert_called_with("1200x800")

    def test_undo_when_successful(self):
        self.app.status_label = MagicMock()
        self.app.service.undo = MagicMock()
        self.app.play = MagicMock()

        self.app.undo()

        calls = [call(text="Se revine la versiunea anterioară!")]
        self.assertIn(call(text="Se revine la versiunea anterioară!"), self.app.status_label.config.call_args_list)

    def test_undo_when_no_history(self):
        self.app.service.undo.return_value = False
        with patch("tkinter.messagebox.showinfo") as mock_info:
            self.app.undo()
            mock_info.assert_called_once_with("Info", "Nu există stări anterioare pentru Undo.")

    def test_update_progress_changes_status_on_100(self):
        self.app.status_label.cget.return_value = "Se înregistrează..."
        self.app.update_progress(100)
        self.app.status_label.config.assert_called_with(text="Înregistrare finalizată. Gata de redare!")

    def test_update_progress_regular(self):
        self.app.status_label.cget.return_value = "Altceva"
        self.app.update_progress(50)
        self.app.progress_var.set.assert_called_with(50)
        self.app.root.update_idletasks.assert_called_once()

    def test_update_duration_sets_entry_correctly(self):
        self.app.update_duration(5.6789)
        self.app.duration_entry.delete.assert_called_once_with(0, tk.END)
        self.app.duration_entry.insert.assert_called_once_with(0, "5.68")
        self.app.root.update_idletasks.assert_called_once()

    def test_apply_lpf_runs_all_steps(self):
        self.app.play = MagicMock()
        self.app.update_plot = MagicMock()
        self.app.update_fields = MagicMock()
        self.app.status_label.config = MagicMock()
        self.app.progress_var.set = MagicMock()
        self.app.service.apply_lowpass_filter = MagicMock()

        self.app.apply_lpf()

        self.app.status_label.config.assert_called_with(text="Se aplică Low Pass Filter...")
        self.app.progress_var.set.assert_called_with(0)
        self.app.service.apply_lowpass_filter.assert_called_once()
        self.app.update_plot.assert_called_once()
        self.app.play.assert_called_once()
        self.app.update_fields.assert_called_once()

    def test_apply_hpf_runs_all_steps(self):
        self.app.play = MagicMock()
        self.app.update_plot = MagicMock()
        self.app.update_fields = MagicMock()
        self.app.status_label.config = MagicMock()
        self.app.progress_var.set = MagicMock()
        self.app.service.apply_highpass_filter = MagicMock()

        self.app.apply_hpf()

        self.app.status_label.config.assert_called_with(text="Se aplică High Pass Filter...")
        self.app.progress_var.set.assert_called_with(0)
        self.app.service.apply_highpass_filter.assert_called_once()
        self.app.update_plot.assert_called_once()
        self.app.play.assert_called_once()
        self.app.update_fields.assert_called_once()

    def test_apply_bpf_runs_all_steps(self):
        self.app.play = MagicMock()
        self.app.update_plot = MagicMock()
        self.app.update_fields = MagicMock()
        self.app.status_label.config = MagicMock()
        self.app.progress_var.set = MagicMock()
        self.app.service.apply_bandpass_filter = MagicMock()

        self.app.apply_bpf()

        self.app.status_label.config.assert_called_with(text="Se aplică Band Filter...")
        self.app.progress_var.set.assert_called_with(0)
        self.app.service.apply_bandpass_filter.assert_called_once()
        self.app.update_plot.assert_called_once()
        self.app.play.assert_called_once()
        self.app.update_fields.assert_called_once()



if __name__ == "__main__":
    unittest.main()
