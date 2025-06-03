import unittest
from unittest.mock import MagicMock, patch
import tkinter as tk
from ui.settings_window import SettingsWindow
from domain.config import Config


class TestSettingsWindow(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # ascunde fereastra principală
        self.config = Config()
        self.config.pitch_factor = 1.2
        self.config.sample_rate = 44100
        self.config.save_directory = "/path/to/save"

        self.window = SettingsWindow(self.root, self.config)

    def tearDown(self):
        self.window.destroy()
        self.root.destroy()

    def test_initial_values_are_displayed(self):
        self.assertEqual(self.window.pitch_factor_entry.get(), "1.2")
        self.assertEqual(self.window.sample_rate_entry.get(), "44100")
        self.assertEqual(self.window.save_dir_entry.get(), "/path/to/save")

    @patch("tkinter.filedialog.askdirectory")
    def test_change_directory_updates_entry(self, mock_askdir):
        mock_askdir.return_value = "/new/path"
        self.window.change_directory()
        self.assertEqual(self.window.save_dir_entry.get(), "/new/path")

    def test_save_settings_updates_config(self):
        self.window.pitch_factor_entry.delete(0, tk.END)
        self.window.pitch_factor_entry.insert(0, "2.0")
        self.window.sample_rate_entry.delete(0, tk.END)
        self.window.sample_rate_entry.insert(0, "22050")
        self.window.save_dir_entry.delete(0, tk.END)
        self.window.save_dir_entry.insert(0, "/new/dir")

        with patch.object(self.window, "destroy") as mock_destroy:
            self.window.save_settings()

        self.assertEqual(self.config.pitch_factor, 2.0)
        self.assertEqual(self.config.sample_rate, 22050)
        self.assertEqual(self.config.save_directory, "/new/dir")
        mock_destroy.assert_called_once()

    @patch("tkinter.messagebox.showerror")
    def test_save_settings_shows_error_on_invalid_input(self, mock_error):
        self.window.pitch_factor_entry.delete(0, tk.END)
        self.window.pitch_factor_entry.insert(0, "invalid")
        self.window.save_settings()
        mock_error.assert_called_once_with("Eroare", "Valorile introduse nu sunt valide.")
