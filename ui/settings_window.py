# /audio_app/ui/settings_window.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from domain.config import Config

class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.config = config
        self.title("Setări Aplicație")
        self.geometry("300x250")

        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self)
        frame.pack(padx=10, pady=10)

        # Pitch Factor
        ttk.Label(frame, text="Pitch Factor:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.pitch_factor_entry = ttk.Entry(frame)
        self.pitch_factor_entry.grid(row=0, column=1, padx=5, pady=5)
        self.pitch_factor_entry.insert(0, str(self.config.pitch_factor))

        # Sample Rate
        ttk.Label(frame, text="Sample Rate:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.sample_rate_entry = ttk.Entry(frame)
        self.sample_rate_entry.grid(row=1, column=1, padx=5, pady=5)
        self.sample_rate_entry.insert(0, str(self.config.sample_rate))

        # Change Save Directory
        ttk.Label(frame, text="Direct. Salvare:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.save_dir_entry = ttk.Entry(frame)
        self.save_dir_entry.grid(row=2, column=1, padx=5, pady=5)
        self.save_dir_entry.insert(0, self.config.save_directory)

        change_dir_btn = ttk.Button(frame, text="Schimbă Director", command=self.change_directory)
        change_dir_btn.grid(row=3, column=0, columnspan=2, pady=10)

        save_btn = ttk.Button(self, text="Salvează", command=self.save_settings)
        save_btn.pack(pady=5)

    def change_directory(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.save_dir_entry.delete(0, tk.END)
            self.save_dir_entry.insert(0, folder_selected)

    def save_settings(self):
        try:
            self.config.pitch_factor = float(self.pitch_factor_entry.get())
            self.config.sample_rate = int(self.sample_rate_entry.get())
            self.config.save_directory = self.save_dir_entry.get()
            self.destroy()
        except ValueError:
            messagebox.showerror("Eroare", "Valorile introduse nu sunt valide.")
