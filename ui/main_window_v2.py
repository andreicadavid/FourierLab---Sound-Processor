import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import sounddevice as sd
import numpy as np
import librosa
import librosa.display
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from service.audio_service import AudioService
from repo.audio_repo import AudioRepository
from domain.config import Config

class MainWindowV2:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicație Procesare Sunet")
        self.root.geometry("1100x800")

        self.config = Config()
        self.service = AudioService(self.config)

        self.setup_menu()
        self.setup_ui()

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Settings", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Audio Devices", command=self.open_audio_devices)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        # --- Meniu Cache ---
        cache_menu = tk.Menu(menubar, tearoff=0)
        cache_menu.add_command(label="Show Cache Panel", command=self.show_cache_panel)
        menubar.add_cascade(label="Cache", menu=cache_menu)
        # --- Meniu Export ---
        export_menu = tk.Menu(menubar, tearoff=0)
        export_menu.add_command(label="Export Onsets", command=self.export_onsets_ui)
        menubar.add_cascade(label="Export", menu=export_menu)

        self.root.config(menu=menubar)

    def setup_ui(self):
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12))

        # Cadrul principal
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        # Sub-cadre pentru coloane
        left_frame = ttk.Frame(frame)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        right_frame = ttk.Frame(frame)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # --- Coloana stângă: Setări generale ---
        ttk.Label(left_frame, text="Durată (secunde):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.duration_entry = ttk.Entry(left_frame)
        self.duration_entry.grid(row=0, column=1, padx=5, pady=5)
        self.duration_entry.insert(0, "5")

        ttk.Label(left_frame, text="Sample Rate:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.sample_rate_entry = ttk.Entry(left_frame)
        self.sample_rate_entry.grid(row=1, column=1, padx=5, pady=5)
        self.sample_rate_entry.insert(0, str(self.config.sample_rate))

        ttk.Label(left_frame, text="Decay (0-1):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.decay_entry = ttk.Entry(left_frame)
        self.decay_entry.grid(row=2, column=1, padx=5, pady=5)
        self.decay_entry.insert(0, "0.5")

        ttk.Label(left_frame, text="Delay (secunde):").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.delay_entry = ttk.Entry(left_frame)
        self.delay_entry.grid(row=3, column=1, padx=5, pady=5)
        self.delay_entry.insert(0, "0.02")

        ttk.Label(left_frame, text="IR Duration (secunde):").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.ir_duration_entry = ttk.Entry(left_frame)
        self.ir_duration_entry.grid(row=4, column=1, padx=5, pady=5)
        self.ir_duration_entry.insert(0, "0.5")

        # --- Coloana dreaptă: Caracteristici spectrale ---
        ttk.Label(right_frame, text="Spectral Centroid:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.spectral_centroid_var = tk.StringVar()
        self.spectral_centroid_entry = ttk.Entry(right_frame, textvariable=self.spectral_centroid_var, state="readonly")
        self.spectral_centroid_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(right_frame, text="Spectral Bandwidth:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.spectral_bandwidth_var = tk.StringVar()
        self.spectral_bandwidth_entry = ttk.Entry(right_frame, textvariable=self.spectral_bandwidth_var, state="readonly")
        self.spectral_bandwidth_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(right_frame, text="Spectral Rolloff:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.spectral_rolloff_var = tk.StringVar()
        self.spectral_rolloff_entry = ttk.Entry(right_frame, textvariable=self.spectral_rolloff_var, state="readonly")
        self.spectral_rolloff_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(right_frame, text="Spectral Contrast:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.spectral_contrast_var = tk.StringVar()
        self.spectral_contrast_entry = ttk.Entry(right_frame, textvariable=self.spectral_contrast_var, state="readonly")
        self.spectral_contrast_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(right_frame, text="Pitch Fundamental (Hz):").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.pitch_var = tk.StringVar()
        self.pitch_entry = ttk.Entry(right_frame, textvariable=self.pitch_var, state="readonly")
        self.pitch_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(right_frame, text="Tuning Adjustment (semitone):").grid(row=5, column=0, sticky="e", padx=5, pady=5)
        self.tuning_var = tk.StringVar()
        self.tuning_entry = ttk.Entry(right_frame, textvariable=self.tuning_var, state="readonly")
        self.tuning_entry.grid(row=5, column=1, padx=5, pady=5)

        # --- Buttons: grupare pe rânduri și categorii ---
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(padx=10, pady=5)

        # Rând 0: Înregistrare, redare, salvare, undo
        ttk.Button(btn_frame, text="Înregistrează", command=self.record).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="Redă", command=self.play).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="Salvează", command=self.save_recording).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(btn_frame, text="Încarcă", command=self.load_recording).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(btn_frame, text="Undo", command=self.undo).grid(row=0, column=4, padx=5, pady=5)

        # Rând 1: Efecte audio
        ttk.Button(btn_frame, text="Pitch Up", command=self.pitch_up).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="Pitch Down", command=self.pitch_down).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="Aplică Reverb", command=self.apply_reverb).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(btn_frame, text="Aplică Echo", command=self.apply_echo).grid(row=1, column=3, padx=5, pady=5)
        ttk.Button(btn_frame, text="Aplică Compressor", command=self.apply_compressor).grid(row=1, column=4, padx=5, pady=5)
        ttk.Button(btn_frame, text="Distortion", command=self.show_distortion_dialog).grid(row=1, column=5, padx=5,
                                                                                           pady=5)
        ttk.Button(btn_frame, text="Equalizer", command=self.show_equalizer_dialog).grid(row=1, column=6, padx=5,
                                                                                         pady=5)

        # Rând 3: Filtre DSP
        ttk.Label(btn_frame, text="Filtre DSP:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        ttk.Button(btn_frame, text="LPF", command=self.apply_lpf).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="HPF", command=self.apply_hpf).grid(row=3, column=2, padx=5, pady=5)
        ttk.Button(btn_frame, text="BPF", command=self.apply_bpf).grid(row=3, column=3, padx=5, pady=5)

        #Redare/Pause
        self.is_playing = False
        self.is_paused = False

        self.play_button = ttk.Button(btn_frame, text="Redă", command=self.toggle_play_pause)
        self.play_button.grid(row=0, column=1, padx=5, pady=5)
        self.stop_button = ttk.Button(btn_frame, text="Stop", command=self.stop_playback)
        self.stop_button.grid(row=0, column=6, padx=5, pady=5)  # sau altă poziție liberă
        style = ttk.Style()
        style.configure("Red.TButton", foreground="white", background="red")

        # --- Time-stretch (poate fi sub butoane, nu în grid) ---
        self.bpm_label = tk.Label(self.root, text="Target BPM:")
        self.bpm_label.pack()
        self.bpm_entry = tk.Entry(self.root)
        self.bpm_entry.pack()
        self.bpm_button = tk.Button(self.root, text="Apply Stretch", command=self.apply_time_stretch)
        self.bpm_button.pack(pady=5)

        # --- Notebook pentru ploturi (footer cu file) ---
        self.plot_notebook = ttk.Notebook(self.root)
        self.plot_notebook.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)

        # Tab-uri pentru fiecare tip de plot
        self.waveform_tab = ttk.Frame(self.plot_notebook)
        self.spectrogram_tab = ttk.Frame(self.plot_notebook)
        self.mel_tab = ttk.Frame(self.plot_notebook)
        self.onset_tab = ttk.Frame(self.plot_notebook)
        self.cqt_tab = ttk.Frame(self.plot_notebook)
        self.mfcc_tab = ttk.Frame(self.plot_notebook)
        self.chroma_tab = ttk.Frame(self.plot_notebook)

        self.plot_notebook.add(self.waveform_tab, text="Waveform")
        self.plot_notebook.add(self.spectrogram_tab, text="Spectrogramă")
        self.plot_notebook.add(self.mel_tab, text="Mel Spectrogram")
        self.plot_notebook.add(self.onset_tab, text="Onsets")
        self.plot_notebook.add(self.cqt_tab, text="CQT")
        self.plot_notebook.add(self.mfcc_tab, text="MFCC")
        self.plot_notebook.add(self.chroma_tab, text="Chroma")

        # Inițial, afișăm waveform-ul
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.waveform_canvas = FigureCanvasTkAgg(self.fig, master=self.waveform_tab)
        self.waveform_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.update_plot()  # Desenează waveform-ul inițial

    def toggle_play_pause(self):
        if not self.is_playing:
            self.is_playing = True
            self.is_paused = False
            self.play_button.config(text="Pause", style="Red.TButton")
            self.stop_button.config(state="normal")
            # Pornește redarea pe thread separat dacă vrei să nu blochezi UI-ul
            threading.Thread(target=self._play_audio, daemon=True).start()
        else:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.play_button.config(text="Redă", style="TButton")
                sd.stop()
            else:
                self.play_button.config(text="Pause", style="Red.TButton")
                threading.Thread(target=self._play_audio, daemon=True).start()

    def _play_audio(self):
        self.service.play()
        # Când redarea s-a terminat, resetează UI-ul (execută pe thread-ul principal)
        self.root.after(0, self._reset_play_ui)

    def stop_playback(self):
        self.is_playing = False
        self.is_paused = False
        self.play_button.config(text="Redă", style="TButton")
        self.stop_button.config(state="disabled")
        sd.stop()

    def show_loading(self, message="Se procesează..."):
        self.loading_win = tk.Toplevel(self.root)
        self.loading_win.title("Loading")
        self.loading_win.geometry("200x80")
        ttk.Label(self.loading_win, text=message).pack(pady=10)
        self.loading_label = ttk.Label(self.loading_win, text="")
        self.loading_label.pack()
        self.loading_running = True
        self.animate_loading()

    def animate_loading(self):
        # Verifică dacă loading_label și loading_win există și nu au fost distruse
        if not hasattr(self, "loading_label") or not hasattr(self, "loading_win"):
            return
        try:
            current = self.loading_label.cget("text")
        except tk.TclError:
            return  # Labelul a fost distrus, ieși din funcție
        if len(current) < 3:
            self.loading_label.config(text=current + ".")
        else:
            self.loading_label.config(text="")
        if getattr(self, "loading_running", False):
            self.root.after(400, self.animate_loading)

    def hide_loading(self):
        self.loading_running = False
        if hasattr(self, "loading_win"):
            try:
                self.loading_win.destroy()
            except Exception:
                pass
            del self.loading_win
        if hasattr(self, "loading_label"):
            del self.loading_label

    def show_cache_panel(self):
        panel = tk.Toplevel(self.root)
        panel.title("Cache History")
        panel.geometry("500x300")

        # Listbox cu pașii cache
        listbox = tk.Listbox(panel, width=70, height=12)
        listbox.pack(padx=10, pady=10, fill="both", expand=True)

        # Populează listbox-ul cu pașii cache
        history = self.service.get_cache_history()
        for entry in history:
            listbox.insert(tk.END, entry)

        def on_select(event):
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                if self.service.load_from_cache(idx):
                    self.update_plot()
                    self.update_fields()
                    self.update_bpm_field()
                    self.service.play()

        # Sari la starea selectată la dublu-click
        listbox.bind("<Double-Button-1>", on_select)

        # Buton explicit de „Jump”
        def jump():
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                if self.service.load_from_cache(idx):
                    self.update_plot()
                    self.update_fields()
                    self.update_bpm_field()
                    self.service.play()

        tk.Button(panel, text="Jump to Selected", command=jump).pack(pady=5)

    def update_plot(self):
        if self.service.recording:
            # Waveform
            self.ax.clear()
            self.ax.plot(self.service.recording.data)
            self.ax.set_title("Semnal audio")
            self.ax.set_xlabel("Eșantion")
            self.ax.set_ylabel("Amplitudine")
            self.ax.grid(True)
            self.waveform_canvas.draw()
            # Nu schimba tab-ul activ!

            # Actualizează toate celelalte ploturi
            self._update_all_plots()

    def _update_all_plots(self):
        # Spectrogramă
        try:
            y = self.service.recording.data
            sr = self.service.recording.sample_rate
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
            ax.set_title('Spectrograma')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, self.spectrogram_tab)
        except Exception:
            pass

        # Chroma
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
            ax.set_title('Chroma')
            fig.colorbar(img, ax=ax)
            self._show_plot_in_tab(fig, self.chroma_tab)
        except Exception:
            pass

        # Mel Spectrogram
        try:
            mel_spec_db, _ = self.service.generate_mel_spectrogram()
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
            ax.set_title("Mel Spectrogram")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, self.mel_tab)
        except Exception:
            pass

        # MFCC
        try:
            mfcc, _ = self.service.generate_mfcc()
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
            ax.set_title("MFCC (Mel Frequency Cepstral Coefficients)")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, self.mfcc_tab)
        except Exception:
            pass

        # CQT
        try:
            cqt_db, _ = self.service.generate_cqt()
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax, cmap='coolwarm')
            ax.set_title("Constant-Q Transform (CQT)")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, self.cqt_tab)
        except Exception:
            pass

        # Onsets
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            fig, ax = plt.subplots(figsize=(8, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.6)
            ax.vlines(onset_times, ymin=min(y), ymax=max(y), color='r', linestyle='--', label='Onsets')
            ax.set_title("Formă de undă cu Onsets")
            ax.set_xlabel("Timp (secunde)")
            ax.set_ylabel("Amplitudine")
            ax.legend()
            self._show_plot_in_tab(fig, self.onset_tab)
        except Exception:
            pass

    def export_onsets_ui(self):
        if self.service.recording is None:
            messagebox.showinfo("Info", "Nu există înregistrare pentru export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
        if path:
            success = self.service.export_onsets(path)
            if success:
                messagebox.showinfo("Succes", f"Onsets exportate în {path}")
            else:
                messagebox.showerror("Eroare", "Exportul a eșuat.")

    def open_audio_devices(self):
        win = tk.Toplevel(self.root)
        win.title("Audio Devices & Buffer Size")

        # Query devices
        devices = sd.query_devices()
        input_devices = [d['name'] for d in devices if d['max_input_channels'] > 0]
        output_devices = [d['name'] for d in devices if d['max_output_channels'] > 0]

        # Input device
        ttk.Label(win, text="Input Device:").grid(row=0, column=0, padx=5, pady=5)
        input_var = tk.StringVar(value=self.config.input_device or input_devices[0])
        input_menu = ttk.Combobox(win, textvariable=input_var, values=input_devices, state="readonly")
        input_menu.grid(row=0, column=1, padx=5, pady=5)

        # Output device
        ttk.Label(win, text="Output Device:").grid(row=1, column=0, padx=5, pady=5)
        output_var = tk.StringVar(value=self.config.output_device or output_devices[0])
        output_menu = ttk.Combobox(win, textvariable=output_var, values=output_devices, state="readonly")
        output_menu.grid(row=1, column=1, padx=5, pady=5)

        # Buffer size
        ttk.Label(win, text="Buffer Size:").grid(row=2, column=0, padx=5, pady=5)
        buffer_var = tk.IntVar(value=self.config.buffer_size)
        buffer_entry = ttk.Entry(win, textvariable=buffer_var)
        buffer_entry.grid(row=2, column=1, padx=5, pady=5)

        def save():
            self.config.input_device = input_var.get()
            self.config.output_device = output_var.get()
            try:
                self.config.buffer_size = int(buffer_var.get())
            except:
                messagebox.showerror("Eroare", "Buffer size trebuie să fie un număr întreg.")
                return
            win.destroy()

        ttk.Button(win, text="Save", command=save).grid(row=3, column=1, pady=10)

    def generate_spectrogram(self):
        if self.service.recording is None:
            messagebox.showinfo("Info","Nu există înregistrare pentru spectrogramă.")
            return
        y = self.service.recording.data
        sr = self.service.recording.sample_rate
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots(figsize=(8, 4))
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        ax.set_title('Spectrograma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        self._show_plot_in_tab(fig, self.spectrogram_tab)
        self.plot_notebook.select(self.spectrogram_tab)

    def generate_chroma(self):
        if self.service.recording is None:
            messagebox.showinfo("Info","Nu există înregistrare pentru chroma.")
            return
        y = self.service.recording.data
        sr = self.service.recording.sample_rate
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        fig, ax = plt.subplots(figsize=(8, 4))
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
        ax.set_title('Chroma')
        fig.colorbar(img, ax=ax)
        self._show_plot_in_tab(fig, self.chroma_tab)
        self.plot_notebook.select(self.chroma_tab)

    def generate_mel_spectrogram(self):
        try:
            mel_spec_db, sr = self.service.generate_mel_spectrogram()
            if mel_spec_db is None:
                messagebox.showinfo("Info", "Nu există înregistrare pentru Mel Spectrogram.")
                return
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
            ax.set_title("Mel Spectrogram")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, self.mel_tab)
            self.plot_notebook.select(self.mel_tab)
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la generarea Mel Spectrogram: {e}")

    def generate_mfcc(self):
        try:
            mfcc, sr = self.service.generate_mfcc()
            if mfcc is None:
                messagebox.showinfo("Info", "Nu există înregistrare pentru MFCC.")
                return
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
            ax.set_title("MFCC (Mel Frequency Cepstral Coefficients)")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, self.mfcc_tab)
            self.plot_notebook.select(self.mfcc_tab)
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la generarea MFCC: {e}")

    def generate_cqt(self):
        try:
            cqt_db, sr = self.service.generate_cqt()
            if cqt_db is None:
                messagebox.showinfo("Info", "Nu există înregistrare pentru Constant-Q Transform (CQT).")
                return
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax, cmap='coolwarm')
            ax.set_title("Constant-Q Transform (CQT)")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, self.cqt_tab)
            self.plot_notebook.select(self.cqt_tab)
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la generarea Constant-Q Transform (CQT): {e}")

    def plot_waveform_with_onsets(self):
        if self.service.recording is None:
            messagebox.showinfo("Info", "Nu există înregistrare pentru detectarea de onsets.")
            return
        y = self.service.recording.data
        sr = self.service.recording.sample_rate
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        fig, ax = plt.subplots(figsize=(8, 4))
        librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.6)
        ax.vlines(onset_times, ymin=min(y), ymax=max(y), color='r', linestyle='--', label='Onsets')
        ax.set_title("Formă de undă cu Onsets")
        ax.set_xlabel("Timp (secunde)")
        ax.set_ylabel("Amplitudine")
        ax.legend()
        self._show_plot_in_tab(fig, self.onset_tab)
        self.plot_notebook.select(self.onset_tab)

    def _show_plot_in_tab(self, fig, tab):
        for widget in tab.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        plt.close(fig)

    def show_spectral_features(self):
        features = self.service.calculate_spectral_features()
        if features is None:
            messagebox.showinfo("Info", "Nu există înregistrare pentru calculul caracteristicilor spectrale.")
            return

#------------------------------------------------------- FUNCTIONALITATI ---------------------------------------------
    def update_pitch_and_tuning_ui(self, pitch_and_tuning):
        """
        Actualizează câmpurile text din UI cu valorile pitch-ului și tuning-ului.
        :param pitch_and_tuning: Dicționar cu pitch-ul detectat și ajustarea tuning-ului.
        """
        self.pitch_var.set(f"{pitch_and_tuning['Pitch Fundamental']:.2f} Hz")
        self.tuning_var.set(f"{pitch_and_tuning['Tuning Adjustment']:.2f} semitone")

    def update_spectral_features_ui(self, features):
        """
        Actualizează câmpurile text din UI cu valorile caracteristicilor spectrale.
        :param features: Dicționar cu valorile caracteristicilor spectrale.
        """
        self.spectral_centroid_var.set(f"{features['Spectral Centroid']:.2f}")
        self.spectral_bandwidth_var.set(f"{features['Spectral Bandwidth']:.2f}")
        self.spectral_rolloff_var.set(f"{features['Spectral Rolloff']:.2f}")
        self.spectral_contrast_var.set(", ".join(f"{v:.2f}" for v in features['Spectral Contrast']))

    def update_fields(self):
        """
        Actualizează toate câmpurile din UI (caracteristici spectrale și pitch/tuning).
        """
        # Calculăm caracteristicile spectrale
        features = self.service.calculate_spectral_features()
        if features:
            self.update_spectral_features_ui(features)

        # Analizăm pitch-ul și tuning-ul
        pitch_and_tuning = self.service.analyze_pitch_and_tuning()
        if pitch_and_tuning:
            self.update_pitch_and_tuning_ui(pitch_and_tuning)

    def update_bpm_field(self):
        if self.service.recording:
            y = self.service.recording.data
            sr = self.service.recording.sample_rate
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(bpm, np.ndarray):
                bpm = float(bpm.item())
            self.bpm_entry.delete(0, tk.END)
            self.bpm_entry.insert(0, f"{bpm:.2f}")

    def record(self):
        try:
            self.config.sample_rate = int(self.sample_rate_entry.get())
            duration = float(self.duration_entry.get())
            self.service.record(duration)
            self.update_plot()
            # Actualizăm toate câmpurile relevante
            self.update_fields()
            self.update_bpm_field()
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la înregistrare: {e}")

    import threading
    import sounddevice as sd

    def play(self):
        if not self.service.recording:
            messagebox.showinfo("Info", "Nu există înregistrare pentru redare.")
            return

        # Actualizează UI-ul instant
        self.is_playing = True
        self.is_paused = False
        self.play_button.config(text="Pause", style="Red.TButton")
        self.stop_button.config(state="normal")

        # Pornește redarea pe thread separat
        threading.Thread(target=self._play_audio_thread, daemon=True).start()

    def _play_audio_thread(self):
        try:
            self.service.play()  # sau direct sd.play(...), dacă vrei
            # Afișează istoricul cache-ului după ce începe redarea
            print(self.service.get_cache_history())
            # Așteaptă să termine redarea
            sd.wait()
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Eroare", f"Eroare la redare: {e}"))
        finally:
            # La finalul redării, resetează UI-ul pe thread-ul principal
            self.root.after(0, self._reset_play_ui)

    def _reset_play_ui(self):
        self.is_playing = False
        self.is_paused = False
        self.play_button.config(text="Redă", style="TButton")
        self.stop_button.config(state="disabled")


    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        ttk.Label(win, text="Director de salvare:").grid(row=0, column=0, padx=5, pady=5)
        save_dir = ttk.Entry(win, width=40)
        save_dir.grid(row=0, column=1, padx=5, pady=5)
        save_dir.insert(0,self.config.save_directory)
        def browse():
            d = filedialog.askdirectory()
            if d: save_dir.delete(0,tk.END); save_dir.insert(0,d)
        ttk.Button(win,text="Browse...",command=browse).grid(row=0,column=2,padx=5)
        ttk.Label(win,text="Factor pitch:").grid(row=1,column=0,padx=5,pady=5)
        pitch = ttk.Entry(win)
        pitch.grid(row=1,column=1,padx=5,pady=5)
        pitch.insert(0,str(self.config.pitch_factor))
        def save():
            self.config.save_directory=save_dir.get()
            try: self.config.pitch_factor=float(pitch.get())
            except: messagebox.showerror("Eroare","Pitch trebuie numeric")
            win.destroy()
        ttk.Button(win,text="Save",command=save).grid(row=2,column=1,pady=10)

    def save_recording(self):
        if self.service.recording:
            initial = self.config.save_directory or os.getcwd()
            path = filedialog.asksaveasfilename(initialdir=initial, defaultextension=".wav",
                                                filetypes=[("WAV files", "*.wav")])
            if path:
                self.config.save_directory = os.path.dirname(path)
                AudioRepository.save(self.service.recording, path)
        else:
            messagebox.showinfo("Info", "Nu există înregistrare.")

    import threading

    def load_recording(self):
        initial = self.config.save_directory or os.getcwd()
        path = filedialog.askopenfilename(initialdir=initial, filetypes=[("WAV files", "*.wav")])
        if not path:
            return  # utilizatorul a anulat

        # 1. Pornește animația de loading
        self.show_loading("Se încarcă fișierul...")

        def task():
            try:
                recording = AudioRepository.load(path)
                self.config.save_directory = os.path.dirname(path)
                self.root.after(0, lambda: self._on_recording_loaded(recording))
            except Exception as e:
                self.root.after(0, self.hide_loading)
                messagebox.showerror("Eroare", f"Eroare la încărcare: {e}")

        # 2. Rulează task-ul pe un thread separat
        threading.Thread(target=task, daemon=True).start()

    def _on_recording_loaded(self, recording):
        self.service.recording = recording
        self.update_plot()
        self.update_fields()
        self.update_bpm_field()
        self.hide_loading()

    def start_playback_ui(self):
        self.is_playing = True
        self.is_paused = False
        self.play_button.config(text="Pause", style="Red.TButton")
        self.stop_button.config(state="normal")

    def pitch_up(self):
        shifted = self.service.pitch_shift(up=True)
        if shifted:
            self.service.recording = shifted
            self.update_plot()
            self.play()
        # Actualizăm toate câmpurile relevante
        self.update_fields()

    def pitch_down(self):
        shifted = self.service.pitch_shift(up=False)
        if shifted:
            self.service.recording = shifted
            self.update_plot()
            self.play()
        # Actualizăm toate câmpurile relevante
        self.update_fields()

    def apply_reverb(self):
        try:
            decay = float(self.decay_entry.get())
            delay = float(self.delay_entry.get())
            ir_dur = float(self.ir_duration_entry.get())
            if self.service.recording is None:
                messagebox.showinfo("Info", "Nu există înregistrare pentru reverb.")
                return
            self.root.config(cursor="wait")
            self.root.update()

            def task():
                try:
                    self.service.apply_reverb_with_ir("C:\Faculta/an_3/Licenta/Licenta_tkinter/IR/bathroom.wav")
                    self.root.after(0, self.on_reverb_done)
                except Exception as e:
                    messagebox.showerror("Eroare", f"Eroare la reverb: {e}")

            threading.Thread(target=task, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la reverb: {e}")

    def on_reverb_done(self):
        self.root.config(cursor="")
        self.update_plot()
        self.play()
        # Actualizăm toate câmpurile relevante
        self.update_fields()

    def apply_time_stretch(self):
        try:
            bpm = float(self.bpm_entry.get())
            if bpm <= 0:
                raise ValueError
            self.service.apply_time_stretch_bpm(bpm)
            self.update_plot()
            self.play()
            self.update_fields()
            # Actualizăm câmpul BPM cu noul BPM țintă
            self.bpm_entry.delete(0, tk.END)
            self.bpm_entry.insert(0, f"{bpm:.2f}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid positive BPM.")

    def apply_echo(self):
        """
        Apelează metoda din AudioService pentru a aplica efectul de echo.
        """
        try:
            decay = float(self.decay_entry.get())
            delay = float(self.delay_entry.get())

            if self.service.recording is None:
                messagebox.showinfo("Info", "Nu există înregistrare pentru aplicarea efectului de echo.")
                return

            self.service.apply_echo(decay=decay, delay=delay)
            self.update_plot()
            self.play()
            # Actualizăm toate câmpurile relevante
            self.update_fields()
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la aplicarea efectului de echo: {e}")

    def apply_compressor(self):
        # Poți folosi un dialog pentru a seta parametrii, sau folosește valori implicite pentru început
        try:
            # Exemplu cu valori implicite (poți extinde cu un dialog pentru parametri)
            threshold_db = -20.0
            ratio = 4.0
            attack_ms = 10.0
            release_ms = 100.0

            self.service.apply_simple_compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                # attack_ms=attack_ms,
                # release_ms=release_ms
            )
            self.update_plot()
            self.play()
            self.update_fields()

        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la aplicarea compresorului: {e}")

    def test_compressor(self):
        if self.service.recording is None:
            messagebox.showinfo("Info", "Nu există înregistrare pentru test.")
            return

        y_original = np.copy(self.service.recording.data)
        sr = self.service.recording.sample_rate

        # Aplică compresorul fără normalizare
        _, y_compressed = self.service.apply_simple_compressor(normalize=False)

        # Calculează RMS și amplitudinea maximă
        def rms(x): return np.sqrt(np.mean(x ** 2))

        rms_orig = rms(y_original)
        rms_comp = rms(y_compressed)
        max_orig = np.max(np.abs(y_original))
        max_comp = np.max(np.abs(y_compressed))

        # Plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_original, label="Original", alpha=0.7)
        ax.plot(y_compressed, label="Compresat (fără normalizare)", alpha=0.7)
        ax.set_title("Comparație semnal original vs. compresat")
        ax.set_xlabel("Eșantion")
        ax.set_ylabel("Amplitudine")
        ax.legend()
        ax.grid(True)
        plt.show()

        # Afișează valorile
        print(f"RMS original: {rms_orig:.4f}, RMS compresat: {rms_comp:.4f}")
        print(f"Max original: {max_orig:.4f}, Max compresat: {max_comp:.4f}")

    def apply_lpf(self):
        try:
            self.service.apply_lowpass_filter(cutoff_hz=1000.0, order=5)
            self.update_plot()
            self.play()
            self.update_fields()
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la LPF: {e}")

    def apply_hpf(self):
        try:
            self.service.apply_highpass_filter(cutoff_hz=1000.0, order=5)
            self.update_plot()
            self.play()
            self.update_fields()
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la HPF: {e}")

    def apply_bpf(self):
        try:
            self.service.apply_bandpass_filter(lowcut_hz=300.0, highcut_hz=3000.0, order=5)
            self.update_plot()
            self.play()
            self.update_fields()
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la BPF: {e}")

    def show_distortion_dialog(self):
        if self.service.recording is None:
            messagebox.showinfo("Info", "Nu există înregistrare pentru aplicarea distorsiunii.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Distorsiune")
        dialog.geometry("350x350")

        sliders_frame = ttk.Frame(dialog)
        sliders_frame.pack(pady=10, fill="x")

        # --- Drive ---
        drive_frame = ttk.Frame(sliders_frame)
        drive_frame.pack(fill="x", pady=5)
        ttk.Label(drive_frame, text="Drive (1-10)").pack()
        drive_var = tk.DoubleVar(value=1.0)
        drive_slider = ttk.Scale(drive_frame, from_=1.0, to=10.0, variable=drive_var, orient="horizontal")
        drive_slider.pack(fill="x", padx=10)
        drive_value_label = ttk.Label(drive_frame, text="1.0")
        drive_value_label.pack()
        drive_var.trace_add("write", lambda *args: drive_value_label.config(text=f"{drive_var.get():.1f}"))

        # --- Tone ---
        tone_frame = ttk.Frame(sliders_frame)
        tone_frame.pack(fill="x", pady=5)
        ttk.Label(tone_frame, text="Tone (0-1)").pack()
        tone_var = tk.DoubleVar(value=0.5)
        tone_slider = ttk.Scale(tone_frame, from_=0.0, to=1.0, variable=tone_var, orient="horizontal")
        tone_slider.pack(fill="x", padx=10)
        tone_value_label = ttk.Label(tone_frame, text="0.50")
        tone_value_label.pack()
        tone_var.trace_add("write", lambda *args: tone_value_label.config(text=f"{tone_var.get():.2f}"))

        # --- Mix ---
        mix_frame = ttk.Frame(sliders_frame)
        mix_frame.pack(fill="x", pady=5)
        ttk.Label(mix_frame, text="Mix (0-1)").pack()
        mix_var = tk.DoubleVar(value=1.0)
        mix_slider = ttk.Scale(mix_frame, from_=0.0, to=1.0, variable=mix_var, orient="horizontal")
        mix_slider.pack(fill="x", padx=10)
        mix_value_label = ttk.Label(mix_frame, text="1.00")
        mix_value_label.pack()
        mix_var.trace_add("write", lambda *args: mix_value_label.config(text=f"{mix_var.get():.2f}"))

        def apply():
            try:
                self.service.apply_distortion(
                    drive=drive_var.get(),
                    tone=tone_var.get(),
                    mix=mix_var.get()
                )
                self.update_plot()
                self.play()
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Eroare", f"Eroare la aplicarea distorsiunii: {e}")

        ttk.Button(dialog, text="Aplică", command=apply).pack(pady=10)

    def show_equalizer_dialog(self):
        if self.service.recording is None:
            messagebox.showinfo("Info", "Nu există înregistrare pentru aplicarea egalizatorului.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Equalizer")
        dialog.geometry("1200x800")

        eq_frame = ttk.Frame(dialog)
        eq_frame.pack(fill="x", padx=10, pady=5)

        freqs = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        band_vars = [tk.DoubleVar(value=0.0) for _ in range(10)]

        # --- Istoric pentru valori slider-e ---
        eq_history = []
        # Salvează valorile inițiale la deschiderea dialogului
        eq_history.append([var.get() for var in band_vars])

        for i, (freq, var) in enumerate(zip(freqs, band_vars)):
            band_frame = ttk.Frame(eq_frame)
            band_frame.grid(row=0, column=i, padx=2)
            ttk.Label(band_frame, text=f"{freq}Hz").pack()
            slider = ttk.Scale(band_frame, from_=12, to=-12, variable=var, orient="vertical", length=200)
            slider.pack()
            value_label = ttk.Label(band_frame, text="0.0 dB")
            value_label.pack()

            def update_label(var=var, label=value_label):
                label.config(text=f"{var.get():.1f} dB")

            var.trace_add("write", lambda *args: update_label())

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill="x", padx=10, pady=5)

        def apply():
            try:
                # Salvează valorile curente înainte de aplicare
                eq_history.append([var.get() for var in band_vars])
                bands = [var.get() for var in band_vars]
                self.service.apply_equalizer(bands)
                self.update_plot()
                self.play()
                update_plots()
            except Exception as e:
                messagebox.showerror("Eroare", f"Eroare la aplicarea egalizatorului: {e}")

        def play():
            try:
                self.service.play()
            except Exception as e:
                messagebox.showerror("Eroare", f"Eroare la redare: {e}")

        def undo():
            try:
                if self.service.undo():
                    self.update_plot()
                    self.service.play()
                    self.update_fields()
                    # --- Undo și pentru slider-e ---
                    if len(eq_history) > 1:
                        eq_history.pop()  # Scoate starea curentă
                        last_bands = eq_history[-1]
                        for var, val in zip(band_vars, last_bands):
                            var.set(val)
                    update_plots()
                else:
                    messagebox.showinfo("Info", "Nu există stări anterioare pentru Undo.")
            except Exception as e:
                messagebox.showerror("Eroare", f"Eroare la aplicarea Undo: {e}")

        ttk.Button(btn_frame, text="Aplică", command=apply).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Redă", command=play).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Undo", command=undo).pack(side="left", padx=5)

        # Notebook pentru ploturi
        plot_notebook = ttk.Notebook(dialog)
        plot_notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab-uri pentru fiecare tip de plot
        waveform_tab = ttk.Frame(plot_notebook)
        spectrogram_tab = ttk.Frame(plot_notebook)
        mel_tab = ttk.Frame(plot_notebook)
        mfcc_tab = ttk.Frame(plot_notebook)
        cqt_tab = ttk.Frame(plot_notebook)

        plot_notebook.add(waveform_tab, text="Waveform")
        plot_notebook.add(spectrogram_tab, text="Spectrogramă")
        plot_notebook.add(mel_tab, text="Mel Spectrogram")
        plot_notebook.add(mfcc_tab, text="MFCC")
        plot_notebook.add(cqt_tab, text="CQT")

        def update_plots():
            # Waveform
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self.service.recording.data)
            ax.set_title("Semnal audio")
            ax.set_xlabel("Eșantion")
            ax.set_ylabel("Amplitudine")
            ax.grid(True)
            self._show_plot_in_tab(fig, waveform_tab)

            # Spectrogramă
            y = self.service.recording.data
            sr = self.service.recording.sample_rate
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
            ax.set_title('Spectrograma')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, spectrogram_tab)

            # Mel Spectrogram
            mel_spec_db, _ = self.service.generate_mel_spectrogram()
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
            ax.set_title("Mel Spectrogram")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, mel_tab)

            # MFCC
            mfcc, _ = self.service.generate_mfcc()
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
            ax.set_title("MFCC (Mel Frequency Cepstral Coefficients)")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, mfcc_tab)

            # CQT
            cqt_db, _ = self.service.generate_cqt()
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax, cmap='coolwarm')
            ax.set_title("Constant-Q Transform (CQT)")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self._show_plot_in_tab(fig, cqt_tab)

        # Inițializăm ploturile
        update_plots()

    def undo(self):
        """
        Revine la ultima stare salvată a înregistrării.
        """
        try:
            if self.service.undo():
                self.update_plot()  # Actualizează graficul
                self.play()  # Redă înregistrarea anterioară
                # Actualizăm toate câmpurile relevante
                self.update_fields()
                self.update_bpm_field()
            else:
                messagebox.showinfo("Info", "Nu există stări anterioare pentru Undo.")
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la aplicarea Undo: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindowV2(root)
    root.mainloop()