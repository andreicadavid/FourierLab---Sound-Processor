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
        self.service.set_progress_callback(self.update_progress)
        # Setăm callback-ul pentru actualizarea duratei
        self.service.set_duration_callback(self.update_duration)

        # Variabile pentru controlul redării
        self.is_playing = False
        self.is_paused = False
        self.playback_thread = None
        self.stop_event = threading.Event()

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
        style.configure("TButton", font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))

        # Inițializăm variabilele pentru entry-uri cu valorile default
        self.duration_entry = ttk.Entry()
        self.duration_entry.insert(0, "5")

        self.sample_rate_entry = ttk.Entry()
        self.sample_rate_entry.insert(0, str(self.config.sample_rate))

        self.decay_entry = ttk.Entry()
        self.decay_entry.insert(0, "0.5")

        self.delay_entry = ttk.Entry()
        self.delay_entry.insert(0, "0.1")

        self.ir_duration_entry = ttk.Entry()
        self.ir_duration_entry.insert(0, "0.5")

        # Inițializăm variabilele pentru caracteristici spectrale
        self.spectral_centroid_var = tk.StringVar()
        self.spectral_bandwidth_var = tk.StringVar()
        self.spectral_rolloff_var = tk.StringVar()
        self.spectral_contrast_var = tk.StringVar()
        self.pitch_var = tk.StringVar()
        self.tuning_var = tk.StringVar()

        # Inițializăm variabilele pentru BPM
        self.bpm_entry = ttk.Entry()

        # Cadrul principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Frame pentru progres și status
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill="x", padx=5, pady=2)

        # Bară de progres
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, length=300, mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(fill="x", padx=5, pady=2)

        # Label pentru status
        self.status_label = ttk.Label(status_frame, text="Introduceți o înregistrare!")
        self.status_label.pack(fill="x", padx=5, pady=2)

        # Frame pentru controale (deasupra ploturilor)
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill="x", padx=5, pady=5)

        # Frame pentru ploturi (50% din înălțime)
        plots_frame = ttk.Frame(main_frame)
        plots_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Controale în 3 coloane ---
        # Frame pentru cele 3 coloane
        columns_frame = ttk.Frame(controls_frame)
        columns_frame.pack(fill="x", padx=5, pady=5)

        # Coloana 1: Setări
        settings_frame = ttk.LabelFrame(columns_frame, text="Setări", padding=5)
        settings_frame.pack(side="left", fill="both", expand=True, padx=5)

        # Setări generale
        for label, default_value in [
            ("Durată (sec):", "5"),
            ("Sample Rate:", str(self.config.sample_rate)),
            ("Decay (0-1):", "0.5"),
            ("Delay (sec):", "0.1"),
            ("IR Duration:", "0.5")
        ]:
            frame = ttk.Frame(settings_frame)
            frame.pack(fill="x", padx=2, pady=2)
            ttk.Label(frame, text=label).pack(side="left", padx=2)
            entry = ttk.Entry(frame)
            entry.insert(0, default_value)
            entry.pack(side="right", fill="x", expand=True, padx=2)

            # Salvăm referința la entry în variabila corespunzătoare
            if label == "Durată (sec):":
                self.duration_entry = entry
            elif label == "Sample Rate:":
                self.sample_rate_entry = entry
            elif label == "Decay (0-1):":
                self.decay_entry = entry
            elif label == "Delay (sec):":
                self.delay_entry = entry
            elif label == "IR Duration:":
                self.ir_duration_entry = entry

        # Separator
        ttk.Separator(settings_frame, orient="horizontal").pack(fill="x", padx=5, pady=5)

        # Time Stretch settings
        frame = ttk.Frame(settings_frame)
        frame.pack(fill="x", padx=2, pady=2)
        ttk.Label(frame, text="Target BPM:").pack(side="left", padx=2)
        self.bpm_entry = ttk.Entry(frame)
        self.bpm_entry.pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(frame, text="Apply", command=self.apply_time_stretch).pack(side="right", padx=2)

        # Coloana 2: Caracteristici spectrale
        spectral_frame = ttk.LabelFrame(columns_frame, text="Caracteristici spectrale", padding=5)
        spectral_frame.pack(side="left", fill="both", expand=True, padx=5)

        for label, var in [
            ("Spectral Centroid:", self.spectral_centroid_var),
            ("Spectral Bandwidth:", self.spectral_bandwidth_var),
            ("Spectral Rolloff:", self.spectral_rolloff_var),
            ("Spectral Contrast:", self.spectral_contrast_var),
            ("Pitch (Hz):", self.pitch_var),
            ("Tuning (semitone):", self.tuning_var)
        ]:
            frame = ttk.Frame(spectral_frame)
            frame.pack(fill="x", padx=2, pady=2)
            ttk.Label(frame, text=label).pack(side="left", padx=2)
            ttk.Entry(frame, textvariable=var, state="readonly").pack(side="right", fill="x", expand=True, padx=2)

        # Coloana 3: Butoane
        buttons_frame = ttk.LabelFrame(columns_frame, text="Controale", padding=5)
        buttons_frame.pack(side="left", fill="both", expand=True, padx=5)

        # Butoane principale
        self.play_button = ttk.Button(buttons_frame, text="Redă", command=self.play)
        self.play_button.pack(fill="x", padx=2, pady=2)

        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_playback, state="disabled")
        self.stop_button.pack(fill="x", padx=2, pady=2)

        for text, command in [
            ("Înregistrează", self.record),
            ("Salvează", self.save_recording),
            ("Încarcă", self.load_recording),
            ("Undo", self.undo)
        ]:
            ttk.Button(buttons_frame, text=text, command=command).pack(fill="x", padx=2, pady=2)

        # --- Efecte și filtre ---
        effects_filters_frame = ttk.Frame(controls_frame)
        effects_filters_frame.pack(fill="x", padx=5, pady=5)

        # Efecte
        effects_frame = ttk.LabelFrame(effects_filters_frame, text="Efecte", padding=5)
        effects_frame.pack(side="left", fill="x", expand=True, padx=5)

        effects = [
            ("Pitch Up", self.pitch_up),
            ("Pitch Down", self.pitch_down),
            ("Reverb", self.apply_reverb),
            ("Echo", self.apply_echo),
            ("Compressor", self.show_compressor_dialog),
            ("Test Compressor", self.test_compressor),
            ("Distortion", self.show_distortion_dialog),
            ("Equalizer", self.show_equalizer_dialog)
        ]

        for text, command in effects:
            ttk.Button(effects_frame, text=text, command=command).pack(side="left", padx=2, pady=2)

        # Filtre
        filters_frame = ttk.LabelFrame(effects_filters_frame, text="Filtre DSP", padding=5)
        filters_frame.pack(side="left", fill="x", expand=True, padx=5)

        filters = [
            ("LPF", self.apply_lpf),
            ("HPF", self.apply_hpf),
            ("BPF", self.apply_bpf)
        ]

        for text, command in filters:
            ttk.Button(filters_frame, text=text, command=command).pack(side="left", padx=2, pady=2)

        # --- Notebook pentru ploturi ---
        self.plot_notebook = ttk.Notebook(plots_frame)
        self.plot_notebook.pack(fill="both", expand=True)

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

    def _play_audio_thread(self):
        try:
            if self.service.recording is None:
                return

            # Redăm audio-ul
            self.service.play()

            # Așteptăm finalizarea redării sau oprirea
            while not self.stop_event.is_set() and sd.get_stream().active:
                sd.wait(100)  # Verificăm la fiecare 100ms

        except Exception as e:
            print(f"Error in playback: {e}")
            self.root.after(0, lambda: messagebox.showerror("Eroare", f"Eroare la redare: {e}"))
        finally:
            self.root.after(0, self._reset_play_ui)

    def stop_playback(self):
        """
        Oprește redarea și înregistrarea audio.
        """
        # Oprim redarea
        if self.service.is_playing:
            sd.stop()
            self.service.is_playing = False
            self.is_playing = False
            self.stop_event.set()

        # Oprim înregistrarea
        if self.service.is_recording:
            self.service.is_recording = False
            if self.service.recording_thread:
                self.service.recording_thread.join()

        # Resetăm UI-ul
        self._reset_play_ui()
        self.status_label.config(text="Standby")
        self.progress_var.set(0)

        # Dezactivăm butonul de stop
        self.stop_button.config(state="disabled")

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

        # Buton explicit de "Jump"
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
        try:
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
        except Exception as e:
            print(f"Error updating plot: {e}")

    def _update_all_plots(self):
        try:
            if not self.service.recording:
                return

            y = self.service.recording.data
            sr = self.service.recording.sample_rate

            # Spectrogramă
            try:
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                fig, ax = plt.subplots(figsize=(8, 4))
                img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
                ax.set_title('Spectrograma')
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                self._show_plot_in_tab(fig, self.spectrogram_tab)
            except Exception as e:
                print(f"Error updating spectrogram: {e}")

            # Chroma
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                fig, ax = plt.subplots(figsize=(8, 4))
                img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
                ax.set_title('Chroma')
                fig.colorbar(img, ax=ax)
                self._show_plot_in_tab(fig, self.chroma_tab)
            except Exception as e:
                print(f"Error updating chroma: {e}")

            # Mel Spectrogram
            try:
                mel_spec_db, _ = self.service.generate_mel_spectrogram()
                fig, ax = plt.subplots(figsize=(8, 4))
                img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
                ax.set_title("Mel Spectrogram")
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                self._show_plot_in_tab(fig, self.mel_tab)
            except Exception as e:
                print(f"Error updating mel spectrogram: {e}")

            # MFCC
            try:
                mfcc, _ = self.service.generate_mfcc()
                fig, ax = plt.subplots(figsize=(8, 4))
                img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
                ax.set_title("MFCC (Mel Frequency Cepstral Coefficients)")
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                self._show_plot_in_tab(fig, self.mfcc_tab)
            except Exception as e:
                print(f"Error updating MFCC: {e}")

            # CQT
            try:
                cqt_db, _ = self.service.generate_cqt()
                fig, ax = plt.subplots(figsize=(8, 4))
                img = librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax, cmap='coolwarm')
                ax.set_title("Constant-Q Transform (CQT)")
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                self._show_plot_in_tab(fig, self.cqt_tab)
            except Exception as e:
                print(f"Error updating CQT: {e}")

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
            except Exception as e:
                print(f"Error updating onsets: {e}")

        except Exception as e:
            print(f"Error in _update_all_plots: {e}")

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
            messagebox.showinfo("Info", "Nu există înregistrare pentru spectrogramă.")
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
            messagebox.showinfo("Info", "Nu există înregistrare pentru chroma.")
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
        try:
            for widget in tab.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()
            plt.close(fig)
        except Exception as e:
            print(f"Error showing plot in tab: {e}")

    def show_spectral_features(self):
        """
        Afișează caracteristicile spectrale ale înregistrării.
        """
        features = self.service.calculate_spectral_features()
        if features is None:
            messagebox.showinfo("Info", "Nu există înregistrare pentru calculul caracteristicilor spectrale.")
            return

        # Actualizăm UI-ul cu valorile calculate
        self.update_spectral_features_ui(features)

    # ------------------------------------------------------- FUNCTIONALITATI ---------------------------------------------
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
        Actualizează toate câmpurile din interfață cu datele curente.
        """
        if self.service.recording is None:
            return

        # Verificăm dacă semnalul este suficient de lung pentru analiză
        if len(self.service.recording.data) < 2048:
            print("Semnalul este prea scurt pentru analiză.")
            return

        # Actualizăm caracteristicile spectrale
        features = self.service.calculate_spectral_features()
        if features:
            self.update_spectral_features_ui(features)

        # Actualizăm pitch și tuning
        pitch_and_tuning = self.service.analyze_pitch_and_tuning()
        if pitch_and_tuning:
            self.update_pitch_and_tuning_ui(pitch_and_tuning)

        # Actualizăm BPM
        self.update_bpm_field()

    def update_bpm_field(self):
        if self.service.recording:
            # Folosim BPM-ul original salvat în service dacă există
            if hasattr(self.service, 'original_bpm') and self.service.original_bpm is not None:
                bpm = self.service.original_bpm
            else:
                # Dacă nu există, îl calculăm
                y = self.service.recording.data
                sr = self.service.recording.sample_rate
                # Folosim beat_track pentru a calcula BPM-ul
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                if isinstance(tempo, np.ndarray):
                    bpm = float(tempo.item())
                else:
                    bpm = float(tempo)
                # Salvăm BPM-ul calculat în service
                self.service.original_bpm = bpm

            self.bpm_entry.delete(0, tk.END)
            self.bpm_entry.insert(0, f"{bpm:.2f}")

    def record(self):
        """
        Începe înregistrarea într-un thread separat.
        """
        try:
            duration = float(self.duration_entry.get())
            self.status_label.config(text="Se înregistrează...")
            self.progress_var.set(0)
            self.service.record(duration)
            # Activăm butonul de stop
            self.stop_button.config(state="normal")
            # Înregistrarea se face în thread separat, așteptăm finalizarea
            self.root.after(100, self._check_recording_status)
        except ValueError:
            messagebox.showerror("Eroare", "Durata trebuie să fie un număr valid.")
            self.status_label.config(text="Înregistrare oprită!")

    def _check_recording_status(self):
        """
        Verifică statusul înregistrării și actualizează interfața.
        """
        if self.service.is_recording:
            self.root.after(100, self._check_recording_status)
        else:
            self.status_label.config(text="Înregistrare finalizată. Gata de redare!")
            self.progress_var.set(100)
            if hasattr(self.service, 'original_bpm'):
                delattr(self.service, 'original_bpm')
            self._on_recording_loaded()

    def play(self):
        """
        Redă audio-ul într-un thread separat.
        """
        if not self.service.recording:
            messagebox.showwarning("Avertisment", "Nu există înregistrare pentru redare.")
            return

        self.status_label.config(text="Se redă...")
        self.progress_var.set(0)
        self.is_playing = True
        self.stop_button.config(state="normal")  # Activăm butonul de stop
        self.service.play()
        # Redarea se face în thread separat, așteptăm finalizarea
        self.root.after(100, self._check_playback_status)

    def _check_playback_status(self):
        """
        Verifică statusul redării și actualizează interfața.
        """
        if self.service.is_playing:
            self.root.after(100, self._check_playback_status)
        else:
            self.status_label.config(text="Standby")
            self._reset_play_ui()

    def _reset_play_ui(self):
        try:
            self.is_playing = False
            self.is_paused = False
            self.stop_event.clear()

            if hasattr(self, 'play_button'):
                self.play_button.config(text="Redă")
            if hasattr(self, 'stop_button'):
                self.stop_button.config(state="disabled")
        except Exception as e:
            print(f"Error resetting UI: {e}")

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        ttk.Label(win, text="Director de salvare:").grid(row=0, column=0, padx=5, pady=5)
        save_dir = ttk.Entry(win, width=40)
        save_dir.grid(row=0, column=1, padx=5, pady=5)
        save_dir.insert(0, self.config.save_directory)

        def browse():
            d = filedialog.askdirectory()
            if d: save_dir.delete(0, tk.END); save_dir.insert(0, d)

        ttk.Button(win, text="Browse...", command=browse).grid(row=0, column=2, padx=5)
        ttk.Label(win, text="Factor pitch:").grid(row=1, column=0, padx=5, pady=5)
        pitch = ttk.Entry(win)
        pitch.grid(row=1, column=1, padx=5, pady=5)
        pitch.insert(0, str(self.config.pitch_factor))

        def save():
            self.config.save_directory = save_dir.get()
            try:
                self.config.pitch_factor = float(pitch.get())
            except:
                messagebox.showerror("Eroare", "Pitch trebuie numeric")
            win.destroy()

        ttk.Button(win, text="Save", command=save).grid(row=2, column=1, pady=10)

    def save_recording(self):
        """
        Salvează înregistrarea într-un thread separat.
        """
        if not self.service.recording:
            messagebox.showwarning("Avertisment", "Nu există înregistrare de salvat.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if file_path:
            self.status_label.config(text="Se salvează...")
            self.progress_var.set(0)
            self.service.save_recording(file_path)
            # Salvarea se face în thread separat, așteptăm finalizarea
            self.root.after(100, self._check_saving_status)

    def _check_recording_status(self):
        """
        Verifică statusul înregistrării și actualizează interfața.
        """
        if self.service.is_recording:
            self.root.after(100, self._check_recording_status)
        else:
            self.status_label.config(text="Standby")
            # Resetăm BPM-ul original pentru a forța recalcularea
            if hasattr(self.service, 'original_bpm'):
                delattr(self.service, 'original_bpm')
            self._on_recording_loaded()

    def _on_recording_loaded(self):
        """
        Actualizează interfața după ce o înregistrare a fost încărcată.
        """
        self.update_plot()
        # Resetăm BPM-ul original pentru a forța recalcularea
        if hasattr(self.service, 'original_bpm'):
            delattr(self.service, 'original_bpm')
        self.update_fields()
        self.update_bpm_field()
        self.show_spectral_features()
        pitch_and_tuning = self.service.analyze_pitch_and_tuning()
        if pitch_and_tuning:
            self.update_pitch_and_tuning_ui(pitch_and_tuning)
        # Actualizăm durata
        if self.service.recording:
            duration = len(self.service.recording.data) / self.service.recording.sample_rate
            self.update_duration(duration)

    def load_recording(self):
        """
        Încarcă un fișier audio într-un thread separat.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if file_path:
            self.status_label.config(text="Se încarcă fișierul...")
            self.progress_var.set(0)
            self.service.load_audio(file_path)
            # Încărcarea se face în thread separat, așteptăm finalizarea
            self.root.after(100, self._check_loading_status)

    def _check_loading_status(self):
        """
        Verifică statusul încărcării și actualizează interfața.
        """
        if self.service.is_loading:
            self.root.after(100, self._check_loading_status)
        else:
            self.status_label.config(text="Standby")
            self._on_recording_loaded()

    def start_playback_ui(self):
        self.is_playing = True
        self.is_paused = False
        self.play_button.config(text="Pause", style="Red.TButton")
        self.stop_button.config(state="normal")

    def pitch_up(self):
        self.status_label.config(text="Se aplică pitch up...")
        self.progress_var.set(0)
        shifted = self.service.pitch_shift(up=True)
        if shifted:
            self.service.recording = shifted
            self.update_plot()
            self.play()
        # Actualizăm toate câmpurile relevante
        self.update_fields()

    def pitch_down(self):
        self.status_label.config(text="Se aplică pitch down...")
        self.progress_var.set(0)
        shifted = self.service.pitch_shift(up=False)
        if shifted:
            self.service.recording = shifted
            self.update_plot()
            self.play()
        # Actualizăm toate câmpurile relevante
        self.update_fields()

    def apply_reverb(self):
        try:
            self.status_label.config(text="Se aplică reverb...")
            self.progress_var.set(0)
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
        # Actualizăm durata
        if self.service.recording:
            duration = len(self.service.recording.data) / self.service.recording.sample_rate
            self.update_duration(duration)

    def apply_time_stretch(self):
        try:
            self.status_label.config(text="Se aplică time stretch...")
            self.progress_var.set(0)
            target_bpm = float(self.bpm_entry.get())
            if target_bpm <= 0:
                raise ValueError("BPM trebuie să fie un număr pozitiv.")

            self.status_label.config(text="Se aplică time stretch...")
            self.progress_var.set(0)

            # Aplicăm time stretch
            result = self.service.apply_time_stretch_bpm(target_bpm)

            if result:
                # Actualizăm interfața
                self.update_plot()
                self.play()
                self.update_fields()

                # Actualizăm câmpul BPM cu noul BPM țintă
                self.bpm_entry.delete(0, tk.END)
                self.bpm_entry.insert(0, f"{target_bpm:.2f}")

                # Actualizăm durata
                if self.service.recording:
                    duration = len(self.service.recording.data) / self.service.recording.sample_rate
                    self.update_duration(duration)

                self.status_label.config(text="Time stretch aplicat")
            else:
                messagebox.showerror("Eroare", "Nu s-a putut aplica time stretch.")

        except ValueError as e:
            messagebox.showerror("Eroare", str(e))
            self.status_label.config(text="Standby")
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la aplicarea time stretch: {e}")
            self.status_label.config(text="Standby")

    def apply_echo(self):
        """
        Apelează metoda din AudioService pentru a aplica efectul de echo.
        """
        try:
            self.status_label.config(text="Se aplică echo...")
            self.progress_var.set(0)
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

    def show_compressor_dialog(self):
        if self.service.recording is None:
            messagebox.showinfo("Info", "Nu există înregistrare pentru aplicarea compresorului.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Compresor")
        dialog.geometry("400x300")

        # Frame pentru controale
        controls_frame = ttk.Frame(dialog)
        controls_frame.pack(pady=10, padx=10, fill="x")

        # Threshold
        threshold_frame = ttk.Frame(controls_frame)
        threshold_frame.pack(fill="x", pady=5)
        ttk.Label(threshold_frame, text="Threshold (dB):").pack(side="left")
        threshold_var = tk.DoubleVar(value=-20.0)
        threshold_slider = ttk.Scale(threshold_frame, from_=-60.0, to=0.0, variable=threshold_var, orient="horizontal")
        threshold_slider.pack(side="left", fill="x", expand=True, padx=5)
        threshold_label = ttk.Label(threshold_frame, text="-20.0")
        threshold_label.pack(side="left")
        threshold_var.trace_add("write", lambda *args: threshold_label.config(text=f"{threshold_var.get():.1f}"))

        # Ratio
        ratio_frame = ttk.Frame(controls_frame)
        ratio_frame.pack(fill="x", pady=5)
        ttk.Label(ratio_frame, text="Ratio:").pack(side="left")
        ratio_var = tk.DoubleVar(value=4.0)
        ratio_slider = ttk.Scale(ratio_frame, from_=1.0, to=20.0, variable=ratio_var, orient="horizontal")
        ratio_slider.pack(side="left", fill="x", expand=True, padx=5)
        ratio_label = ttk.Label(ratio_frame, text="4.0")
        ratio_label.pack(side="left")
        ratio_var.trace_add("write", lambda *args: ratio_label.config(text=f"{ratio_var.get():.1f}"))

        # Normalize
        normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Normalize", variable=normalize_var).pack(pady=5)

        def apply():
            try:
                result = self.service.apply_simple_compressor(
                    threshold_db=threshold_var.get(),
                    ratio=ratio_var.get(),
                    normalize=normalize_var.get()
                )
                if result:
                    self.update_plot()
                    self.play()
                    self.update_fields()
                    # Actualizăm durata
                    duration = len(result.data) / result.sample_rate
                    self.update_duration(duration)
                    dialog.destroy()
                    messagebox.showinfo("Succes", "Compresor aplicat cu succes!")
                else:
                    messagebox.showerror("Eroare", "Nu s-a putut aplica compresorul!")
            except Exception as e:
                messagebox.showerror("Eroare", f"Eroare la aplicarea compresorului: {str(e)}")

        def preview():
            try:
                result = self.service.apply_simple_compressor(
                    threshold_db=threshold_var.get(),
                    ratio=ratio_var.get(),
                    normalize=normalize_var.get()
                )
                if result:
                    self.service.play()
            except Exception as e:
                messagebox.showerror("Eroare", f"Eroare la preview: {str(e)}")

        # Butoane
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Preview", command=preview).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Aplică", command=apply).pack(side="left", padx=5)

    def apply_compressor(self):
        """Aplică compresorul pe înregistrare"""
        self.status_label.config(text="Se aplică compressor...")
        self.progress_var.set(0)
        self.show_compressor_dialog()

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
        tone_slider = ttk.Scale(tone_frame, from_=0.0, to=1.0, variable=tone_var, orient="horizontal", length=200)
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
                self.status_label.config(text="Se aplică distortion...")
                self.progress_var.set(0)
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
        self.status_label.config(text="Se aplică compressor...")
        self.progress_var.set(0)
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
            self.status_label.config(text="Se revine la versiunea anterioară!")
            self.progress_var.set(0)
            if self.service.undo():
                # Resetăm BPM-ul original pentru a forța recalcularea
                if hasattr(self.service, 'original_bpm'):
                    delattr(self.service, 'original_bpm')
                self.update_plot()  # Actualizează graficul
                self.play()  # Redă înregistrarea anterioară
                # Actualizăm toate câmpurile relevante
                self.update_fields()
                #self.update_bpm_field()
                # Actualizăm durata
                if self.service.recording:
                    duration = len(self.service.recording.data) / self.service.recording.sample_rate
                    self.update_duration(duration)
            else:
                messagebox.showinfo("Info", "Nu există stări anterioare pentru Undo.")
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la aplicarea Undo: {e}")

    def update_progress(self, value):
        """
        Actualizează bara de progres.
        """
        self.progress_var.set(value)
        self.root.update_idletasks()
        # Dacă progresul ajunge la 100 și statusul este 'Se înregistrează...', actualizăm statusul
        if value >= 100 and self.status_label.cget("text") == "Se înregistrează...":
            self.status_label.config(text="Înregistrare finalizată. Gata de redare!")


    def update_duration(self, duration):
        """
        Actualizează durata afișată în interfață.
        :param duration: Durata în secunde
        """
        try:
            self.duration_entry.delete(0, tk.END)
            self.duration_entry.insert(0, f"{duration:.2f}")
            #self.status_label.config(text=f"Durată: {duration:.2f} secunde")
            self.root.update_idletasks()  # Forțăm actualizarea UI-ului
        except Exception as e:
            print(f"Error updating duration: {e}")

    def apply_lpf(self):
        try:
            self.status_label.config(text="Se aplică Low Pass Filter...")
            self.progress_var.set(0)
            self.service.apply_lowpass_filter(cutoff_hz=1000.0, order=5)
            self.update_plot()
            self.play()
            self.update_fields()
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la LPF: {e}")

    def apply_hpf(self):
        try:
            self.status_label.config(text="Se aplică High Pass Filter...")
            self.progress_var.set(0)
            self.service.apply_highpass_filter(cutoff_hz=1000.0, order=5)
            self.update_plot()
            self.play()
            self.update_fields()
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la HPF: {e}")

    def apply_bpf(self):
        try:
            self.status_label.config(text="Se aplică Band Filter...")
            self.progress_var.set(0)
            self.service.apply_bandpass_filter(lowcut_hz=300.0, highcut_hz=3000.0, order=5)
            self.update_plot()
            self.play()
            self.update_fields()
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la BPF: {e}")

    def test_compressor(self):
        if self.service.recording is None:
            messagebox.showinfo("Info", "Nu există înregistrare pentru test.")
            return

        # Salvăm semnalul original
        y_original = np.copy(self.service.recording.data)
        sr = self.service.recording.sample_rate

        print("\n=== TEST COMPRESOR ===")
        print(f"Lungime semnal original: {len(y_original)} eșantioane")
        print(f"Durată: {len(y_original) / sr:.2f} secunde")

        # Calculăm statistici pentru semnalul original
        rms_orig = np.sqrt(np.mean(y_original ** 2))
        max_orig = np.max(np.abs(y_original))
        min_orig = np.min(y_original)
        print("\nStatistici semnal original:")
        print(f"RMS: {rms_orig:.4f}")
        print(f"Max: {max_orig:.4f}")
        print(f"Min: {min_orig:.4f}")
        print(f"Dynamic Range: {20 * np.log10(max_orig / rms_orig):.2f} dB")

        # Aplicăm compresorul cu setări moderate
        result = self.service.apply_simple_compressor(
            threshold_db=-20.0,
            ratio=4.0,
            normalize=True
        )

        if result is None:
            print("Eroare la aplicarea compresorului!")
            return

        y_compressed = result.data

        # Calculăm statistici pentru semnalul compresat
        rms_comp = np.sqrt(np.mean(y_compressed ** 2))
        max_comp = np.max(np.abs(y_compressed))
        min_comp = np.min(y_compressed)
        print("\nStatistici semnal compresat:")
        print(f"RMS: {rms_comp:.4f}")
        print(f"Max: {max_comp:.4f}")
        print(f"Min: {min_comp:.4f}")
        print(f"Dynamic Range: {20 * np.log10(max_comp / rms_comp):.2f} dB")

        # Calculăm diferențele
        print("\nDiferențe:")
        print(f"RMS Change: {20 * np.log10(rms_comp / rms_orig):.2f} dB")
        print(f"Max Change: {20 * np.log10(max_comp / max_orig):.2f} dB")
        print(f"Dynamic Range Change: {20 * np.log10((max_comp / rms_comp) / (max_orig / rms_orig)):.2f} dB")

        # Afișăm un grafic de comparație
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Grafic pentru semnalul original
        ax1.plot(y_original, label="Original", alpha=0.7)
        ax1.set_title("Semnal Original")
        ax1.set_xlabel("Eșantion")
        ax1.set_ylabel("Amplitudine")
        ax1.grid(True)
        ax1.legend()

        # Grafic pentru semnalul compresat
        ax2.plot(y_compressed, label="Compresat", alpha=0.7)
        ax2.set_title("Semnal Compresat")
        ax2.set_xlabel("Eșantion")
        ax2.set_ylabel("Amplitudine")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # Redăm ambele semnale pentru comparație
        print("\nRedare semnal original...")
        sd.play(y_original, sr)
        sd.wait()

        print("Redare semnal compresat...")
        sd.play(y_compressed, sr)
        sd.wait()

    def __del__(self):
        # Curățăm resursele la închiderea aplicației
        try:
            with self.lock:
                sd.stop()
        except:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindowV2(root)
    root.mainloop()