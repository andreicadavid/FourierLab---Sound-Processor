import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import sounddevice as sd
import numpy as np
import threading
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.io.wavfile as wav

# Variabile globale
fs = 44100  # Frecvența de eșantionare
pitch_factor = 2.0  # Factor de pitch-up (ex. 2.0 = o octavă mai sus)
real_time_stream = None
stream_active = False  # Starea fluxului audio în timp real
recording_cut = None
positive_frequencies = None
positive_fft = None
nrinregitrare = 1
recording_active = False
recording_thread = None
recording = None

# Funcție pentru pitch shift prin resampling de calitate superioară
def pitch_shift_with_librosa(data, sr, pitch_factor):
    # Resamplează semnalul pentru a schimba pitch-ul
    target_sr = int(sr * pitch_factor)  # Crește frecvența de eșantionare
    resampled = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    # Resamplează înapoi la rata originală
    shifted = librosa.resample(resampled, orig_sr=target_sr, target_sr=sr)

    # Normalizează semnalul pentru a evita distorsiunile
    shifted = shifted / np.max(np.abs(shifted)) if np.max(np.abs(shifted)) > 0 else shifted

    return shifted

# Funcție pentru redare în timp real cu pitch-up
def toggle_real_time_playback():
    global real_time_stream, stream_active

    if not stream_active:
        # Pornește fluxul audio
        stream_active = True
        playback_button.config(text="Oprește Pitch-Up", style="Record.TButton")

        def audio_callback(indata, outdata, frames, time, status):
            if not stream_active:
                raise sd.CallbackStop()
            try:
                # Aplică pitch shift pe datele primite
                shifted = pitch_shift_with_librosa(indata[:, 0], sr=fs, pitch_factor=pitch_factor)

                # Ajustează dimensiunea pentru a se potrivi cu `frames`
                if len(shifted) > frames:
                    shifted = shifted[:frames]
                elif len(shifted) < frames:
                    shifted = np.pad(shifted, (0, frames - len(shifted)), mode='constant')

                # Asigură forma de ieșire
                outdata[:] = np.expand_dims(shifted, axis=1)
            except Exception as e:
                print(f"Error in audio_callback: {e}")

        real_time_stream = sd.Stream(samplerate=fs, channels=1, callback=audio_callback)

        # Pornește fluxul într-un thread separat
        threading.Thread(target=real_time_stream.start, daemon=True).start()
    else:
        # Oprește fluxul audio
        stream_active = False
        playback_button.config(text="Redare Pitch-Up")
        if real_time_stream:
            real_time_stream.stop()
            real_time_stream.close()

# Funcție pentru redarea înregistrării
def play_recording():
    global recording_cut
    if recording_cut is not None:
        print("Redare...")
        sd.play(recording_cut, int(sample_rate_input.get()))
        sd.wait()  # Așteaptă până se termină redarea
        print("Redare finalizată!")
    else:
        print("Nu există nicio înregistrare. Apasă pe 'Înregistrează' pentru a începe.")

# Funcție pentru oprirea înregistrării
def stop_recording():
    global recording_active, recording_cut, positive_frequencies, positive_fft, nrinregitrare

    if recording_active:
        print("Oprire înregistrare...")
        recording_active = False
        style.configure("Record.TButton", foreground="black")  # Resetează culoarea textului în negru
        record_button.config(text="Înregistrează")  # Resetează textul butonului

        # Decupează primele 100 ms pentru a elimina vârfurile inițiale
        fs = int(sample_rate_input.get())
        cut_time = 0.1  # Timp de tăiere (100 ms)
        cut_samples = int(cut_time * fs)
        recording_cut = recording[cut_samples:]

        # Aplică FFT (transformata Fourier rapidă)
        fft_result = np.fft.fft(recording_cut.flatten())  # Calcul FFT
        frequencies = np.fft.fftfreq(len(fft_result), 1 / fs)  # Frecvențele corespunzătoare

        # Filtrarea frecvențelor pozitive (spectrul real)
        positive_frequencies = frequencies[:len(frequencies) // 2]
        positive_fft = np.abs(fft_result)[:len(frequencies) // 2]

        # Salvează înregistrarea ca fișier WAV
        filename = f'C:\\Faculta\\an_3\\Licenta\\ProiectLicena\\recordings\\inregistrare{nrinregitrare}.wav'
        wav.write(filename, fs,
                  np.int16(recording_cut * 32767))  # Conversia valorilor float în int16 pentru a salva fișierul
        print(f"Fișierul a fost salvat ca {filename}")
        nrinregitrare += 1

        # Actualizează graficul
        ax.clear()
        ax.plot(positive_frequencies, positive_fft)
        ax.set_title("Spectrul de frecvențe")
        ax.set_xlabel("Frecvență (Hz)")
        ax.set_ylabel("Amplitudine")
        ax.grid(True)
        canvas.draw()  # Redesenare grafic

# Funcție pentru începerea/oprierea înregistrării
def toggle_recording():
    global recording_active, recording, recording_cut, positive_frequencies, positive_fft, nrinregitrare, recording_thread

    if not recording_active:
        try:
            fs = int(sample_rate_input.get())
            seconds = float(duration_input.get())
        except ValueError:
            print("Te rog să introduci valori numerice valide pentru sample rate și durata.")
            return

        print("Începere înregistrare...")
        recording_active = True
        style.configure("Record.TButton", foreground="red")
        record_button.config(text="Oprește Înregistrarea")

        def record_audio():
            global recording
            recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float64')
            sd.wait()
            stop_recording()

        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
    else:
        stop_recording()

# Creează interfața grafică
root = tk.Tk()
root.title("Audio Visualizer with Pitch-Up")

# Stil pentru butoane
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12))
style.configure("Record.TButton", font=("Helvetica", 12), foreground="black")

# Layout pentru input-uri
input_frame = ttk.Frame(root)
input_frame.pack(padx=10, pady=10)

# Câmp pentru Sample Rate
ttk.Label(input_frame, text="Sample Rate:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
sample_rate_input = ttk.Entry(input_frame)
sample_rate_input.grid(row=0, column=1, padx=5, pady=5)
sample_rate_input.insert(0, "44100")

# Câmp pentru Durată
ttk.Label(input_frame, text="Durată (secunde):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
duration_input = ttk.Entry(input_frame)
duration_input.grid(row=1, column=1, padx=5, pady=5)
duration_input.insert(0, "10")

# Creează și afișează graficul în interfață
fig, ax = plt.subplots(figsize=(8, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(padx=10, pady=10)

# Adaugă butonul de înregistrare
record_button = ttk.Button(root, text="Înregistrează", command=toggle_recording, style="Record.TButton")
record_button.pack(pady=10)

# Adaugă butonul de redare normală
play_button = ttk.Button(root, text="Redă Înregistrarea", command=play_recording, style="TButton")
play_button.pack(pady=10)

# Adaugă butonul de redare cu pitch-up
playback_button = ttk.Button(root, text="Redare Pitch-Up", command=toggle_real_time_playback, style="TButton")
playback_button.pack(pady=10)

# Rulează aplicația
root.mainloop()