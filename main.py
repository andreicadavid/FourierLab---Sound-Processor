import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Parametri
fs = 44100  # Rata de eșantionare (samples per second)
seconds = 10  # Durata înregistrării (în secunde)
nrinregitrare = 1
# Înregistrarea sunetului
print("Începere înregistrare...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float64')
sd.wait()  # Așteaptă să se termine înregistrarea
nrinregitrare+=1;
print("Înregistrare finalizată!")

# Decupează primele 100 ms pentru a elimina vârfurile inițiale
cut_time = 0.1  # Timp de tăiere (100 ms)
cut_samples = int(cut_time * fs)
recording_cut = recording[cut_samples:]

# Aplică FFT (transformata Fourier rapidă)
fft_result = np.fft.fft(recording_cut.flatten())  # Calcul FFT
frequencies = np.fft.fftfreq(len(fft_result), 1/fs)  # Frecvențele corespunzătoare

# Filtrarea frecvențelor pozitive (spectrul real)
positive_frequencies = frequencies[:len(frequencies)//2]
positive_fft = np.abs(fft_result)[:len(frequencies)//2]

# Vizualizează spectrul de frecvențe
plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, positive_fft)
plt.title("Spectrul de frecvențe")
plt.xlabel("Frecvență (Hz)")
plt.ylabel("Amplitudine")
plt.grid(True)
plt.show()

# Salvează înregistrarea ca fișier WAV
filename = f'C:\\Faculta\\an_3\\Licenta\\ProiectLicena\\recordings\\inregistrare{nrinregitrare}.wav'
wav.write(filename, fs, np.int16(recording_cut * 32767))  # Conversia valorilor float în int16 pentru a salva fișierul
print(f"Fișierul a fost salvat ca {filename}")

# Redă sunetul înregistrat
print("Redare...")
sd.play(recording_cut, fs)
sd.wait()  # Așteaptă până se termină redarea
