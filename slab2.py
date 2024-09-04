import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import soundfile as sf

# Función para cargar un archivo .wav
def cargar_wav(file_path):
    sig, fs = sf.read(file_path)
    return sig, fs

# Función para calcular la media
def calc_media(valores):
    total = sum(valores)
    return total / len(valores)

# Función para calcular la desviación estándar
def calc_desv(valores, media):
    suma_difcuad = sum((x - media) ** 2 for x in valores)
    return (suma_difcuad / len(valores)) ** 0.5

# Función para calcular la relación señal-ruido (SNR)
def snr(valores, ruido):
    fsum = 0
    ssum = 0
    potv = 0
    potr = 0

    # Calcular la potencia de la señal
    for x in valores:
        fsum += x ** 2
    potv = fsum / len(valores)
    
    # Calcular la potencia del ruido
    for x in ruido:
        ssum += x ** 2
    potr = ssum / len(ruido)

    # Calcular el SNR en decibelios
    return 10 * np.log10(potv / potr)

# Función para calcular la SNR de las señales separadas por FastICA
def snr_ica(sigs_ica, r_sig):
    snr_vals_ica = []
    for i, sig in enumerate(sigs_ica):
        snr_val = snr(sig, r_sig)
        snr_vals_ica.append(snr_val)
        print(f"SNR para la señal separada por FastICA {i+1}: {snr_val:.2f} dB")
    return snr_vals_ica

# Rutas a los archivos de audio de los tres micrófonos y el ruido ambiental
wavs = ["j1 cel1-e.wav", "j2 cel2-e.wav", "j3 cel3-e.wav"]
r = "rt-e.wav"

# Cargar señales desde los archivos .wav
sigs = []
muestras = []

for file_path in wavs:
    sig, fs = cargar_wav(file_path)
    # Asegurarse de que las señales sean unidimensionales
    if len(sig.shape) > 1:
        sig = sig[:, 0]
    sigs.append(sig)
    muestras.append(fs)

r_sig, fs_ruido = cargar_wav(r)

# Verificacion de que todas las señales posean la misma frecuencia de muestreo
if not all(fs == muestras[0] for fs in muestras):
    raise ValueError("Todas las señales deben tener la misma frecuencia de muestreo.")
if fs_ruido != muestras[0]:
    raise ValueError("La frecuencia de muestreo del ruido debe coincidir con las señales de los micrófonos.")

# Mantener la misma longitud
min_length = min(len(sig) for sig in sigs)
sigs = [sig[:min_length] for sig in sigs]
r_sig = r_sig[:min_length]

# Calcular la SNR para cada micrófono
snr_vals = []
for i, sig in enumerate(sigs):
    snr_val = snr(sig, r_sig)
    snr_vals.append(snr_val)
    print(f"SNR para el Micrófono {i+1}: {snr_val:.2f} dB")
    
# Cargar los archivos de audio separados por ICA
sigs_ica = []
for i in range(3):
    sig_ica, _ = cargar_wav(f'Voice_{i+1}.wav')
    sigs_ica.append(sig_ica[:min_length])

# Calcular la SNR para las señales separadas por ICA
snr_vals_ica = snr_ica(sigs_ica, r_sig)

# Análisis temporal
plt.figure(figsize=(12, 8))
for i, sig in enumerate(sigs):
    t = np.arange(len(sig)) / muestras[i]
    plt.subplot(3, 1, i + 1)
    plt.plot(t, sig)
    plt.title(f'Señal del Micrófono {i+1} en el dominio del tiempo')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
plt.tight_layout()
plt.show()

# Análisis espectral o en frecuencia(FFT)
plt.figure(figsize=(12, 8))
for i, sig in enumerate(sigs):
    n = len(sig)
    yf = fft(sig)
    xf = fftfreq(n, 1 / muestras[i])[:n//2]
     # Filtrar las frecuencias fuera del rango de interés en este caso de 0 a 2000 hz
    lim_inf = 0
    lim_sup = 2000
    indices = np.where((xf >= lim_inf) & (xf <= lim_sup))
    xf_f = xf[indices]
    yf_f = 2.0/n * np.abs(yf[:n//2][indices])
    
    plt.subplot(3, 1, i + 1)
    plt.plot(xf_f, yf_f)
    plt.title(f'Señal del Micrófono {i+1} en el dominio de la frecuencia')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.xlim(lim_inf, lim_sup)
plt.tight_layout()
plt.show()
