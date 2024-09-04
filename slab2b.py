import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Función para cargar un archivo .wav
def cargar_wav(file_path):
    sig, fs = sf.read(file_path)
    return sig, fs

# Rutas a los archivos de audio de los tres micrófonos
wavs = ["j1 cel1-e.wav", "j2 cel2-e.wav", "j3 cel3-e.wav"]

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

# Verificacion de que todas las señales posean la misma frecuencia de muestreo
if not all(fs == muestras[0] for fs in muestras):
    raise ValueError("Todas las señales deben tener la misma frecuencia de muestreo.")

# Asegurarse de mantener la misma longitud
min_length = min(len(sig) for sig in sigs)
sigs = [sig[:min_length] for sig in sigs]

# Parámetros del beamformer
c = 343  # Velocidad del sonido en el aire (m/s)

# Posiciones de los micrófonos (en metros)
mic_positions = np.array([
    [0, 0],  # Micro 1
    [1.16, 1.16], # Micro 2
    [2.32, 0] # Micro 3
]) 

# Posiciones de las fuentes (en metros)
source_positions = np.array([
    [1.16, 0],  # Persona 1
    [0, 1.16],  # Persona 2
    [2.32, 1.16]  # Persona 3
])

# Función para alinear señales según los retardos
def align_sigs(sigs, mic_positions, source_positions, fs):
    num_sources = len(source_positions)
    aligned_sigs = np.zeros((num_sources, len(sigs[0])))
    
    for i, src_pos in enumerate(source_positions):
        delays = np.zeros(len(mic_positions))
        for j, mic_pos in enumerate(mic_positions):
            distance = np.linalg.norm(mic_pos - src_pos)
            delays[j] = distance / c
        
        # Aplicar los retardos a las señales
        delay_samples = np.round(delays * fs).astype(int)
        aligned_sigs[i] = np.zeros(len(sigs[0]))
        for j in range(len(sigs)):
            sig = sigs[j]
            aligned_sigs[i] += np.roll(sig, int(-delay_samples[j]))[:len(sig)]
    
    return aligned_sigs

# Aplicar el beamforming para alinear las señales
aligned_sigs = align_sigs(sigs, mic_positions, source_positions, muestras[0])

# Guardar las señales separadas como archivos .wav
fs = muestras[0]  # Usar la frecuencia de muestreo común
for i in range(aligned_sigs.shape[0]):
    sf.write(f'beamformed_{i+1}.wav', aligned_sigs[i], fs)

# Graficar las señales separadas
plt.figure(figsize=(12, 8))
for i in range(aligned_sigs.shape[0]):
    plt.subplot(3, 1, i + 1)
    plt.plot(aligned_sigs[i])
    plt.title(f'Señal separada por Beamforming {i+1}')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
plt.tight_layout()
plt.show()
