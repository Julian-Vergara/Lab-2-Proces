import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import FastICA

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
    #Hacer que las señales sean unidimensionales
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

# Convertir las señales a una matriz de mezclas
X = np.array(sigs).T

# Aplicar FastICA
ica = FastICA(n_components=3)  # 3 fuentes independientes
S_ = ica.fit_transform(X)  # Fuentes estimadas
A_ = ica.mixing_  # Matriz de mezclado estimada

# Guardar las señales separadas como archivos .wav
fs = muestras[0]  # Usar la frecuencia de muestreo común
for i in range(S_.shape[1]):
    sf.write(f'Voice_{i+1}.wav', S_[:, i], fs)

# Graficar las señales originales y separadas
plt.figure(figsize=(12, 8))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(S_[:, i])
    plt.title(f'Señal separada {i+1}')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
plt.tight_layout()
plt.show()

