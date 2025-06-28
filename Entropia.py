import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

def downsample_cpu(imagen_np):
    H, W, C = imagen_np.shape
    H_new, W_new = H // 4, W // 4
    img_reshaped = imagen_np.reshape(H_new, 4, W_new, 4, C)
    img_reducida = img_reshaped.mean(axis=(1,3))
    return img_reducida.astype(np.uint8)

def calcular_entropia_canal(canal, niveles=256):
    hist, _ = np.histogram(canal, bins=range(niveles+1), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calcular_entropia_total(imagen_np):
    r, g, b = imagen_np[...,0], imagen_np[...,1], imagen_np[...,2]
    h_r = calcular_entropia_canal(r)
    h_g = calcular_entropia_canal(g)
    h_b = calcular_entropia_canal(b)
    return (h_r + h_g + h_b)/3, (h_r, h_g, h_b)

def calcular_preservacion(H_orig, H_red):
    return H_orig / H_red

def mostrar_imagenes(original_np, reducida_np):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(original_np)
    plt.title("Imagen Original")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(reducida_np)
    plt.title("Reducida (CPU)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === MAIN ===

ruta_original = '4096/prueba3.jpg'
ruta_destino_cpu = '1024_cpu/resultado_prueba3.jpg'

# Cargar imagen original
imagen_pil = Image.open(ruta_original).convert('RGB')
imagen_np = np.array(imagen_pil)

# Reducción por CPU con medición de tiempo
start_time = time.time()
imagen_reducida_np = downsample_cpu(imagen_np)
end_time = time.time()
tiempo_cpu = end_time - start_time

# Guardar imagen reducida
os.makedirs("1024_cpu", exist_ok=True)
Image.fromarray(imagen_reducida_np).save(ruta_destino_cpu)

# Cálculo de entropía
H_orig, (Hr_o, Hg_o, Hb_o) = calcular_entropia_total(imagen_np)
H_red, (Hr_r, Hg_r, Hb_r) = calcular_entropia_total(imagen_reducida_np)
Pr = calcular_preservacion(H_orig, H_red)

# Resultados por consola
print("=== RESULTADOS CPU ===")
print(f"Tiempo de ejecución: {tiempo_cpu:.4f} segundos")
print(f"Entropía original: H = {H_orig:.4f} (R: {Hr_o:.4f}, G: {Hg_o:.4f}, B: {Hb_o:.4f})")
print(f"Entropía reducida: H = {H_red:.4f} (R: {Hr_r:.4f}, G: {Hg_r:.4f}, B: {Hb_r:.4f})")
print(f"Preservación de entropía: Pr = {Pr:.4f}")

# Mostrar imágenes
mostrar_imagenes(imagen_np, imagen_reducida_np)
