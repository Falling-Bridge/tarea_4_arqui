import numpy as np
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

# --- Cargar imagen y convertir a escala de grises
img = Image.open("e4283bd8-b862-41e6-a37a-0f052f716e5d.png").convert('L')  # Escala de grises
img_np = np.array(img).astype(np.float32)
height, width = img_np.shape

# --- Validación
assert height == 4096 and width == 4096, "La imagen debe tener resolución 4096x4096"

# --- Parámetros de salida
out_height, out_width = height // 4, width // 4
out_np = np.zeros((out_height, out_width), dtype=np.float32)

# --- Kernel CUDA
kernel_code = """
__global__ void downsample_4x4(float *input, float *output, int width_in, int width_out) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= width_out || out_y >= width_out) return;

    float sum = 0.0;
    for (int dy = 0; dy < 4; ++dy) {
        for (int dx = 0; dx < 4; ++dx) {
            int in_x = out_x * 4 + dx;
            int in_y = out_y * 4 + dy;
            sum += input[in_y * width_in + in_x];
        }
    }

    output[out_y * width_out + out_x] = sum / 16.0;
}
"""

mod = SourceModule(kernel_code)
func = mod.get_function("downsample_4x4")

# --- Reservar memoria en GPU
input_gpu = cuda.mem_alloc(img_np.nbytes)
output_gpu = cuda.mem_alloc(out_np.nbytes)

# --- Transferencia de datos
cuda.memcpy_htod(input_gpu, img_np)

# --- Configurar ejecución
block_size = (16, 16, 1)
grid_size = (out_width // 16, out_height // 16, 1)

# --- Medir tiempo
start = time.time()

func(input_gpu, output_gpu,
     np.int32(width), np.int32(out_width),
     block=block_size, grid=grid_size)

# --- Transferencia de vuelta
cuda.memcpy_dtoh(out_np, output_gpu)
end = time.time()

print(f"Tiempo de ejecución en GPU: {end - start:.4f} segundos")

# --- Guardar imagen reducida
img_reducida = Image.fromarray(out_np.astype(np.uint8))
img_reducida.save("imagen_reducida_gpu.png")
print("Imagen reducida guardada como 'imagen_reducida_gpu.png'")