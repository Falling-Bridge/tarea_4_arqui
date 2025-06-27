import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from PIL import Image
from io import BytesIO

def downsample_cpu(imagen_np):
    H, W, C = imagen_np.shape
    H_new, W_new = H // 4, W // 4
    img_reshaped = imagen_np.reshape(H_new, 4, W_new, 4, C)
    img_reducida = img_reshaped.mean(axis=(1,3))
    return img_reducida.astype(np.uint8)
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
# === MAIN ===

    plt.show()
# URL de imagen RAW en GitHub
url = 'https://raw.githubusercontent.com/Falling-Bridge/tarea_4_arqui/main/4096/prueba3.jpg'

try:
    # Cargar imagen original
    imagen_pil = Image.open(url).convert('RGB')
    imagen_np = np.array(imagen_pil)   
except Exception as e:
    print(f"An error occurred: {e}")
    image_data = None

cuda_kernel_code = """
__global__ void downsample_kernel(const float* input_img, float* output_img, int img_width) {
    // Calculate global thread index corresponding to output block coordinates
    int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    int output_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the bounds of the output image
    if (output_x >= img_width / 4 || output_y >= img_width / 4) {
        return;
    }

    // Calculate the starting pixel coordinates of the corresponding 4x4 block in the input image
    int start_x = output_x * 4;
    int start_y = output_y * 4;

    float sum = 0.0f;
    // Iterate through the 4x4 block in the input image and sum pixel values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            // Calculate the linear index in the input image
            int input_idx = (start_y + i) * img_width + (start_x + j);
            sum += input_img[input_idx];
        }
    }

    // Calculate the average and store in the output image
    float average = sum / 16.0f;

    // Calculate the linear index in the output image
    int output_idx = output_y * (img_width / 4) + output_x;
    output_img[output_idx] = average;
}
"""
print("CUDA kernel code defined.")

module = SourceModule(cuda_kernel_code)
downsample_kernel = module.get_function("downsample_kernel")

print("CUDA kernel compiled and function obtained.")

if 'image_data_flat' not in locals() or image_data_flat is None or image_data_flat.shape != (4096*4096,):
    print("Error: Image data is missing or has an incorrect shape.")
else:
    # Input image data size in bytes (float32 is 4 bytes)
    input_size_bytes = image_data_flat.nbytes
    print(f"Input image data size: {input_size_bytes} bytes")

    # Allocate memory on GPU for input image
    try:
        gpu_input_img = cuda.mem_alloc(input_size_bytes)
        print("GPU memory allocated for input image.")
    except Exception as e:
        print(f"Error allocating GPU memory for input image: {e}")
        gpu_input_img = None


    # Output image size (1024x1024)
    output_width = 1024
    output_height = 1024
    output_size = output_width * output_height

    # Output image data size in bytes (float32 is 4 bytes)
    output_size_bytes = output_size * np.float32(0).itemsize
    print(f"Output image data size: {output_size_bytes} bytes")

    # Allocate memory on GPU for output image
    try:
        gpu_output_img = cuda.mem_alloc(output_size_bytes)
        print("GPU memory allocated for output image.")
    except Exception as e:
        print(f"Error allocating GPU memory for output image: {e}")
        gpu_output_img = None

    if gpu_input_img is not None and gpu_output_img is not None:
        print("GPU memory has been allocated for both input and output images.")

# Data a GPU 

if 'image_data_flat' not in locals() or image_data_flat is None:
    print("Error: Input image data (image_data_flat) is not available.")
elif 'gpu_input_img' not in locals() or gpu_input_img is None:
     print("Error: GPU input memory allocation (gpu_input_img) is not available.")
else:
    try:
        cuda.memcpy_htod(gpu_input_img, image_data_flat)
        print("Successfully transferred input image data from CPU to GPU.")
    except Exception as e:
        print(f"Error transferring data to GPU: {e}")

# 1. Define the block size
block_dim = (16, 16, 1)
print(f"Block dimensions set to: {block_dim}")

# 2. Calculate the required grid size based on output image dimensions (1024x1024)
output_width = 1024
output_height = 1024
block_dim_x = block_dim[0]
block_dim_y = block_dim[1]

# Calculate grid dimensions using ceiling division
grid_dim_x = (output_width + block_dim_x - 1) // block_dim_x
grid_dim_y = (output_height + block_dim_y - 1) // block_dim_y

# 3. Define the grid dimensions as a tuple
grid_dim = (grid_dim_x, grid_dim_y, 1)

# 4. Print the calculated block and grid dimensions
print(f"Grid dimensions calculated as: {grid_dim}")

# 1. Check if necessary components are available
if 'downsample_kernel' not in locals() or downsample_kernel is None:
    print("Error: CUDA kernel function 'downsample_kernel' is not available.")
    can_launch_kernel = False
elif 'gpu_input_img' not in locals() or gpu_input_img is None:
    print("Error: GPU input memory (gpu_input_img) is not available.")
    can_launch_kernel = False
elif 'gpu_output_img' not in locals() or gpu_output_img is None:
    print("Error: GPU output memory (gpu_output_img) is not available.")
    can_launch_kernel = False
elif 'original_width' not in locals():
     print("Error: Input image width (original_width) is not available.")
     can_launch_kernel = False
else:
    can_launch_kernel = True

# 2. Launch the kernel if all components are available
if can_launch_kernel:
    try:
        # Launch the kernel
        downsample_kernel(
            gpu_input_img,
            gpu_output_img,
            np.int32(original_width), # Pass image width as a kernel argument
            block=block_dim,
            grid=grid_dim
        )
        # 3. Print confirmation message
        print("CUDA kernel 'downsample_kernel' launched successfully.")
    except Exception as e:
        print(f"Error launching CUDA kernel: {e}")
else:
    print("CUDA kernel cannot be launched due to missing components.")
# 1. Check if the GPU output memory and the output image dimensions are available.
if 'gpu_output_img' not in locals() or gpu_output_img is None:
    print("Error: GPU output memory (gpu_output_img) is not available.")
elif 'output_width' not in locals() or 'output_height' not in locals():
    print("Error: Output image dimensions (output_width, output_height) are not available.")
else:
    # 2. Create an empty NumPy array on the CPU to store the downsampled image data.
    try:
        cpu_output_img = np.empty((output_height, output_width), dtype=np.float32)
        print(f"CPU array created with shape {cpu_output_img.shape} and dtype {cpu_output_img.dtype}")

        # 3. Use the cuda.memcpy_dtoh() function to transfer the data from the GPU memory to the CPU NumPy array.
        cuda.memcpy_dtoh(cpu_output_img, gpu_output_img)

        # 4. Print a confirmation message.
        print("Downsampled data successfully transferred from GPU to CPU.")

    except Exception as e:
        # If an error occurs during the transfer, print an informative error message.
        print(f"Error transferring data from GPU to CPU: {e}")
# 1. Check if necessary components are available
if 'downsample_kernel' not in locals() or downsample_kernel is None:
    print("Error: CUDA kernel function 'downsample_kernel' is not available.")
    can_measure_time = False
elif 'gpu_input_img' not in locals() or gpu_input_img is None:
    print("Error: GPU input memory (gpu_input_img) is not available.")
    can_measure_time = False
elif 'gpu_output_img' not in locals() or gpu_output_img is None:
    print("Error: GPU output memory (gpu_output_img) is not available.")
    can_measure_time = False
elif 'original_width' not in locals():
     print("Error: Input image width (original_width) is not available.")
     can_measure_time = False
elif 'block_dim' not in locals():
     print("Error: Block dimensions (block_dim) are not available.")
     can_measure_time = False
elif 'grid_dim' not in locals():
     print("Error: Grid dimensions (grid_dim) are not available.")
     can_measure_time = False
else:
    # 2. If all components are available, set can_measure_time to True.
    can_measure_time = True
    print("All necessary components for time measurement are available.")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

if 'cpu_output_img' not in locals() or cpu_output_img is None:
    print("Error: Downsampled data (cpu_output_img) is not available for visualization or saving.")
else:
    print("Downsampled data is available. Attempting to save and visualize.")
    # Attempt to save the image
    try:
        # Scale pixel values and convert to uint8 for grayscale image
        # Assuming the float values are between 0 and 255 or need scaling to that range
        # If the values are normalized between 0 and 1, scale by 255
        if np.max(cpu_output_img) <= 1.0: # Simple check if values might be normalized
             scaled_img_data = (cpu_output_img * 255).astype(np.uint8)
        else: # Otherwise, assume they are in a larger range and clip/convert directly
             scaled_img_data = np.clip(cpu_output_img, 0, 255).astype(np.uint8)

        img_to_save = Image.fromarray(scaled_img_data, 'L') # 'L' mode for grayscale
        save_path = 'downsampled_image.png'
        img_to_save.save(save_path)
        print(f"Downsampled image successfully saved to {save_path}")
    except Exception as e:
        print(f"Error saving the downsampled image: {e}")

    # Attempt to visualize the image
    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(cpu_output_img, cmap='gray') # Use gray colormap for grayscale
        plt.title("Downsampled Image")
        plt.axis('off')
        plt.show()
        print("Downsampled image visualization attempted.")
    except Exception as e:
        print(f"Error visualizing the downsampled image: {e}")