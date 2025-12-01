#include "image_core.hpp"
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Kernels (Forward declaration)
__global__ void grayscale_kernel(const uint8_t *input, uint8_t *output,
                                 int width, int height, int channels);

ImageCUDA::ImageCUDA(int w, int h, int c) : width(w), height(h), channels(c) {
  size = width * height * channels * sizeof(uint8_t);
  CHECK_CUDA(cudaMalloc(&d_data, size));
  // Allocate temp buffer assuming max size (same as input for now)
  CHECK_CUDA(cudaMalloc(&d_temp, size));
}

ImageCUDA::~ImageCUDA() {
  CHECK_CUDA(cudaFree(d_data));
  CHECK_CUDA(cudaFree(d_temp));
}

void ImageCUDA::load_data(const uint8_t *host_data) {
  CHECK_CUDA(cudaMemcpy(d_data, host_data, size, cudaMemcpyHostToDevice));
}

std::vector<uint8_t> ImageCUDA::get_data() const {
  std::vector<uint8_t> host_vec(size);
  CHECK_CUDA(cudaMemcpy(host_vec.data(), d_data, size, cudaMemcpyDeviceToHost));
  return host_vec;
}

// --- Kernels ---

__global__ void grayscale_kernel(const uint8_t *input, uint8_t *output,
                                 int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = (y * width + x) * channels;
  int out_idx = y * width + x;

  if (channels == 3) {
    float r = input[idx];
    float g = input[idx + 1];
    float b = input[idx + 2];
    // Standard luminance formula
    output[out_idx] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
  } else {
    output[out_idx] = input[idx];
  }
}

void ImageCUDA::to_grayscale() {
  if (channels == 1)
    return; // Already gray

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // We write to d_temp, which will become the new d_data (1 channel)
  // Note: We need to handle the channel change in the class state
  grayscale_kernel<<<grid, block>>>(d_data, d_temp, width, height, channels);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Update state
  std::swap(d_data, d_temp);
  channels = 1;
  size = width * height * sizeof(uint8_t);
  // Note: d_temp is now the old RGB buffer, which is larger. That's fine.
}

// Naive Adaptive Thresholding Kernel
// Naive Adaptive Thresholding Kernel
// Uses a local window loop for mean calculation.
__global__ void adaptive_threshold_kernel(const uint8_t *input, uint8_t *output,
                                          int width, int height,
                                          int window_size, float c) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int half_win = window_size / 2;
  int sum = 0;
  int count = 0;

  // Compute local mean
  for (int dy = -half_win; dy <= half_win; ++dy) {
    for (int dx = -half_win; dx <= half_win; ++dx) {
      int nx = x + dx;
      int ny = y + dy;

      // Clamp to boundary
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        sum += input[ny * width + nx];
        count++;
      }
    }
  }

  float mean = (float)sum / count;
  uint8_t pixel = input[y * width + x];

  // Binarize
  if (pixel < (mean - c)) {
    output[y * width + x] = 0; // Black (Foreground/Text)
  } else {
    output[y * width + x] = 255; // White (Background)
  }
}

void ImageCUDA::apply_adaptive_threshold(int window_size, float c) {
  if (channels != 1) {
    std::cerr << "Error: Adaptive threshold requires grayscale image."
              << std::endl;
    return;
  }

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // Write to d_temp
  adaptive_threshold_kernel<<<grid, block>>>(d_data, d_temp, width, height,
                                             window_size, c);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Swap buffers
  std::swap(d_data, d_temp);
}

// Morphological Dilation (Max)
__global__ void dilation_kernel(const uint8_t *input, uint8_t *output,
                                int width, int height, int window_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int half_win = window_size / 2;
  uint8_t max_val = 0;

  for (int dy = -half_win; dy <= half_win; ++dy) {
    for (int dx = -half_win; dx <= half_win; ++dx) {
      int nx = x + dx;
      int ny = y + dy;

      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        uint8_t val = input[ny * width + nx];
        if (val > max_val)
          max_val = val;
      }
    }
  }
  output[y * width + x] = max_val;
}

// Morphological Erosion (Min)
__global__ void erosion_kernel(const uint8_t *input, uint8_t *output, int width,
                               int height, int window_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int half_win = window_size / 2;
  uint8_t min_val = 255;

  for (int dy = -half_win; dy <= half_win; ++dy) {
    for (int dx = -half_win; dx <= half_win; ++dx) {
      int nx = x + dx;
      int ny = y + dy;

      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        uint8_t val = input[ny * width + nx];
        if (val < min_val)
          min_val = val;
      }
    }
  }
  output[y * width + x] = min_val;
}

void ImageCUDA::apply_morphology(const std::string &op, int window_size) {
  if (channels != 1) {
    std::cerr << "Error: Morphology requires grayscale image." << std::endl;
    return;
  }

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  if (op == "dilation") {
    dilation_kernel<<<grid, block>>>(d_data, d_temp, width, height,
                                     window_size);
  } else if (op == "erosion") {
    erosion_kernel<<<grid, block>>>(d_data, d_temp, width, height, window_size);
  } else {
    std::cerr << "Unknown morphology operation: " << op << std::endl;
    return;
  }

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  std::swap(d_data, d_temp);
}

void ImageCUDA::synchronize() { CHECK_CUDA(cudaDeviceSynchronize()); }
