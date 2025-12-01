#pragma once
#include <cstdint>
#include <string>
#include <vector>

class ImageCUDA {
public:
  ImageCUDA(int width, int height, int channels);
  ~ImageCUDA();

  // Load data from host (e.g. from Python/Numpy)
  void load_data(const uint8_t *host_data);

  // Get data back to host
  std::vector<uint8_t> get_data() const;

  // Processing methods
  void to_grayscale();
  void apply_adaptive_threshold(int window_size, float c);
  void apply_morphology(const std::string &op, int window_size);
  void synchronize();

  // Getters
  int get_width() const { return width; }
  int get_height() const { return height; }
  int get_channels() const { return channels; }

private:
  int width;
  int height;
  int channels;
  size_t size;

  uint8_t *d_data; // Device pointer
  // We might need a secondary buffer for double-buffering
  uint8_t *d_temp;
};
