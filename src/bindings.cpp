#include "image_core.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(bill_cuda, m) {
  m.doc() = "CUDA-accelerated Image Processing for Bill Accelerator";

  py::class_<ImageCUDA>(m, "ImageCUDA")
      .def(py::init<int, int, int>())
      .def("load_data",
           [](ImageCUDA &self, py::array_t<uint8_t> array) {
             py::buffer_info buf = array.request();
             if (buf.ndim != 3 && buf.ndim != 2) {
               throw std::runtime_error("Number of dimensions must be 2 or 3");
             }

             // Verify total size matches expected size
             size_t input_size = buf.size * sizeof(uint8_t);
             // We can't check internal size easily without exposing it, but we
             // can check dimensions Assuming row-major
             int h = buf.shape[0];
             int w = buf.shape[1];
             int c = (buf.ndim == 3) ? buf.shape[2] : 1;

             if (w != self.get_width() || h != self.get_height() ||
                 c != self.get_channels()) {
               throw std::runtime_error(
                   "Input array shape does not match ImageCUDA dimensions");
             }

             self.load_data(static_cast<uint8_t *>(buf.ptr));
           })
      .def("get_data",
           [](ImageCUDA &self) {
             auto vec = self.get_data();
             std::vector<ssize_t> shape;
             if (self.get_channels() == 1) {
               shape = {self.get_height(), self.get_width()};
             } else {
               shape = {self.get_height(), self.get_width(),
                        self.get_channels()};
             }

             // Explicitly create numpy array and copy data
             // This is safer than relying on vector buffer
             py::array_t<uint8_t> result(shape);
             py::buffer_info buf = result.request();
             uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
             std::memcpy(ptr, vec.data(), vec.size() * sizeof(uint8_t));

             return result;
           })
      .def("to_grayscale", &ImageCUDA::to_grayscale)
      .def("apply_adaptive_threshold", &ImageCUDA::apply_adaptive_threshold)
      .def("apply_morphology", &ImageCUDA::apply_morphology)
      .def("synchronize", &ImageCUDA::synchronize);
}
