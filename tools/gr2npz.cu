/*
 * @Description:
 * @Date: 2020-12-24 14:04:06
 * @LastEditors: PengyuWang
 * @LastEditTime: 2020-12-24 14:22:35
 * @FilePath: /sampling/tools/gr2npz.cu
 */
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
#include <iostream>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for myPyObject.cast<std::vector<T>>()
#include <vector>
#include "graph.cuh"


namespace py = pybind11;

int main() {
  py::scoped_interpreter guard{};

  py::module np = py::module::import("numpy");
  // py::object random = np.attr("random");
  // py::module scipy = py::module::import("scipy.optimize");

  // Load created module containing f_a(x) = a*x^2
  // py::module myModule = py::module::import("MyPythonModule.MyFunctionality");

  // Create some data for fitting
  std::vector<double> xValues(11, 0);
  std::vector<double> yValues(11, 0);
  for (int i = -5; i < 6; ++i) {
    xValues[i + 5] = i;
    yValues[i + 5] = i * i;
  }

  // Cast data to numpy arrays
  py::array_t<double> pyXValues = py::cast(xValues);
  py::array_t<double> pyYValues = py::cast(yValues);

  // The return value contains the optimal values and the covariance matrix.
  // Get the optimal values
  py::object optVals = retVals.attr("__getitem__")(0);

  // Cast return value back to std::vector and show the result
  std::vector<double> retValsStd = optVals.cast<std::vector<double>>();
  std::cout << "Fitted parameter a = " << retValsStd[0] << std::endl;

  return 0;
}

py::array_t<uint> my_fft1d_complex(py::array_t<> input) {

    if (input.ndim() != 1)
        throw std::runtime_error("input dim must be 1");

    vector<complex<float>> in, out;
    auto r1 = input.unchecked<1>();
    for (int i = 0; i < input.size(); i++)
    {
        in.push_back(r1(i));
    }

    fft1d(in, out, in.size());

    py::array_t<std::complex<float>> result(out.size());
    auto r2 = result.mutable_unchecked<1>();

    for (int i = 0; i < out.size(); i++)
    {
        r2(i) = out[i];
    }

    return result;

}
