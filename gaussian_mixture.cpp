#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <math.h>

#include "jzq.h"
#include "poisson_disk_sampling.h"

template <int N, typename T>
Vec<N, T> sampleBilinear(unsigned int flw_width, unsigned int flw_height, PyArrayObject *I, const V2f &x)
{

    int x0 = static_cast<int>(std::floor(x[0]));
    int x1 = x0 + 1;
    int y0 = static_cast<int>(std::floor(x[1]));
    int y1 = y0 + 1;

    if (x0 < 0 || x1 >= flw_width || y0 < 0 || y1 >= flw_height)
    {
        return Vec<N, T>{0};
    }

    float x_diff = x[0] - x0;
    float y_diff = x[1] - y0;

    Vec<N, T> result{0};
    for (int n = 0; n < N; ++n)
    {
        float v00 = *(float *)PyArray_GETPTR3(I, y0, x0, n);
        float v01 = *(float *)PyArray_GETPTR3(I, y1, x0, n);
        float v10 = *(float *)PyArray_GETPTR3(I, y0, x1, n);
        float v11 = *(float *)PyArray_GETPTR3(I, y1, x1, n);

        result[n] = (v00 * (1 - x_diff) * (1 - y_diff) +
                     v10 * x_diff * (1 - y_diff) +
                     v01 * (1 - x_diff) * y_diff +
                     v11 * x_diff * y_diff);
    }
    return result;
}

const float SQR(float x) { return x * x; }

static std::vector<V2f> genPts(unsigned int msk_height, unsigned int msk_width, PyArrayObject *mask,
                               float radius, unsigned int max_sample_attempts = 30, unsigned int seed = 0)
{
    const auto x_min = std::array<float, 2>{{0.f, 0.f}};
    const auto x_max = std::array<float, 2>{{float(msk_width - 1), float(msk_height - 1)}};
    const auto samples = thinks::poisson_disk_sampling::PoissonDiskSampling(radius, x_min, x_max);

    std::vector<V2f> result;
    uint8_t *data = reinterpret_cast<uint8_t *>(PyArray_DATA(mask));

    for (const auto &sample : samples)
    {
        int x = sample[0];
        int y = sample[1];
        if (x >= 0 && x < msk_width && y >= 0 && y < msk_height)
        {
            if (data[y * msk_width + x] > 64)
            {
                result.push_back(V2f(sample[0], sample[1]));
            }
        }
    }

    return result;
}

static PyObject *generate(PyObject *self, PyObject *args)
{
    PyArrayObject *mask;
    PyArrayObject *flow;
    unsigned int radius = 10, sigma = 10, max_sample_attempts = 30, seed = 0;
    if (!PyArg_ParseTuple(args, "O!O!|IIII",
                          &PyArray_Type, &mask, &PyArray_Type, &flow,
                          &radius, &sigma, &max_sample_attempts, &seed))
    {
        return NULL;
    }

    npy_intp *msk_shape = PyArray_SHAPE(mask);
    const unsigned int msk_height = msk_shape[0];
    const unsigned int msk_width = msk_shape[1];

    npy_intp *flw_shape = PyArray_SHAPE(flow);

    const unsigned int flw_height = flw_shape[0];
    const unsigned int flw_width = flw_shape[1];
    const unsigned int flw_channel = flw_shape[2]; // 2

    const std::vector<V2f> keyPs = genPts(msk_height, msk_width, mask, radius, max_sample_attempts, seed);

    std::vector<V2f> Ps = keyPs;
    for (int i = 0; i < Ps.size(); i++)
    {
        Ps[i] = Ps[i] + sampleBilinear<2, float>(flw_width, flw_height, flow, Ps[i]);
    }

    PyObject *ks_result = PyList_New(keyPs.size());
    for (size_t i = 0; i < keyPs.size(); i++)
    {
        PyObject *point = Py_BuildValue("(ff)", keyPs[i][0], keyPs[i][1]);
        PyList_SetItem(ks_result, i, point);
    }

    PyObject *ps_result = PyList_New(Ps.size());
    for (size_t i = 0; i < Ps.size(); i++)
    {
        PyObject *point = Py_BuildValue("(ff)", Ps[i][0], Ps[i][1]);
        PyList_SetItem(ps_result, i, point);
    }

    PyObject *result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, ks_result);
    PyTuple_SetItem(result, 1, ps_result);

    // Py_DECREF(mask);
    // Py_DECREF(flow);

    return result;

    // std::stringstream ss;
    // ss << "Mask shape: (" << msk_height << ", " << msk_width << ")";
    // ss << "\nPoisson: (" << keyPs.size() << ") [" << keyPs[0][0] << ", " << keyPs[0][1] << "])";
    // ss << "\nBilinear: (" << Ps.size() << ") [" << Ps[0][0] << ", " << Ps[0][1] << "])";
    // ss << "\nFlow shape: (" << flw_height << ", " << flw_width << ", " << flw_channel << ")";

    // return PyUnicode_FromString(ss.str().c_str());
}

static PyMethodDef ModuleMethods[] = {
    {"generate", generate, METH_VARARGS, "Generate Gaussian Mixture given Mask and Flow"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "gaussian_mixture",
    NULL,
    -1,
    ModuleMethods};

PyMODINIT_FUNC PyInit_gaussian_mixture(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}