#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "poisson_disk_sampling.h"

static PyObject* poisson_disk_sampling(PyObject* self, PyObject* args) {
    float radius;
    PyObject *x_min_obj, *x_max_obj;
    unsigned int max_sample_attempts = 30;
    unsigned int seed = 0;

    if (!PyArg_ParseTuple(args, "fOO|II", &radius, &x_min_obj, &x_max_obj, &max_sample_attempts, &seed)) {
        return NULL;
    }

    std::array<float, 2> x_min, x_max;
    for (int i = 0; i < 2; i++) {
        x_min[i] = (float)PyFloat_AsDouble(PyList_GetItem(x_min_obj, i));
        x_max[i] = (float)PyFloat_AsDouble(PyList_GetItem(x_max_obj, i));
    }

    auto result = thinks::poisson_disk_sampling::PoissonDiskSampling<float, 2>(
        radius, x_min, x_max, max_sample_attempts, seed);

    PyObject* py_result = PyList_New(result.size());
    for (size_t i = 0; i < result.size(); i++) {
        PyObject* point = Py_BuildValue("(ff)", result[i][0], result[i][1]);
        PyList_SetItem(py_result, i, point);
    }

    return py_result;
}

static PyMethodDef PoissonMethods[] = {
    {"poisson_disk_sampling", poisson_disk_sampling, METH_VARARGS, "Poisson disk sampling"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef poissondiskmodule = {
    PyModuleDef_HEAD_INIT,
    "poisson_disk_module",
    NULL,
    -1,
    PoissonMethods
};

PyMODINIT_FUNC PyInit_poisson_disk_module(void) {
    return PyModule_Create(&poissondiskmodule);
}