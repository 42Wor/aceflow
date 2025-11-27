#include <Python.h>
#include <numpy/arrayobject.h>
#include "_rnn_ops.h"

static PyObject* rnn_forward_python(PyObject* self, PyObject* args) {
    PyArrayObject *input_arr, *hidden_arr, *cell_arr, *weights_arr, *biases_arr;
    int batch_size, seq_len, input_size, hidden_size, num_layers, rnn_type, bidirectional;
    
    if (!PyArg_ParseTuple(args, "OOOOiiiiiii", 
                         &input_arr, &hidden_arr, &cell_arr, &weights_arr, &biases_arr,
                         &batch_size, &seq_len, &input_size, &hidden_size, 
                         &num_layers, &rnn_type, &bidirectional)) {
        return NULL;
    }
    
    // Get pointers to numpy array data
    float* input_data = (float*)PyArray_DATA(input_arr);
    float* hidden_data = (float*)PyArray_DATA(hidden_arr);
    float* cell_data = (float*)PyArray_DATA(cell_arr);
    float* weights_data = (float*)PyArray_DATA(weights_arr);
    float* biases_data = (float*)PyArray_DATA(biases_arr);
    
    // Create output arrays
    npy_intp output_dims[] = {batch_size, seq_len, hidden_size};
    npy_intp final_hidden_dims[] = {num_layers, batch_size, hidden_size};
    npy_intp final_cell_dims[] = {num_layers, batch_size, hidden_size};
    
    PyArrayObject* output_arr = (PyArrayObject*)PyArray_SimpleNew(3, output_dims, NPY_FLOAT32);
    PyArrayObject* final_hidden_arr = (PyArrayObject*)PyArray_SimpleNew(3, final_hidden_dims, NPY_FLOAT32);
    PyArrayObject* final_cell_arr = (PyArrayObject*)PyArray_SimpleNew(3, final_cell_dims, NPY_FLOAT32);
    
    if (!output_arr || !final_hidden_arr || !final_cell_arr) {
        Py_XDECREF(output_arr);
        Py_XDECREF(final_hidden_arr);
        Py_XDECREF(final_cell_arr);
        return NULL;
    }
    
    float* output_data = (float*)PyArray_DATA(output_arr);
    float* final_hidden_data = (float*)PyArray_DATA(final_hidden_arr);
    float* final_cell_data = (float*)PyArray_DATA(final_cell_arr);
    
    // Initialize output arrays to zero
    memset(output_data, 0, batch_size * seq_len * hidden_size * sizeof(float));
    memset(final_hidden_data, 0, num_layers * batch_size * hidden_size * sizeof(float));
    memset(final_cell_data, 0, num_layers * batch_size * hidden_size * sizeof(float));
    
    // Call C implementation
    int result = rnn_forward(input_data, hidden_data, cell_data, output_data,
                           final_hidden_data, final_cell_data, weights_data, biases_data,
                           batch_size, seq_len, input_size, hidden_size, num_layers,
                           (rnn_type_t)rnn_type, bidirectional);
    
    if (result != 0) {
        Py_DECREF(output_arr);
        Py_DECREF(final_hidden_arr);
        Py_DECREF(final_cell_arr);
        PyErr_SetString(PyExc_RuntimeError, "RNN forward pass failed");
        return NULL;
    }
    
    return Py_BuildValue("(NNN)", output_arr, final_hidden_arr, final_cell_arr);
}

static PyMethodDef module_methods[] = {
    {"rnn_forward", rnn_forward_python, METH_VARARGS, "Optimized RNN forward pass"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef rnn_extension = {
    PyModuleDef_HEAD_INIT,
    "_rnn_ops",
    "C-optimized RNN operations for AceFlow",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__rnn_ops(void) {
    import_array();
    return PyModule_Create(&rnn_extension);
}