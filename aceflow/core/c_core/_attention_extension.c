#include <Python.h>
#include <numpy/arrayobject.h>
#include "_attention_ops.h"

static PyObject* bahdanau_attention_forward_python(PyObject* self, PyObject* args) {
    PyArrayObject *decoder_hidden_arr, *encoder_outputs_arr;
    PyArrayObject *w1_arr, *w2_arr, *v_arr;
    int batch_size, seq_len, hidden_size, attention_type;
    
    if (!PyArg_ParseTuple(args, "OOOiiiiO!O!O!", 
                         &decoder_hidden_arr, &encoder_outputs_arr,
                         &batch_size, &seq_len, &hidden_size, &attention_type,
                         &PyArray_Type, &w1_arr, &PyArray_Type, &w2_arr, &PyArray_Type, &v_arr)) {
        // Try without weight arrays (for dot product attention)
        PyErr_Clear();
        if (!PyArg_ParseTuple(args, "OOiiii", 
                             &decoder_hidden_arr, &encoder_outputs_arr,
                             &batch_size, &seq_len, &hidden_size, &attention_type)) {
            return NULL;
        }
        w1_arr = w2_arr = v_arr = NULL;
    }
    
    // Get pointers to numpy array data
    float* decoder_hidden_data = (float*)PyArray_DATA(decoder_hidden_arr);
    float* encoder_outputs_data = (float*)PyArray_DATA(encoder_outputs_arr);
    
    float* w1_data = NULL;
    float* w2_data = NULL;
    float* v_data = NULL;
    
    if (w1_arr) w1_data = (float*)PyArray_DATA(w1_arr);
    if (w2_arr) w2_data = (float*)PyArray_DATA(w2_arr);
    if (v_arr) v_data = (float*)PyArray_DATA(v_arr);
    
    // Create output arrays
    npy_intp attention_weights_dims[] = {batch_size, seq_len};
    npy_intp context_vector_dims[] = {batch_size, hidden_size};
    
    PyArrayObject* attention_weights_arr = (PyArrayObject*)PyArray_SimpleNew(2, attention_weights_dims, NPY_FLOAT32);
    PyArrayObject* context_vector_arr = (PyArrayObject*)PyArray_SimpleNew(2, context_vector_dims, NPY_FLOAT32);
    
    if (!attention_weights_arr || !context_vector_arr) {
        Py_XDECREF(attention_weights_arr);
        Py_XDECREF(context_vector_arr);
        return NULL;
    }
    
    float* attention_weights_data = (float*)PyArray_DATA(attention_weights_arr);
    float* context_vector_data = (float*)PyArray_DATA(context_vector_arr);
    
    // Initialize output arrays to zero
    memset(attention_weights_data, 0, batch_size * seq_len * sizeof(float));
    memset(context_vector_data, 0, batch_size * hidden_size * sizeof(float));
    
    // Call C implementation
    int result = bahdanau_attention_forward(
        decoder_hidden_data, encoder_outputs_data,
        attention_weights_data, context_vector_data,
        w1_data, w2_data, v_data,
        batch_size, seq_len, hidden_size,
        (attention_type_t)attention_type
    );
    
    if (result != 0) {
        Py_DECREF(attention_weights_arr);
        Py_DECREF(context_vector_arr);
        
        switch (result) {
            case -1:
                PyErr_SetString(PyExc_MemoryError, "Memory allocation failed in attention forward pass");
                break;
            case -2:
                PyErr_SetString(PyExc_ValueError, "Missing weights for concat attention");
                break;
            case -3:
                PyErr_SetString(PyExc_ValueError, "Missing weights for general attention");
                break;
            default:
                PyErr_SetString(PyExc_RuntimeError, "Attention forward pass failed");
        }
        return NULL;
    }
    
    return Py_BuildValue("(NN)", attention_weights_arr, context_vector_arr);
}

static PyMethodDef module_methods[] = {
    {"bahdanau_attention_forward", bahdanau_attention_forward_python, METH_VARARGS, 
     "Optimized Bahdanau attention forward pass"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef attention_extension = {
    PyModuleDef_HEAD_INIT,
    "_attention_ops",
    "C-optimized attention operations for AceFlow",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__attention_ops(void) {
    import_array();
    return PyModule_Create(&attention_extension);
}