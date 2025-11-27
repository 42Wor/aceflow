#ifndef ACEFLOW_RNN_OPS_H
#define ACEFLOW_RNN_OPS_H

#include <stddef.h>

// RNN types
typedef enum {
    RNN_TYPE_RNN = 0,
    RNN_TYPE_LSTM = 1,
    RNN_TYPE_GRU = 2
} rnn_type_t;

// RNN forward function
int rnn_forward(float* input,           // [batch_size, seq_len, input_size]
                float* hidden,          // [num_layers, batch_size, hidden_size] 
                float* cell,            // [num_layers, batch_size, hidden_size] (LSTM only)
                float* output,          // [batch_size, seq_len, hidden_size] (output)
                float* final_hidden,    // [num_layers, batch_size, hidden_size]
                float* final_cell,      // [num_layers, batch_size, hidden_size] (LSTM only)
                const float* weights,   // Flattened weights array
                const float* biases,    // Flattened biases array  
                int batch_size,
                int seq_len,
                int input_size,
                int hidden_size,
                int num_layers,
                rnn_type_t rnn_type,
                int bidirectional);

// LSTM cell forward pass
void lstm_cell_forward(float* input,          // [batch_size, input_size]
                       float* hidden,         // [batch_size, hidden_size]
                       float* cell,           // [batch_size, hidden_size]  
                       float* output,         // [batch_size, hidden_size]
                       const float* w_ih,     // [4*hidden_size, input_size]
                       const float* w_hh,     // [4*hidden_size, hidden_size]
                       const float* b_ih,     // [4*hidden_size]
                       const float* b_hh,     // [4*hidden_size]
                       int batch_size,
                       int input_size, 
                       int hidden_size);

// GRU cell forward pass  
void gru_cell_forward(float* input,          // [batch_size, input_size]
                      float* hidden,         // [batch_size, hidden_size]
                      float* output,         // [batch_size, hidden_size]
                      const float* w_ih,     // [3*hidden_size, input_size]
                      const float* w_hh,     // [3*hidden_size, hidden_size] 
                      const float* b_ih,     // [3*hidden_size]
                      const float* b_hh,     // [3*hidden_size]
                      int batch_size,
                      int input_size,
                      int hidden_size);

// Utility functions
void sigmoid_forward(float* x, int n);
void tanh_forward(float* x, int n);
void softmax_forward(float* x, int n, int dim);

#endif