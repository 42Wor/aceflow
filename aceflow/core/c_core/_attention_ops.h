#ifndef ACEFLOW_ATTENTION_OPS_H
#define ACEFLOW_ATTENTION_OPS_H

#include <stddef.h>

// Attention types
typedef enum {
    ATTENTION_DOT = 0,
    ATTENTION_GENERAL = 1,
    ATTENTION_CONCAT = 2
} attention_type_t;

// Bahdanau attention forward pass
int bahdanau_attention_forward(float* decoder_hidden,      // [batch_size, hidden_size]
                               float* encoder_outputs,     // [batch_size, seq_len, hidden_size]
                               float* attention_weights,   // [batch_size, seq_len] (output)
                               float* context_vector,      // [batch_size, hidden_size] (output)
                               const float* w1,           // [hidden_size, hidden_size] (optional)
                               const float* w2,           // [hidden_size, hidden_size] (optional)
                               const float* v,            // [hidden_size] (optional)
                               int batch_size,
                               int seq_len,
                               int hidden_size,
                               attention_type_t attention_type);

// Utility functions for attention
void softmax_forward(float* x, int batch_size, int seq_len);
void tanh_forward(float* x, int n);
void matrix_multiply(float* C, const float* A, const float* B, int m, int n, int k);
void matrix_add(float* C, const float* A, const float* B, int m, int n);

#endif