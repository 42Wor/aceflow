#include "_rnn_ops.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Sigmoid activation
void sigmoid_forward(float* x, int n) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

// Tanh activation  
void tanh_forward(float* x, int n) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; i++) {
        x[i] = tanhf(x[i]);
    }
}

// Matrix multiplication: C = A * B
static void matmul(float* C, const float* A, const float* B, 
                   int m, int n, int k) {
    // Simple matrix multiplication - can be replaced with BLAS
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Matrix multiplication with bias: C = A * B + bias
static void matmul_bias(float* C, const float* A, const float* B, 
                        const float* bias, int m, int n, int k) {
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = bias[j];
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// LSTM cell forward pass
void lstm_cell_forward(float* input, float* hidden, float* cell,
                       float* output, const float* w_ih, const float* w_hh,
                       const float* b_ih, const float* b_hh,
                       int batch_size, int input_size, int hidden_size) {
    
    int gate_size = 4 * hidden_size;
    
    // Temporary buffers
    float* gates = (float*)malloc(batch_size * gate_size * sizeof(float));
    float* input_gates = gates;
    float* forget_gates = gates + hidden_size;
    float* cell_gates = gates + 2 * hidden_size; 
    float* output_gates = gates + 3 * hidden_size;
    
    // Input projection: gates = input * w_ih + b_ih
    matmul_bias(gates, input, w_ih, b_ih, batch_size, gate_size, input_size);
    
    // Hidden projection: gates += hidden * w_hh + b_hh  
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int b = 0; b < batch_size; b++) {
        for (int g = 0; g < gate_size; g++) {
            float hidden_sum = b_hh[g];
            for (int h = 0; h < hidden_size; h++) {
                hidden_sum += hidden[b * hidden_size + h] * w_hh[h * gate_size + g];
            }
            gates[b * gate_size + g] += hidden_sum;
        }
    }
    
    // Apply activations
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < hidden_size; h++) {
            int idx = b * gate_size;
            
            // Input gate
            input_gates[idx + h] = 1.0f / (1.0f + expf(-input_gates[idx + h]));
            
            // Forget gate  
            forget_gates[idx + h] = 1.0f / (1.0f + expf(-forget_gates[idx + h]));
            
            // Cell gate
            cell_gates[idx + h] = tanhf(cell_gates[idx + h]);
            
            // Output gate
            output_gates[idx + h] = 1.0f / (1.0f + expf(-output_gates[idx + h]));
        }
    }
    
    // Update cell state: cell = forget_gate * cell + input_gate * cell_gate
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < hidden_size; h++) {
            int idx = b * gate_size;
            int cell_idx = b * hidden_size + h;
            
            cell[cell_idx] = forget_gates[idx + h] * cell[cell_idx] + 
                            input_gates[idx + h] * cell_gates[idx + h];
        }
    }
    
    // Compute output: output = output_gate * tanh(cell)
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < hidden_size; h++) {
            int idx = b * gate_size;
            int cell_idx = b * hidden_size + h;
            int out_idx = b * hidden_size + h;
            
            output[out_idx] = output_gates[idx + h] * tanhf(cell[cell_idx]);
        }
    }
    
    free(gates);
}

// GRU cell forward pass
void gru_cell_forward(float* input, float* hidden, float* output,
                      const float* w_ih, const float* w_hh,
                      const float* b_ih, const float* b_hh,
                      int batch_size, int input_size, int hidden_size) {
    
    int gate_size = 3 * hidden_size;
    
    // Temporary buffers
    float* gates = (float*)malloc(batch_size * gate_size * sizeof(float));
    float* reset_gates = gates;
    float* update_gates = gates + hidden_size;
    float* new_gates = gates + 2 * hidden_size;
    
    // Reset and update gates: gates[:, :2*hidden_size] = input * w_ih[:, :2*hidden_size] + b_ih[:, :2*hidden_size]
    matmul_bias(gates, input, w_ih, b_ih, batch_size, 2 * hidden_size, input_size);
    
    // Add hidden contribution to reset/update gates
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int b = 0; b < batch_size; b++) {
        for (int g = 0; g < 2 * hidden_size; g++) {
            float hidden_sum = b_hh[g];
            for (int h = 0; h < hidden_size; h++) {
                hidden_sum += hidden[b * hidden_size + h] * w_hh[h * gate_size + g];
            }
            gates[b * gate_size + g] += hidden_sum;
        }
    }
    
    // Apply sigmoid to reset and update gates
    sigmoid_forward(gates, batch_size * 2 * hidden_size);
    
    // Compute new gate: new_gate = input * w_ih[:, 2*hidden_size:] + b_ih[:, 2*hidden_size:]
    const float* w_ih_new = w_ih + 2 * hidden_size * input_size;
    const float* b_ih_new = b_ih + 2 * hidden_size;
    matmul_bias(new_gates, input, w_ih_new, b_ih_new, batch_size, hidden_size, input_size);
    
    // Add reset-gated hidden contribution to new gate
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < hidden_size; h++) {
            float reset_sum = b_hh[2 * hidden_size + h];
            for (int h2 = 0; h2 < hidden_size; h2++) {
                reset_sum += (hidden[b * hidden_size + h2] * reset_gates[b * gate_size + h2]) * 
                           w_hh[h2 * gate_size + 2 * hidden_size + h];
            }
            new_gates[b * gate_size + h] += reset_sum;
        }
    }
    
    // Apply tanh to new gate
    tanh_forward(new_gates, batch_size * hidden_size);
    
    // Compute output: output = (1 - update_gate) * new_gate + update_gate * hidden
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < hidden_size; h++) {
            int gate_idx = b * gate_size;
            int out_idx = b * hidden_size + h;
            
            output[out_idx] = (1.0f - update_gates[gate_idx + h]) * new_gates[gate_idx + h] +
                             update_gates[gate_idx + h] * hidden[out_idx];
        }
    }
    
    free(gates);
}

// Main RNN forward function
int rnn_forward(float* input, float* hidden, float* cell, float* output,
                float* final_hidden, float* final_cell, const float* weights,
                const float* biases, int batch_size, int seq_len, int input_size,
                int hidden_size, int num_layers, rnn_type_t rnn_type, int bidirectional) {
    
    if (bidirectional) {
        // For simplicity, this implementation handles unidirectional only
        // Bidirectional would require separate forward/backward passes
        printf("Bidirectional RNN not yet implemented in C extension\n");
        return -1;
    }
    
    int weights_per_layer, biases_per_layer;
    
    // Calculate weight sizes based on RNN type
    switch (rnn_type) {
        case RNN_TYPE_LSTM:
            weights_per_layer = 4 * hidden_size * (input_size + hidden_size);
            biases_per_layer = 8 * hidden_size;  // 4 gates * 2 (input+hidden)
            break;
        case RNN_TYPE_GRU:
            weights_per_layer = 3 * hidden_size * (input_size + hidden_size);
            biases_per_layer = 6 * hidden_size;  // 3 gates * 2 (input+hidden)
            break;
        case RNN_TYPE_RNN:
        default:
            weights_per_layer = hidden_size * (input_size + hidden_size);
            biases_per_layer = 2 * hidden_size;
            break;
    }
    
    // Process each layer
    for (int layer = 0; layer < num_layers; layer++) {
        const float* layer_weights = weights + layer * weights_per_layer;
        const float* layer_biases = biases + layer * biases_per_layer;
        
        float* layer_input = (layer == 0) ? input : output;
        int layer_input_size = (layer == 0) ? input_size : hidden_size;
        
        float* layer_hidden = hidden + layer * batch_size * hidden_size;
        float* layer_cell = cell + layer * batch_size * hidden_size;
        
        // Process each time step
        for (int t = 0; t < seq_len; t++) {
            float* step_input = layer_input + t * batch_size * layer_input_size;
            float* step_output = output + t * batch_size * hidden_size;
            
            switch (rnn_type) {
                case RNN_TYPE_LSTM: {
                    const float* w_ih = layer_weights;
                    const float* w_hh = layer_weights + 4 * hidden_size * layer_input_size;
                    const float* b_ih = layer_biases;
                    const float* b_hh = layer_biases + 4 * hidden_size;
                    
                    lstm_cell_forward(step_input, layer_hidden, layer_cell, step_output,
                                     w_ih, w_hh, b_ih, b_hh, batch_size, layer_input_size, hidden_size);
                    
                    // Update hidden state for next time step
                    memcpy(layer_hidden, step_output, batch_size * hidden_size * sizeof(float));
                    break;
                }
                    
                case RNN_TYPE_GRU: {
                    const float* w_ih = layer_weights;
                    const float* w_hh = layer_weights + 3 * hidden_size * layer_input_size;
                    const float* b_ih = layer_biases;
                    const float* b_hh = layer_biases + 3 * hidden_size;
                    
                    gru_cell_forward(step_input, layer_hidden, step_output,
                                    w_ih, w_hh, b_ih, b_hh, batch_size, layer_input_size, hidden_size);
                    
                    // Update hidden state for next time step
                    memcpy(layer_hidden, step_output, batch_size * hidden_size * sizeof(float));
                    break;
                }
                    
                case RNN_TYPE_RNN:
                default: {
                    // Simple RNN: output = tanh(input * W_ih + hidden * W_hh + b_ih + b_hh)
                    const float* w_ih = layer_weights;
                    const float* w_hh = layer_weights + hidden_size * layer_input_size;
                    const float* b_ih = layer_biases;
                    const float* b_hh = layer_biases + hidden_size;
                    
                    // input * W_ih + b_ih
                    matmul_bias(step_output, step_input, w_ih, b_ih, batch_size, hidden_size, layer_input_size);
                    
                    // Add hidden * W_hh + b_hh
                    #ifdef _OPENMP
                    #pragma omp parallel for collapse(2)
                    #endif
                    for (int b = 0; b < batch_size; b++) {
                        for (int h = 0; h < hidden_size; h++) {
                            float hidden_sum = b_hh[h];
                            for (int h2 = 0; h2 < hidden_size; h2++) {
                                hidden_sum += layer_hidden[b * hidden_size + h2] * w_hh[h2 * hidden_size + h];
                            }
                            step_output[b * hidden_size + h] += hidden_sum;
                        }
                    }
                    
                    // Apply tanh activation
                    tanh_forward(step_output, batch_size * hidden_size);
                    
                    // Update hidden state for next time step
                    memcpy(layer_hidden, step_output, batch_size * hidden_size * sizeof(float));
                    break;
                }
            }
        }
        
        // Save final hidden state for this layer
        memcpy(final_hidden + layer * batch_size * hidden_size, 
               layer_hidden, batch_size * hidden_size * sizeof(float));
        
        if (rnn_type == RNN_TYPE_LSTM) {
            memcpy(final_cell + layer * batch_size * hidden_size,
                   layer_cell, batch_size * hidden_size * sizeof(float));
        }
    }
    
    return 0;
}