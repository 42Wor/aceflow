#include "_attention_ops.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Softmax along sequence dimension
void softmax_forward(float* x, int batch_size, int seq_len) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int b = 0; b < batch_size; b++) {
        float* batch_scores = x + b * seq_len;
        
        // Find max for numerical stability
        float max_score = batch_scores[0];
        for (int i = 1; i < seq_len; i++) {
            if (batch_scores[i] > max_score) {
                max_score = batch_scores[i];
            }
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            batch_scores[i] = expf(batch_scores[i] - max_score);
            sum_exp += batch_scores[i];
        }
        
        // Normalize
        if (sum_exp > 0.0f) {
            for (int i = 0; i < seq_len; i++) {
                batch_scores[i] /= sum_exp;
            }
        }
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
void matrix_multiply(float* C, const float* A, const float* B, int m, int n, int k) {
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

// Matrix addition: C = A + B
void matrix_add(float* C, const float* A, const float* B, int m, int n) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < m * n; i++) {
        C[i] = A[i] + B[i];
    }
}

// Bahdanau attention forward pass
int bahdanau_attention_forward(float* decoder_hidden, float* encoder_outputs,
                              float* attention_weights, float* context_vector,
                              const float* w1, const float* w2, const float* v,
                              int batch_size, int seq_len, int hidden_size,
                              attention_type_t attention_type) {
    
    // Temporary buffer for attention scores
    float* attention_scores = (float*)malloc(batch_size * seq_len * sizeof(float));
    if (!attention_scores) {
        return -1; // Memory allocation failed
    }
    
    switch (attention_type) {
        case ATTENTION_CONCAT: {
            // Bahdanau-style additive attention
            if (!w1 || !w2 || !v) {
                free(attention_scores);
                return -2; // Missing weights for concat attention
            }
            
            #ifdef _OPENMP
            #pragma omp parallel for collapse(2)
            #endif
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < seq_len; s++) {
                    float score = 0.0f;
                    
                    // Compute: v^T * tanh(W1 * h_enc + W2 * h_dec)
                    for (int h = 0; h < hidden_size; h++) {
                        float enc_contrib = 0.0f;
                        float dec_contrib = 0.0f;
                        
                        // W1 * encoder_outputs
                        for (int h2 = 0; h2 < hidden_size; h2++) {
                            enc_contrib += encoder_outputs[b * seq_len * hidden_size + s * hidden_size + h2] * 
                                          w1[h2 * hidden_size + h];
                        }
                        
                        // W2 * decoder_hidden
                        for (int h2 = 0; h2 < hidden_size; h2++) {
                            dec_contrib += decoder_hidden[b * hidden_size + h2] * 
                                          w2[h2 * hidden_size + h];
                        }
                        
                        // tanh(enc_contrib + dec_contrib)
                        float activated = tanhf(enc_contrib + dec_contrib);
                        
                        // v^T * activated
                        score += v[h] * activated;
                    }
                    
                    attention_scores[b * seq_len + s] = score;
                }
            }
            break;
        }
        
        case ATTENTION_GENERAL: {
            // Luong-style general attention
            if (!w1) {
                free(attention_scores);
                return -3; // Missing weights for general attention
            }
            
            // attention_scores = decoder_hidden * W * encoder_outputs^T
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int b = 0; b < batch_size; b++) {
                // Transform encoder outputs: transformed = encoder_outputs * W
                float* transformed_enc = (float*)malloc(seq_len * hidden_size * sizeof(float));
                if (!transformed_enc) {
                    free(attention_scores);
                    return -1;
                }
                
                matrix_multiply(transformed_enc, 
                               encoder_outputs + b * seq_len * hidden_size,
                               w1, seq_len, hidden_size, hidden_size);
                
                // Compute scores: decoder_hidden * transformed_enc^T
                for (int s = 0; s < seq_len; s++) {
                    float score = 0.0f;
                    for (int h = 0; h < hidden_size; h++) {
                        score += decoder_hidden[b * hidden_size + h] * 
                                transformed_enc[s * hidden_size + h];
                    }
                    attention_scores[b * seq_len + s] = score;
                }
                
                free(transformed_enc);
            }
            break;
        }
        
        case ATTENTION_DOT:
        default: {
            // Dot product attention (Luong-style)
            #ifdef _OPENMP
            #pragma omp parallel for collapse(2)
            #endif
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < seq_len; s++) {
                    float score = 0.0f;
                    for (int h = 0; h < hidden_size; h++) {
                        score += decoder_hidden[b * hidden_size + h] * 
                                encoder_outputs[b * seq_len * hidden_size + s * hidden_size + h];
                    }
                    attention_scores[b * seq_len + s] = score;
                }
            }
            break;
        }
    }
    
    // Apply softmax to get attention weights
    softmax_forward(attention_scores, batch_size, seq_len);
    
    // Copy to output
    memcpy(attention_weights, attention_scores, batch_size * seq_len * sizeof(float));
    
    // Compute context vector: weighted sum of encoder outputs
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int b = 0; b < batch_size; b++) {
        // Initialize context vector to zero
        for (int h = 0; h < hidden_size; h++) {
            context_vector[b * hidden_size + h] = 0.0f;
        }
        
        // Weighted sum
        for (int s = 0; s < seq_len; s++) {
            float weight = attention_weights[b * seq_len + s];
            for (int h = 0; h < hidden_size; h++) {
                context_vector[b * hidden_size + h] += weight * 
                    encoder_outputs[b * seq_len * hidden_size + s * hidden_size + h];
            }
        }
    }
    
    free(attention_scores);
    return 0;
}