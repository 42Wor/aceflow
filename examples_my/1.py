

from aceflow import version

print(f"AceFlow version: {version}")
def get_recommended_rnn_type(task_type, data_size, sequence_length):
    """
    Recommend RNN type based on task characteristics
    
    Args:
        task_type: 'translation', 'summarization', 'chatbot', 'classification'
        data_size: 'small' (<10K samples), 'medium' (10K-100K), 'large' (>100K)
        sequence_length: 'short' (<50), 'medium' (50-200), 'long' (>200)
    """
    
    recommendations = {
        'translation': {
            'small': {'short': 'gru', 'medium': 'lstm', 'long': 'bilstm'},
            'medium': {'short': 'lstm', 'medium': 'bilstm', 'long': 'bilstm'},
            'large': {'short': 'bilstm', 'medium': 'bilstm', 'long': 'bilstm'}
        },
        'summarization': {
            'small': {'short': 'gru', 'medium': 'lstm', 'long': 'bilstm'},
            'medium': {'short': 'lstm', 'medium': 'bilstm', 'long': 'bilstm'},
            'large': {'short': 'bilstm', 'medium': 'bilstm', 'long': 'bilstm'}
        },
        'chatbot': {
            'small': {'short': 'gru', 'medium': 'lstm', 'long': 'lstm'},
            'medium': {'short': 'lstm', 'medium': 'lstm', 'long': 'bilstm'},
            'large': {'short': 'lstm', 'medium': 'bilstm', 'long': 'bilstm'}
        }
    }
    
    return recommendations.get(task_type, {}).get(data_size, {}).get(sequence_length, 'lstm')

# Usage example:
recommended = get_recommended_rnn_type('translation', 'medium', 'medium')
print(f"Recommended RNN type: {recommended}")  # Output: 'bilstm'