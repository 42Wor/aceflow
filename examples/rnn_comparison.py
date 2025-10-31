"""
Compare different RNN types for sequence-to-sequence tasks
"""

import torch
import time
import json
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Trainer

def benchmark_rnn_types():
    """Benchmark different RNN types"""
    
    print("🧪 RNN Type Comparison Benchmark")
    print("=" * 50)
    
    # Sample data
    english_sentences = [
        "hello world", "how are you", "good morning", "what is your name",
        "i love programming", "the weather is nice", "see you later",
        "thank you", "have a nice day", "where is the station"
    ] * 5  # Repeat to have more data

    french_sentences = [
        "bonjour le monde", "comment allez vous", "bonjour", "quel est votre nom",
        "j aime la programmation", "le temps est agreable", "a plus tard",
        "merci", "passez une bonne journee", "ou est la gare"
    ] * 5

    # Initialize tokenizers
    src_tokenizer = Tokenizer(name="english_benchmark")
    tgt_tokenizer = Tokenizer(name="french_benchmark")
    src_tokenizer.fit(english_sentences)
    tgt_tokenizer.fit(french_sentences)

    # RNN types to benchmark
    rnn_configs = [
        {'type': 'rnn', 'name': 'Simple RNN', 'bidirectional': False},
        {'type': 'lstm', 'name': 'LSTM', 'bidirectional': False},
        {'type': 'gru', 'name': 'GRU', 'bidirectional': False},
        {'type': 'bilstm', 'name': 'Bidirectional LSTM', 'bidirectional': True},
        {'type': 'bigru', 'name': 'Bidirectional GRU', 'bidirectional': True},
    ]

    # Create data loader
    train_loader = create_data_loader(
        english_sentences, french_sentences,
        src_tokenizer, tgt_tokenizer,
        batch_size=4, max_length=15
    )

    results = {}

    for config in rnn_configs:
        print(f"\n🔍 Testing {config['name']}...")
        
        # Create model
        model = Seq2SeqModel(
            src_vocab_size=len(src_tokenizer),
            tgt_vocab_size=len(tgt_tokenizer),
            hidden_size=128,
            num_layers=2,
            rnn_type=config['type'],
            use_attention=True,
            bidirectional=config['bidirectional']
        )

        # Get model info
        model_info = model.get_rnn_info()
        
        # Measure training speed
        trainer = Trainer(model, learning_rate=0.001)
        
        # Time one epoch of training
        start_time = time.time()
        train_loss, train_acc = trainer.train_epoch(train_loader)
        training_time = time.time() - start_time
        
        # Measure inference speed
        test_sentence = "hello world"
        encoded = src_tokenizer.encode(test_sentence, return_tensors='list')
        input_tensor = torch.tensor([encoded], dtype=torch.long)
        
        start_time = time.time()
        with torch.no_grad():
            output_sequence = model.beam_search(input_tensor, beam_width=3)
        inference_time = time.time() - start_time
        
        # Decode result
        translation = tgt_tokenizer.decode(output_sequence)
        
        # Store results
        results[config['type']] = {
            'name': config['name'],
            'parameters': model_info['total_parameters'],
            'training_time_per_epoch': training_time,
            'inference_time': inference_time,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'translation_example': translation
        }
        
        print(f"  ✅ Parameters: {model_info['total_parameters']:,}")
        print(f"  ⏱️  Training time/epoch: {training_time:.2f}s")
        print(f"  ⚡ Inference time: {inference_time:.4f}s")
        print(f"  📊 Train Loss: {train_loss:.4f}")
        print(f"  🎯 Example: 'hello world' -> '{translation}'")

    # Display comparison
    print("\n📊 RNN Type Comparison Summary:")
    print("=" * 60)
    print(f"{'RNN Type':<20} {'Params':<12} {'Train Time':<12} {'Inference Time':<15} {'Loss':<8}")
    print("-" * 60)
    
    for rnn_type, result in results.items():
        print(f"{result['name']:<20} {result['parameters']:<12,} {result['training_time_per_epoch']:<12.2f} "
              f"{result['inference_time']:<15.4f} {result['train_loss']:<8.4f}")

    # Save results
    with open("logs/rnn_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmark results saved to: logs/rnn_benchmark.json")
    
    return results

def get_rnn_recommendations():
    """Provide RNN type recommendations based on use case"""
    
    recommendations = {
        "translation": {
            "best": "bilstm",
            "alternatives": ["bigru", "lstm"],
            "reason": "Bidirectional models capture context from both directions"
        },
        "chatbot": {
            "best": "lstm", 
            "alternatives": ["gru", "bilstm"],
            "reason": "LSTM balances performance and memory for conversational context"
        },
        "summarization": {
            "best": "bilstm",
            "alternatives": ["lstm", "bigru"],
            "reason": "Bidirectional context helps understand document structure"
        },
        "time_series": {
            "best": "lstm",
            "alternatives": ["gru", "rnn"],
            "reason": "LSTM handles long-term dependencies in sequential data"
        },
        "real_time": {
            "best": "gru",
            "alternatives": ["rnn", "lstm"],
            "reason": "GRU offers good performance with faster inference"
        }
    }
    
    print("\n💡 RNN Type Recommendations:")
    print("=" * 50)
    for use_case, rec in recommendations.items():
        print(f"\n🎯 {use_case.upper()}:")
        print(f"   Best: {rec['best']}")
        print(f"   Alternatives: {', '.join(rec['alternatives'])}")
        print(f"   Reason: {rec['reason']}")

if __name__ == "__main__":
    results = benchmark_rnn_types()
    get_rnn_recommendations()