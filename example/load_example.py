from aceflow import Seq2Seq, Tokenizer, ACEFormat
import numpy as np

def load_and_test_model():
    # Load the saved model
    print("Loading model from 'seq2seq_model.ace'...")
    model = ACEFormat.load_model("seq2seq_model.ace")
    print("Model loaded successfully!")
    
    # Create tokenizers (you need to use the same vocab as during training)
    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()
    
    # Rebuild vocab with the same data (or save/load tokenizers separately)
    src_texts = ["hello world", "how are you", "good morning", "nice day"]
    tgt_texts = ["hola mundo", "como estas", "buenos dias", "buen dia"]
    
    src_tokenizer.build_vocab(src_texts)
    tgt_tokenizer.build_vocab(tgt_texts)
    
    print(f"Source vocab size: {src_tokenizer.vocab_size}")
    print(f"Target vocab size: {tgt_tokenizer.vocab_size}")
    
    # Test predictions
    test_cases = ["hello", "how", "good", "nice"]
    
    print("\nTesting predictions:")
    for test_text in test_cases:
        test_encoded = src_tokenizer.encode(test_text, max_length=10).reshape(1, -1)
        prediction = model.predict(test_encoded)
        decoded = tgt_tokenizer.decode(prediction[0])
        print(f"Input: '{test_text}' -> Output: '{decoded}'")
    
    # Test with known training examples
    print("\nTesting with training examples:")
    for i, (src, tgt) in enumerate(zip(src_texts, tgt_texts)):
        test_encoded = src_tokenizer.encode(src, max_length=10).reshape(1, -1)
        prediction = model.predict(test_encoded)
        decoded = tgt_tokenizer.decode(prediction[0])
        print(f"Example {i+1}: '{src}' -> '{decoded}' (expected: '{tgt}')")

if __name__ == "__main__":
    load_and_test_model()