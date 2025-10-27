
import torch
from aceflow import Seq2SeqModel, Tokenizer
from aceflow.trainers import Trainer
from aceflow.utils.data_loader import create_data_loader
# Save tokenizers
src_tokenizer.save("src_tokenizer.pkl")
tgt_tokenizer.save("tgt_tokenizer.pkl")

# Load model for inference
loaded_model = Seq2SeqModel.load("translation_model.ace")

# Example inference
test_sentence = "hello world"
test_encoded = src_tokenizer.encode(test_sentence)
print(f"Encoded input: {test_encoded}")
test_tensor = torch.tensor([test_encoded], dtype=torch.long)

with torch.no_grad():
    output_sequence = loaded_model.beam_search(test_tensor, beam_width=3)
    translated = tgt_tokenizer.decode(output_sequence)
    print(f"Input: {test_sentence}")
    print(f"Translation: {translated}")