import torch
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class TextProcessor:
    def __init__(self, tokenizer=None, max_length=512):
        self.tokenizer = tokenizer or get_tokenizer("basic_english")  # Use basic English tokenizer by default
        self.max_length = max_length
    
    def process_text(self, text):
        """
        tokenizes and processes the given text.
        returns a tensor of tokenized and padded text.
        """
        if isinstance(text, str):  # Check if the input is a string
            print(f"processing text: {text}")  # Debugging statement
            tokens = self.tokenizer(text)
            tokens = [ord(c) for c in text]  # Convert characters to token representations (if needed)
            
            # Ensure tokens do not exceed max_length
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]  # Truncate to max length
            tokens_tensor = torch.tensor(tokens)
            return tokens_tensor
        else:
            raise TypeError("input should be a string.")  # Raise error if input is not a string

    def process_batch(self, batch_texts):
        """
        processes a batch of text data.
        pads the sequences to the maximum length in the batch.
        """
        if not all(isinstance(text, str) for text in batch_texts):  # Ensure all inputs are strings
            raise TypeError("All inputs in batch should be strings.")
        
        tokenized_batch = [self.process_text(text) for text in batch_texts]
        padded_batch = pad_sequence(tokenized_batch, batch_first=True, padding_value=0)
        return padded_batch

# Example usage
if __name__ == "__main__":
    processor = TextProcessor()
    
    # process a single text
    text = "this is a test sentence."
    processed_text = processor.process_text(text)
    print(f"processed text shape: {processed_text.shape}")

    # Process a batch of texts
    batch_texts = ["this is sentence one.", "this is sentence two."]
    padded_batch = processor.process_batch(batch_texts)
    print(f"padded batch shape: {padded_batch.shape}")
