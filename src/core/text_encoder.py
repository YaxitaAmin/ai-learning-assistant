import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from collections import Counter
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=512, vocab_size=10000, max_length=100):
        super(TextEncoder, self).__init__()
        
        # Tokenizer for splitting text into words
        self.tokenizer = get_tokenizer('basic_english')
        
        # Vocabulary size and embedding dimension
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Embedding layer for converting words to vectors
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # RNN layer (e.g., GRU or LSTM) to process the word embeddings sequentially
        self.rnn = nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        
        # Fully connected layer to produce the final embedding for the entire text
        self.fc = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Padding index for padding sequences to a fixed length
        self.padding_idx = 0
    
    def build_vocab(self, texts):
        """
        Build a vocabulary from the given texts.
        """
        counter = Counter()
        for text in texts:
            tokens = self.tokenizer(text)
            counter.update(tokens)
        vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common(self.vocab_size - 1))}
        vocab['<PAD>'] = self.padding_idx  # Add padding token to vocab
        return vocab
    
    def text_to_tensor(self, text, vocab):
        """
        Convert text into a tensor of word indices.
        """
        tokens = self.tokenizer(text)
        token_indices = [vocab.get(token, vocab['<PAD>']) for token in tokens]
        # Pad sequences to a fixed length
        token_indices = token_indices[:self.max_length] + [self.padding_idx] * (self.max_length - len(token_indices))
        return torch.tensor(token_indices)
    
    def forward(self, x):
        """
        Forward pass of the TextEncoder.
        x: input tensor of word indices (batch_size, sequence_length)
        """
        # Get word embeddings for the input sequence
        embedded = self.embeddings(x)
        
        # Pass through RNN
        _, hidden = self.rnn(embedded)
        
        # Final text embedding from RNN output (hidden state)
        text_embedding = self.fc(hidden[-1])
        return text_embedding
    
    def encode_text(self, text, vocab):
        """
        Encode a text string using the TextEncoder.
        """
        tensor_input = self.text_to_tensor(text, vocab)
        tensor_input = tensor_input.unsqueeze(0)  # Add batch dimension
        return self(tensor_input)

# Test function to demonstrate usage
def test_text_encoder():
    # Example text data
    texts = ["This is a sample sentence.", "Text encoding is fun!", "PyTorch makes it easy."]
    
    # Create encoder
    encoder = TextEncoder(embedding_dim=512, vocab_size=10000)
    
    # Build vocabulary from text data
    vocab = encoder.build_vocab(texts)
    
    # Encode sample text
    sample_text = "This is a sample sentence."
    embedding = encoder.encode_text(sample_text, vocab)
    
    print("Text Embedding Shape:", embedding.shape)
    return embedding

# Uncomment to test
if __name__ == "__main__":
    test_text_encoder()
