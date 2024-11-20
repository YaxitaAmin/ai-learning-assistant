# 5. src/models/learning_path.py
import torch
import torch.nn as nn
from typing import List, Dict

class PersonalizedLearningPath(nn.Module):
    def __init__(self, num_concepts: int, embedding_dim: int = 128):
        super().__init__()
        self.concept_embeddings = nn.Embedding(num_concepts, embedding_dim)
        self.path_rnn = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_concepts)
        )
        
    def forward(self, concept_sequence: torch.Tensor, user_state: torch.Tensor):
        # encode concept sequence
        concept_embeds = self.concept_embeddings(concept_sequence)
        
        # update sequence with user state
        seq_length = concept_embeds.size(1)
        user_state_expanded = user_state.unsqueeze(1).expand(-1, seq_length, -1)
        enhanced_sequence = torch.cat([concept_embeds, user_state_expanded], dim=-1)
        
        # generate path predictions
        lstm_out, _ = self.path_rnn(enhanced_sequence)
        next_concepts = self.predictor(lstm_out[:, -1, :])
        
        return next_concepts