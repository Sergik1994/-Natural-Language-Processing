import torch
from torch import nn
from typing import Tuple
import torch.nn.functional as F
import numpy as np
import json

with open('utils/vocab2int.json', 'r') as f:
    vocab_to_int = json.load(f)

EMBEDDING_DIM = 64 # embedding_dim 
HIDDEN_SIZE = 32
VOCAB_SIZE = len(vocab_to_int)

class ConcatAttention(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 32
            ) -> None:
        
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.align  = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.tanh   = nn.Tanh()

    def forward(
            self, 
            lstm_outputs: torch.Tensor, # BATCH_SIZE x SEQ_LEN x HIDDEN_SIZE
            final_hidden: torch.Tensor  # BATCH_SIZE x HIDDEN_SIZE
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        att_weights = self.linear(lstm_outputs)
        # print(f'After linear: {att_weights.shape, final_hidden.unsqueeze(2).shape}')
        att_weights = torch.bmm(att_weights, final_hidden.unsqueeze(2))
        # print(f'After bmm: {att_weights.shape}')
        att_weights = F.softmax(att_weights.squeeze(2), dim=1)
        # print(f'After softmax: {att_weights.shape}')
        cntxt       = torch.bmm(lstm_outputs.transpose(1, 2), att_weights.unsqueeze(2))
        # print(f'Context: {cntxt.shape}')
        concatted   = torch.cat((cntxt, final_hidden.unsqueeze(2)), dim=1)
        # print(f'Concatted: {concatted.shape}')
        att_hidden  = self.tanh(self.align(concatted.squeeze(-1)))
        # print(f'Att Hidden: {att_hidden.shape}')
        return att_hidden, att_weights



embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

class LSTMConcatAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.embedding = embedding_layer
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True)
        self.attn = ConcatAttention(HIDDEN_SIZE)
        self.clf = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
    
    def forward(self, x):
        embeddings = self.embedding(x)
        outputs, (h_n, _) = self.lstm(embeddings)
        att_hidden, att_weights = self.attn(outputs, h_n.squeeze(0))
        out = self.clf(att_hidden)
        return out, att_weights


