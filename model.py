import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import SOS_token

class TNNCell(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, hidden_size, embedding_size, batch_size):
        super(TNNCell, self).__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size

        self.i2h = nn.Linear(embedding_size * 2, hidden_size)
        self.h2h = nn.Linear(hidden_size, embedding_size * 2)

        self.sourceEmbeddingHelper = nn.Embedding(source_vocab_size, embedding_size)
        source_indices = torch.tensor(range(source_vocab_size))
        source_indices = source_indices.repeat(self.batch_size, 1).transpose(0, 1)
        self.source_dict = self.sourceEmbedding(source_indices)

        self.targetEmbeddingHelper = nn.Embedding(target_vocab_size, embedding_size)
        target_indices = torch.tensor(range(target_vocab_size))
        target_indices = target_indices.repeat(self.batch_size, 1).transpose(0, 1)
        self.target_dict = self.targetEmbedding(target_indices)


    def forward(self, input_x, input_y, noisy_rate=0):

        combined = torch.cat((input_x, input_y), dim=1)
        noise = 2 * torch.rand(combined.shape) - 1
        combined = (1 - noisy_rate) * combined + noisy_rate * noise

        hidden = F.relu(self.i2h(combined))
        combined = self.h2h(hidden)
        combined = F.tanh(combined)

        output_x = combined[:, 0:self.embedding_size]
        output_y = combined[:, self.embedding_size:]

        return output_x, output_y 


    def sourceEmbedding(self, tensor):
        return torch.sign(self.sourceEmbeddingHelper(tensor)).detach()


    def targetEmbedding(self, tensor):
        return torch.sign(self.targetEmbeddingHelper(tensor)).detach()
