import math
import torch
from torch import nn
import torch.nn.functional as F

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
    self.positional_embedding = nn.Parameter(torch.randn(config.max_len, config.embed_dim))
    self.dropout = nn.Dropout(config.embed_dropout)

  def forward(self, inputs):
    pass



class Decoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_blocks)])

  def forward(self, inputs, encoder_outputs):
    pass
