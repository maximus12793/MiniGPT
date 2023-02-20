from torch import nn
import torch.nn.functional as F

"""
Example GPT1 configuration.

class GPT1Config(GPTConfig):
  num_attention_heads = 12
  num_blocks = 12 # number of layers
  embed_dim = 768
  vocab_size = 50_257

1. Embedding layer
2. Stack of transformer encoder
3. Final layer 
"""

class GPTSimple(nn.Module):
  def __init__(self, config):
      super(GPTSimple, self).__init__()
      # Save config for other callers.
      self.config = config
      # Define embedding layer.
      self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
      # Define a single layer of the Transformer encoder.
      self.encoder_layer = nn.TransformerEncoderLayer(
          d_model=config.embed_dim,  # Number of expected features in the input (i.e., the embedding dimension).
          nhead=config.num_attention_heads,  # Number of attention heads.
      )
      # Define the stack of Transformer encoder layers.
      self.encoder = nn.TransformerEncoder(
          self.encoder_layer,  # Transformer encoder layer.
          num_layers=config.num_blocks,  # Number of encoder layers to stack.
      )
      # Linear layer to project the output of the last Transformer encoder
      # layer back into the original vocabulary space.
      self.ff = nn.Linear(
          config.embed_dim,  # Number of expected features in the input (i.e., the size of the last hidden layer of the Transformer encoder).
          config.embed_dim,  # Number of output features (i.e., the size of the last hidden layer of the Transformer encoder).
      )
      # Final output layer, when softmax is applied we will get a distribution
      # of vocab_size back.
      self.output = nn.Linear(
          config.embed_dim,  # Number of expected features in the input (i.e., the size of the last hidden layer of the Transformer encoder).
          config.vocab_size, # Number of output features (i.e., the size of the vocabulary).
      )
      # Initialize the weights of the model using Xavier initialization.
      nn.init.xavier_uniform_(self.embedding.weight)
      nn.init.xavier_uniform_(self.ff.weight)
      nn.init.xavier_uniform_(self.output.weight)

  def forward(self, input_ids):
      # Get the embedding of the input sequence.
      embedded = self.embedding(input_ids)
      # Transpose the output of the embedding layer so that it has shape (seq_len, batch_size, embed_dim).
      embedded = embedded.transpose(0, 1)
      # Apply the Transformer encoder to the embedded input sequence.
      encoded = self.encoder(embedded)
      # Get the last output of the Transformer encoder, which represents the output of the entire sequence.
      last_encoded = encoded[-1]
      # Apply the feedforward layer to the last encoded sequence, followed by the GELU activation.
      ff_output = self.ff(last_encoded)
      gelu_output = nn.GELU(ff_output)
      # Project the output of the feedforward layer back into the original vocabulary space.
      output = self.output(gelu_output)
      # Apply the Softmax function to the output to get a probability distribution over the vocabulary.
      probs = F.softmax(output, dim=-1)
      return probs
