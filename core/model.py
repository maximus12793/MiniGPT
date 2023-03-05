import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
import ipdb

"""
Example GPT1 configuration.

class GPT1Config(GPTConfig):
  num_attention_heads = 12
  num_blocks = 12 # number of layers
  embed_dim = 768 # number of embedding dimensions
  vocab_size = 50_257 # vocabulary size

1. Embedding layer
2. Stack of transformer encoder
3. Final layer 
"""

class GPTConfig:
  attn_dropout = 0.1
  embed_dropout = 0.1
  ff_dropout = 0.1

  def __init__(
        self, vocab_size, max_len, **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        for key, value in kwargs.items():
            setattr(self, key, value)

class GPT2Config(GPTConfig):
  num_attention_heads = 12 # number of attention heads in transformer
  num_blocks = 12 # number of layers
  embed_dim = 768 # number of embedding dimensions


class GPTSimple(pl.LightningModule):
    def __init__(self, config):
        super(GPTSimple, self).__init__()
        # Save config for other callers.
        self.config = config
        # Define learning rate.
        self.learning_rate = 1e-3
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
            config.vocab_size,  # Number of output features (i.e., the size of the vocabulary).
        )
        # Applies the Gaussian Error Linear Units function.
        self.gelu = nn.GELU()
        # Initialize the weights of the model using Xavier initialization.
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.ff.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, input_ids, attention_mask):
        try:
            # Get the embedding of the input sequence.
            embedded = self.embedding(input_ids)
        except:
            ipdb.set_trace()
        # Apply the attention mask to the embedding.
        embedded = embedded * attention_mask.unsqueeze(-1)
        # Transpose the output of the embedding layer so that it has shape (seq_len, batch_size, embed_dim).
        embedded = embedded.transpose(0, 1)
        # Apply the Transformer encoder to the embedded input sequence.
        encoded = self.encoder(embedded)
        # Get the last output of the Transformer encoder, which represents the output of the entire sequence.
        last_encoded = encoded[-1]
        # Apply the feedforward layer to the last encoded sequence, followed by the GELU activation.
        ff_output = self.ff(last_encoded)
        gelu_output = self.gelu(ff_output)
        # Project the output of the feedforward layer back into the original vocabulary space.
        output = self.output(gelu_output)
        # Apply the Softmax function to the output to get a probability distribution over the vocabulary.
        probs = F.softmax(output, dim=-1)
        # Note: Returns a distribution over all vocab_size.
        return probs

    def training_step(self, batch, batch_idx):
        # Get the inputs from the batch.
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Shifted by 1
        shifted_inputs = torch.roll(input_ids, shifts=1)
        shifted_inputs[:, 0] = 0 # Set the first token to 0

        # Compute the logits.
        input_logits = self(input_ids, attention_mask)
        target_logits = self(shifted_inputs, attention_mask)

        # Flatten the logits and targets tensors to be 2D.
        input_logits = input_logits.view(-1, self.config.vocab_size)
        target_logits = target_logits.view(-1, self.config.vocab_size)

        # Calculate the loss and ignore the padding token.
        loss = F.cross_entropy(input_logits, target_logits)
        # Log the training loss.
        self.log("train_loss", loss, on_epoch=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
