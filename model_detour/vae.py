import torch
from torch import nn
import torch.nn.functional as F

class TextVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(TextVAE, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTMCell(latent_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        # Encode the input sequence
        x = self.embedding(x)
        x, (hidden, cell) = self.encoder(x)
        z_mean = self.mean(hidden.squeeze(0))
        z_logvar = self.logvar(hidden.squeeze(0))
        return z_mean, z_logvar
    
    def reparameterize(self, z_mean, z_logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        return z

    def decode(self, z, max_len, device):
        batch_size = z.size(0)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(batch_size, self.hidden_dim).to(device)

        outputs = []
        for i in range(max_len):
            hidden, cell = self.decoder(z, (hidden, cell))
            output = self.output(hidden)
            # Do I need unsqueeze here?
            outputs.append(output)

            # Use the last output as the next input
            _, predicted = torch.max(output, 1)
            input = predicted.unsqueeze(1)
            z = self.embedding(input)

        outputs = torch.stack(outputs, dim=1)
        return outputs