import pytorch_lightning as pl
from torch import optim
import torch
import torch.nn.functional as F



"""
The goal of this script is to do the following

1. Ensure training loop works correctly
2. Run training and eval on codeparrot data
3. Get lightning visuals working
"""


class GPTSimpleTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, input_ids):
        return self.model(input_ids)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

