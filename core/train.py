import pytorch_lightning as pl
from torch import optim
import torch.functional as F

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

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        outputs = self(input_ids)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

