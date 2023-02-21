import sys

sys.path.append("../")

import pytorch_lightning as pl
from core.tiny import GPTSimple
from core.config import GPT1Config
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
import ipdb

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
SPECIAL_TOKENS = {
    "bos_token": "<|endoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "[PAD]",
    "additional_special_tokens": [
        "[SYS]",
        "[USR]",
        "[KG]",
        "[SUB]",
        "[PRED]",
        "[OBJ]",
        "[TRIPLE]",
        "[SEP]",
        "[Q]",
        "[DOM]",
    ],
}
tokenizer.add_special_tokens(SPECIAL_TOKENS)

# Load the CodeParrot dataset
# dataset = load_dataset("codeparrot/github-code", streaming=True, languages=["Python"])

# Set up the DataLoader for the training data
train_dataset = load_dataset(
    "huggingface-course/codeparrot-ds-train", streaming=True, split="train"
)
train_dataset = train_dataset.map(
    lambda x: tokenizer(x["content"], truncation=True, padding="max_length"),
    batched=True,
    remove_columns=["content", "repo_name", "path", "copies", "size", "license"],
)
train_dataset = train_dataset.with_format(
    "numpy", columns=["input_ids", "attention_mask"]
)
train_loader = DataLoader(train_dataset, batch_size=32)

# Set up the DataLoader for the validation data
valid_dataset = load_dataset(
    "huggingface-course/codeparrot-ds-valid", streaming=True, split="validation"
)
valid_dataset = valid_dataset.map(
    lambda x: tokenizer(x["content"], truncation=True, padding="max_length"),
    batched=True,
    remove_columns=["content", "repo_name", "path", "copies", "size", "license"],
)
valid_dataset = valid_dataset.with_format(
    "numpy", columns=["input_ids", "attention_mask"]
)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# Define the GPTSimple model and the optimizer
config = GPT1Config(vocab_size=50257, max_len=512)
model = GPTSimple(config)

# Set up the PyTorch Lightning trainer
trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss")],
)

# Train the model
trainer.fit(model, train_loader, valid_loader)
