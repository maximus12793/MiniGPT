import sys
sys.path.append('../')

import pytorch_lightning as pl
from tiny import GPTSimple
from train import GPTSimpleTrainer
from config import GPT1Config
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer
from datasets import load_dataset

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
train_dataset = train_dataset.shuffle(seed=42, buffer_size=1000)
train_dataset = train_dataset.with_format("torch")
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
valid_dataset = valid_dataset.shuffle(seed=42, buffer_size=1000)
valid_dataset = valid_dataset.with_format("torch")
valid_loader = DataLoader(valid_dataset, batch_size=32)# Load the preprocessed GPT-2 dataset from Hugging Face.

# Define the GPTSimple model and the optimizer
config = GPT1Config(vocab_size=50257, max_len=512)
model = GPTSimple(config)

# Create a GPTSimpleTrainer object and pass it to the trainer
trainer = pl.Trainer(
    gpus=0,
    accelerator='cpu',
    max_epochs=10,
)
trainer.fit(model, train_loader)
