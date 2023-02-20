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

# vocab_size = 50257
class GPT1Config(GPTConfig):
  num_attention_heads = 12
  num_blocks = 12 # number of layers
  embed_dim = 768