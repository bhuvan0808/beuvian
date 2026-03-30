from dataclasses import dataclass


@dataclass
class BUVNConfig:
    """Configuration class for the BUVN foundation model."""
    vocab_size: int = 50000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False
    rope_theta: float = 10000.0
    gradient_checkpointing: bool = False

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.d_model > 0 and self.n_layers > 0 and self.vocab_size > 0
        assert 0.0 <= self.dropout < 1.0, f"dropout must be in [0, 1), got {self.dropout}"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BUVNConfig":
        """Creates a BUVNConfig from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)
