from dataclasses import dataclass

@dataclass
class BUVNConfig:
    """Configuration class for the BUVN-1.1 foundation model (~120M parameters)."""
    vocab_size: int = 50000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False # False means no bias in LayerNorm and Linear layers (a la PaLM)
    
    # RoPE config
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BUVNConfig":
        """Creates a BUVNConfig object from a dictionary."""
        return cls(**config_dict)
