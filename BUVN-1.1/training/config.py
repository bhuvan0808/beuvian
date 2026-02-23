import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    max_iters: int
    lr: float
    min_lr: float
    warmup_iters: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    eval_interval: int
    eval_iters: int
    log_interval: int
    checkpoint_dir: str
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(**d)

@dataclass
class DataConfig:
    data_dir: str
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(**d)

@dataclass
class AppConfig:
    model: Dict[str, Any]
    training: TrainingConfig
    data: DataConfig

    @classmethod
    def load(cls, yaml_path: str) -> "AppConfig":
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        return cls(
            model=cfg.get('model', {}),
            training=TrainingConfig.from_dict(cfg.get('training', {})),
            data=DataConfig.from_dict(cfg.get('data', {}))
        )
