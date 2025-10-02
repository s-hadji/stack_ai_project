from dataclasses import dataclass
from typing import Optional, List, Dict
import torch.nn as nn

@dataclass
class ModelHandle:
    name: str
    module: nn.Module
    inputs: List[str]
    outputs: List[str]
    hf_id: Optional[str] = None
    version: Optional[str] = None
    weights_hash: Optional[str] = None

class Registry:
    def __init__(self):
        self._models: Dict[str, ModelHandle] = {}

    def add(self, name: str, module: nn.Module, *, inputs: list[str], outputs: list[str], **meta) -> None:
        if name in self._models:
            raise ValueError(f"Model '{name}' already exists.")
        self._models[name] = ModelHandle(name, module, inputs, outputs, **meta)

    def get(self, name: str) -> ModelHandle:
        if name not in self._models:
            raise KeyError(f"Unknown model '{name}'")
        return self._models[name]

    def exists(self, name: str) -> bool:
        return name in self._models
