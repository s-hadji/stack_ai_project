import torch
from .graph import Graph
from .runtime import Executor

def to_torch(graph: Graph, *, compile: bool = False):
    m = Executor(graph)
    if compile:
        try:
            m = torch.compile(m)  # optional
        except Exception:
            pass
    return m

# convenience
Graph.to_torch = lambda self, compile=False: to_torch(self, compile=compile)
Graph.enable_history = lambda self, *a, **k: (setattr(self, "_history_enabled", True) or self)
