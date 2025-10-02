from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Tuple
import torch.nn as nn

@dataclass(frozen=True)
class PortRef:
    node: str
    port: str

@dataclass
class Node:
    name: str
    module: nn.Module
    inputs: List[str]
    outputs: List[str]
    is_function: bool = False

@dataclass
class Edge:
    srcs: List[PortRef]
    dsts: List[PortRef]

class FunctionModule(nn.Module):
    def __init__(self, fn: Callable[..., Dict[str, Any]], input_names: List[str], output_names: List[str]):
        super().__init__()
        self.fn = fn
        self.input_names = input_names
        self.output_names = output_names
    def forward(self, **kwargs):
        out = self.fn(**kwargs)
        if not isinstance(out, dict):
            raise TypeError("Function must return a dict")
        return {k: out[k] for k in self.output_names}

class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._inputs: Dict[str, PortRef] = {}
        self._outputs: Dict[str, PortRef] = {}
        self._history_enabled = False
        self._watch: List[Tuple[str, Optional[str]]] = []  # (pattern, label)

    # ---- public API ----

    def add_model(self, *, name: str, module: nn.Module, inputs: list[str], outputs: list[str]):
        self._assert_new_node(name)
        self.nodes[name] = Node(name, module, inputs, outputs, is_function=False)
        return self

    def add_function(self, *, name: str, fn: Callable, inputs: list[str] | None = None, outputs: list[str] | None = None):
        self._assert_new_node(name)
        if inputs is None:  inputs = []
        if outputs is None: outputs = ["out"]
        self.nodes[name] = Node(name, FunctionModule(fn, inputs, outputs), inputs, outputs, is_function=True)
        return self

    def connect(self,
                sources: str | list[str],
                target: str | list[str] | None = None,
                *,
                via: str | Callable | None = None,
                as_: list[str] | None = None):
        srcs = self._normalize_ports(sources)
        dsts = self._normalize_ports(target) if target is not None else []

        if via is None:
            if not dsts:
                raise ValueError("Direct connection requires a target")
            for d in dsts:
                for s in srcs:
                    self.edges.append(Edge([s], [d]))
            return self

        # via is set
        if isinstance(via, str):
            via_name = via
            if via_name not in self.nodes or not self.nodes[via_name].is_function:
                raise ValueError(f"'via' refers to unknown function node '{via_name}'")
        else:
            via_name = f"fn_auto_{len([n for n in self.nodes if self.nodes[n].is_function])}"
            if as_ is None:
                as_ = [f"a{i}" for i in range(len(srcs))]
            self.add_function(name=via_name, fn=lambda **kw: via(**kw), inputs=as_, outputs=["out"])

        # connect sources -> via inputs
        if as_ is None:
            raise ValueError("You must provide 'as_' to map sources to function inputs")
        if len(as_) != len(srcs):
            raise ValueError("len(sources) must match len(as_)")
        for s, pname in zip(srcs, as_):
            self.edges.append(Edge([s], [PortRef(via_name, pname)]))

        # if targets exist: via first output -> each dst
        if dsts:
            first_out = self.nodes[via_name].outputs[0]
            for d in dsts:
                self.edges.append(Edge([PortRef(via_name, first_out)], [d]))
        return self

    def input(self, name: str):
        class _Binder:
            def __init__(self, outer, in_name): self.outer, self.in_name = outer, in_name
            def to(self, port: str):
                n, p = port.split(".")
                outer = self.outer
                if n not in outer.nodes or p not in outer.nodes[n].inputs:
                    raise ValueError(f"Invalid graph input mapping to {port}")
                outer._inputs[self.in_name] = PortRef(n, p)
                return outer
        return _Binder(self, name)

    def output(self, name: str):
        class _Binder:
            def __init__(self, outer, out_name): self.outer, self.out_name = outer, out_name
            def from_(self, port: str):
                n, p = port.split(".")
                outer = self.outer
                if n not in outer.nodes or p not in outer.nodes[n].outputs:
                    raise ValueError(f"Invalid graph output mapping from {port}")
                outer._outputs[self.out_name] = PortRef(n, p)
                return outer
        return _Binder(self, name)

    # history (minimal)
    def enable_history(self):
        self._history_enabled = True
        return self
    def watch(self, pattern: str, label: str | None = None):
        self._watch.append((pattern, label))
        return self

    # utils
    def _normalize_ports(self, ref: str | list[str]) -> List[PortRef]:
        if ref is None: return []
        refs = [ref] if isinstance(ref, str) else ref
        out: List[PortRef] = []
        for r in refs:
            if "." not in r: raise ValueError("Port must be 'node.port'")
            n, p = r.split(".")
            if n not in self.nodes: raise ValueError(f"Unknown node '{n}'")
            out.append(PortRef(n, p))
        return out

    def _assert_new_node(self, name: str):
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists.")
