from __future__ import annotations
from typing import Dict, Any, Tuple, List, Set, Optional
import fnmatch
import torch
import torch.nn as nn

from .graph import Graph, PortRef, Edge

# -----------------------------------------------------------------------------
# History helper
# -----------------------------------------------------------------------------
class History:
    def __init__(self):
        self.events = []
    def log(self, **event):
        self.events.append(event)
    def export_jsonl(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

# -----------------------------------------------------------------------------
# Utilities for history summarization
# -----------------------------------------------------------------------------
def _match_port(patterns: List[str], node: str, port: str) -> bool:
    full = f"{node}.{port}"
    for pat in patterns:
        if fnmatch.fnmatch(full, pat):
            return True
    return False

def _summarize_value(val: Any, *, tensor_policy: str, text_limit: int) -> Any:
    if isinstance(val, torch.Tensor):
        meta = {
            "type": "tensor",
            "shape": list(val.shape),
            "dtype": str(val.dtype),
            "device": str(val.device),
        }
        if tensor_policy == "off":
            return {"type": "tensor"}
        if tensor_policy == "shape":
            return meta
        if tensor_policy == "sample":
            try:
                flat = val.detach().flatten()
                meta["sample"] = flat[: min(5, flat.numel())].tolist()
            except Exception:
                pass
            return meta
        if tensor_policy == "full":
            try:
                meta["data"] = val.detach().cpu().tolist()
            except Exception:
                meta["data"] = "<unavailable>"
            return meta
        return meta

    if isinstance(val, str):
        if len(val) > text_limit:
            return val[:text_limit] + f"... <{len(val)-text_limit} chars more>"
        return val

    if isinstance(val, dict):
        return {"type": "dict", "keys": list(val.keys()), "size": len(val)}

    if isinstance(val, (list, tuple)):
        return {"type": "list", "size": len(val)}

    return val

# -----------------------------------------------------------------------------
# Executor with dynamic/partial execution
# -----------------------------------------------------------------------------
class Executor(nn.Module):
    """
    nn.Module compilé depuis un Graph.
    - Exécution *partielle* : on n'exécute que les nœuds dont toutes les entrées
      sont disponibles à partir des inputs fournis à forward().
    - Sélection des sorties : via paramètre spécial `_only=["digit", "text"]`
      pour ne construire que certaines sorties du graphe.
    - `_strict=True` : lève si une sortie demandée n'est pas calculable
      avec les inputs fournis.
    """
    def __init__(self, graph: Graph):
        super().__init__()
        self.g = graph
        self.nodes = nn.ModuleDict({n: spec.module for n, spec in graph.nodes.items()})
        self.history = History()

        self._cfg = getattr(self.g, "_history_cfg", {
            "keep": ["*"],
            "tensor_policy": "shape",
            "text_limit": 200,
            "with_inputs": True,
            "with_outputs": True,
            "with_graph_io": True,
        })

        # Build inbound edges per (node,port)
        self._in_edges: Dict[Tuple[str,str], List[Edge]] = {}
        deps_by_node: Dict[str, Set[str]] = {n:set() for n in self.g.nodes}
        for e in self.g.edges:
            for d in e.dsts:
                for s in e.srcs:
                    deps_by_node[d.node].add(s.node)
                self._in_edges.setdefault((d.node, d.port), []).append(e)
        self._deps_by_node = deps_by_node  # may be useful

        # Optional: a global topological order of nodes (used as a tiebreaker)
        self.order = self._topo_order()

    # ---------------- Topological order (for deterministic scheduling) --------
    def _topo_order(self) -> List[str]:
        from collections import deque
        incoming = {n:set() for n in self.g.nodes}
        outgoing = {n:set() for n in self.g.nodes}
        for e in self.g.edges:
            for s in e.srcs:
                for d in e.dsts:
                    incoming[d.node].add(s.node)
                    outgoing[s.node].add(d.node)
        q = deque([n for n in self.g.nodes if len(incoming[n]) == 0])
        order, seen = [], set()
        while q:
            u = q.popleft()
            order.append(u); seen.add(u)
            for v in list(outgoing[u]):
                if v in seen:
                    continue
                incoming[v].discard(u)
                if len(incoming[v]) == 0:
                    q.append(v)
        if len(order) != len(self.g.nodes):
            # We keep a best-effort order; cycles would be caught during forward
            pass
        return order or list(self.g.nodes.keys())

    # ---------------- History helpers ----------------------------------------
    def _maybe_log_graph_input(self, name: str, value: Any):
        if not self.g._history_enabled or not self._cfg.get("with_graph_io", True):
            return
        sv = _summarize_value(value,
                              tensor_policy=self._cfg["tensor_policy"],
                              text_limit=self._cfg["text_limit"])
        self.history.log(ev="graph_input", name=name, value=sv)

    def _maybe_log_graph_output(self, name: str, value: Any):
        if not self.g._history_enabled or not self._cfg.get("with_graph_io", True):
            return
        sv = _summarize_value(value,
                              tensor_policy=self._cfg["tensor_policy"],
                              text_limit=self._cfg["text_limit"])
        self.history.log(ev="graph_output", name=name, value=sv)

    def _maybe_log_node_input(self, node: str, port: str, value: Any):
        if not self.g._history_enabled or not self._cfg.get("with_inputs", True):
            return
        if not _match_port(self._cfg["keep"], node, port):
            return
        sv = _summarize_value(value,
                              tensor_policy=self._cfg["tensor_policy"],
                              text_limit=self._cfg["text_limit"])
        self.history.log(ev="node_input", node=node, port=port, value=sv)

    def _maybe_log_node_output(self, node: str, port: str, value: Any):
        if not self.g._history_enabled or not self._cfg.get("with_outputs", True):
            return
        if not _match_port(self._cfg["keep"], node, port):
            return
        sv = _summarize_value(value,
                              tensor_policy=self._cfg["tensor_policy"],
                              text_limit=self._cfg["text_limit"])
        self.history.log(ev="node_output", node=node, port=port, value=sv)

    # ---------------- Public API ---------------------------------------------
    def available_outputs(self, **graph_inputs) -> List[str]:
        """
        Retourne la liste des sorties du graphe qui seraient calculables
        avec les inputs fournis.
        """
        cache: Set[Tuple[str,str]] = set()
        # seed with provided graph inputs
        provided_ports = set()
        for gname, pref in self.g._inputs.items():
            if gname in graph_inputs:
                provided_ports.add((pref.node, pref.port))

        # Dynamic "can-run" propagation
        done_nodes: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for n in self.order:
                if n in done_nodes:
                    continue
                spec = self.g.nodes[n]
                # A node can run if all its inputs are available
                ready = True
                for p in spec.inputs:
                    if (n,p) in provided_ports or (n,p) in cache:
                        continue
                    edges = self._in_edges.get((n,p), [])
                    if not edges:
                        ready = False
                        break
                    # at least one value must be available for this input
                    got = False
                    for e in edges:
                        for s in e.srcs:
                            if (s.node, s.port) in cache:
                                got = True; break
                        if got: break
                    if not got:
                        ready = False
                        break
                if ready:
                    # mark outputs as available
                    for op in spec.outputs:
                        cache.add((n, op))
                    done_nodes.add(n)
                    changed = True

        # Which graph outputs are now available?
        avail = []
        for out_name, pref in self.g._outputs.items():
            if (pref.node, pref.port) in cache or (pref.node, pref.port) in provided_ports:
                avail.append(out_name)
        return sorted(avail)

    def forward(
        self,
        _only: Optional[List[str]] = None,
        _strict: bool = False,
        **graph_inputs
    ) -> Dict[str, Any]:
        """
        Exécution dynamique :
        - si `_only` est spécifié, ne vise que ces sorties (n'exécute que ce qui est nécessaire).
        - sinon, produit toutes les sorties calculables avec les inputs fournis.
        - si `_strict=True`, lève une erreur si une des sorties demandées n'est pas calculable.

        Exemple :
            m(image=img)                          -> ne calcule que les sorties atteignables depuis 'image'
            m(image=img, _only=["digit"])         -> ne calcule que 'digit' (et dépendances)
            m(topic="x", _only=["text","digit"])  -> calcule 'text' + 'digit' si possible
        """
        cache: Dict[Tuple[str,str], Any] = {}

        # Seed cache with provided graph inputs
        provided_ports = set()
        for gname, pref in self.g._inputs.items():
            if gname in graph_inputs:
                val = graph_inputs[gname]
                cache[(pref.node, pref.port)] = val
                provided_ports.add((pref.node, pref.port))
                self._maybe_log_graph_input(gname, val)

        # Determine which graph outputs we're targeting
        all_outputs = list(self.g._outputs.keys())
        target_outputs = list(_only) if _only else all_outputs

        # If _only is set, we compute a backward-needed set of nodes/ports
        needed_nodes: Optional[Set[str]] = None
        if _only:
            needed_nodes = self._backward_needed_nodes(target_outputs)

        # Dynamic scheduling: run any node whose inputs are all available.
        # Stop when no more nodes can run.
        ran_nodes: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for n in self.order:
                if n in ran_nodes:
                    continue
                if needed_nodes is not None and n not in needed_nodes:
                    # Not needed for requested outputs -> skip
                    continue
                spec = self.g.nodes[n]

                # Are all inputs available?
                kwargs = {}
                ready = True
                for p in spec.inputs:
                    if (n,p) in cache:
                        kwargs[p] = cache[(n,p)]
                        continue
                    edges = self._in_edges.get((n,p), [])
                    if not edges:
                        ready = False
                        break
                    # If single source, pass the single value
                    if len(edges) == 1 and len(edges[0].srcs) == 1:
                        s = edges[0].srcs[0]
                        if (s.node, s.port) not in cache:
                            ready = False
                            break
                        kwargs[p] = cache[(s.node, s.port)]
                    else:
                        # Bundle dict from all available sources for this port
                        bundle = {}
                        any_avail = False
                        for e in edges:
                            for s in e.srcs:
                                key = (s.node, s.port)
                                if key in cache:
                                    bundle[f"{s.node}.{s.port}"] = cache[key]
                                    any_avail = True
                        if not any_avail:
                            ready = False
                            break
                        kwargs[p] = bundle

                if not ready:
                    continue

                # Log node inputs
                for k,v in kwargs.items():
                    self._maybe_log_node_input(n, k, v)

                # Run node
                out = self.nodes[n](**kwargs) if kwargs else self.nodes[n]()
                if not isinstance(out, dict):
                    if len(spec.outputs) != 1:
                        raise TypeError(f"Node '{n}' returned non-dict with multiple outputs declared")
                    out = {spec.outputs[0]: out}

                # Store outputs
                for op in spec.outputs:
                    if op not in out:
                        raise RuntimeError(f"Node '{n}' missing output '{op}'")
                    cache[(n, op)] = out[op]
                    self._maybe_log_node_output(n, op, out[op])

                if self.g._history_enabled:
                    self.history.log(ev="node_done", node=n)

                ran_nodes.add(n)
                changed = True

        # Build final outputs: if `_only` set, limit to requested ones; else return all computed.
        final: Dict[str, Any] = {}
        missing: List[str] = []

        def has_value(pref: PortRef) -> bool:
            return (pref.node, pref.port) in cache

        if target_outputs:
            for out_name in target_outputs:
                pref = self.g._outputs.get(out_name)
                if pref is None:
                    missing.append(out_name)
                    continue
                if has_value(pref):
                    val = cache[(pref.node, pref.port)]
                    final[out_name] = val
                    self._maybe_log_graph_output(out_name, val)
                else:
                    missing.append(out_name)

        # If _only is not provided, we add any other computed outputs (best-effort)
        if not _only:
            for out_name, pref in self.g._outputs.items():
                if out_name in final:
                    continue
                if has_value(pref):
                    val = cache[(pref.node, pref.port)]
                    final[out_name] = val
                    self._maybe_log_graph_output(out_name, val)

        if _strict and missing:
            raise RuntimeError(
                "Some requested outputs are not computable with provided inputs: "
                + ", ".join(missing)
            )

        return final

    # ---------------- Backward traversal: which nodes are needed? -------------
    def _backward_needed_nodes(self, outputs: List[str]) -> Set[str]:
        """
        Retourne l'ensemble des nœuds nécessaires pour produire les sorties demandées
        (traversée inverse sur les arêtes).
        """
        # Build reverse adjacency: dst_node -> {src_nodes}
        rev: Dict[str, Set[str]] = {n:set() for n in self.g.nodes}
        for e in self.g.edges:
            for d in e.dsts:
                for s in e.srcs:
                    rev[d.node].add(s.node)

        need: Set[str] = set()
        stack: List[str] = []
        # seed with nodes that directly feed requested outputs
        for out_name in outputs:
            pref = self.g._outputs.get(out_name)
            if pref is None:
                continue
            need.add(pref.node)
            stack.append(pref.node)

        while stack:
            u = stack.pop()
            for v in rev.get(u, ()):
                if v not in need:
                    need.add(v)
                    stack.append(v)

        return need
