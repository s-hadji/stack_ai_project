def to_graphviz(graph, path: str):
    # Minimal placeholder to avoid hard dependency; can be implemented later.
    with open(path, "w", encoding="utf-8") as f:
        f.write("digraph G {\n")
        for name, node in graph.nodes.items():
            f.write(f"  {name} [label=\"{name}\"];\n")
        for e in graph.edges:
            for s in e.srcs:
                for d in e.dsts:
                    f.write(f"  {s.node} -> {d.node};\n")
        f.write("}\n")
