from stack_ai import Graph
import torch.nn as nn

class A(nn.Module):
    def forward(self): return {"out": "x"}

class B(nn.Module):
    def forward(self, x): return {"y": x + "!"}

def test_connect_direct():
    g = Graph()
    g.add_model(name="A", module=A(), inputs=[], outputs=["out"])
    g.add_model(name="B", module=B(), inputs=["x"], outputs=["y"])
    g.connect("A.out", "B.x")
    g.output("res").from_("B.y")
    m = g.to_torch()
    assert m()["res"] == "x!"

def test_connect_via_function():
    g = Graph()
    g.add_model(name="A", module=A(), inputs=[], outputs=["out"])
    g.add_model(name="B", module=B(), inputs=["x"], outputs=["y"])
    g.add_function(name="twice", inputs=["a"], outputs=["out"], fn=lambda a: {"out": a+a})
    g.connect("A.out", "B.x", via="twice", as_=["a"])
    g.output("res").from_("B.y")
    m = g.to_torch()
    assert m()["res"] == "xx!"
