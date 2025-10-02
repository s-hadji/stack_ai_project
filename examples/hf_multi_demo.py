from stack_ai import Graph
from stack_ai.io.hf import load_hf_text_generation, load_hf_text_classification

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

# (Option) baisser la verbosité Transformers pour éviter le spam
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# =============================================================================
# 0) Device
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================================================================
# 1) Charger les modèles HF (texte)
# =============================================================================
gen = load_hf_text_generation("sshleifer/tiny-gpt2", max_new_tokens=24, do_sample=False)
sent = load_hf_text_classification("distilbert-base-uncased-finetuned-sst-2-english")
rephraser = load_hf_text_generation("sshleifer/tiny-gpt2", max_new_tokens=32, do_sample=False)

# =============================================================================
# 2) Définir et entraîner un petit modèle MNIST
# =============================================================================
def prepare_mnist_loaders(batch_size=64):
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)   # [B,1,28,28] -> [B,16,28,28]
        self.pool = nn.MaxPool2d(2,2)                                      # -> [B,16,14,14]
        self.fc   = nn.Linear(16*14*14, 10)

    def forward(self, image):  # image: [B,1,28,28]
        x = self.pool(F.relu(self.conv(image)))
        x = x.flatten(1)
        logits = self.fc(x)                                               # [B,10]
        return {"logits": logits}

def train_mnist(model, train_loader, epochs=1, lr=1e-3, device=torch.device("cpu")):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        total = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)               # {"logits": [B,10]}
            loss = loss_fn(out["logits"], labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[MNIST] epoch {ep+1}/{epochs} - loss={total:.4f}")
    model.eval()
    return model

@torch.no_grad()
def eval_mnist(model, test_loader, device=torch.device("cpu"), max_batches=20):
    model.to(device).eval()
    correct, total = 0, 0
    for bi, (images, labels) in enumerate(test_loader):
        if bi >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)["logits"]
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total   += labels.numel()
    acc = correct / max(1,total)
    print(f"[MNIST] quick accuracy on {total} samples: {acc*100:.2f}%")

# — Data & training —
train_loader, test_loader = prepare_mnist_loaders(batch_size=64)
mnist_model = MNISTNet()
mnist_model = train_mnist(mnist_model, train_loader, epochs=1, lr=1e-3, device=device)
eval_mnist(mnist_model, test_loader, device=device, max_batches=20)

# =============================================================================
# 3) Construire le graphe (texte + MNIST en parallèle)
# =============================================================================
g = Graph()

# ---- Pipeline texte
g.add_model(name="llm_gen", module=gen, inputs=["prompt"], outputs=["text"])
g.add_model(name="sentiment", module=sent, inputs=["text"], outputs=["label", "score"])
g.add_model(name="rephraser", module=rephraser, inputs=["prompt"], outputs=["text"])

g.add_function(
    name="templater",
    inputs=["topic"],
    outputs=["prompt"],
    fn=lambda topic: {"prompt": f"Write two short sentences about: {topic}"},
)
def combine(text, label, score):
    tag = f"[{label} {score:.2f}]"
    return {"final": f"{tag}\n{text}\n"}
g.add_function(name="combine", inputs=["text","label","score"], outputs=["final"], fn=combine)

g.add_function(
    name="prompt_rephrase",
    inputs=["x"],
    outputs=["prompt"],
    fn=lambda x: {"prompt": f"Rewrite more clearly and concisely:\n{x}"},
)

# Connexions texte
g.input("topic").to("templater.topic")
g.connect("templater.prompt", "llm_gen.prompt")
g.connect("llm_gen.text", "sentiment.text")
g.connect(["llm_gen.text", "sentiment.label", "sentiment.score"], via="combine", as_=["text","label","score"])
g.connect("combine.final", "prompt_rephrase.x")
g.connect("prompt_rephrase.prompt", "rephraser.prompt")
g.output("text").from_("rephraser.text")

# ---- Pipeline image (MNIST)
g.add_model(name="mnist", module=mnist_model.to("cpu"), inputs=["image"], outputs=["logits"])

def mnist_post(logits):
    probs = F.softmax(logits, dim=-1)      # [B,10]
    digit = int(probs.argmax(dim=-1)[0].item())
    prob  = float(probs.max(dim=-1).values[0].item())
    return {"digit": digit, "prob": prob}

g.add_function(name="mnist_post", inputs=["logits"], outputs=["digit","prob"], fn=mnist_post)
g.input("image").to("mnist.image")
g.connect("mnist.logits", "mnist_post.logits")
g.output("digit").from_("mnist_post.digit")
g.output("digit_prob").from_("mnist_post.prob")

# =============================================================================
# 4) Historique
# =============================================================================
g.enable_history(
    keep=["*.text", "sentiment.*", "combine.final", "templater.prompt", "mnist.*", "mnist_post.*"],
    tensor_policy="shape",
    text_limit=400,
    with_inputs=True,
    with_outputs=True,
    with_graph_io=True,
)

# =============================================================================
# 5) Compiler -> objet Torch
# =============================================================================
m = g.to_torch()  # nn.Module (Executor)

print("---- Torch / Module Infos ----")
print("torch.__version__        :", torch.__version__)
print("isinstance(m, nn.Module) :", isinstance(m, nn.Module))
print("Sub-modules (nodes)      :", list(m.nodes.keys()))
print()

# =============================================================================
# 6) “Plot” texte rapide du graphe
# =============================================================================
def ascii_graph(nodes: Dict[str, nn.Module], edges: List[Tuple[str, str]]):
    uniq = sorted(set(edges))
    print("---- Graph (ASCII) ----")
    for s, d in uniq:
        print(f"{s}  -->  {d}")
    print()

edge_pairs = []
for e in g.edges:
    for s in e.srcs:
        for d in e.dsts:
            edge_pairs.append((s.node, d.node))
ascii_graph(g.nodes, edge_pairs)

# =============================================================================
# 7) Démonstrations d’exécution partielle
# =============================================================================
# Prélève un batch test MNIST
test_images, test_labels = next(iter(test_loader))
img0, label0 = test_images[0:1], test_labels[0].item()  # [1,1,28,28], int

# A) Image seule -> n'exécute QUE la branche MNIST
print("\n-- Run: image only (MNIST branch) --")
print("available_outputs(image=img0):", m.available_outputs(image=img0))
mnist_out = m(image=img0)  # pas de 'topic' => la branche texte ne tourne pas
print("outputs (keys):", mnist_out.keys())
print("digit:", mnist_out.get("digit"), "prob:", f"{mnist_out.get('digit_prob', 0):.2f}", "| true:", label0)

# B) Texte seul -> n'exécute QUE la branche texte
print("\n-- Run: topic only (text branch) --")
print("available_outputs(topic='tips'):", m.available_outputs(topic="tips"))
text_out = m(topic="productivity tips")  # pas d'image => MNIST ne tourne pas
print("outputs (keys):", text_out.keys())
print("text sample:\n", text_out["text"][:200], "...")

# C) Deux entrées -> les deux branches s'exécutent
print("\n-- Run: topic + image (both branches) --")
both_out = m(topic="productivity tips", image=img0)
print("outputs (keys):", both_out.keys())

# D) Cibler explicitement un sous-ensemble de sorties
print("\n-- Run: image + _only=['digit'] --")
only_digit_out = m(image=img0, _only=["digit"])
print("outputs (keys):", only_digit_out.keys())

# E) Mode strict: lève si non calculable
print("\n-- Try: _only=['text'] with image only (should be missing) --")
try:
    _ = m(image=img0, _only=["text"], _strict=True)
except Exception as e:
    print("strict error:", e)

# =============================================================================
# 8) Historique : aperçu + export
# =============================================================================
events = getattr(m, "history", None).events if hasattr(m, "history") else []
print("\n---- History (events summary) ----")
print(f"Total events recorded: {len(events)}")
for ev in events[:10]:
    print(ev)
if len(events) > 10:
    print("...")

try:
    m.history.export_jsonl("run_history.jsonl")
    print('\nHistory exported to "run_history.jsonl"')
except Exception as e:
    print("History export failed:", e)

# =============================================================================
# 9) Mini-évaluation graphe sur quelques images test (MNIST only)
#    -> grâce à l’exécution partielle, on ne fait tourner QUE la branche image
# =============================================================================
@torch.no_grad()
def graph_eval_accuracy_mnist_only(module, test_loader, n_batches=5):
    correct, total = 0, 0
    for bi, (images, labels) in enumerate(test_loader):
        if bi >= n_batches:
            break
        for i in range(min(len(images), 8)):
            x = images[i:i+1]  # [1,1,28,28]
            y = labels[i].item()
            out = module(image=x, _only=["digit"])   # <-- branche image uniquement
            pred = out["digit"]
            correct += int(pred == y)
            total   += 1
    acc = correct / max(1,total)
    print(f"[Graph/MNIST partial] quick eval on {total} samples: {acc*100:.2f}%")

graph_eval_accuracy_mnist_only(m, test_loader, n_batches=5)
