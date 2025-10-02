# stack-ai

A tiny library to **stack AI models** (LLMs, vision, adapters, custom functions) using Hugging Face to load model
as a **graph** and compile it into a single **PyTorch `nn.Module`**.

### Highlights
- Minimal API: `add_model`, `add_function`, `connect`, `input`, `output`, `to_torch`.
- Connect **one or many sources** to **one or many targets**.
- Insert **functions** inline with `via=` or as named nodes with `add_function`.
- Optional **HuggingFace** helpers (`pip install stack-ai[hf]`).
- Simple **history** logger and **GraphViz** export (optional).

See `examples/` for runnable demos.


## Installation

- Basic :  
  `pip install stack-ai`

- With Hugging Face helpers :  
  `pip install "stack-ai[hf]"`

## Exemples
- **Local minimal** : `python examples/mvp_concat.py`
- **Multi-model (Hugging Face)** : `python examples/hf_multi_demo.py`
