from typing import Any
def load_hf_text_generation(model_id: str, **kwargs) -> Any:
    """
    Lazy helper that returns a callable nn.Module-like object which accepts `prompt`
    and returns { "text": generated_text } using transformers pipeline.
    Requires `pip install stack-ai[hf]`.
    """
    from transformers import pipeline
    import torch.nn as nn
    class HFGen(nn.Module):
        def __init__(self, pipe):
            super().__init__()
            self.pipe = pipe
        def forward(self, prompt):
            out = self.pipe(prompt, **kwargs)
            text = out[0]["generated_text"] if isinstance(out, list) else str(out)
            return {"text": text}
    pipe = pipeline("text-generation", model=model_id)
    return HFGen(pipe)
