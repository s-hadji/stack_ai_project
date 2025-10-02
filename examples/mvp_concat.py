from stack_ai import Graph
import torch.nn as nn

class Img2Txt(nn.Module):
    def forward(self, image): return {"text": "[VIS] cat on mat"}

class T2T(nn.Module):
    def forward(self, input): return {"text": f"{input} (normalized)"}

class LLM(nn.Module):
    def forward(self, prompt): return {"text": f"LLM says: {prompt[:50]}..."}

g = Graph()
g.add_model(name="img2txt", module=Img2Txt(), inputs=["image"], outputs=["text"])
g.add_model(name="t2t", module=T2T(), inputs=["input"], outputs=["text"])
g.add_model(name="llm", module=LLM(), inputs=["prompt"], outputs=["text"])

g.input("image").to("img2txt.image")
g.input("user_text").to("t2t.input")

g.add_function(name="concat", inputs=["a","b"], outputs=["text"],
               fn=lambda a,b: {"text": a + "\n" + b})
g.connect(["img2txt.text","t2t.text"], "llm.prompt", via="concat", as_=["a","b"])

g.output("final_text").from_("llm.text")

model = g.to_torch()
print(model(image="dummy", user_text="Bonjour le monde")["final_text"])
