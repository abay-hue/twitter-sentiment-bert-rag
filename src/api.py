import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = os.environ.get("MODEL_DIR", "models/bert")

app = FastAPI(title="Sentiment API")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

class Inp(BaseModel):
    text: str

@app.post("/predict")
def predict(inp: Inp):
    with torch.no_grad():
        enc = tokenizer(inp.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        label = int(torch.argmax(logits, dim=-1).item())
        return {"label": label, "probs": probs}
