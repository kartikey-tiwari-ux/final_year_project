"""Text explainability: per-token saliency (gradient x input) for the DistilBERT
branch. No extra dependency (SHAP/LIME/Captum) needed — see PROJECT_ARCHITECTURE.md.

This produces a per-token importance score for one text at a time (used by the demo
app and for qualitative examples in the report) — it is NOT a rigorously validated
attribution method, and is documented as exploratory, per RESEARCH_GAP.md point 4.
"""
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"


class TextSaliencyExplainer:
    def __init__(self, classifier_weights: torch.Tensor, classifier_bias: torch.Tensor):
        """classifier_weights/bias: the trained LogisticRegression's coef_/intercept_
        for the predicted class, used to turn the frozen embedding gradient into a
        per-token saliency score without needing a differentiable classifier head."""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.classifier_weights = classifier_weights  # shape (hidden,)
        self.classifier_bias = classifier_bias

    def explain(self, text: str, max_length: int = 96) -> Tuple[List[str], List[float]]:
        enc = self.tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        embeddings_layer = self.model.get_input_embeddings()
        input_embeds = embeddings_layer(enc["input_ids"]).clone().detach().requires_grad_(True)

        out = self.model(inputs_embeds=input_embeds, attention_mask=enc["attention_mask"])
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        # Score = dot product with the predicted class's logistic-regression weight
        # vector — differentiating this back to the input embeddings gives a saliency
        # signal for which tokens pushed the pooled representation toward that class.
        score = (pooled.squeeze(0) * self.classifier_weights).sum() + self.classifier_bias
        score.backward()

        grads = input_embeds.grad.squeeze(0)  # (seq_len, hidden)
        saliency = (grads * input_embeds.squeeze(0)).sum(dim=-1).abs()  # gradient x input
        saliency = (saliency / saliency.max().clamp(min=1e-9)).tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0).tolist())
        return tokens, saliency
