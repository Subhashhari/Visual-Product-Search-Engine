from __future__ import annotations

import io
import json
import os
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image, ImageOps


MODEL_NAME = os.getenv("BLIP2_MODEL_NAME", "blip2_image_text_matching")
MODEL_TYPE = os.getenv("BLIP2_MODEL_TYPE", "pretrain")
CLIP_WEIGHT = float(os.getenv("CLIP_WEIGHT", "0.7"))

app = FastAPI(title="BLIP-2 Re-ranking Service")
_model_bundle: tuple[Any, Any, Any, str] | None = None


def load_model():
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    from lavis.models import load_model_and_preprocess

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=MODEL_NAME,
        model_type=MODEL_TYPE,
        is_eval=True,
        device=device,
    )
    _model_bundle = model, vis_processors, txt_processors, device
    return _model_bundle


def read_image(raw: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


def score_candidates(image: Image.Image, captions: list[str]) -> list[float]:
    model, vis_processors, txt_processors, device = load_model()
    if not captions:
        return []

    image_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)
    image_batch = image_tensor.repeat(len(captions), 1, 1, 1)
    text_batch = [txt_processors["eval"](caption or "clothing product") for caption in captions]

    with torch.no_grad():
        output = model({"image": image_batch, "text_input": text_batch}, match_head="itm")
        probabilities = F.softmax(output, dim=1)[:, 1]
    return probabilities.detach().float().cpu().tolist()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
    }


@app.post("/rerank")
async def rerank(
    image: UploadFile = File(...),
    candidates: str = Form(...),
) -> dict[str, Any]:
    query_image = read_image(await image.read())
    rows = json.loads(candidates)
    captions = [str(row.get("caption") or "") for row in rows]
    blip_scores = score_candidates(query_image, captions)

    results = []
    for row, blip_score in zip(rows, blip_scores):
        clip_score = float(row.get("clip_score") or 0.0)
        final_score = CLIP_WEIGHT * clip_score + (1.0 - CLIP_WEIGHT) * float(blip_score)
        results.append(
            {
                "id": row.get("id"),
                "clip_score": clip_score,
                "blip2_score": float(blip_score),
                "final_score": final_score,
                "metadata": row.get("metadata") or {},
            }
        )
    results.sort(key=lambda row: row["final_score"], reverse=True)
    return {"results": results}
