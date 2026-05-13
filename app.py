from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps


APP_TITLE = "Visual Product Search Engine"
DEFAULT_INDEX_NAME = "vr-clothing-gallery"
DEFAULT_NAMESPACE = "finetuned-alpha-0.7"
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}


@dataclass(frozen=True)
class Settings:
    blip2_server_url: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    gallery_csv: str
    captions_csv: str
    image_root: str
    yolo_model_path: str
    clip_checkpoint: str
    clip_model: str
    clip_pretrained: str
    candidate_k: int
    timeout_seconds: int


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        blip2_server_url=os.getenv("BLIP2_SERVER_URL", "").rstrip("/"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE),
        gallery_csv=os.getenv("GALLERY_CSV", ""),
        captions_csv=os.getenv("CAPTIONS_CSV", ""),
        image_root=os.getenv("IMAGE_ROOT", ""),
        yolo_model_path=os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"),
        clip_checkpoint=os.getenv("CLIP_CHECKPOINT", ""),
        clip_model=os.getenv("CLIP_MODEL", "ViT-L-14"),
        clip_pretrained=os.getenv("CLIP_PRETRAINED", "openai"),
        candidate_k=env_int("CANDIDATE_K", 50),
        timeout_seconds=env_int("BLIP2_TIMEOUT_SECONDS", 120),
    )


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def normalize_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def center_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def manual_crop_controls(image: Image.Image) -> Image.Image:
    width, height = image.size
    st.write("Manual crop")
    col_a, col_b = st.columns(2)
    x1 = col_a.slider("Left", 0, max(width - 1, 1), 0)
    x2 = col_a.slider("Right", 1, width, width)
    y1 = col_b.slider("Top", 0, max(height - 1, 1), 0)
    y2 = col_b.slider("Bottom", 1, height, height)
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return image.crop((x1, y1, x2, y2))


@st.cache_resource(show_spinner=False)
def load_yolo(model_path: str):
    try:
        from ultralytics import YOLO

        return YOLO(model_path)
    except Exception as exc:  # noqa: BLE001 - shown to user in sidebar.
        return exc


def crop_with_yolo(image: Image.Image, model_path: str) -> tuple[Image.Image, str]:
    model = load_yolo(model_path)
    if isinstance(model, Exception):
        return center_crop(image), f"YOLO unavailable ({model}). Using center crop."

    results = model.predict(image, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return center_crop(image), "No YOLO box found. Using center crop."

    areas = []
    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    x1, y1, x2, y2 = boxes.xyxy[int(np.argmax(areas))].cpu().numpy()
    width, height = image.size
    pad_x = 0.04 * (x2 - x1)
    pad_y = 0.04 * (y2 - y1)
    box = (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )
    return image.crop(box), "YOLO crop selected from the largest detected product region."


@st.cache_resource(show_spinner=False)
def load_clip(model_name: str, pretrained: str, checkpoint_path: str):
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, preprocess, device


def encode_query_image(image: Image.Image, settings: Settings) -> list[float]:
    import torch

    model, preprocess, device = load_clip(
        settings.clip_model,
        settings.clip_pretrained,
        settings.clip_checkpoint,
    )
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze(0).cpu().numpy().astype(float).tolist()


@st.cache_resource(show_spinner=False)
def get_pinecone_index(api_key: str, index_name: str):
    from pinecone import Pinecone

    return Pinecone(api_key=api_key).Index(index_name)


def query_index(vector: list[float], settings: Settings) -> list[dict[str, Any]]:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set.")

    index = get_pinecone_index(settings.pinecone_api_key, settings.pinecone_index_name)
    response = index.query(
        vector=vector,
        top_k=settings.candidate_k,
        namespace=settings.pinecone_namespace,
        include_metadata=True,
    )
    matches = response.get("matches", []) if isinstance(response, dict) else response.matches
    candidates = []
    for match in matches:
        metadata = match.get("metadata", {}) if isinstance(match, dict) else (match.metadata or {})
        score = match.get("score", 0.0) if isinstance(match, dict) else match.score
        match_id = match.get("id", "") if isinstance(match, dict) else match.id
        candidates.append({"id": match_id, "clip_score": float(score), "metadata": metadata})
    return candidates


@st.cache_data(show_spinner=False)
def load_caption_lookup(captions_csv: str) -> dict[str, str]:
    if not captions_csv or not Path(captions_csv).exists():
        return {}

    df = pd.read_csv(captions_csv)
    if "image_name" not in df.columns:
        return {}
    caption_col = "blip2_caption" if "blip2_caption" in df.columns else df.columns[-1]
    return dict(zip(df["image_name"].astype(str), df[caption_col].fillna("").astype(str)))


def enrich_candidates(candidates: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    captions = load_caption_lookup(settings.captions_csv)
    enriched = []
    for candidate in candidates:
        metadata = dict(candidate.get("metadata") or {})
        image_name = str(
            metadata.get("image_name")
            or metadata.get("filename")
            or metadata.get("path")
            or candidate.get("id")
        )
        caption = str(metadata.get("caption") or metadata.get("blip2_caption") or captions.get(image_name, ""))
        enriched.append(
            {
                **candidate,
                "image_name": image_name,
                "item_id": metadata.get("item_id", metadata.get("product_id", "")),
                "caption": caption,
            }
        )
    return enriched


def request_blip2_rerank(
    query_crop: Image.Image,
    candidates: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[dict[str, Any]], str]:
    if not settings.blip2_server_url:
        return candidates, "BLIP2_SERVER_URL is not set. Showing CLIP/Pinecone ranking only."

    payload_candidates = [
        {
            "id": row["id"],
            "caption": row.get("caption", ""),
            "clip_score": row.get("clip_score", 0.0),
            "metadata": row.get("metadata", {}),
        }
        for row in candidates
    ]

    files = {"image": ("query_crop.png", to_png_bytes(query_crop), "image/png")}
    data = {"candidates": json.dumps(payload_candidates)}
    try:
        response = requests.post(
            f"{settings.blip2_server_url}/rerank",
            files=files,
            data=data,
            headers=NGROK_HEADERS,
            timeout=settings.timeout_seconds,
        )
        response.raise_for_status()
        by_id = {str(row["id"]): row for row in response.json().get("results", [])}
    except Exception as exc:  # noqa: BLE001 - this keeps the demo alive during viva.
        return candidates, f"Remote BLIP-2 re-rank failed: {exc}. Showing CLIP/Pinecone ranking only."

    reranked = []
    for candidate in candidates:
        remote = by_id.get(str(candidate["id"]), {})
        reranked.append(
            {
                **candidate,
                "blip2_score": remote.get("blip2_score"),
                "final_score": remote.get("final_score", candidate.get("clip_score", 0.0)),
            }
        )
    reranked.sort(key=lambda row: row.get("final_score", row.get("clip_score", 0.0)), reverse=True)
    return reranked, "Remote BLIP-2 image-text matching re-rank applied."


def blip2_health(settings: Settings) -> tuple[bool, str]:
    if not settings.blip2_server_url:
        return False, "BLIP2_SERVER_URL is not set."

    try:
        response = requests.get(
            f"{settings.blip2_server_url}/health",
            headers=NGROK_HEADERS,
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - surfaced in the sidebar.
        return False, str(exc)

    cuda = "GPU" if payload.get("cuda") else "CPU"
    return True, f"{payload.get('model_name', 'BLIP-2')} ready on {cuda}"


def local_image_path(image_name: str, image_root: str) -> Path | None:
    if not image_root:
        return None
    root = Path(image_root)
    direct = root / image_name
    if direct.exists():
        return direct
    matches = list(root.rglob(Path(image_name).name))
    return matches[0] if matches else None


def render_candidate(row: dict[str, Any], rank: int, settings: Settings) -> None:
    with st.container(border=True):
        cols = st.columns([1, 2])
        with cols[0]:
            path = local_image_path(row.get("image_name", ""), settings.image_root)
            if path and path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.caption("Catalog image path not available")
        with cols[1]:
            st.subheader(f"Rank {rank}")
            if row.get("item_id"):
                st.write(f"Item ID: `{row['item_id']}`")
            st.write(f"Image: `{row.get('image_name', row.get('id', ''))}`")
            if row.get("caption"):
                st.caption(row["caption"])
            metric_cols = st.columns(3)
            metric_cols[0].metric("CLIP", f"{row.get('clip_score', 0.0):.4f}")
            blip_score = row.get("blip2_score")
            metric_cols[1].metric("BLIP-2", "N/A" if blip_score is None else f"{blip_score:.4f}")
            metric_cols[2].metric("Final", f"{row.get('final_score', row.get('clip_score', 0.0)):.4f}")


def sidebar(settings: Settings) -> int:
    with st.sidebar:
        st.header("Runtime")
        top_k = st.slider("Results", 5, 30, 10, step=5)
        st.text_input("BLIP-2 server", value=settings.blip2_server_url or "not set", disabled=True)
        st.text_input("Pinecone index", value=settings.pinecone_index_name, disabled=True)
        st.text_input("Namespace", value=settings.pinecone_namespace, disabled=True)
        if st.button("Check BLIP-2 server"):
            ok, message = blip2_health(settings)
            if ok:
                st.success(message)
            else:
                st.error(message)
        st.caption("Local config is read from .env or environment variables.")
    return top_k


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    settings = load_settings()
    top_k = sidebar(settings)

    st.title(APP_TITLE)
    st.write("Upload a fashion product image, confirm the detected crop, then retrieve visually and semantically similar catalog items.")

    uploaded = st.file_uploader("Query image", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        st.info("Waiting for an input image.")
        return

    image = normalize_image(Image.open(uploaded))
    upload_signature = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("upload_signature") != upload_signature:
        st.session_state.upload_signature = upload_signature
        st.session_state.confirmed_crop = False
        st.session_state.manual_crop = False

    yolo_crop, crop_note = crop_with_yolo(image, settings.yolo_model_path)
    st.session_state.setdefault("confirmed_crop", False)
    st.session_state.setdefault("manual_crop", False)

    st.session_state.manual_crop = st.checkbox("Adjust crop manually", value=st.session_state.manual_crop)
    crop = manual_crop_controls(image) if st.session_state.manual_crop else yolo_crop

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Product crop")
        st.image(crop, use_container_width=True)
        st.caption(crop_note)

    actions = st.columns([1, 1, 4])
    if actions[0].button("Confirm crop", type="primary"):
        st.session_state.confirmed_crop = True
    if actions[1].button("Re-crop"):
        st.session_state.confirmed_crop = False
        st.session_state.manual_crop = True
        st.rerun()

    if not st.session_state.confirmed_crop:
        st.stop()

    with st.spinner("Encoding query, searching the ANN index, and re-ranking candidates..."):
        try:
            vector = encode_query_image(crop, settings)
            candidates = query_index(vector, settings)
            candidates = enrich_candidates(candidates, settings)
            results, rerank_note = request_blip2_rerank(crop, candidates, settings)
        except Exception as exc:  # noqa: BLE001 - Streamlit should show setup gaps cleanly.
            st.error(f"Search pipeline could not run: {exc}")
            st.stop()

    st.success(rerank_note)
    st.subheader("Retrieved products")
    if not results:
        st.warning("No results returned from the vector index.")
        return

    for rank, row in enumerate(results[:top_k], start=1):
        render_candidate(row, rank, settings)


if __name__ == "__main__":
    main()
