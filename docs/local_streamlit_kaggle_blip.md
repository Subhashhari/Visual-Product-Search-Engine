# Run Streamlit locally and BLIP-2 on Kaggle

This is the intended setup when your laptop has no GPU:

- Laptop: Streamlit app, YOLO crop, CLIP query encoding, Pinecone search, result display.
- Kaggle GPU: BLIP-2 image-text matching server only.

## 1. Prepare the local dataset paths

Download the Kaggle dataset version shown in your screenshot to your laptop.

Your local dataset folder should contain:

```text
version_5/
  img/
  blip2_captions_gallery.csv
  blip2_captions_train.csv
  clip_best.pt
  gallery.csv
  query.csv
  train.csv
```

`IMAGE_ROOT` must point to `version_5`, because the CSV image paths look like:

```text
img/img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg
```

## 2. Start BLIP-2 on Kaggle GPU

In a Kaggle notebook with GPU enabled:

```bash
git clone YOUR_REPO_URL
cd Visual-Product-Search-Engine
pip install -r requirements-blip2-server.txt
```

If you have an ngrok auth token, set it:

```bash
export NGROK_AUTHTOKEN="YOUR_NGROK_TOKEN"
```

Start the BLIP-2 API and ngrok tunnel:

```bash
python remote_server/run_blip2_ngrok.py
```

Copy the printed URL:

```text
BLIP-2 service public URL: https://xxxx.ngrok-free.app
```

Keep this Kaggle notebook running.

## 3. Configure the local Streamlit app

On your laptop:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
BLIP2_SERVER_URL=https://xxxx.ngrok-free.app
PINECONE_API_KEY=YOUR_PINECONE_KEY
PINECONE_INDEX_NAME=vr-clothing-gallery
PINECONE_NAMESPACE=finetuned-alpha-0.7
GALLERY_CSV=/absolute/path/to/version_5/gallery.csv
CAPTIONS_CSV=/absolute/path/to/version_5/blip2_captions_gallery.csv
IMAGE_ROOT=/absolute/path/to/version_5
CLIP_CHECKPOINT=/absolute/path/to/version_5/clip_best.pt
```

## 4. Run Streamlit locally

Install local app dependencies:

```bash
pip install -r requirements-streamlit.txt
```

Run the app:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## 5. Demo flow

1. Click **Check BLIP-2 server** in the sidebar.
2. Upload a query image.
3. Confirm the YOLO crop, or use manual crop.
4. The local app runs CLIP and Pinecone search.
5. The local app sends only the cropped query image and candidate captions to the Kaggle BLIP-2 URL.
6. Results are shown locally with CLIP, BLIP-2, and final scores.
