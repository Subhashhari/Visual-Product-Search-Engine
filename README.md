# Visual Product Search Engine

![Visual Product Search Engine Architecture](https://img.shields.io/badge/Project-Visual%20Search-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-red)

A query-by-image visual product search engine built for fashion retail. This project addresses the limitation of text-based product search by allowing users to search for visually similar products using an uploaded image. 

The system utilizes a state-of-the-art cross-modal retrieval pipeline combining **YOLO** for product localization, fine-tuned **CLIP** for vector retrieval, and **BLIP-2** for semantic image-text matching (ITM) re-ranking, all orchestrated over a **Pinecone** vector database.

## Features

- **Query-by-Image:** Upload any fashion product image to find similar catalog items.
- **Smart Localization:** YOLO-based automatic detection of the main clothing product, with manual crop adjustment capabilities to reduce background noise.
- **Cross-Modal Retrieval:** Fast Approximate Nearest Neighbor (ANN) search using Pinecone, powered by image-text fused embeddings from CLIP and BLIP-2.
- **Semantic Re-ranking:** High-precision re-ranking using BLIP-2 Image-Text Matching (ITM) to prioritize semantically aligned products.
- **Interactive Demo:** A fully functional end-to-end Streamlit web application.

## Architecture

### 1. Offline Indexing Pipeline
1. **Product Localization:** Gallery images are cropped using YOLO to isolate the product.
2. **Semantic Captioning:** BLIP-2 generates descriptive captions for each gallery product.
3. **Embedding Fusion:** CLIP encodes both the product crop and the caption. The embeddings are fused using a weighted combination ($\alpha=0.7$).
4. **Vector Database:** Fused vectors are upserted into Pinecone namespaces for fast retrieval.

### 2. Online Retrieval Pipeline
1. User uploads a query image via the Streamlit app.
2. YOLO proposes a product crop (user can confirm or adjust).
3. The query crop is encoded using a fine-tuned CLIP checkpoint.
4. Top-K candidate products are retrieved from the Pinecone vector index.
5. Candidates are sent to a remote BLIP-2 ITM service for semantic re-ranking.
6. Final ranked results are displayed alongside similarity scores.

## Dataset & Evaluation

Built and evaluated on the **DeepFashion In-Shop Clothes Retrieval** dataset. 
- **Query Images:** 14,218
- **Gallery Images:** 12,612
- **Ground Truth:** Matches defined by shared `item_id`.

### Results
Through rigorous ablation studies, the best configuration (**Fine-tuned CLIP + BLIP-2 ITM Re-ranking** with fusion weight $\alpha=0.7$) achieved a **Mean Recall@15 of 0.9368** across evaluation seeds, significantly outperforming baseline vision-only retrieval.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Subhashhari/Visual-Product-Search-Engine.git
   cd Visual-Product-Search-Engine
   ```

2. **Install dependencies:**
   There are separate requirement files for different components. To run the main Streamlit app:
   ```bash
   pip install -r requirements-streamlit.txt
   ```
   *(For CPU environments, use `requirements-streamlit-cpu.txt`)*

3. **Remote BLIP-2 Service (Optional but recommended for full re-ranking):**
   Set up the remote service on a GPU instance using:
   ```bash
   pip install -r requirements-blip2-server.txt
   ```

4. **Environment Variables:**
   Rename `.env.example` to `.env` and fill in your Pinecone API keys and other configuration variables.

5. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```


Submitted to the International Institute of Information Technology Bangalore (IIIT-B).
