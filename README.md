# multimodal-search

> **Content-Based Recommendation & Search Engine** — ML backend microservice powering product discovery for [Tied Up with Creativity](https://github.com/KraisuN-1010/tiedUpWithCreativity), a live full-stack bracelet e-commerce storefront.

---

## Overview

This repository houses a specialized **Machine Learning backend microservice** that enables semantic, multimodal product search — accepting either a natural language text query or a raw image upload and returning visually and contextually similar products ranked by confidence score.

The engine is built around **OpenAI's CLIP model** (`clip-vit-base-patch32`) and a **FAISS vector similarity index**, with a lightweight **FastAPI** gateway as the public-facing REST API. It is designed from the ground up for **zero-cost cloud deployment** while maintaining production-grade performance characteristics.

---

## Architecture: The Optimization Story

### The Problem — Original Design (Local CLIP, Heavy Stack)

The initial architecture embedded the full CLIP model pipeline directly inside the Docker container, pulling in PyTorch and the Hugging Face `transformers` library as local dependencies.

| Metric | Original Architecture |
|---|---|
| Docker Image Size | **~8.75 GB** |
| RAM at Runtime | Exceeded free-tier limits |
| Cloud Deployment | ❌ OOM crashes on Render, Railway, and similar free tiers |

The result was a microservice that was technically correct but operationally undeployable on any cost-constrained infrastructure.

---

### The Solution — Decoupled Serverless Inference

The architecture was **refactored to fully decouple heavy deep learning inference** from the local execution layer. The new design offloads all CLIP model computation to the **Hugging Face Serverless Inference API**, treating it as a remote feature extraction endpoint.

The local service now only handles what it is uniquely good at: **ultra-fast vector similarity search**.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Incoming Request                              │
│               (Text Query  OR  Image Binary Upload)                  │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FastAPI  (Uvicorn, async)                         │
│           CORS: Trusted storefront domains only                      │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│            Hugging Face Serverless Inference API                     │
│        clip-vit-base-patch32  →  512-D feature vector                │
│                   (Remote · No local GPU/CPU model load)             │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  Returns float32[512]
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 FAISS IndexFlatIP  (Local, In-Memory)                │
│         Pre-computed 512-D vectors · Inner Product similarity        │
│                      Sub-5ms query execution                         │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│              file_mapping.txt  →  Supabase Storage URLs              │
│         Returns: ranked product image URLs + confidence scores       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Performance & Infrastructure Metrics

| Metric | Value | Notes |
|---|---|---|
| **Vector Search Latency** | **< 5 ms** | FAISS `IndexFlatIP` inner product matching |
| **Docker Image Size** | **~657 MB** | Down from 8.75 GB — a **92.5% reduction** |
| **Runtime RAM Usage** | **< 90 MB** | Stable on Render Free Tier (512 MB cap) |
| **OOM Crashes** | **None** | Confirmed stable on restricted cloud environments |
| **Index Dimensionality** | 512-D float32 vectors | CLIP `clip-vit-base-patch32` output space |

> **Engineering Note:** The 92.5% image size reduction was achieved by switching from a `python:3.10` base image bundling PyTorch + Transformers to a `python:3.10-slim` base image with only `faiss-cpu`, `numpy`, `requests`, and `pillow` as runtime dependencies.

---

## Tech Stack

| Layer | Component | Version / Notes |
|---|---|---|
| **Web Framework** | FastAPI + Uvicorn | Async ASGI server |
| **HTTP Client** | Requests | HF Inference API calls |
| **Vector Search** | FAISS-CPU | `1.8.0` · `IndexFlatIP` layout |
| **AI Model** | OpenAI CLIP `clip-vit-base-patch32` | Hosted on Hugging Face Serverless |
| **Numerical Ops** | NumPy | Strictly `< 2.0.0` — FAISS C-extension ABI compatibility |
| **Image Processing** | Pillow | Binary image pre-processing before API dispatch |
| **Media Storage** | Supabase Storage Bucket | Public URL pointers via `file_mapping.txt` |
| **Container Runtime** | Docker (`python:3.10-slim`) | Optimized CPU-only production image |

> ⚠️ **Dependency Constraint:** NumPy is pinned to `<2.0.0`. FAISS's compiled C-extensions rely on the NumPy 1.x C-ABI. Installing NumPy 2.x **will break** the FAISS bindings at runtime with a silent import error.

---

## Repository Structure

```
multimodal-search/
│
├── app/
│   ├── engine.py           # Core FAISS index builder + HF Inference API request logic
│   └── main.py             # FastAPI routing, CORS configuration, endpoint definitions
│
├── data/
│   ├── images/             # Local folder used during initial data collection & indexing
│   ├── file_mapping.txt    # Maps local FAISS index IDs → Supabase Storage public URLs
│   └── vector_index.bin    # Serialized pre-computed 512-D FAISS index (binary)
│
├── Dockerfile              # Optimized CPU-only production container blueprint
└── requirements.txt        # Pinned dependency version tree
```

---

## Local Development Setup

### Prerequisites

- Docker installed and running
- A valid **Hugging Face API Token** — generate one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with at least `read` scope.

---

### Step 1 — Set Your Hugging Face Token

**PowerShell (Windows):**
```powershell
$env:HF_TOKEN="hf_your_token_here"
```

**Bash (Linux / macOS):**
```bash
export HF_TOKEN="hf_your_token_here"
```

---

### Step 2 — Build the FAISS Vector Index

Run the engine script to extract CLIP feature vectors for all images in `data/images/` and serialize the FAISS index to `data/vector_index.bin`.

```bash
python app/engine.py
```

> This step makes outbound calls to the Hugging Face Inference API for each image. Ensure `HF_TOKEN` is set in your environment before running.

---

### Step 3 — Build the Docker Image

```bash
docker build -t multimodal-search-api .
```

---

### Step 4 — Run the Container Locally

**PowerShell (Windows):**
```powershell
docker run -it --rm -p 8000:8000 `
  -e HF_TOKEN=$env:HF_TOKEN `
  -v "$(pwd)/data:/app/data" `
  multimodal-search-api
```

**Bash (Linux / macOS):**
```bash
docker run -it --rm -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$(pwd)/data:/app/data" \
  multimodal-search-api
```

> The `-v` volume mount injects the pre-built `vector_index.bin` and `file_mapping.txt` from your local `data/` directory into the container at runtime, keeping the Docker image itself stateless and reusable.

---

### Step 5 — Access the Interactive API Docs

Once the server boots, open your browser to:

```
http://localhost:8000/docs
```

FastAPI's built-in **Swagger UI** provides a fully interactive testing environment for all endpoints — no external client required.

---

## API Endpoints

### `GET /health`

Verifies that the service is online and the FAISS index is loaded.

**Response:**
```json
{
  "status": "online",
  "index_loaded": true,
  "vector_count": 128
}
```

---

### `POST /search/text`

Accepts a natural language query string. The text is encoded into a 512-D CLIP vector via the Hugging Face API, then matched against the product index.

**Request Body:**
```json
{
  "text": "minimalist silver bracelet",
  "top_k": 3
}
```

**Response:**
```json
{
  "query": "minimalist silver bracelet",
  "results": [
    {
      "rank": 1,
      "url": "https://<project>.supabase.co/storage/v1/object/public/products/silver-chain-thin.jpg",
      "score": 0.9142
    },
    {
      "rank": 2,
      "url": "https://<project>.supabase.co/storage/v1/object/public/products/sterling-cuff.jpg",
      "score": 0.8873
    },
    {
      "rank": 3,
      "url": "https://<project>.supabase.co/storage/v1/object/public/products/delicate-link.jpg",
      "score": 0.8601
    }
  ]
}
```

---

### `POST /search/image`

Accepts a multipart form-data image file upload. The image binary is forwarded to the Hugging Face Inference API to extract a CLIP embedding, which is then queried against the index for visually similar product recommendations.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | `binary` | The image file to match against (JPEG, PNG) |
| `top_k` | `integer` | Number of results to return (default: `5`) |

**Response:**
```json
{
  "results": [
    {
      "rank": 1,
      "url": "https://<project>.supabase.co/storage/v1/object/public/products/gold-beaded.jpg",
      "score": 0.9388
    },
    {
      "rank": 2,
      "url": "https://<project>.supabase.co/storage/v1/object/public/products/pearl-strand.jpg",
      "score": 0.9011
    }
  ]
}
```

---

## Cloud Deployment

This service is designed for **zero-cost production deployment** on platforms such as [Render](https://render.com) or [Railway](https://railway.app).

### Deployment Steps (Render / Railway)

1. **Connect your GitHub repository** to the platform dashboard.
2. Set the **build command** to use the provided `Dockerfile`.
3. In the platform's **Environment Variables** panel, add:

   | Key | Value |
   |---|---|
   | `HF_TOKEN` | `hf_your_token_here` |

4. Mount or pre-bake the `data/` directory. Since `vector_index.bin` is a binary artifact, either:
   - Commit it directly to the repository (acceptable for small indices), or
   - Use a persistent disk mount if the platform supports it.

5. Deploy. The service will boot and expose the FastAPI server on the configured port.

---

### ⚠️ Cold Start Warning (Render Free Tier)

**Render's free tier automatically spins down idle services after 15 minutes of inactivity.**

When a new inbound request arrives after a period of inactivity, the platform boots the container from scratch before serving the response. This **cold start sequence takes approximately 30 seconds** on the first request after a quiet period.

Once the container is warm, **all subsequent requests revert to normal sub-5ms vector search performance.**

| State | First Request Latency | Subsequent Request Latency |
|---|---|---|
| **Cold (after 15 min idle)** | ~30 seconds (boot) | — |
| **Warm (active)** | < 5 ms | < 5 ms |

> **Mitigation options:** Use an uptime monitoring service (e.g., UptimeRobot on a free plan) to ping `/health` every 10 minutes, keeping the container warm during business hours.

---

## Related Repositories

| Repository | Description |
|---|---|
| [`tiedUpWithCreativity`](https://github.com/KraisuN-1010/tiedUpWithCreativity) | Full-stack storefront frontend (React, Vite, Supabase) |

---

## License

This project is private and proprietary. All rights reserved.
