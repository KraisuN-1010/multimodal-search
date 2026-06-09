import os
import glob
import time
import requests
import faiss
import numpy as np
from PIL import Image

# Baseline directory management
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX = os.path.join(BASE_DIR, "data", "vector_index.bin")
DEFAULT_MAPPING = os.path.join(BASE_DIR, "data", "file_mapping.txt")

# Hugging Face Inference configuration
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-base-patch32"
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_ACTUAL_HF_TOKEN_HERE") # Fallback string if env variable isn't set yet
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf_embedding(payload, is_image=False):
    """
    Sends raw text or image binary data to Hugging Face Inference API for vector extraction.
    """
    for attempt in range(3): 
        if is_image:
            response = requests.post(API_URL, headers=HEADERS, data=payload)
        else:
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503: 
            print("Hugging Face model is initializing, waiting 5 seconds...", flush=True)
            time.sleep(5)
        else:
            raise Exception(f"Hugging Face API Error ({response.status_code}): {response.text}")
            
    raise Exception("Failed to fetch vectors from Hugging Face after multiple retries.")

def build_vector_index(image_folder_path, index_output_path=DEFAULT_INDEX, mapping_output_path=DEFAULT_MAPPING):
    image_paths = glob.glob(os.path.join(image_folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder_path, "*.jpeg")) + \
                  glob.glob(os.path.join(image_folder_path, "*.png"))
    
    if not image_paths:
        print(f"Error: No images found in local path: {image_folder_path}", flush=True)
        return

    print(f">>> Found {len(image_paths)} images! Streaming targets to Hugging Face API... <<<", flush=True)
    embeddings_list = []
    id_to_path_mapping = []

    for index, path in enumerate(image_paths):
        try:
            # Read image as raw bytes instead of loading heavy tensor objects into local RAM
            with open(path, "rb") as f:
                img_bytes = f.read()
                
            vector = query_hf_embedding(img_bytes, is_image=True)
            
            # Extract flattened array if response is multi-dimensional
            if isinstance(vector[0], list):
                vector = np.array(vector).flatten()
                
            embeddings_list.append(vector)
            id_to_path_mapping.append(path)
            print(f"Indexed [{index + 1}/{len(image_paths)}]: {os.path.basename(path)}", flush=True)
        except Exception as e:
            print(f"Failed to process image {path}: {str(e)}", flush=True)

    embedding_matrix = np.array(embeddings_list).astype('float32')
    vector_dimension = embedding_matrix.shape[1] 
    
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(embedding_matrix)

    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)
    faiss.write_index(index, index_output_path)
    
    with open(mapping_output_path, "w") as f:
        for path in id_to_path_mapping:
            f.write(f"{path}\n")
    print(f"\nSuccess! Compiled 512-D index tracking files saved.", flush=True)


def multimodal_search(query_text=None, query_image_path=None, top_k=3, index_path=DEFAULT_INDEX, mapping_path=DEFAULT_MAPPING):
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError("Database index files missing. Run build_vector_index first.")

    index = faiss.read_index(index_path)
    with open(mapping_path, "r") as f:
        id_to_path_mapping = [line.strip() for line in f.readlines()]

    query_vector = None

    if query_text:
        print(f"Requesting API extraction for text query: '{query_text}'", flush=True)
        vector = query_hf_embedding({"inputs": query_text}, is_image=False)
        
        if isinstance(vector[0], list):
            vector = vector[0]
        query_vector = np.array([vector]).astype('float32')

    elif query_image_path:
        print(f"Requesting API extraction for image query asset: '{os.path.basename(query_image_path)}'", flush=True)
        with open(query_image_path, "rb") as f:
            img_bytes = f.read()
        vector = query_hf_embedding(img_bytes, is_image=True)
        if isinstance(vector[0], list):
            vector = np.array(vector).flatten()
        query_vector = np.array([vector]).astype('float32')

    if query_vector is None:
        return []

    # FAISS Cosine/Inner Product Matrix Similarity Evaluation Loop
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(id_to_path_mapping):
            results.append({
                "image_path": id_to_path_mapping[idx],
                "confidence_score": float(score)
            })
            
    return results

if __name__ == "__main__":
    print(">>> TARGET TRIGGERED: Rebuilding 500-Image Vector Index Database via Remote API... <<<", flush=True)
    data_img_dir = os.path.join(BASE_DIR, "data", "images")
    build_vector_index(image_folder_path=data_img_dir)