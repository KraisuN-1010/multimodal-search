import os
import time
import requests
import faiss
import numpy as np
from supabase import create_client, Client

# Baseline directory management
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX = os.path.join(BASE_DIR, "data", "vector_index.bin")
DEFAULT_MAPPING = os.path.join(BASE_DIR, "data", "file_mapping.txt")

# Hugging Face Inference configuration
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-base-patch32"
HF_TOKEN = os.getenv("HF_TOKEN", "ACTUAL_HF_TOKEN_HERE")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Supabase Credentials & Configurations
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://vspiwiteykkpqfsqgakh.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_ANON_OR_SERVICE_ROLE_KEY")
SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "product-images")

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def query_hf_embedding(payload: bytes | dict, is_image: bool = False) -> list:
    """Sends raw data to Hugging Face Serverless Inference to extract CLIP feature vectors."""
    for attempt in range(3): 
        if is_image:
            response = requests.post(API_URL, headers=HEADERS, data=payload)
        else:
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503: 
            print("Model initializing, retrying in 5 seconds...", flush=True)
            time.sleep(5)
        else:
            raise Exception(f"Hugging Face API Error ({response.status_code}): {response.text}")
    raise Exception("Failed to fetch embedding vectors after max retries.")


def build_vector_index_from_cloud(index_output_path=DEFAULT_INDEX, mapping_output_path=DEFAULT_MAPPING):
    """
    Connects to Supabase Storage, pulls all images from the /products folder directly into memory,
    extracts CLIP embeddings via Hugging Face, and builds the localized FAISS mapping files.
    """
    print(f">>> Querying Supabase Storage Bucket '{SUPABASE_BUCKET_NAME}/products'... <<<", flush=True)
    
    try:
        # 1. List all objects inside the 'products' subfolder
        bucket_objects = supabase.storage.from_(SUPABASE_BUCKET_NAME).list("products")
    except Exception as e:
        print(f"Failed to connect or list files from Supabase: {str(e)}", flush=True)
        return

    if not bucket_objects:
        print("Error: No assets found in your remote Supabase path.", flush=True)
        return

    embeddings_list = []
    id_to_url_mapping = []  

    # Construct base public URL layout
    supabase_base_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET_NAME}/products"

    print(f">>> Found {len(bucket_objects)} assets online. Starting stream-indexing pipeline... <<<", flush=True)

    for index, obj in enumerate(bucket_objects):
        filename = obj['name']
        
        # Skip system placeholders or folders if they show up in the API return array
        if filename == ".emptyFolderPlaceholder" or filename == "":
            continue
            
        try:
            # 2. Generate the absolute public URL for this asset
            supabase_public_url = f"{supabase_base_url}/{filename}"
            
            # 3. Stream the raw image bytes straight into your runtime RAM over HTTP
            img_response = requests.get(supabase_public_url)
            if img_response.status_code != 200:
                print(f"Failed to fetch file binary from storage: {filename}", flush=True)
                continue
                
            img_bytes = img_response.content

            # 4. Extract CLIP features using the downloaded file bytes
            vector = query_hf_embedding(img_bytes, is_image=True)
            if isinstance(vector[0], list):
                vector = np.array(vector).flatten()
                
            embeddings_list.append(vector)
            id_to_url_mapping.append(supabase_public_url)
            
            print(f"Cloud Indexed [{index + 1}/{len(bucket_objects)}]: {filename}", flush=True)
        except Exception as e:
            print(f"Skipping cloud asset {filename} due to pipeline failure: {str(e)}", flush=True)

    if not embeddings_list:
        print("Error: No vectors compiled. Indexing aborted.", flush=True)
        return

    # Initialize a high-performance Inner Product Flat Index for normalized cosine similarity
    embedding_matrix = np.array(embeddings_list).astype('float32')
    vector_dimension = embedding_matrix.shape[1] 
    
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(embedding_matrix)

    # Serialize files to persistent system storage space
    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)
    faiss.write_index(index, index_output_path)
    
    with open(mapping_output_path, "w") as f:
        for url in id_to_url_mapping:
            f.write(f"{url}\n")
            
    print(f"\nSuccess! 512-D local tracking files compiled via live Supabase synchronization stream.", flush=True)


def multimodal_search(query_text: str = None, query_image_path: str = None, top_k: int = 3, index_path=DEFAULT_INDEX, mapping_path=DEFAULT_MAPPING) -> list:
    """Executes a vector search over the local FAISS index file using text or image queries."""
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError("Compiled index artifacts missing. Synchronize with cloud storage first.")

    index = faiss.read_index(index_path)
    with open(mapping_path, "r") as f:
        id_to_url_mapping = [line.strip() for line in f.readlines()]

    query_vector = None

    if query_text:
        vector = query_hf_embedding({"inputs": query_text}, is_image=False)
        if isinstance(vector[0], list):
            vector = vector[0]
        query_vector = np.array([vector]).astype('float32')

    elif query_image_path:
        with open(query_image_path, "rb") as f:
            img_bytes = f.read()
        vector = query_hf_embedding(img_bytes, is_image=True)
        if isinstance(vector[0], list):
            vector = np.array(vector).flatten()
        query_vector = np.array([vector]).astype('float32')

    if query_vector is None:
        return []

    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(id_to_url_mapping):
            results.append({
                "image_path": id_to_url_mapping[idx],  
                "confidence_score": float(score)
            })
            
    return results

if __name__ == "__main__":
    # Triggers cloud synchronization automatically when executed directly
    build_vector_index_from_cloud()