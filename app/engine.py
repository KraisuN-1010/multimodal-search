import os
import glob
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_NAME = "openai/clip-vit-base-patch32"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX = os.path.join(BASE_DIR, "data", "vector_index.bin")
DEFAULT_MAPPING = os.path.join(BASE_DIR, "data", "file_mapping.txt")

def get_model_and_processor():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor

def build_vector_index(image_folder_path, index_output_path=DEFAULT_INDEX, mapping_output_path=DEFAULT_MAPPING):
    image_paths = glob.glob(os.path.join(image_folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder_path, "*.jpeg")) + \
                  glob.glob(os.path.join(image_folder_path, "*.png"))
    
    if not image_paths:
        print(f"Error: No images found in local path: {image_folder_path}", flush=True)
        return

    print(">>> Found images! Loading CLIP model to index data... <<<", flush=True)
    model, processor = get_model_and_processor()
    
    embeddings_list = []
    id_to_path_mapping = []

    for index, path in enumerate(image_paths):
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            embeddings_list.append(image_features.numpy().flatten())
            id_to_path_mapping.append(path)
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
    print(f"\nSuccess! Compiled index tracking files saved.", flush=True)


def multimodal_search(query_text=None, query_image_path=None, top_k=3, index_path=DEFAULT_INDEX, mapping_path=DEFAULT_MAPPING):
    """
    Scans the compiled vector database and outputs the top matching image assets/URLs.
    """
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Database index files missing at {index_path}. Run build_vector_index first.")

    index = faiss.read_index(index_path)
    with open(mapping_path, "r") as f:
        id_to_path_mapping = [line.strip() for line in f.readlines()]

    model, processor = get_model_and_processor()
    query_vector = None

    if query_text:
        print(f"Encoding text query: '{query_text}'", flush=True)
        inputs = processor(text=[query_text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        query_vector = text_features.numpy().astype('float32')

    elif query_image_path:
        print(f"Encoding image query asset: '{os.path.basename(query_image_path)}'", flush=True)
        image = Image.open(query_image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        query_vector = image_features.numpy().astype('float32')

    if query_vector is None:
        return []

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
    print(">>> TARGET TRIGGERED: Rebuilding 500-Image Vector Index Database... <<<", flush=True)
    data_img_dir = os.path.join(BASE_DIR, "data", "images")
    build_vector_index(image_folder_path=data_img_dir)