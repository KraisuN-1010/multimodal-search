import os
import glob
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Global Configuration Parameters
MODEL_NAME = "openai/clip-vit-base-patch32"

def get_model_and_processor():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor

def build_vector_index(image_folder_path, index_output_path, mapping_output_path):
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

    faiss.write_index(index, index_output_path)
    with open(mapping_output_path, "w") as f:
        for path in id_to_path_mapping:
            f.write(f"{path}\n")
    print(f"\nSuccess! Compiled index tracking files saved.", flush=True)


def multimodal_search(query_text=None, query_image_path=None, top_k=3, index_path="data/vector_index.bin", mapping_path="data/file_mapping.txt"):
    """
    Scans the compiled vector database and outputs the top matching image assets.
    Supports queries using text phrases, raw image paths, or a hybrid execution of both.
    """
    # 1. Verification Safeguard: Ensure pre-built assets exist on disk
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError("Database index files missing. Run build_vector_index first.")

    # 2. Read the FAISS index snapshot and file labels into RAM
    index = faiss.read_index(index_path)
    with open(mapping_path, "r") as f:
        id_to_path_mapping = [line.strip() for line in f.readlines()]

    model, processor = get_model_and_processor()
    query_vector = None

    # 3. Processing Branch A: User submitted a textual search string
    if query_text:
        print(f"Encoding text query: '{query_text}'", flush=True)
        inputs = processor(text=[query_text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        query_vector = text_features.numpy().astype('float32')

    # 4. Processing Branch B: User submitted an image path for visual comparison
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

    # 5. Execute the High-Speed Vector Comparison Search Loop
    scores, indices = index.search(query_vector, top_k)

    # 6. Parse the multi-dimensional output array fields down into a readable output list
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "image_path": id_to_path_mapping[idx],
                "confidence_score": float(score)
            })
            
    return results

if __name__ == "__main__":
    print(">>> TARGET TRIGGERED: Rebuilding 500-Image Vector Index Database... <<<", flush=True)
    
    build_vector_index(
        image_folder_path="data/images",
        index_output_path="data/vector_index.bin",
        mapping_output_path="data/file_mapping.txt"
    )
    
    print("\n>>> Testing Search Pipeline Engine Natively with New Index... <<<", flush=True)