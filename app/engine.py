# app/engine.py
import os
import glob
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def build_vector_index(image_folder_path, index_output_path, mapping_output_path):
    image_paths = glob.glob(os.path.join(image_folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder_path, "*.jpeg")) + \
                  glob.glob(os.path.join(image_folder_path, "*.png"))
    
    if not image_paths:
        print(f"Error: No images found in local path: {image_folder_path}", flush=True)
        return

    # load the AI models 
    print(">>> Found images! Loading CLIP model and weights from Hugging Face hub... <<<", flush=True)
    MODEL_NAME = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print(">>> Model initialized successfully. Generating embeddings now... <<<", flush=True)
    
    embeddings_list = []
    id_to_path_mapping = []

    # Process each image through CLIP
    for index, path in enumerate(image_paths):
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            embeddings_list.append(image_features.numpy().flatten())
            id_to_path_mapping.append(path)
            print(f"[{index + 1}/{len(image_paths)}] Successfully processed {os.path.basename(path)}", flush=True)
            
        except Exception as e:
            print(f"Failed to process image {path}: {str(e)}", flush=True)

    #  Initialize and Populate the FAISS Index
    embedding_matrix = np.array(embeddings_list).astype('float32')
    vector_dimension = embedding_matrix.shape[1] 
    
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(embedding_matrix)

    # Persist the Index and Mapping files to disk
    faiss.write_index(index, index_output_path)
    
    with open(mapping_output_path, "w") as f:
        for path in id_to_path_mapping:
            f.write(f"{path}\n")

    print(f"\nSuccess! Saved FAISS index to {index_output_path}", flush=True)
    print(f"Saved tracking mapping file to {mapping_output_path}", flush=True)

if __name__ == "__main__":
    print(">>> TARGET TRIGGERED: Starting offline search data workflow pipeline... <<<", flush=True)
    build_vector_index(
        image_folder_path="data/images",
        index_output_path="data/vector_index.bin",
        mapping_output_path="data/file_mapping.txt"
    )