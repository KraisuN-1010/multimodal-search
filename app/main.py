import os
import io
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from app.engine import multimodal_search, build_vector_index_from_cloud

app = FastAPI(
    title="Tied Up with Creativity - Multimodal Search API",
    description="Optimized CLIP + FAISS Vector search engine backend microservice."
)

# CORS Middleware Configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextQueryRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3

@app.get("/health")
def health_check():
    """
    Validates infrastructure availability and server connectivity status.
    """
    return {"status": "online", "message": "Search infrastructure engine fully initialized."}

@app.post("/sync")
def synchronize_database_index():
    """
    Administrative Endpoint: Connects to Supabase, streams all bucket images into 
    transient RAM, extracts CLIP vectors, and compiles the localized FAISS index.
    """
    try:
        build_vector_index_from_cloud()
        return {"success": True, "message": "FAISS vector index and cloud file mappings compiled successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synchronization failed: {str(e)}")

@app.post("/search/text")
def search_by_text(payload: TextQueryRequest):
    """
    Processes conceptual text queries to fetch matching items from the Supabase vector index.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Search phrase cannot be empty.")
    try:
        results = multimodal_search(query_text=payload.text, top_k=payload.top_k)
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(3)):
    """
    Processes multi-part image uploads to discover visually similar catalog assets.
    """
    extension = file.filename.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png"]:
        raise HTTPException(status_code=400, detail="Invalid layout type. Only JPG, JPEG, and PNG accepted.")
        
    try:
        # Load raw file binary directly to avoid memory overflows
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Save a localized transient copy for engine preprocessing
        temp_filename = f"temp_upload_{file.filename}"
        image.save(temp_filename)
        
        results = multimodal_search(query_image_path=temp_filename, top_k=top_k)
        
        # Cleanup storage artifact post-execution
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))