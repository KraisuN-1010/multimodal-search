import os
import io
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from app.engine import multimodal_search

app = FastAPI(
    title="Tied Up with Creativity - Multimodal Search API",
    description="CLIP + FAISS Vector search engine backend microservice."
)

# Enable CORS so React production frontend can securely query this API
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
    return {"status": "online", "message": "Search infrastructure engine fully initialized."}

@app.post("/search/text")
def search_by_text(payload: TextQueryRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Search phrase cannot be empty.")
    try:
        results = multimodal_search(query_text=payload.text, top_k=payload.top_k)
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(3)):
    extension = file.filename.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png"]:
        raise HTTPException(status_code=400, detail="Invalid image layout type. Only JPG, JPEG, and PNG accepted.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        temp_filename = f"temp_upload_{file.filename}"
        image.save(temp_filename)
        
        results = multimodal_search(query_image_path=temp_filename, top_k=top_k)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))