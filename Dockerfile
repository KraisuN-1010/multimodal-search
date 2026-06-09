FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for FAISS and image processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Expose port 8000 for standard Web REST API traffic
EXPOSE 8000

# Set environment variable to make sure module imports map perfectly
ENV PYTHONPATH=/app

# Command to run the high-performance FastAPI production server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]