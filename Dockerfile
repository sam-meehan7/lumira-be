FROM python:3.12.1-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
ENV PORT=8000
EXPOSE $PORT

# Start the application
CMD uvicorn youtube_pinecone.api:app --host 0.0.0.0 --port $PORT