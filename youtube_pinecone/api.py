import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as VectorstorePinecone
from ask_vector_and_supplment_with_ai import answer_question
from handle_video import process_video
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002"],  # Update this with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize embeddings model
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Pydantic models for request/response
class VideoUploadRequest(BaseModel):
    videoUrl: str

class QuestionRequest(BaseModel):
    question: str

class VideoResponse(BaseModel):
    video_id: str
    title: str
    url: str
    thumbnail: str
    author: str

class AnswerResponse(BaseModel):
    answer: str
    vectorResults: List[dict]

# API endpoints

@app.post("/api/upload")
async def upload_video(request: VideoUploadRequest):
    try:
        success = process_video(request.videoUrl, pc)
        if success:
            return {"message": "Video processed and indexed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process video")
    except Exception as e:
        logging.error(f"Error in upload_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    try:
        logging.info(f"Received question: {request.question}")
        answer, vector_results = answer_question(request.question, pc, video_id=None)
        logging.info(f"Answer: {answer[:100]}...")  # Log the first 100 characters of the answer
        logging.info(f"Vector results: {[result.metadata['video_id'] for result in vector_results]}")
        logging.info(f"Number of vector results: {len(vector_results)}")
        for i, result in enumerate(vector_results):
            logging.info(f"Result {i+1}: Video ID: {result.metadata['video_id']}, Title: {result.metadata['title']}")
        return AnswerResponse(answer=answer, vectorResults=[result.metadata for result in vector_results])
    except Exception as e:
        logging.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos", response_model=List[VideoResponse])
async def get_indexed_videos():
    logging.info("Received request to /api/videos endpoint")  # Add this line
    try:
        index = pc.Index("videos-index")
        results = index.query(vector=[0]*1536, top_k=10000, include_metadata=True)
        videos = {}
        for match in results['matches']:
            video_id = match['metadata']['video_id']
            if video_id not in videos:
                videos[video_id] = VideoResponse(
                    video_id=video_id,
                    title=match['metadata']['title'],
                    url=match['metadata']['url'],
                    thumbnail=match['metadata']['thumbnail'],
                    author=match['metadata']['author']
                )
        return list(videos.values())
    except Exception as e:
        logging.error(f"Error in get_indexed_videos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask/{video_id}")
async def ask_question_about_video(video_id: str, request: QuestionRequest):
    try:
        answer, vector_results = answer_question(request.question, pc, video_id)
        return AnswerResponse(answer=answer, vectorResults=[result.metadata for result in vector_results])
    except Exception as e:
        logging.error(f"Error in ask_question_about_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/index-stats")
async def get_index_stats():
    try:
        index = pc.Index("videos-index")
        stats = index.describe_index_stats()
        unique_videos = set()
        sample_vectors = index.query(vector=[0]*1536, top_k=10, include_metadata=True)

        # Extract only the necessary information from sample vectors
        simplified_samples = []
        for match in sample_vectors['matches']:
            unique_videos.add(match['metadata']['video_id'])
            simplified_samples.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': {
                    'video_id': match['metadata']['video_id'],
                    'title': match['metadata']['title'],
                    'url': match['metadata']['url'],
                    'start': match['metadata']['start'],
                    'end': match['metadata']['end'],
                }
            })

        return {
            "total_vector_count": stats['total_vector_count'],
            "dimension": stats['dimension'],
            "unique_videos": list(unique_videos),
            "sample_vectors": simplified_samples
        }
    except Exception as e:
        logging.error(f"Error in get_index_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)