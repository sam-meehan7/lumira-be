import os
from fastapi import FastAPI, HTTPException, Depends
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
from auth import get_current_user


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
async def upload_video(
    request: VideoUploadRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        success = process_video(request.videoUrl, pc, user_id)
        if success:
            return {"message": "Video processed and indexed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process video")
    except Exception as e:
        logging.error(f"Error in upload_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos", response_model=List[VideoResponse])
async def get_indexed_videos(user_id: str = Depends(get_current_user)):
    try:
        index = pc.Index("videos-index")
        # Filter by user_id
        results = index.query(
            vector=[0]*1536,
            top_k=10000,
            include_metadata=True,
            filter={"user_id": user_id}
        )
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

@app.post("/api/ask")
async def ask_question(
    request: QuestionRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        answer, vector_results = answer_question(request.question, pc, user_id=user_id)
        return AnswerResponse(
            answer=answer,
            vectorResults=[result.metadata for result in vector_results]
        )
    except Exception as e:
        logging.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask/{video_id}")
async def ask_question_about_video(
    video_id: str,
    request: QuestionRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        answer, vector_results = answer_question(
            request.question,
            pc,
            video_id=video_id,
            user_id=user_id
        )
        return AnswerResponse(
            answer=answer,
            vectorResults=[result.metadata for result in vector_results]
        )
    except Exception as e:
        logging.error(f"Error in ask_question_about_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)