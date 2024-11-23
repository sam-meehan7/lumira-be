import os
import logging
from pytube import YouTube
from pytube.exceptions import PytubeError
import moviepy.editor as mp
from openai import OpenAI
from pydub import AudioSegment
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from pydub import AudioSegment
import yt_dlp
from pinecone import Pinecone, ServerlessSpec
from db import insert_video, get_user_content_hours
from fastapi import HTTPException



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize OpenAI Embeddings
try:
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI Embeddings: {str(e)}")
    raise

def download_audio(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'data/%(id)s/%(id)s.%(ext)s',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info['id']
            audio_path = f"data/{video_id}/{video_id}.mp3"
            return audio_path, info
    except Exception as e:
        logging.error(f"Error occurred while downloading audio: {str(e)}")
        return None, None

def transcribe_audio(audio_path):
    try:
        chunk_seconds_length = 30
        audio = AudioSegment.from_mp3(audio_path)
        chunk_duration_ms = chunk_seconds_length * 1000
        total_duration_ms = len(audio)
        transcriptions = []

        for start_ms in range(0, total_duration_ms, chunk_duration_ms):
            end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
            chunk = audio[start_ms:end_ms]
            chunk_path = f"{audio_path}_{start_ms // chunk_duration_ms}.mp3"
            chunk.export(chunk_path, format="mp3")

            response = call_openai_api(chunk_path)
            if response:
                transcriptions.append({
                    "start": start_ms // 1000,
                    "end": end_ms // 1000,
                    "text": response.text.strip()
                })
            os.remove(chunk_path)  # Clean up temporary chunk file
        return transcriptions
    except Exception as e:
        logging.error(f"Error occurred while transcribing audio: {str(e)}")
        return []

def call_openai_api(audio_file):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open(audio_file, "rb") as file:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=file)
        return transcript
    except Exception as e:
        logging.error(f"Error occurred while calling OpenAI API: {str(e)}")
        return None

def upload_to_pinecone(transcriptions, metadata, pc, user_id):
    try:
        index_name = "videos-index"
        index = pc.Index(index_name, spec=ServerlessSpec(cloud="aws", region="eu-west-1"))

        window = 2
        stride = 1

        combined_transcriptions = []
        for i in range(0, len(transcriptions), stride):
            i_end = min(len(transcriptions) - 1, i + window)
            text = ' '.join(item['text'] for item in transcriptions[i:i_end+1])
            combined_transcriptions.append({
                "start": transcriptions[i]["start"],
                "end": transcriptions[i_end]["end"],
                "text": text
            })

        unique_ids = [f"{metadata['video_id']}_{i}" for i in range(len(combined_transcriptions))]
        texts = [t["text"] for t in combined_transcriptions]
        embeddings = embed_model.embed_documents(texts)

        pinecone_metadata = [{
            "text": t["text"],
            "start": t["start"],
            "end": t["end"],
            "user_id": user_id,
            **metadata
        } for t in combined_transcriptions]

        index.upsert(vectors=zip(unique_ids, embeddings, pinecone_metadata))
        logging.info(f"Successfully uploaded {len(combined_transcriptions)} transcriptions to Pinecone")
    except Exception as e:
        logging.error(f"Error occurred while uploading to Pinecone: {str(e)}")
        raise

async def check_content_hours_limit(user_id, video_duration):
    """Check if adding a new video would exceed the user's content limit."""
    SECONDS_LIMIT = 300  # 1 hour in seconds
    logging.info(f"Checking content limit for user {user_id}")

    current_seconds = await get_user_content_hours(user_id)
    current_seconds = current_seconds * 3600

    if current_seconds + video_duration > SECONDS_LIMIT:
        logging.warning(
            f"Content limit exceeded for user {user_id}. "
            f"Current: {current_seconds} seconds, Attempted to add: {video_duration} seconds"
        )
        raise HTTPException(
            status_code=402,  # Payment Required
            detail={
                "error": "content_limit_reached",
                "message": "Content limit reached",
                "current_usage": current_seconds,
                "attempted_add": video_duration,
                "limit": SECONDS_LIMIT,
                "upgrade_required": True
            }
        )

    logging.info(f"Content limit check passed for user {user_id}")
    return True

async def process_video(video_url, pc, user_id):
    try:
        logging.info(f"Starting video processing for URL: {video_url}, User: {user_id}")

        # Get video info without downloading
        logging.info("Fetching video information...")
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': False,  # Changed to False to see yt-dlp logs
            'no_warnings': False,  # Changed to False to see warnings
            'extract_flat': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            logging.info(f"Video information retrieved: {info['title']}")

        # Check content hours limit
        await check_content_hours_limit(user_id, info['duration'])

        # Continue with video processing
        logging.info("Downloading audio...")
        audio_path, info = download_audio(video_url)
        if not audio_path or not info:
            logging.error("Failed to download audio")
            return False

        logging.info(f"Audio downloaded successfully to: {audio_path}")
        logging.info("Transcribing audio...")
        transcriptions = transcribe_audio(audio_path)
        if not transcriptions:
            logging.error("Failed to transcribe audio")
            return False

        logging.info(f"Successfully transcribed {len(transcriptions)} segments")

        # Save to database
        logging.info("Saving video information to database...")
        await insert_video(user_id, {**info, 'url': video_url})

        # Upload to Pinecone
        logging.info("Preparing metadata for Pinecone upload...")
        metadata = {
            "video_id": info['id'],
            "title": info['title'],
            "url": video_url,
            "thumbnail": info['thumbnail'],
            "author": info['uploader'],
            "channel_id": info['channel_id'],
            "channel_url": info['channel_url'],
            "duration": info['duration']
        }

        logging.info("Uploading transcriptions to Pinecone...")
        upload_to_pinecone(transcriptions, metadata, pc, user_id)

        logging.info(f"Successfully processed video: {info['title']}")
        return True

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise