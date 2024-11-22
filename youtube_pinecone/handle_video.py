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

def process_video(video_url, pc, user_id):
    try:
        audio_path, info = download_audio(video_url)
        if audio_path:
            transcriptions = transcribe_audio(audio_path)
            if transcriptions:
                metadata = {
                    "video_id": info['id'],
                    "title": info['title'],
                    "url": video_url,
                    "thumbnail": info['thumbnail'],
                    "author": info['uploader'],
                    "channel_id": info['channel_id'],
                    "channel_url": info['channel_url']
                }
                upload_to_pinecone(transcriptions, metadata, pc, user_id)
                logging.info(f"Successfully processed video: {video_url}")
                return True
            else:
                logging.warning(f"No transcriptions generated for video: {video_url}")
                return False
        else:
            logging.warning(f"Failed to download audio for video: {video_url}")
            return False
    except Exception as e:
        logging.error(f"Error occurred while processing video: {str(e)}")
        return False