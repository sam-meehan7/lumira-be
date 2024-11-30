import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "Missing Supabase credentials. "
        "Please ensure SUPABASE_URL and SUPABASE_SERVICE_KEY "
        "are set in your .env file"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def insert_video(user_id: str, video_info: dict):
    try:
        # First, prepare and insert/update the video record
        video_data = {
            "video_id": video_info['id'],
            "title": video_info['title'],
            "duration_seconds": video_info['duration'],
            "url": video_info['url'],
            "thumbnail": video_info.get('thumbnail'),
            "author": video_info.get('uploader'),
            "channel_id": video_info.get('channel_id'),
            "channel_url": video_info.get('channel_url')
        }

        # Upsert the video record
        video_result = supabase.table('videos')\
            .upsert(video_data, on_conflict='video_id')\
            .execute()

        # Then create the user-video association
        user_video_data = {
            "user_id": user_id,
            "video_id": video_info['id']
        }

        # Insert the user-video link
        user_video_result = supabase.table('user_videos')\
            .upsert(user_video_data, on_conflict='user_id,video_id')\
            .execute()

        return video_result.data[0]
    except Exception as e:
        logger.error(f"Failed to insert video: {str(e)}")
        raise Exception(f"Failed to insert video: {str(e)}")

async def get_user_content_hours(user_id: str) -> float:
    try:
        result = supabase.table('user_videos')\
            .select('duration_seconds')\
            .eq('user_id', user_id)\
            .execute()

        total_seconds = sum(row['duration_seconds'] for row in result.data)
        return total_seconds / 3600  # Convert to hours
    except Exception as e:
        logger.error(f"Failed to get user content hours: {str(e)}")
        raise Exception(f"Failed to get user content hours: {str(e)}")

async def set_payment_preference(user_id: str, willing_to_pay: bool):
    try:
        data = {
            "user_id": user_id,
            "willing_to_pay": willing_to_pay
        }

        # Use upsert with explicit on_conflict parameter
        result = supabase.table('user_payment_preferences')\
            .upsert(
                data,
                on_conflict='user_id'  # Specify the column to check for conflicts
            )\
            .execute()
        return result.data[0]
    except Exception as e:
        logger.error(f"Failed to set payment preference: {str(e)}")
        raise Exception(f"Failed to set payment preference: {str(e)}")

async def get_user_video_ids(user_id: str) -> List[str]:
    try:
        result = supabase.table('user_videos')\
            .select('video_id')\
            .eq('user_id', user_id)\
            .execute()
        return [row['video_id'] for row in result.data]
    except Exception as e:
        logger.error(f"Failed to get user video IDs: {str(e)}")
        raise Exception(f"Failed to get user video IDs: {str(e)}")