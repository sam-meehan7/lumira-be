from pathlib import Path
import json
from tqdm.auto import tqdm
import os
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
import time

load_dotenv()


# Load video dictionary
def load_video_dict(path):
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"Error reading video_dict.json: {e}")
        return {}


def create_chunks(audio_path, chunk_minutes_length=2):
    file_name = os.path.basename(audio_path)
    base_name, _ = os.path.splitext(file_name)
    chunk_dir = os.path.join("data", base_name)

    print(file_name, base_name, chunk_dir)

    # Ensure the directory exists
    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir)

    chunk_pattern = f"{base_name}_{{}}.mp3"
    audio_list = []

    audio = AudioSegment.from_mp3(audio_path)
    chunk_duration_ms = chunk_minutes_length * 60 * 1000  # Convert to milliseconds

    for i, chunk in enumerate(audio[::chunk_duration_ms]):
        chunk_name = chunk_pattern.format(i)
        chunk_path = os.path.join(chunk_dir, chunk_name)

        if not os.path.exists(chunk_path):
            chunk.export(chunk_path, format="mp3")
        chunk_duration_sec = len(chunk) / 1000  # Duration in seconds
        audio_list.append((chunk_path, chunk_duration_sec))

    return audio_list


def call_openai_api(audio_file):
    MAX_RETRIES = 3
    RETRY_DELAY = 59  # seconds
    for attempt in range(MAX_RETRIES):
        try:
            client = OpenAI()
            with open(audio_file, "rb") as file:
                response = client.audio.transcriptions.create(
                    model="whisper-1", file=file
                )
            return response
        except Exception as e:
            print(f"OpenAI API error on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None
    return None


def is_transcription_completed(chunk_id):
    try:
        with open("data/youtube-transcriptions.jsonl", "r", encoding="utf-8") as fp:
            for line in fp:
                transcription = json.loads(line)
                if transcription.get("id", "") == chunk_id:
                    return True
    except FileNotFoundError:
        # File not found, so no transcriptions have been completed
        return False
    return False


def transcribe(audio_list, video_meta):
    print(f"↪ Chunk size: {len(audio_list)}")
    transcriptions = []
    total_time_elapsed = 0  # Keeps track of the elapsed time

    for audio, duration in audio_list:
        audio_file = Path(audio)
        chunk_id = f"{audio_file.stem}-t{int(total_time_elapsed)}"

        # Check if the transcription for this chunk is already completed
        if is_transcription_completed(chunk_id):
            print(f"\t↪ Skipping already transcribed {audio}...")
            total_time_elapsed += duration
            continue

        print(f"\t↪ Transcribing {audio}...")
        response = call_openai_api(audio)

        if not response:  # Add your error handling here
            print(f"\t↪ Error or no response for {audio}.")
            continue

        text = response.text.strip()

        start_time = int(total_time_elapsed)
        end_time = int(start_time + duration)

        meta = {
            **video_meta,
            "id": chunk_id,
            "text": text,
            "start": start_time,
            "end": end_time,
        }
        transcriptions.append(meta)
        total_time_elapsed += duration

    return transcriptions


def transcribe_audio(audio_path, video_meta, chunk_minutes_length=2):
    try:
        audio_list = create_chunks(audio_path, chunk_minutes_length)
        transcriptions = transcribe(audio_list, video_meta)
        return transcriptions
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        error_title = video_meta.get("title", "Unknown Title")
        return error_title


def main():
    video_dict = load_video_dict("data/video_dict.json")

    with open("data/youtube-transcriptions.jsonl", "w", encoding="utf-8") as fp, open(
        "data/error.txt", "a", encoding="utf-8"
    ) as error_file:
        # Iterate over the keys (video IDs) in the video_dict
        for video_id in tqdm(video_dict.keys()):
            video_meta = video_dict[video_id]
            # Check if the video has already been processed
            if video_meta.get("is_processed"):
                print(f"Skipping already processed video {video_id}")
                continue

            # Construct the path to the mp3 file using the video_id
            mp3_path = f"./data/{video_id}/{video_id}.mp3"
            if not Path(mp3_path).exists():
                error_message = f"File not found for video {video_id}"
                print(error_message)
                error_file.write(f"{error_message}\n")
                continue

            for meta in transcribe_audio(mp3_path, video_meta, 2):
                if isinstance(meta, str):  # It's an error title
                    error_file.write(f"{meta}\n")
                else:
                    json.dump(meta, fp)
                    fp.write("\n")
            # Mark the video as processed
            video_meta["is_processed"] = True

    # Save the updated video_dict with the processed flags
    with open("data/video_dict.json", "w", encoding="utf-8") as f:
        json.dump(video_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
