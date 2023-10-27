from pathlib import Path
import json
import whisper
from tqdm.auto import tqdm

# Initialize an empty dictionary to store video details
video_dict = {}

try:
    # Read video_dict from the 'data' directory
    with open('data/video_dict.json', 'r', encoding='utf-8') as json_file:
        video_dict = json.load(json_file)
except Exception as e:
    print(f"Error reading video_dict.json: {e}")

# Get a list of MP3 audio files from the 'data' directory
paths = [str(x) for x in Path('./data').rglob('*.mp3')]

try:
    # Load the whisper model
    model = whisper.load_model("base")
except Exception as e:
    print(f"Error loading whisper model: {e}")
    exit()

# Open the JSONL file in write mode
with open("data/youtube-transcriptions.jsonl", "w", encoding="utf-8") as fp:
    # Loop through each audio file path
    for i, path in enumerate(tqdm(paths)):
        try:
            # Extract the video ID from the file name
            _id = Path(
                path
            ).stem  # Using pathlib's stem method to get the file's root name

            # Transcribe to get speech-to-text data
            result = model.transcribe(path)
            segments = result['segments']

            # Get the video metadata
            video_meta = video_dict.get(_id, {})

            # Loop through each segment and merge metadata
            for j, segment in enumerate(segments):
                meta = {
                    **video_meta,
                    "id": f"{_id}-t{segments[j]['start']}",
                    "text": segment["text"].strip(),
                    "start": segment['start'],
                    "end": segment['end'],
                }
                # Write to the JSONL file immediately
                json.dump(meta, fp)
                fp.write('\n')
        except Exception as e:
            print(f"Error processing file {path}: {e}")
            # Write the video title to an error.txt file
            with open("data/error.txt", "a", encoding="utf-8") as error_file:
                error_title = video_meta.get("title", "Unknown Title")
                error_file.write(f"{error_title}\n")
