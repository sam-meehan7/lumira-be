from pathlib import Path
import json
import whisper
from tqdm.auto import tqdm

# Initialize an empty dictionary to store video details
video_dict = {}

# Read video_dict from the 'data' directory
with open('data/video_dict.json', 'r') as json_file:
    video_dict = json.load(json_file)

# Get a list of MP3 audio files from the 'data' directory
paths = [str(x) for x in Path('./data').glob('*.mp3')]

# Load the whisper model
model = whisper.load_model("base")

# Initialize an empty list to store data
data = []

# Loop through each audio file path
for i, path in enumerate(tqdm(paths)):
    # Extract the video ID from the file name
    _id = Path(path).stem  # Using pathlib's stem method to get the file's root name

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
        data.append(meta)


# Write data to a JSONL file in the 'data' directory
with open("data/youtube-transcriptions.jsonl", "w", encoding="utf-8") as fp:
    for line in tqdm(data):
        json.dump(line, fp)
        fp.write('\n')
