import whisper
from tqdm.auto import tqdm
from pathlib import Path
import json
import json


video_dict = {}
with open('video_dict.json', 'r') as json_file:
    video_dict = json.load(json_file)


# get list of MP3 audio files
paths = [str(x) for x in Path('./').glob('*.mp3')]
model = whisper.load_model("base")
data = []

for i, path in enumerate(tqdm(paths)):
    _id = path.split('/')[-1][:-4]
    # transcribe to get speech-to-text data
    result = model.transcribe(path)
    segments = result['segments']
    # get the video metadata...
    video_meta = video_dict[_id]
    for j, segment in enumerate(segments):
        # merge segments data and videos_meta data
        meta = {
            **video_meta,
            **{
                "id": f"{_id}-t{segments[j]['start']}",
                "text": segment["text"].strip(),
                "start": segment['start'],
                "end": segment['end'],
            },
        }
        data.append(meta)

with open("youtube-transcriptions.jsonl", "w", encoding="utf-8") as fp:
    for line in tqdm(data):
        json.dump(line, fp)
        fp.write('\n')
