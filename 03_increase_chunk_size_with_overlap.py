import json
from tqdm import tqdm

new_data = []

window = 6  # number of sentences to combine
stride = 3  # number of sentences to 'stride' over, used to create overlap

# Open the JSONL file
data = []
with open('youtube-transcriptions.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

for i in tqdm(range(0, len(data), stride)):
    i_end = min(len(data) - 1, i + window)
    if data[i]['title'] != data[i_end]['title']:
        # in this case we skip this entry as we have start/end of two videos
        continue
    text = ' '.join(item['text'] for item in data[i:i_end])
    new_data.append(
        {
            'start': data[i]['start'],
            'end': data[i_end]['end'],
            'text': text,
            'id': data[i]['id'],
            'url': data[i]['url'],
        }
    )

# Save to new JSONL file
with open("youtube-transcriptions-part-2.jsonl", "w", encoding="utf-8") as fp:
    for line in tqdm(new_data):
        json.dump(line, fp)
        fp.write('\n')
