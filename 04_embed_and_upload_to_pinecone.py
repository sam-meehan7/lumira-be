import os
import json
import pinecone
from dotenv import load_dotenv
import pandas as pd

from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm  # for progress bar


load_dotenv()


# get API key from app.pinecone.io and environment from console
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENV']
)

index = pinecone.Index('youtube')

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")


with open('youtube-transcriptions-part-2.jsonl', 'r') as f:
    dataset = [json.loads(line) for line in f]

data = pd.DataFrame(dataset)

batch_size = 100

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i + batch_size)
    # get batch of data
    batch = data.iloc[i:i_end]
    # generate unique ids for each chunk
    ids = [x['id'] for _, x in batch.iterrows()]
    # get text to embed
    texts = [x['text'] for _, x in batch.iterrows()]
    # embed text
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {
            'text': x['text'],
            'url': x['url'],
            'start': x['start'],
            'end': x['end'],
            'title': x['title']
        }
        for _, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

print(index.describe_index_stats())
