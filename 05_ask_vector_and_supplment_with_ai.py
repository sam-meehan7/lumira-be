import os
import pinecone
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import json

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


load_dotenv()

chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')

# get API key from app.pinecone.io and environment from console
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENV']
)

index = pinecone.Index('youtube')

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(index, embed_model.embed_query, text_field)


def augment_prompt(query: str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)

    # save results to json file
    with open('results.json', 'w') as f:
        json.dump([result.__dict__ for result in results], f)

    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt


question = "How big is the engine in the BMW XM?"

augmented_prompt = augment_prompt(question)

template = "You are an expert on cars and talk like a Irish car sales lad. {add_aug_prompt_here}"

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


result = chat(
    chat_prompt.format_prompt(
        add_aug_prompt_here=augmented_prompt, text=""
    ).to_messages()
)

print(result)
