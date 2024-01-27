import os
from pinecone import Pinecone as PineconePincone

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as VectorstorePinecone

from dotenv import load_dotenv
import cohere

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
# init client
co = cohere.Client(os.environ["COHERE_API_KEY"])

chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model='gpt-4-0613')

pc = PineconePincone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("donedeal-car-reviews")

embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

text_field = "text"  # the metadata field that contains our text

# pylint: disable=not-callable
vectorstore = VectorstorePinecone(index, embed_model.embed_query, text_field)


def search_vectorstore(query: str):
    results = vectorstore.similarity_search(query, k=25)

    # Convert results to format suitable for Cohere reranking
    documents_to_rerank = [{"text": result.page_content} for result in results]

    # Use Cohere to rerank the documents
    reranked_response = co.rerank(
        query=query, documents=documents_to_rerank, model="rerank-english-v2.0"
    )
    reranked_indices = [result.index for result in reranked_response]

    # Take the top 3 reranked results
    top_3_results = [results[i] for i in reranked_indices[:3]]

    return top_3_results


def augment_prompt(query: str, vector_results):
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in vector_results])
    # feed into an augmented prompt
    augmented_prompt = f"""
    Extra Context:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt


def answer_question(question):
    vector_results = search_vectorstore(question)

    augmented_prompt = augment_prompt(question, vector_results)

    template = "You are a car expert. Use the given context to tell the user why this car is a good fit fot them. Start off by giving them the basics: make, model, spec etc."

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{add_aug_prompt_here}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    result = chat(
        chat_prompt.format_prompt(
            add_aug_prompt_here=augmented_prompt, text=""
        ).to_messages()
    )
    return [result.content, vector_results]
