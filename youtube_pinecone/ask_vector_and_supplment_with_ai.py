import os
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as VectorstorePinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import logging

load_dotenv()

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
# init client
co = cohere.Client(os.environ["COHERE_API_KEY"])
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model='gpt-4o', temperature=0.1, max_tokens=3500)

embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

text_field = "text"  # the metadata field that contains our text

def initialize_vectorstore(pc):
    index = pc.Index("videos-index", spec=ServerlessSpec(cloud="aws", region="eu-west-1"))
    return PineconeVectorStore(index, embed_model, text_field)

def search_vectorstore(query: str, vectorstore, video_id=None):
    logging.info(f"Searching vectorstore for query: {query}")
    logging.info(f"Video ID filter: {video_id}")

    if video_id:
        filter = {"video_id": video_id}
        results = vectorstore.similarity_search(query, k=25, filter=filter)
    else:
        results = vectorstore.similarity_search(query, k=25)

    logging.info(f"Number of results before reranking: {len(results)}")
    for i, result in enumerate(results[:5]):  # Log details of the first 5 results
        logging.info(f"Result {i+1}: Video ID: {result.metadata['video_id']}, Title: {result.metadata['title']}, Score: {result.score if hasattr(result, 'score') else 'N/A'}")

    documents_to_rerank = [{"text": result.page_content} for result in results]
    reranked_response = co.rerank(query=query, documents=documents_to_rerank, model="rerank-english-v2.0")
    reranked_indices = [result.index for result in reranked_response]
    top_3_results = [results[i] for i in reranked_indices[:3]]

    # Sort the top 3 results by start time
    sorted_top_3_results = sorted(top_3_results, key=lambda x: float(x.metadata['start']))

    logging.info(f"Top 3 video IDs after reranking: {[result.metadata['video_id'] for result in sorted_top_3_results]}")

    return sorted_top_3_results

def augment_prompt(query: str, vector_results):
    source_knowledge = "\n\n".join([f"{x.metadata['title']}\n{x.page_content}" for x in vector_results])
    augmented_prompt = f"""
    # CONTEXT
    {source_knowledge}

    # USER QUESTION
    {query}
    """
    return augmented_prompt

def answer_question(question, pc, video_id=None):
    vectorstore = initialize_vectorstore(pc)
    vector_results = search_vectorstore(question, vectorstore, video_id=video_id)
    augmented_prompt = augment_prompt(question, vector_results)


    template = """
    # MISSION
    You are an AI assistant specializing in answering questions based on video content. Your primary goal is to provide accurate, concise, and relevant information from video transcriptions.

    # INSTRUCTIONS
    1. Analyze the given context, which contains transcriptions from the top 3 most relevant videos.
    2. Focus on directly answering the user's question using information from these transcriptions.
    3. If the context doesn't contain enough information to fully answer the question, clearly state what you can answer based on the available information.
    4. Provide timestamps or video titles when referencing specific information, if available in the context.

    # OUTPUT FORMAT
    1. Begin with a concise answer to the user's question.
    2. Follow with supporting details and explanations from the video content.
    3. If relevant, include a brief summary of key points at the end.

    # RULES
    1. Never invent or assume information not present in the given context.
    2. If the question cannot be answered using the provided context, clearly state this and explain why.
    3. Maintain a neutral, informative tone throughout your response.
    4. Use clear and simple language to ensure accessibility for a wide audience.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{add_aug_prompt_here}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    formatted_messages = chat_prompt.format_prompt(add_aug_prompt_here=augmented_prompt, text="").to_messages()
    result = chat.invoke(formatted_messages)
    return [result.content, vector_results]