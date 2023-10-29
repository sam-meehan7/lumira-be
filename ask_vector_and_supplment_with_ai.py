import os
import pinecone
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
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

# get API key from app.pinecone.io and environment from console
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENV']
)

index = pinecone.Index('youtube')

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

text_field = "text"  # the metadata field that contains our text

# pylint: disable=not-callable
vectorstore = Pinecone(index, embed_model.embed_query, text_field)


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


# Useful method to see data compared with and without reranking
def test_ranking(query: str):
    # get results from the vectorstore without reranking
    unranked_results = vectorstore.similarity_search(query, k=15)
    unranked_content = [result.page_content for result in unranked_results]

    # get top 3 reranked results
    ranked_results = search_vectorstore(query)
    ranked_content = [result.page_content for result in ranked_results]

    # Display the content more clearly
    print("===== Unranked Content =====")
    for i, content in enumerate(
        unranked_content[:3], start=1
    ):  # Displaying the top 3 for brevity
        print(f"--- Result {i} ---")
        print(content)
        print("\n")

    print("===== Ranked Content =====")
    for i, content in enumerate(ranked_content, start=1):
        print(f"--- Result {i} ---")
        print(content)
        print("\n")

    return unranked_content, ranked_content


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

    template = "You are a friendly AI who helps answer queries from users. They will usually provide some extra context to help. You should lean into this context to answer the question."

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{add_aug_prompt_here}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # Comment back in to see comparison of ranked vs unranked results
    # test_ranking(question)

    result = chat(
        chat_prompt.format_prompt(
            add_aug_prompt_here=augmented_prompt, text=""
        ).to_messages()
    )
    return [result.content, vector_results]
