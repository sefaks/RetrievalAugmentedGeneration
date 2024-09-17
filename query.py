import argparse
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from openai import AzureOpenAI

from get_embedings import get_embeddings
from langchain_core.prompts import ChatPromptTemplate



CHROMA_PATH = "data/chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""

load_dotenv()

api_key = os.getenv("openai_key")
api_version = os.getenv("openai_version")
azure_endpoint =  os.getenv("openai_endpoint")
deployment = os.getenv("openai_model")

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)

def query_openai(prompt):
    
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
        top_p=1,
        n=1
    )
    return response.choices[0].message.content.strip()

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    embedding_function = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db.similarity_search_with_score(query_text, k=5)
    if not results:
        print("No relevant documents found.")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response_text = query_openai(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()