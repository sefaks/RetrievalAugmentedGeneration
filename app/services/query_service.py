


from langchain_chroma import Chroma
from get_embedings import get_embeddings
from langchain_core.prompts import ChatPromptTemplate
from app.services.dependencies import query_openai

from query import PROMPT_TEMPLATE


CHROMA_PATH = "/Users/ysefakose/Desktop/LLM/Rag/data/chroma"  

def process_query(query_text: str):
    """
    Gelen sorguyu işleyen ve sonuç dönen ana fonksiyon.
    """
    # Embedding işlevini al
    embedding_function = get_embeddings()
    # Chroma DB'yi yükle
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Benzer dokümanları ara
    results = db.similarity_search_with_score(query_text, k=5)
    if not results:
        raise ValueError("No relevant documents found.")
    
    # Bağlamı oluştur
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Prompt hazırlama
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # OpenAI yanıtı al
    response_text = query_openai(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    return response_text, sources
