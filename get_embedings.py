import openai
import os
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings  # Güncellenmiş import
from dotenv import load_dotenv


load_dotenv()

api_key =os.getenv("openai_key")
deployment = os.getenv("embedding_deployment")
version = os.getenv("embedding_api_version")
endpoint = os.getenv("openai_endpoint")

def get_embeddings():  
    try:  
        embeddings = AzureOpenAIEmbeddings(  
            api_key=api_key,  
            model=deployment,  
            api_version=version,  
            azure_endpoint=endpoint  
        )  
        return embeddings  
    except Exception as e:  
        print(f"Hata oluştu: {e}")  
        return None 