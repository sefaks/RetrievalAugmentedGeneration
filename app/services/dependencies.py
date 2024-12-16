

from langchain_chroma import Chroma
import os
import dotenv
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Örnek bir sabit PROMPT_TEMPLATE
PROMPT_TEMPLATE = """
Bir ilkokul öğrencisine anlatacak şekilde basitçe soruya cevap vermeni istiyorum. Ona basit ve anlamsal şekilde anlatabilirsin. 
Aşağıda sorunun cevabının içeriği yer almaktadır. Ona göre cevap ver.:
{context}
---
Cevap vereceğin soru bu şekilde, bu soruya yukarıdaki bilgileri kullanarak cevap ver: {question}
"""

api_key = os.getenv("openai_key")
api_version = os.getenv("openai_version")
azure_endpoint =  os.getenv("openai_endpoint")
deployment = os.getenv("openai_model")

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    azure_deployment=deployment
    
)


def query_openai(prompt):
    # OpenAI API çağrısı burada yapılır
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7,
        top_p=1,
        n=1
    )
    return response.choices[0].message.content.strip()
