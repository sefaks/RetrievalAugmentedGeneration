
from fastapi import FastAPI
from app.router import query_router

app = FastAPI(title="Query API", description="An API for querying RAG and OpenAI")

# Router'ı dahil et
app.include_router(query_router.router)
