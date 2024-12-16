

from fastapi import APIRouter, HTTPException
from app.schemas.query_schema import QueryRequest, QueryResponse
from app.services.query_service import process_query

router = APIRouter(prefix="/query", tags=["Query"])

@router.post("/", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    RAG tabanlı bir sorgu işlemi yapan endpoint.
    """
    try:
        response, sources = process_query(request.query_text)
        return QueryResponse(response=response, sources=sources)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()  # Konsola tam hata bilgisini yazdırır
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
