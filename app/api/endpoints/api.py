from fastapi import APIRouter

from app.api.endpoints.v1 import docs_router


api_router = APIRouter()
api_router.include_router(docs_router, prefix="/v1", tags=["documents"])
