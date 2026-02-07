from contextlib import asynccontextmanager
from fastapi import FastAPI
from paddleocr import PaddleOCR
import easyocr

from app.core.config import GPU
from app.core.logging import logger
from app.api.endpoints.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup (OCR engine loading) and shutdown events."""
    
    logger.info("Loading PaddleOCR engine...")
    app.state.paddle_engine = PaddleOCR(use_angle_cls=False, lang="ar")

    logger.info("Loading EasyOCR engine...")
    app.state.easy_engine = easyocr.Reader(["ar", "en"], gpu=GPU)

    logger.info("All AI models ready")
    yield


app = FastAPI(
    title="OCR Infrastructure",
    description="Auto-detects language (Arabic/English) and routes to the optimal OCR engine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api")
