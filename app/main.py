from contextlib import asynccontextmanager
from fastapi import FastAPI
from paddleocr import PaddleOCR
import easyocr

from app.core.broker import RabbitMQBroker
from app.core.config import settings
from app.core.logging import logger
from app.api.endpoints.api import api_router
from app.core.storage import BlobDownloader
from app.services.worker import create_message_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup (OCR engine loading) and shutdown events."""

    logger.info("Loading PaddleOCR engine...")
    app.state.paddle_engine = PaddleOCR(use_angle_cls=False, lang="ar")

    logger.info("Loading EasyOCR engine...")
    app.state.easy_engine = easyocr.Reader(["ar", "en"], gpu=settings.GPU)

    logger.info("All AI models ready")

    logger.info("Connecting to message broker...")
    broker = RabbitMQBroker(settings.MESSAGE_BROKER_URL)
    await broker.connect()
    app.state.broker = broker
    logger.info("Message broker connected")

    blob = BlobDownloader(
        conn_str=settings.BLOB_CONNECTION_STR,
        container=settings.BLOB_STORAGE_CONTAINER_NAME,
    )
    app.state.blob = blob

    handler = create_message_handler(
        paddle_engine=app.state.paddle_engine,
        easy_engine=app.state.easy_engine,
        blob=blob,
    )

    logger.info(f"Starting consumer on queue: {settings.OCR_QUEUE_NAME}")
    await broker.consume(settings.OCR_QUEUE_NAME, handler)
    logger.info("OCR worker is now listening for messages")

    yield

    logger.info("Shutting down OCR worker...")
    await broker.close()
    await blob.close()
    logger.info("OCR worker shut down cleanly")


app = FastAPI(
    title="OCR Infrastructure",
    description="Auto-detects language (Arabic/English) and routes to the optimal OCR engine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api")


@app.get("/", status_code=204)
async def health():
    """Health check endpoint for Docker."""
    return
