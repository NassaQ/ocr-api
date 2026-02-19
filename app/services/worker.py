import asyncio
from datetime import datetime, timezone
import json
import os
from typing import Callable
import uuid

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from app.core.storage import BlobDownloader
from app.core.logging import logger
from app.db.session import AsyncSessionLocal
from app.models.models import Documents, ProcessingStatus
from app.core.config import settings
from app.api.deps import get_timestamp


SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
SUPPORTED_PDF_TYPES = {".pdf"}
SUPPORTED_TEXT_TYPES = {".txt"}

def get_file_extension(filename: str) -> str:
    """Extract lowercase file extension."""
    _, ext = os.path.splitext(filename)
    return ext.lower()



async def db_operation_with_retry(operation: Callable, *args, **kwargs):
    """
    Execute an async DB operation with exponential backoff retry.
    Catches sqlalchemy OperationalError (connection resets, timeouts)
    and retries up to SQL_MAX_RETRIES times before re-raising.
    """
    last_exception = None

    for attempt in range(1, settings.SQL_MAX_RETRIES + 1):
        try:
            return await operation(*args, **kwargs)
        except OperationalError as e:
            last_exception = e
            if attempt < settings.SQL_MAX_RETRIES:
                delay = settings.SQL_RETRY_DELAY_BASE**attempt
                logger.warning(
                    f"DB operation failed (attempt {attempt}/{settings.SQL_MAX_RETRIES}). "
                    f"Retrying in {delay}s... Error: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"DB operation failed after {settings.SQL_MAX_RETRIES} attempts. Error: {e}"
                )

    raise last_exception

async def _update_status_inner(doc_id: int, status: str, error_message: str | None = None):
    """Core DB logic for updating ProcessingStatus — called via retry wrapper."""

    async with AsyncSessionLocal() as session:
        query = select(ProcessingStatus).where(
            ProcessingStatus.doc_id == doc_id,
            ProcessingStatus.stage_name == "OCR",
        )
        record = (await session.execute(query)).scalar_one_or_none()

        if not record:
            logger.error(f"No ProcessingStatus record found for doc_id={doc_id}")
            return

        record.status = status

        if status == "Processing":
            record.start_time = datetime.now(timezone.utc)
        elif status in ("Finished", "Failed"):
            record.end_time = datetime.now(timezone.utc)

        if error_message:
            record.error_message = error_message

        await session.commit()

async def update_status(doc_id: int, status: str, error_message: str | None = None):
    """Update the ProcessingStatus record for a document, with retry on connection errors."""
    await db_operation_with_retry(_update_status_inner, doc_id, status, error_message)

async def _update_mongo_doc_id_inner(doc_id: int, mongo_doc_id: str):
    """Core DB logic for updating Documents.mongo_doc_id — called via retry wrapper."""
    async with AsyncSessionLocal() as session:
        query = select(Documents).where(Documents.doc_id == doc_id)
        doc = (await session.execute(query)).scalar_one_or_none()

        if doc:
            doc.mongo_doc_id = mongo_doc_id
            await session.commit()

async def update_mongo_doc_id(doc_id: int, mongo_doc_id: str):
    """Update the Documents.mongo_doc_id placeholder after processing, with retry on connection errors."""
    await db_operation_with_retry(_update_mongo_doc_id_inner, doc_id, mongo_doc_id)


async def process_document(message: dict, paddle_engine, easy_engine, blob: BlobDownloader):
    """
    Main processing function called for each message from the queue.
    Downloads the file, runs OCR, saves output locally, updates DB status.
    """

    doc_id = message["doc_id"]
    file_path = message["file_path"]
    filename = message["filename"]

    logger.info(f"Processing doc_id={doc_id}, filename={filename}")

    await update_status(doc_id, "Processing")

    try:
        file_content = await blob.download(file_path)
        logger.info(
            f"Downloaded {len(file_content)} bytes from blob for doc_id={doc_id}"
        )

        ext = get_file_extension(filename)
        _filename = filename.replace(" ", "_")
        batch_id = get_timestamp()

        extracted_text = ""
        file_metadata = {
            "original_filename": filename,
            "file_type": ext,
            "upload_timestamp": batch_id,
            "model_usage_log": [],
            "page_count": 1,
            "source_file_path": "",
            "text_file_path": "",
            "extraction_details": [],
            "status": "success",
            "overall_confidence": 1.0,
            "error_message": None,
        }

        original_save_name = f"{batch_id}_SOURCE_{_filename}"
        original_file_path = os.path.join(settings.OUTPUT_DIR, original_save_name)
        with open(original_file_path, "wb") as f:
            f.write(file_content)
        file_metadata["source_file_path"] = original_file_path

        if ext in SUPPORTED_TEXT_TYPES:
            extracted_text, detail = process_text_file(file_content)
            file_metadata["extraction_details"].append(detail)
            file_metadata["model_usage_log"].append("None (Text File)")

        elif ext in SUPPORTED_PDF_TYPES:
            extracted_text, pdf_meta = process_pdf(file_content, paddle_engine, easy_engine)
            file_metadata.update(pdf_meta)

        elif ext in SUPPORTED_IMAGE_TYPES:
            extracted_text, detail = process_image(file_content, paddle_engine, easy_engine)
            file_metadata["extraction_details"].append(detail)
            file_metadata["model_usage_log"].append(detail["method"])
            file_metadata["overall_confidence"] = detail["confidence"]

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # TODO: move output to mongo db
        text_filename = f"{batch_id}_TARGET_{_filename}.txt"
        text_file_path = os.path.join(settings.OUTPUT_DIR, text_filename)
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        file_metadata["text_file_path"] = text_file_path

        # TODO: move it to mongo db
        details_filename = f"Details_{batch_id}_{_filename}.json"
        details_path = os.path.join(settings.OUTPUT_DIR, details_filename)
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(file_metadata, f, ensure_ascii=False, indent=4)

        await update_status(doc_id, "Finished")
        placeholder_id = str(uuid.uuid4())
        await update_mongo_doc_id(doc_id, placeholder_id)

        logger.info(f"Finished processing doc_id={doc_id}, output at {text_file_path}")

        # TODO: remove the processed file from local storage

    except Exception as e:
        logger.error(f"Failed to process doc_id={doc_id}: {e}")
        await update_status(doc_id, "Failed", error_message=str(e))
        raise

def process_text_file(file_content: bytes) -> tuple[str, dict]:
    """Process a plain text file — direct read, no OCR needed."""
    text = file_content.decode("utf-8")
    metadata = {
        "page": 1,
        "method": "Direct Read",
        "confidence": 1.0,
    }
    return text, metadata

def create_message_handler(paddle_engine, easy_engine, blob: BlobDownloader):
    """
    Factory that creates the message callback with access to the OCR engines and blob client.
    Returns an async callback suitable for broker.consume().
    """

    async def handle_message(message: dict):
        try:
            await process_document(message, paddle_engine, easy_engine, blob)
        except Exception as e:
            logger.error(f"Message handler caught error for doc_id={message.get('doc_id')}: {e}")

    return handle_message