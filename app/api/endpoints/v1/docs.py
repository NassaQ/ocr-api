from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from typing import List
import numpy as np
import cv2
import fitz
import json
import os

from app.core.config import OUTPUT_DIR
from app.core.logging import logger
from app.api.deps import (
    get_paddle_engine,
    get_easy_engine,
    get_timestamp,
    smart_ocr_pipeline,
)


router = APIRouter()


@router.post("/docs")
async def process_documents(request: Request, files: List[UploadFile] = File(...)):
    """
    Process multiple documents (PDFs, images, text files).
    Auto-detects content type and language, saves source files, extracted text, and metadata.
    """
    paddle_engine = get_paddle_engine(request)
    easy_engine = get_easy_engine(request)

    batch_id = get_timestamp()
    logger.info(f"Starting batch {batch_id} with {len(files)} files")

    batch_metadata = []

    for file in files:
        filename = file.filename or "unknown"
        logger.info(f"Processing: {filename}")
        file_content = await file.read()
        safe_filename = filename.replace(" ", "_")

        file_metadata = {
            "original_filename": filename,
            "file_type": file.content_type,
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

        extracted_full_text = ""

        try:
            original_save_name = f"{batch_id}_SOURCE_{safe_filename}"
            original_file_path = os.path.join(OUTPUT_DIR, original_save_name)
            with open(original_file_path, "wb") as f:
                f.write(file_content)
            file_metadata["source_file_path"] = original_file_path

            if filename.endswith(".txt"):
                extracted_full_text = file_content.decode("utf-8")
                file_metadata["extraction_details"].append(
                    {"page": 1, "method": "Direct Read", "confidence": 1.0}
                )
                file_metadata["model_usage_log"].append("None (Text File)")

            elif file.content_type == "application/pdf":
                pdf_doc = fitz.open(stream=file_content, filetype="pdf")
                file_metadata["page_count"] = len(pdf_doc)
                full_doc_text_list = []
                total_ocr_conf = 0.0
                ocr_pages_count = 0

                for page_num in range(len(pdf_doc)):
                    page = pdf_doc.load_page(page_num)
                    page_log = {"page": page_num + 1, "method": "", "confidence": 1.0}
                    page_text_content: List[str] = []

                    text = page.get_text()
                    if text.strip():
                        page_text_content.append(text)
                        page_log["method"] = "Direct Text"

                    image_list = page.get_images()
                    if image_list:
                        for img in image_list:
                            xref = img[0]
                            base_image = pdf_doc.extract_image(xref)
                            nparr = np.frombuffer(base_image["image"], np.uint8)
                            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            ocr_text, conf, model_name = smart_ocr_pipeline(
                                paddle_engine, easy_engine, img_cv
                            )

                            if ocr_text and ocr_text.strip():
                                page_text_content.append(f"\n[Image]: {ocr_text}")
                                page_log["method"] += f" + {model_name}"
                                file_metadata["model_usage_log"].append(
                                    f"Page {page_num + 1}: {model_name}"
                                )
                                total_ocr_conf += conf
                                ocr_pages_count += 1

                    if not text.strip() and not image_list:
                        pix = page.get_pixmap()
                        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                            pix.h, pix.w, pix.n
                        )
                        if pix.n >= 3:
                            img_cv = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                        else:
                            img_cv = img_data

                        ocr_text, conf, model_name = smart_ocr_pipeline(
                            paddle_engine, easy_engine, img_cv
                        )

                        page_text_content.append(str(ocr_text))
                        page_log["method"] = f"Full Page {model_name}"
                        page_log["confidence"] = conf
                        file_metadata["model_usage_log"].append(
                            f"Page {page_num + 1}: {model_name}"
                        )
                        total_ocr_conf += conf
                        ocr_pages_count += 1

                    full_doc_text_list.append("\n".join(page_text_content))
                    file_metadata["extraction_details"].append(page_log)

                extracted_full_text = "\n-------------------\n".join(full_doc_text_list)
                if ocr_pages_count > 0:
                    file_metadata["overall_confidence"] = round(
                        total_ocr_conf / ocr_pages_count, 2
                    )

            elif file.content_type in ["image/jpeg", "image/png", "image/jpg"]:
                nparr = np.frombuffer(file_content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                text, conf, model_name = smart_ocr_pipeline(
                    paddle_engine, easy_engine, img
                )

                extracted_full_text = str(text)
                file_metadata["extraction_details"].append(
                    {"page": 1, "method": f"Direct {model_name}", "confidence": conf}
                )
                file_metadata["model_usage_log"].append(model_name)
                file_metadata["overall_confidence"] = conf

            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            text_filename = f"{batch_id}_TARGET_{safe_filename}.txt"
            text_file_path = os.path.join(OUTPUT_DIR, text_filename)
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(extracted_full_text)

            file_metadata["text_file_path"] = text_file_path
            batch_metadata.append(file_metadata)

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            file_metadata["status"] = "error"
            file_metadata["error_message"] = str(e)
            batch_metadata.append(file_metadata)

    details_filename = f"Batch_Details_{batch_id}.json"
    details_path = os.path.join(OUTPUT_DIR, details_filename)
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(batch_metadata, f, ensure_ascii=False, indent=4)

    return {
        "status": "batch_complete",
        "batch_id": batch_id,
        "processed_files_count": len(files),
        "output_directory": OUTPUT_DIR,
    }
