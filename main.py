from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Tuple
from paddleocr import PaddleOCR
import easyocr
import numpy as np
import cv2
import fitz  # PyMuPDF
import json
import os
import re
from datetime import datetime
import logging


app = FastAPI(
    title="Intelligent Document Processing Pipeline", 
    description="Auto-detects language (Arabic/English) and routes to the optimal OCR engine.",
    version="8.0.0"
)

# Define the output directory for processed files
OUTPUT_DIR = "processed_documents"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Initialize PaddleOCR (Default Engine) ---
# Used for speed and high accuracy with English/Numbers.
print("--- Loading PaddleOCR (Speed Engine)... ---")
# Suppress PaddleOCR internal logging to keep the terminal clean
logging.getLogger("ppocr").setLevel(logging.ERROR)
# Note: 'use_angle_cls=False' disables angle classification for performance
paddle_engine = PaddleOCR(use_angle_cls=False, lang='ar')

# --- Initialize EasyOCR (Fallback Engine) ---
# Used specifically for Arabic text to ensure correct character connectivity.
print("--- Loading EasyOCR (Arabic Expert)... ---")
# gpu=False ensures the server runs on CPU (prevents overheating on non-GPU laptops)
easy_engine = easyocr.Reader(['ar', 'en'], gpu=False)

print("--- All AI Models Ready! ---")


# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def get_timestamp() -> str:
    """Returns the current timestamp string to ensure unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def contains_arabic(text: str) -> bool:
    """
    Checks if the text contains Arabic characters using Regex.
    Arabic Unicode range: \u0600-\u06FF
    """
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    return bool(arabic_pattern.search(text))

def ocr_with_paddle(img_array: np.ndarray) -> Tuple[str, float]:
    """
    Runs PaddleOCR on the image.
    Returns: Extracted text and average confidence score.
    """
    try:
        result = paddle_engine.ocr(img_array)
        extracted_text = ""
        confidence_scores = []
        
        if result:
            data = result[0]
            # Handle dictionary format (newer Paddle versions)
            if isinstance(data, dict) and 'rec_texts' in data:
                extracted_text = " ".join(data.get('rec_texts', []))
                confidence_scores = data.get('rec_scores', [])
            # Handle list format (legacy Paddle versions)
            elif isinstance(data, list):
                texts = []
                for item in data:
                    if isinstance(item, list) and len(item) >= 2:
                        texts.append(item[1][0])
                        confidence_scores.append(item[1][1])
                extracted_text = " ".join(texts)
        
        avg_conf = round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else 0.0
        return extracted_text, avg_conf
    except Exception as e:
        print(f"Paddle Error: {e}")
        return "", 0.0

def ocr_with_easy(img_array: np.ndarray) -> Tuple[str, float]:
    """
    Runs EasyOCR with paragraph handling enabled.
    This is critical for Arabic to maintain correct right-to-left reading order.
    """
    try:
        # detail=0 returns only text. 
        # paragraph=True groups words into coherent sentences/paragraphs.
        results = easy_engine.readtext(img_array, detail=0, paragraph=True)
        
        final_text = "\n".join(results)
        
        # Note: When paragraph=True is used, confidence scores per word are not returned.
        # We assign a static high confidence if text is found, assuming EasyOCR's reliability.
        avg_conf = 0.95 if final_text.strip() else 0.0
        
        return final_text, avg_conf
    except Exception as e:
        print(f"EasyOCR Error: {e}")
        return "", 0.0

def smart_ocr_pipeline(img_array: np.ndarray) -> Tuple[str, float, str]:
    """
    The Routing Logic:
    1. Runs PaddleOCR first (Fastest).
    2. Checks the result for Arabic characters.
    3. If Arabic is detected, re-runs with EasyOCR (Best for Arabic Layout).
    4. Otherwise, returns the PaddleOCR result.
    
    Returns: (extracted_text, confidence_score, model_name)
    """
    # Step 1: Initial pass with PaddleOCR
    text, conf = ocr_with_paddle(img_array)
    used_model = "paddle"

    # Step 2: Language Check
    if contains_arabic(text):
        # Arabic detected, switching engine
        easy_text, easy_conf = ocr_with_easy(img_array)
        
        # Only overwrite if EasyOCR actually returned text
        if easy_text.strip():
            return easy_text, easy_conf, "easyocr (auto-switched)"
            
    return text, conf, used_model


# ==============================================================================
# 3. API Endpoints
# ==============================================================================

@app.post("/process-documents")
async def process_documents(files: List[UploadFile] = File(...)):
    """
    Main Endpoint.
    - Accepts multiple files (PDFs, Images, Text).
    - Auto-detects content type and language.
    - Saves original source, extracted text, and metadata logs.
    """
    
    batch_id = get_timestamp()
    print(f"--- Starting Batch {batch_id} | Files: {len(files)} ---")
    
    batch_metadata = []

    for file in files:
        print(f"Processing: {file.filename}...")
        file_content = await file.read()
        safe_filename = file.filename.replace(" ", "_")
        
        # Initialize metadata structure
        file_metadata = {
            "original_filename": file.filename,
            "file_type": file.content_type,
            "upload_timestamp": batch_id,
            "model_usage_log": [],      # Tracks which model was used per page/image
            "page_count": 1,
            "source_file_path": "",
            "text_file_path": "",
            "extraction_details": [],
            "status": "success",
            "overall_confidence": 1.0,
            "error_message": None
        }

        extracted_full_text = ""

        try:
            # ------------------------------------------------------------------
            # Step 1: Save the Source File (Dataset Preservation)
            # ------------------------------------------------------------------
            original_save_name = f"{batch_id}_SOURCE_{safe_filename}"
            original_file_path = os.path.join(OUTPUT_DIR, original_save_name)
            with open(original_file_path, "wb") as f:
                f.write(file_content)
            file_metadata["source_file_path"] = original_file_path

            # ------------------------------------------------------------------
            # Step 2: Content Processing based on File Type
            # ------------------------------------------------------------------
            
            # --- Type A: Text Files (.txt) ---
            if file.filename.endswith(".txt"):
                extracted_full_text = file_content.decode("utf-8")
                file_metadata["extraction_details"].append({
                    "page": 1, "method": "Direct Read", "confidence": 1.0
                })
                file_metadata["model_usage_log"].append("None (Text File)")

            # --- Type B: PDF Files (Hybrid Parsing) ---
            elif file.content_type == "application/pdf":
                pdf_doc = fitz.open(stream=file_content, filetype="pdf")
                file_metadata["page_count"] = len(pdf_doc)
                full_doc_text_list = []
                total_ocr_conf = 0
                ocr_pages_count = 0
                
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc.load_page(page_num)
                    page_log = {"page": page_num + 1, "method": "", "confidence": 1.0}
                    page_text_content = []

                    # A) Attempt Direct Text Extraction
                    text = page.get_text()
                    if text.strip():
                        page_text_content.append(text)
                        page_log["method"] = "Direct Text"
                    
                    # B) Extract Embedded Images
                    image_list = page.get_images()
                    if image_list:
                        for img in image_list:
                            xref = img[0]
                            base_image = pdf_doc.extract_image(xref)
                            nparr = np.frombuffer(base_image["image"], np.uint8)
                            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            # Run Smart Pipeline
                            ocr_text, conf, model_name = smart_ocr_pipeline(img_cv)
                            
                            if ocr_text.strip():
                                page_text_content.append(f"\n[Image]: {ocr_text}")
                                page_log["method"] += f" + {model_name}"
                                file_metadata["model_usage_log"].append(f"Page {page_num+1}: {model_name}")
                                total_ocr_conf += conf
                                ocr_pages_count += 1

                    # C) Fallback for Scanned Pages (Full Page OCR)
                    if not text.strip() and not image_list:
                        pix = page.get_pixmap()
                        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                        # Convert RGB to BGR for OpenCV
                        if pix.n >= 3: 
                            img_cv = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                        else: 
                            img_cv = img_data
                        
                        # Run Smart Pipeline
                        ocr_text, conf, model_name = smart_ocr_pipeline(img_cv)
                        
                        page_text_content.append(ocr_text)
                        page_log["method"] = f"Full Page {model_name}"
                        page_log["confidence"] = conf
                        file_metadata["model_usage_log"].append(f"Page {page_num+1}: {model_name}")
                        total_ocr_conf += conf
                        ocr_pages_count += 1
                    
                    full_doc_text_list.append("\n".join(page_text_content))
                    file_metadata["extraction_details"].append(page_log)

                extracted_full_text = "\n-------------------\n".join(full_doc_text_list)
                if ocr_pages_count > 0:
                    file_metadata["overall_confidence"] = round(total_ocr_conf / ocr_pages_count, 2)

            # --- Type C: Image Files (JPG/PNG) ---
            elif file.content_type in ["image/jpeg", "image/png", "image/jpg"]:
                nparr = np.frombuffer(file_content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Run Smart Pipeline
                text, conf, model_name = smart_ocr_pipeline(img)
                
                extracted_full_text = text
                file_metadata["extraction_details"].append({
                    "page": 1, "method": f"Direct {model_name}", "confidence": conf
                })
                file_metadata["model_usage_log"].append(model_name)
                file_metadata["overall_confidence"] = conf

            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            # ------------------------------------------------------------------
            # Step 3: Save Extracted Text (Target Data)
            # ------------------------------------------------------------------
            text_filename = f"{batch_id}_TARGET_{safe_filename}.txt"
            text_file_path = os.path.join(OUTPUT_DIR, text_filename)
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(extracted_full_text)
            
            file_metadata["text_file_path"] = text_file_path
            batch_metadata.append(file_metadata)

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            file_metadata["status"] = "error"
            file_metadata["error_message"] = str(e)
            batch_metadata.append(file_metadata)

    # ------------------------------------------------------------------
    # Step 4: Save Metadata Log to JSON
    # ------------------------------------------------------------------
    details_filename = f"Batch_Details_{batch_id}.json"
    details_path = os.path.join(OUTPUT_DIR, details_filename)
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(batch_metadata, f, ensure_ascii=False, indent=4)

    return {
        "status": "batch_complete",
        "batch_id": batch_id,
        "processed_files_count": len(files),
        "output_directory": OUTPUT_DIR
    }