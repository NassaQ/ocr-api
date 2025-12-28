# 📄 Intelligent OCR & Document Processing Pipeline

> **Version:** 1.0.0 (Stable)  
> **Tech Stack:** Python 3.11, FastAPI, PaddleOCR, EasyOCR, PyMuPDF  
> **Status:** MVP (Minimum Viable Product)

## 📌 Project Overview
This project is a high-performance **Document Management System backend** designed to extract text from mixed-language documents (Arabic & English). It features a **Smart Routing System** that automatically detects the document language and selects the optimal OCR engine:
* **PaddleOCR:** Used for high-speed processing of English text, numbers, and tables.
* **EasyOCR:** Used for high-accuracy processing of Arabic text with correct right-to-left sentence reconstruction.

Additionally, the pipeline acts as a **Dataset Builder**, archiving original files, extracted text, and metadata logs for future model training.

---

## ⚙️ System Architecture & Logic

The system follows a "Hybrid & Smart" approach to balance speed and accuracy:

1.  **Input:** Accepts Images (JPG, PNG) or PDFs (Scanned or Digital).
2.  **PDF Parsing:** Uses `PyMuPDF` to attempt direct text extraction first. If failed, it converts pages to images.
3.  **The "Smart Router":**
    * **Step 1:** Runs `PaddleOCR` (Fast pass).
    * **Step 2:** Checks extracted text for Arabic characters (`Regex`).
    * **Step 3 (Decision):**
        * If **English/Numbers**: Returns PaddleOCR result immediately.
        * If **Arabic**: Switches to `EasyOCR` with `paragraph=True` to fix cursive connectivity and reading order.
4.  **Output:** Saves the Source File, Target Text File, and JSON Metadata.

---

## 🚀 Key Features

* **⚡ Auto-Routing Engine:** No need to manually select the language; the API decides per file.
* **🧠 Hybrid PDF Processing:** Handles digital PDFs (text layer) and scanned PDFs (OCR layer) simultaneously.
* **🇸🇦 Optimized for Arabic:** Solves the common "disjointed letters" and "wrong reading order" issues in Arabic OCR.
* **📂 Dataset Generation:** Automatically organizes processed files into a structured dataset for future ML tasks.
* **📊 Detailed Logging:** Generates a JSON report for every batch containing confidence scores, models used, and processing methods per page.