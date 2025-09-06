# Chunkify PDF Annotation Tool

**Chunkify** is a React application demonstrating a practical workflow for:
1. **Loading and displaying PDFs** with [React-PDF](https://github.com/wojtekmaj/react-pdf).
2. **Performing OCR** via [Tesseract.js](https://github.com/naptha/tesseract.js) to obtain bounding boxes and text.
3. **Visualizing bounding boxes** over a `<canvas>` overlay.
4. **Extracting text** (by paragraphs or other methods) and **splitting** (“chunkifying”) it into smaller segments.
5. **Manipulating text chunks** (merge, delete, reverse, shuffle).
6. Optionally **running iterative prompts** (e.g., with an AI/LLM) on each chunk.

The overall goal is to support an **interactive workflow** for labeling, reorganizing, or summarizing extracted text.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Key Features](#key-features)
  - [1. Loading PDFs](#1-loading-pdfs)
  - [2. OCR Extraction and Bounding Boxes](#2-ocr-extraction-and-bounding-boxes)
  - [3. Text Chunkification](#3-text-chunkification)
  - [4. Chunk Manipulations](#4-chunk-manipulations)
  - [5. Iterative Prompting](#5-iterative-prompting)
- [File Structure](#file-structure)
- [Usage Flow](#usage-flow)
- [Local Storage Notes](#local-storage-notes)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Key Features

### 1. Loading PDFs
- In the **center panel**, you can **upload** a `.pdf` file.
- Uses **React-PDF** `<Document>` and `<Page>` to render the PDF pages.

### 2. OCR Extraction and Bounding Boxes
- **Tesseract.js** powers the OCR process.
- Choose a bounding box level: `symbol`, `word`, `line`, or `paragraph`.
- Clicking **“Chunkify”** in the OCR toolbox:
  1. Converts a PDF page to an image `<canvas>`.
  2. Tesseract reads the image for text blocks.
  3. Resulting bounding boxes are stored in **localStorage** and drawn on an overlay `<canvas>`.

- If **Process All Pages** is enabled, it processes each page. Otherwise, only the current page is handled.

### 3. Text Chunkification
- **Paragraph-level OCR** automatically extracts paragraphs as chunks.
- Or, if you have text in `extractedText`, you can **split** it into chunks by a chosen length (default `500` characters).

### 4. Chunk Manipulations
- **Merge All**: Combine all chunks into a single chunk.
- **Delete All**: Remove all chunks.
- **Reverse**: Reverse the order of the chunks array.
- **Shuffle**: Randomly reorder the chunks array.
- **Selective Merge**:
  1. Click **“Merge”** on one chunk to select it as the “source.”
  2. Click **“Merge”** on another chunk to merge them, removing the source chunk from the list.

### 5. Iterative Prompting
- In the **right panel**, you can enter a prompt (e.g. “Summarize”).
- Clicking **“Run Prompt on Chunks”** calls a mock function (`dummySendPrompt`) on each chunk.
- The result is displayed under each chunk, simulating how you might integrate a real AI or LLM.

---

## File Structure

```plaintext
├─ PdfAnnotationTool.tsx   # Main "Chunkify" UI component
├─ components
│   ├─ Toolbox.tsx         # Chunk size, tool selection (OCR / pdfplumber), etc.
│   ├─ ChunkList.tsx       # Renders text chunks + merge/delete/prompt results
│   └─ switch.tsx          # A simple toggle switch component
├─ utils
│   ├─ pdf-extract-utils.ts   # Placeholder logic for PDF text extraction
│   ├─ chunk-utils.ts         # "chunkText" helper to split text
│   ├─ bounding-box-utils.ts  # Optional bounding box overlay function
│   └─ ocr-util.ts            # Tesseract setup, OCR, bounding box + localStorage logic
└─ ...

____

## Tool Kit and Local Installation instructions coming soon.