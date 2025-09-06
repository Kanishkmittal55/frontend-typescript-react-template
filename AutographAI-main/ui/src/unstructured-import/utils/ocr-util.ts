import Tesseract, { PSM , Worker, Block, createWorker} from "tesseract.js";
import * as pdfjs from "pdfjs-dist";

// 5 lines above for context
export interface BBoxesIndex {
  [pdfFile: string]: {
    [level: string]: {
      processedAllPages: boolean;
      pages: {
        [pageNum: string]: Array<{
          text: string;
          bbox: { x0: number; y0: number; x1: number; y1: number };
        }>;
      };
    };
  };
}

// Utility to read BBOX_INDEX from localStorage (or return an empty object)
export function getBBoxesIndex(): BBoxesIndex {
  const raw = localStorage.getItem("BBOX_INDEX");
  if (!raw) return {};
  return JSON.parse(raw);
}

// Utility to save BBOX_INDEX to localStorage
export function setBBoxesIndex(index: BBoxesIndex) {
  localStorage.setItem("BBOX_INDEX", JSON.stringify(index));
}



export async function createOCRWorker(
  onProgress?: (info: { status: string; progress: number }) => void
) {
    // 1) Create the worker (optionally with a global logger for progress)
  const worker = await createWorker('eng', 1, {
    logger: (info) => {
      console.log("Global OCR progress:", info.status, info.progress);
      if (onProgress) {
        onProgress({ status: info.status, progress: info.progress });
      }
    },
  });

  // 4) Optionally, set Tesseract parameters 
  //    (similar to the old "setParameters" call)
  await worker.setParameters({
    tessedit_pageseg_mode: PSM.AUTO,
  });

  // 5) Return the fully-initialized worker
  return worker;
}

export async function convertPdfPageToImage(pdfFile: string, pageNumber: number, scale: number = 1): Promise<{ image: string; viewport: pdfjs.PageViewport }> {
    try {
      // Load the PDF file
      const loadingTask = pdfjs.getDocument(pdfFile);
      const pdf = await loadingTask.promise;
  
      // Get the specified page
      const page = await pdf.getPage(pageNumber);
  
      // Get the viewport for the PDF page
      const viewport = page.getViewport({ scale });
  
      // Render the page to a canvas
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
  
      if (!context) {
        throw new Error("Failed to get canvas 2D context");
      }
  
      canvas.width = viewport.width;
      canvas.height = viewport.height;
  
      const renderContext = {
        canvasContext: context,
        viewport,
      };
      await page.render(renderContext).promise;
  
      // Convert the canvas to a data URL (image)
      const image = canvas.toDataURL("image/png");
  
      return { image, viewport };
    } catch (error) {
      console.error("Error during PDF to image conversion:", error);
      throw error;
    }
  }

  const adjustCanvasSizeForPaper = (canvas: HTMLCanvasElement, viewport: pdfjs.PageViewport, paperSize: string) => {
    const { width, height } = viewport;
  
    if (paperSize === "A4") {
      canvas.width = 595; // A4 width in points
      canvas.height = (595 / width) * height; // Scale height proportionally
    } else if (paperSize === "A3") {
      canvas.width = 841; // A3 width in points
      canvas.height = (841 / width) * height; // Scale height proportionally
    } else if (paperSize === "Letter") {
      canvas.width = 612; // Letter width in points
      canvas.height = (612 / width) * height; // Scale height proportionally
    } else {
      // Default: Use viewport dimensions
      canvas.width = width;
      canvas.height = height;
    }
  };

  
export async function performOCRAndDrawBoundingBoxes(worker: Worker, pdfFile: string, pageNumber: number , canvasId: string, paperSize: string, boundingBoxLevel: "symbol" | "word" | "line" | "paragraph"): Promise<void> {
   // Convert the PDF Page to an image using the helper function
   try {
    // Convert the PDF page to an image using the helper function
    const {  image, viewport } = await convertPdfPageToImage(pdfFile, pageNumber);

    // Resize the canvas based on the PDF dimensions
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    if (!canvas) {
      throw new Error(`Canvas with ID "${canvasId}" not found.`);
    }

    adjustCanvasSizeForPaper(canvas, viewport, paperSize); // Pass the `selectedPaperSize` from state
    
    // Perform OCR with bounding box output enabled
    const { data } = await worker.recognize(image, {}, { blocks: true });

    console.log("Full OCR Data:", data);

    let boundingBoxes: Array<{
      text: string;
      bbox: { x0: number; y0: number; x1: number; y1: number };
    }> = [];
    
    if (!data.blocks) {
      console.warn("No blocks found in data.");
      return;
    }
    
    if (boundingBoxLevel === "paragraph") {
      // paragraphs => each block may have multiple paragraphs
      const paragraphs = data.blocks.flatMap((block) => block.paragraphs);
      boundingBoxes = paragraphs.map((para) => ({
        text: para.text,
        bbox: para.bbox, // { x0, y0, x1, y1 }
      }));
    } else if (boundingBoxLevel === "line") {
      // lines => flatten blocks->paragraphs->lines
      const lines = data.blocks.flatMap((block) =>
        block.paragraphs.flatMap((p) => p.lines)
      );
      boundingBoxes = lines.map((line) => ({
        text: line.text,
        bbox: line.bbox,
      }));
    } else if (boundingBoxLevel === "word") {
      // words => flatten blocks->paragraphs->lines->words
      const words = data.blocks.flatMap((block) =>
        block.paragraphs.flatMap((p) =>
          p.lines.flatMap((l) => l.words)
        )
      );
      boundingBoxes = words.map((w) => ({
        text: w.text,
        bbox: w.bbox,
      }));
    } else {
      // symbols => flatten blocks->paragraphs->lines->words->symbols
      const symbols = data.blocks.flatMap((block) =>
        block.paragraphs.flatMap((p) =>
          p.lines.flatMap((l) =>
            l.words.flatMap((w) => w.symbols)
          )
        )
      );
      boundingBoxes = symbols.map((s) => ({
        text: s.text,
        bbox: s.bbox,
      }));
    }
    console.log("Flattened boundingBoxes:", boundingBoxes);

    // (A) Store them in localStorage with a simpler key:
    const key = makeBBoxStorageKey(pdfFile, pageNumber, boundingBoxLevel);
    localStorage.setItem(key, JSON.stringify(boundingBoxes));
    

    // Now `boundingBoxes` is iterable and in the correct format
    console.log("Parsed bounding boxes:", boundingBoxes);

    // Draw bounding boxes on the canvas
    drawBoundingBoxes(canvasId, boundingBoxes);

   
  } catch (error) {
    console.error("Error during OCR processing:", error);
  }

  }

// Put this next to your other exports in ocr-util.ts

export function makeBBoxStorageKey(
  pdfFile: string,
  pageNumber: number,
  boundingBoxLevel: string
): string {
  // Example: "BBOX::blob:http://localhost:3000/abc::page5::word"
  return `BBOX::${pdfFile}::page${pageNumber}::${boundingBoxLevel}`;
}

  

  export function drawBoundingBoxes(
    canvasId: string,
    boxes: { text: string | number; bbox: { x0: string | number; y0: string | number; x1: string | number; y1: string | number } }[]
  ): void {
    // Get the canvas and its 2D rendering context
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement | null;
    if (!canvas) {
      throw new Error(`Canvas with ID "${canvasId}" not found.`);
    }
  
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Failed to get 2D context for canvas.");
    }
  
    // Clear the canvas
    context.clearRect(0, 0, canvas.width, canvas.height);
  
    // Set bounding box style
    context.strokeStyle = "red"; // Red border for bounding boxes
    context.lineWidth = 2;
  
    // Loop through each bounding box and draw it
    boxes.forEach(({ text, bbox }) => {
      const x0 = typeof bbox.x0 === "string" ? parseFloat(bbox.x0) : bbox.x0;
      const y0 = typeof bbox.y0 === "string" ? parseFloat(bbox.y0) : bbox.y0;
      const x1 = typeof bbox.x1 === "string" ? parseFloat(bbox.x1) : bbox.x1;
      const y1 = typeof bbox.y1 === "string" ? parseFloat(bbox.y1) : bbox.y1;
  
      const width = x1 - x0;
      const height = y1 - y0;
  
      // Draw the background for the bounding box
      context.fillStyle = "rgba(255, 255, 150, 0.8)"; // Light yellow background with transparency
      context.fillRect(x0, y0, width, height);

      // Draw the border for the bounding box
      context.strokeRect(x0, y0, width, height);

      // Draw text inside the bounding box
      context.fillStyle = "black"; // Reset the text color
      context.textBaseline = "middle"; // Align text to the middle vertically
      context.textAlign = "center"; // Align text to the center horizontally
      const textX = x0 + width / 2; // Center horizontally in the bounding box
      const textY = y0 + height / 2; // Center vertically in the bounding box
      context.fillText(String(text), textX, textY, width); // Draw the text inside the box, with wrapping

    });
  }
  