// src/report-generation/PipelineView.tsx
import React, { useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

// Worker config if needed
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface ChunkData {
  text: string;
  isLabel: boolean;
}

interface PipelineViewProps {
  chunks: ChunkData[];
  currentChunkIndex: number;
  setCurrentChunkIndex: React.Dispatch<React.SetStateAction<number>>;
  currentPrompt: string;
  setCurrentPrompt: React.Dispatch<React.SetStateAction<string>>;
  globalContext: string;
  setGlobalContext: React.Dispatch<React.SetStateAction<string>>;
  responses: string[];
  setResponses: React.Dispatch<React.SetStateAction<string[]>>;
  triplets: Array<{
    id: string;
    subject: string;
    predicate: string;
    object: string;
  }>;
  setTriplets: React.Dispatch<
    React.SetStateAction<
      Array<{
        id: string;
        subject: string;
        predicate: string;
        object: string;
      }>
    >
  >;
}

const PipelineView: React.FC<PipelineViewProps> = ({
  chunks,
  currentChunkIndex,
  setCurrentChunkIndex,
  currentPrompt,
  setCurrentPrompt,
  globalContext,
  setGlobalContext,
  responses,
  setResponses,
  triplets,
  setTriplets,
}) => {
  // PDF state
  const [pdfFile, setPdfFile] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);

  const handleFileChange = (evt: React.ChangeEvent<HTMLInputElement>) => {
    const file = evt.target.files?.[0];
    if (file) {
      setPdfFile(URL.createObjectURL(file));
      setCurrentPage(1);
    }
  };

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
    setCurrentPage(1);
  };

  const handleNextPage = () => {
    if (currentPage < numPages) setCurrentPage((p) => p + 1);
  };
  const handlePrevPage = () => {
    if (currentPage > 1) setCurrentPage((p) => p - 1);
  };

  const zoomIn = () => setZoomLevel((prev) => Math.min(prev + 0.25, 3.0));
  const zoomOut = () => setZoomLevel((prev) => Math.max(prev - 0.25, 0.5));

  const handleRemovePDF = () => {
    if (!pdfFile) return;
    const confirmRemove = window.confirm("Are you sure you want to remove this PDF?");
    if (!confirmRemove) return;
    setPdfFile(null);
    setNumPages(0);
    setCurrentPage(1);
  };

  // For chunk navigation
  const handlePrevChunk = () => {
    setCurrentChunkIndex((prev) => Math.max(0, prev - 1));
  };
  const handleNextChunk = () => {
    setCurrentChunkIndex((prev) => Math.min(chunks.length - 1, prev + 1));
  };

  // Process prompt => store response
  const handleProcessPrompt = () => {
    if (!chunks.length) return;
    const newResponses = [...responses];
    // Example LLM call => we just mock a string
    const chunkText = chunks[currentChunkIndex].text;
    const mockResponse = `LLM Output for chunk #${currentChunkIndex + 1}:\nPrompt: ${currentPrompt}\nContext: ${globalContext}\n\n(Your LLM answer here)`;

    newResponses[currentChunkIndex] = mockResponse;
    setResponses(newResponses);
  };

  return (
    <div
      className="w-full"
      style={{
        minHeight: "100vh",
        borderBottom: "2px solid #ccc",
        display: "flex",
        flexDirection: "row",
        overflow: "auto",
      }}
    >
      {/* LEFT SIDE: PDF Viewer */}
      <div
        className="bg-gray-200"
        style={{ flex: 1, borderRight: "1px solid #999", display: "flex", flexDirection: "column" }}
      >
        {/* If pdfFile isn't loaded => file input */}
        {!pdfFile && (
          <div className="p-2">
            <input
              type="file"
              accept="application/pdf"
              onChange={handleFileChange}
              className="bg-white border text-sm p-1 rounded"
            />
          </div>
        )}
        {/* If pdfFile => show controls + PDF */}
        {pdfFile && (
          <>
            <div className="relative flex items-center justify-between p-2 bg-gray-300 text-sm">
              <div className="flex items-center gap-2">
                <button
                  onClick={handlePrevPage}
                  disabled={currentPage === 1}
                  className="bg-blue-500 px-3 py-1 text-white rounded disabled:bg-gray-500"
                >
                  Prev
                </button>
                <span>
                  Page {currentPage} of {numPages}
                </span>
                <button
                  onClick={handleNextPage}
                  disabled={currentPage === numPages}
                  className="bg-blue-500 px-3 py-1 text-white rounded disabled:bg-gray-500"
                >
                  Next
                </button>
              </div>

              {/* Zoom controls */}
              <div className="flex items-center gap-2">
                <button
                  onClick={zoomOut}
                  className="bg-green-500 px-3 py-1 rounded text-white"
                >
                  -
                </button>
                <span>{Math.round(zoomLevel * 100)}%</span>
                <button
                  onClick={zoomIn}
                  className="bg-green-500 px-3 py-1 rounded text-white"
                >
                  +
                </button>
              </div>

              {/* "X" to remove PDF */}
              <button
                onClick={handleRemovePDF}
                className="text-white bg-red-500 px-2 py-1 rounded"
              >
                X
              </button>
            </div>

            {/* PDF Display Area */}
            <div className="flex-1 overflow-auto flex justify-center items-start">
              <div
                style={{
                  transform: `scale(${zoomLevel})`,
                  transformOrigin: "top center",
                }}
              >
                <Document
                  file={pdfFile}
                  onLoadSuccess={onDocumentLoadSuccess}
                  className="flex-1"
                >
                  <Page pageNumber={currentPage} renderAnnotationLayer={false} renderTextLayer={false} />
                </Document>
              </div>
            </div>
          </>
        )}
      </div>

      {/* RIGHT SIDE: The pipeline sub-windows */}
      <div style={{ flex: 1.5, display: "flex", flexDirection: "column", padding: "1rem" }}>
        {/* CHUNK WINDOW */}
        <div className="border mb-2 p-2">
          <h3 className="font-bold mb-2">Chunk Window</h3>
          {chunks.length === 0 ? (
            <p>No Chunks Loaded</p>
          ) : (
            <>
              <p>
                Showing chunk {currentChunkIndex + 1} of {chunks.length}
              </p>
              <pre className="text-sm bg-gray-100 p-1 rounded">
                {chunks[currentChunkIndex]?.text}
              </pre>
              <div className="mt-2 flex gap-2">
                <button
                  onClick={handlePrevChunk}
                  disabled={currentChunkIndex === 0}
                  className="bg-blue-500 text-white px-2 py-1 rounded text-xs"
                >
                  Prev Chunk
                </button>
                <button
                  onClick={handleNextChunk}
                  disabled={currentChunkIndex === chunks.length - 1}
                  className="bg-blue-500 text-white px-2 py-1 rounded text-xs"
                >
                  Next Chunk
                </button>
              </div>
            </>
          )}
        </div>

        {/* PROMPT WINDOW */}
        <div className="border mb-2 p-2">
          <h3 className="font-bold mb-2">Prompt Window</h3>
          <textarea
            className="w-full h-24 p-1 border rounded text-sm"
            placeholder="Type your prompt here..."
            value={currentPrompt}
            onChange={(e) => setCurrentPrompt(e.target.value)}
          />
          <div className="mt-2">
            <button
              onClick={handleProcessPrompt}
              disabled={!chunks.length}
              className="bg-green-500 text-white px-3 py-1 rounded text-sm"
            >
              Process Prompt
            </button>
          </div>
        </div>

        {/* RESPONSE WINDOW */}
        <div className="border mb-2 p-2">
          <h3 className="font-bold mb-2">Response Window</h3>
          {chunks.length > 0 ? (
            <pre className="text-sm bg-gray-100 p-1 rounded h-24 overflow-auto">
              {responses[currentChunkIndex] || "(No response yet)"}
            </pre>
          ) : (
            <p>No chunks = no response</p>
          )}
        </div>

        {/* CONTEXT WINDOW */}
        <div className="border p-2">
          <h3 className="font-bold mb-2">Context Window</h3>
          <textarea
            className="w-full h-24 p-1 border rounded text-sm"
            placeholder="Global context / knowledge..."
            value={globalContext}
            onChange={(e) => setGlobalContext(e.target.value)}
          />
        </div>
      </div>
    </div>
  );
};

export default PipelineView;
