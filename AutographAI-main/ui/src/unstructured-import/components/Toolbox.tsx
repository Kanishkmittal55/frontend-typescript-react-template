import React from "react";

interface ToolboxProps {
  selectedTool: "pdfplumber" | "ocr" | "default";
  setSelectedTool: (tool: "pdfplumber" | "ocr" | "default") => void;
  chunkSize: number;
  setChunkSize: (size: number) => void;
  onChunkify: () => void;
  isLoading: boolean;
}

export const Toolbox: React.FC<ToolboxProps> = ({
  selectedTool,
  setSelectedTool,
  chunkSize,
  setChunkSize,
  onChunkify,
  isLoading,
}) => {
  return (
    <div className="bg-gray-700 p-2 rounded">
      <h2 className="text-xl font-bold mb-2 text-blue-300">Toolbox</h2>
      {/* Tool selection */}
      <div className="flex flex-col gap-2 mb-4">
        <label className="flex items-center gap-2">
          <input
            type="radio"
            value="default"
            checked={selectedTool === "default"}
            onChange={() => setSelectedTool("default")}
          />
          Pdf.js ( Regex )
        </label>
        <label className="flex items-center gap-2">
          <input
            type="radio"
            value="ocr"
            checked={selectedTool === "ocr"}
            onChange={() => setSelectedTool("ocr")}
          />
          OCR Extraction
        </label>
      </div>
    </div>
  );
};
