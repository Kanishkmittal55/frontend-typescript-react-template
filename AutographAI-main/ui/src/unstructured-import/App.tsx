import React, { useState, useEffect, useRef } from "react";
import Split from "react-split";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import GraphModal from "./components/GraphModal";

// =============================
//   DATA MODELS & INTERFACES
// =============================

// For each chunk, we store multiple versions in a dictionary:
// chunkVersions[1] is after chunk-splitting (no LLM calls).
// chunkVersions[2] is after applying a prompt to version 1, etc.
interface ChunkData {
  chunkVersions: {
    [version: number]: string;
  };
}

// We'll also store each version's prompt in a dictionary
type VersionPrompts = {
  [version: number]: string;
};

// ================
//   OPENAI CALL
// ================
const OPENAI_API_KEY = "<no-key>";

/**
 * Chat completions call to OpenAI.
 */
async function callOpenAIGraphPrompt(
  systemPrompt: string,
  userInput: string,
  model = "gpt-4o"
): Promise<string> {
  // Minor fix to avoid syntax errors:
  const userMessage = `context: \`\`\`${userInput}\`\`\`\n\n output: `;

  try {
    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userMessage },
        ],
        temperature: 0.7,
      }),
    });

    if (!resp.ok) {
      console.error("OpenAI HTTP error:", resp.status, resp.statusText);
      return "[Error: OpenAI request failed]";
    }

    const data = await resp.json();
    const assistantMsg = data.choices?.[0]?.message?.content || "";
    return assistantMsg.trim() || "[No output]";
  } catch (err) {
    console.error("OpenAI call error:", err);
    return "[Error: Exception in OpenAI call]";
  }
}

// =============================
//   RECURSIVE SPLIT
// =============================
function recursiveSplitText(
  text: string,
  chunk_size: number,
  chunk_overlap: number
): string[] {
  const results: string[] = [];
  let start = 0;
  while (start < text.length) {
    let end = start + chunk_size;
    if (end > text.length) end = text.length;
    results.push(text.slice(start, end));
    start += chunk_size - chunk_overlap;
    if (start < 0) break;
  }
  return results;
}

// ======================================================
//   HELPERS: PARSE & EXTRACT ACTIVE INGREDIENT NAMES
// ======================================================
function parseActiveIngredientNames(rawOutput: string): string[] {
  const foundNames: string[] = [];
  console.log("I am parsing ingredients");

  try {
    console.log("The detected output is a json struct processing here only.");
    console.log("Raw chunk output before parse:", rawOutput);

    let clean = rawOutput.replace(/^```json\s*/i, ""); // remove ```json
    clean = rawOutput.replace(/^```lang\s*/i, "") 
    clean = clean.replace(/```$/, "");                    // Remove closing ```
    clean = clean.replace(/```/g, "");                    // If more stray backticks appear

    const parsed = JSON.parse(clean.trim());
    console.log("Parsed:", parsed);

    // If parsed is an array of items, check each item
    if (Array.isArray(parsed)) {
      parsed.forEach((item: any) => {
        console.log("Each item :", item);

        // Check node_1
        if (
          item.node_1 &&
          item.node_1.is_active_ingredient === true &&
          typeof item.node_1.name === "string"
        ) {
          foundNames.push(item.node_1.name);
        } else {
          console.log("No active ingredient in node_1 or not true");
        }

        // Check node_2
        if (
          item.node_2 &&
          item.node_2.is_active_ingredient === true &&
          typeof item.node_2.name === "string"
        ) {
          foundNames.push(item.node_2.name);
        } else {
          console.log("No active ingredient in node_2 or not true");
        }
      });
    } else {
      console.log("Not a clean JSON array, using collectNames fallback...");
      collectNames(parsed, foundNames);
    }
  } catch (err) {
    // If JSON parse fails, do naive fallback
    console.error("Error while parsing JSON or subsequent logic:", err);

    const lowered = rawOutput.toLowerCase();
    if (
      (lowered.includes("is_active_ingredient") || lowered.includes("isactiveingredient")) &&
      lowered.includes("true")
    ) {
      const nameRegex = /"name"\s*:\s*"([^"]+)"/g;
      let match;
      while ((match = nameRegex.exec(rawOutput)) !== null) {
        foundNames.push(match[1]);
      }
    }
  }

  return foundNames;
}

/**
 * Recursively traverse any object/array to find:
 * - is_active_ingredient: true (or isActiveIngredient: true)
 * - name: string
 */
function collectNames(obj: any, foundNames: string[]) {
  if (!obj || typeof obj !== "object") return;

  const hasActiveIngredientKey =
    (typeof obj.is_active_ingredient === "boolean" && obj.is_active_ingredient) ||
    (typeof obj.isActiveIngredient === "boolean" && obj.isActiveIngredient);

  if (hasActiveIngredientKey && typeof obj.name === "string") {
    foundNames.push(obj.name);
  }

  if (Array.isArray(obj)) {
    for (const item of obj) {
      collectNames(item, foundNames);
    }
  } else {
    for (const key in obj) {
      if (typeof obj[key] === "object") {
        collectNames(obj[key], foundNames);
      }
    }
  }
}

// =============================
//   MAIN COMPONENT
// =============================
const MergedUnstructuredImport: React.FC = () => {
  // Raw extracted text from .txt
  const [extractedText, setExtractedText] = useState<string>("");

  // Word/char counts
  const [wordCount, setWordCount] = useState<number>(0);
  const [charCount, setCharCount] = useState<number>(0);

  // The chunk array. Each chunk has multiple versions
  const [chunks, setChunks] = useState<ChunkData[]>([]);

  // Chunks used for sending to the api
  const [apiChunks, setApiChunks] = useState<string[]>([])

  // "Chunkify" (split) config
  const [rcsChunkSize, setRcsChunkSize] = useState<number>(1500);
  const [rcsChunkOverlap, setRcsChunkOverlap] = useState<number>(150);

  // Show/hide chunkify modal
  const [showChunkifyModal, setShowChunkifyModal] = useState(false);

  // We have multiple "versions" (1,2,3...). The user chooses which version to display.
  const [currentVersion, setCurrentVersion] = useState<number>(1);

  // Each version has a prompt
  const [versionPrompts, setVersionPrompts] = useState<VersionPrompts>({});

  // "Run Prompt" loading
  const [isLoading, setIsLoading] = useState(false);

  // The global system prompt
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful assistant that returns structured text for knowledge graph building."
  );

  // Concurrency modal
  const [showConcurrencyModal, setShowConcurrencyModal] = useState(false);
  const [maxThreads, setMaxThreads] = useState<number>(1);
  const [usedThreads, setUsedThreads] = useState<number>(0);

  // Track which chunks failed
  const [failedChunks, setFailedChunks] = useState<number[]>([]);

  // We store discovered active ingredient names here
  const [activeIngredientsSoFar, setActiveIngredientsSoFar] = useState<string[]>([]);

  // -- graph‑modal state --
  const [showGraphModal, setShowGraphModal] = useState(false);

  // const [graphUrl, setGraphUrl] = useState<string | null>(null);

  const [initialGraphUrl, setInitialGraphUrl] = useState<string | null>(null);

  const [refinedGraphUrl, setRefinedGraphUrl] = useState<string | null>(null);

  const [monteCarloPlotUrl, setMonteCarloPlotUrl] = useState<string | null>(null);

    // 🔍 add next to the other urls you already track
  const [top10JsonUrl, setTop10JsonUrl] = useState<string | null>(null);


  const [logLines, setLogLines] = useState<string[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);


  // ======= Restore from localStorage if present =======
  useEffect(() => {
    console.log("The UNSTRUCTURED_IMPORT_STATE is :", localStorage.getItem("UNSTRUCTURED_IMPORT_STATE"))
    const saved = localStorage.getItem("UNSTRUCTURED_IMPORT_STATE");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setExtractedText(parsed.extractedText || "");
        setWordCount(parsed.wordCount || 0);
        setCharCount(parsed.charCount || 0);
        setChunks(parsed.chunks || []);
        setRcsChunkSize(parsed.rcsChunkSize || 1500);
        setRcsChunkOverlap(parsed.rcsChunkOverlap || 150);
        setCurrentVersion(parsed.currentVersion || 1);
        setVersionPrompts(parsed.versionPrompts || {});
        setSystemPrompt(parsed.systemPrompt || systemPrompt);
      } catch (err) {
        console.warn("Failed to parse localStorage:", err);
      }
    }
  }, []);

  // ======= Persist minimal states to localStorage =======
  useEffect(() => {
    const payload = {
      extractedText,
      wordCount,
      charCount,
      chunks,
      rcsChunkSize,
      rcsChunkOverlap,
      currentVersion,
      versionPrompts,
      systemPrompt,
    };
    localStorage.setItem("UNSTRUCTURED_IMPORT_STATE", JSON.stringify(payload));
  }, [
    extractedText,
    wordCount,
    charCount,
    chunks,
    rcsChunkSize,
    rcsChunkOverlap,
    currentVersion,
    versionPrompts,
    systemPrompt,
  ]);

  useEffect(() => () => {
    if (eventSourceRef.current) eventSourceRef.current.close();
  }, []);  

  //  Close model helper
  const handleCloseGraphModal = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setShowGraphModal(false);
    setInitialGraphUrl(null);
    setRefinedGraphUrl(null);
    setTop10JsonUrl(null);
    setMonteCarloPlotUrl(null);
    setLogLines([]);
  };
  

    // "Drop Cache"
  const handleDropCache = () => {
    localStorage.removeItem("UNSTRUCTURED_IMPORT_STATE");
    setExtractedText("");
    setWordCount(0);
    setCharCount(0);
    setChunks([]);
    setRcsChunkSize(1500);
    setRcsChunkOverlap(150);
    setCurrentVersion(1);
    setVersionPrompts({});
    setSystemPrompt(
      "You are a helpful assistant that returns structured text for knowledge graph building."
    );
    setActiveIngredientsSoFar([]);
    setFailedChunks([]);
    alert("Local cache cleared!");
  }

  const BACKEND = "http://localhost:7860";

  const handleChunksToGraph = async () => {
    if (apiChunks.length === 0) {
      alert("Please upload and process a text file first.");
      return;
    }
  
    /* 🔸 open / reset activity modal */
    setShowGraphModal(true);
    setLogLines([]);
    setInitialGraphUrl(null);
    setTop10JsonUrl(null);
    setRefinedGraphUrl(null);
    setMonteCarloPlotUrl(null);

  
    try {
      /* kick off backend job */
      const resp = await fetch(`${BACKEND}/chunksToKG`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chunks: apiChunks, prompt: systemPrompt }),
      });
  
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  
      const { job_id } = await resp.json();
      setLogLines((p) => [...p, `Job started: ${job_id}`]);
  
      /* 🛰️  start live‑log SSE */
      const es = new EventSource(`${BACKEND}/jobLogs/${job_id}`);
      eventSourceRef.current = es;
  
      es.addEventListener("log", (e: MessageEvent) =>
        setLogLines((p) => [...p, e.data])
      );
  
      /* whenever backend emits a finished file, preview + (optionally) download */
      es.addEventListener("file", (e: MessageEvent) => {
        // backend now sends: {"name":"Seed-Graph.html","url":"/static/src/Seed-Graph.html"}
        const { name, url } = JSON.parse(e.data) as { name: string; url: string };
        setLogLines((p) => [...p, `File ready: ${name}`]);
  
        // show inside modal
        // Preview in the modal
        const fullUrl = `${BACKEND}${url}`;   // e.g. http://localhost:7860/static/src/Seed-Graph.html
        console.log("The full url is : ", fullUrl)
        if (name.toLowerCase().includes("seed")) {
              setInitialGraphUrl(fullUrl);
        } else if(name.toLowerCase().includes("combined")) {
              setMonteCarloPlotUrl(fullUrl);
        } else if(name.toLowerCase().includes("top10")) {
            setTop10JsonUrl(fullUrl);
        } else if(name.toLowerCase().includes("refined")) {
            setRefinedGraphUrl(fullUrl);
        } else {
            console.log("Error the link is not recognized :", fullUrl)
        }

        // still trigger browser download if you want to keep that behaviour
        // const a = document.createElement("a");
        // a.href = fullUrl;
        // a.download = name;
        // a.style.display = "none";
        // document.body.appendChild(a);
        // a.click();
        // a.remove();
      });
  
      es.onerror = () => {
        setLogLines((p) => [...p, "[error] SSE connection closed"]);
        es.close();
      };
    } catch (err) {
      console.error("Error kicking off job:", err);
      setLogLines((p) => [...p, `[frontend error] ${err}`]);
    }
  };
  

  // Upload .txt
  const handleUploadTextFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const reader = new FileReader();
      reader.onload = () => {
        const text = reader.result as string;
        setExtractedText(text);

        const words = text.trim().split(/\s+/).filter(Boolean);
        setWordCount(words.length);
        setCharCount(text.length);

        // Single chunk for entire doc
        const singleChunk: ChunkData = { chunkVersions: { 1: text } };
        setChunks([singleChunk]);
        setCurrentVersion(1);
        setVersionPrompts({});
        setActiveIngredientsSoFar([]);
        setFailedChunks([]);
      };
      reader.readAsText(file);
    } catch (err) {
      console.error("Error reading .txt file:", err);
    }
  };

  // Actually chunkify => version 1
  const doChunkify = () => {
    if (!extractedText) {
      alert("No text found. Please upload a .txt file first.");
      return;
    }
    const splitted = recursiveSplitText(extractedText, rcsChunkSize, rcsChunkOverlap); // return string[]
    
    // Setting the item i.e. the chunks list in the local storage 
    localStorage.setItem("Chunks", JSON.stringify(splitted))

    const newArr: ChunkData[] = splitted.map((text) => ({
      chunkVersions: { 1: text },
    }));
    setApiChunks(splitted)
    setChunks(newArr);
    setCurrentVersion(1);
    setVersionPrompts({});
    setActiveIngredientsSoFar([]);
    setFailedChunks([]);
  };

  const handleChunkifyConfirm = () => {
    doChunkify();
    setShowChunkifyModal(false);
  };

  // Open concurrency modal
  const handleOpenConcurrencyModal = () => {
    setShowConcurrencyModal(true);
  };

  const handleConcurrencyConfirm = async () => {
    setShowConcurrencyModal(false);
    await handleRunAllChunks();
  };

  // =============== RUN ALL CHUNKS (Concurrency) ===============
  async function handleRunAllChunks() {
    if (currentVersion <= 1) {
      alert("Version 1 is the base text—no LLM calls needed. Increase version first.");
      return;
    }
    const prevVersion = currentVersion - 1;
    const promptForThisVersion = versionPrompts[currentVersion] || "";
    if (!promptForThisVersion.trim()) {
      alert(`No prompt set for version ${currentVersion}. Please fill a prompt below first.`);
      return;
    }

    // Build a queue of all chunks that do NOT have currentVersion
    const queue = chunks
      .map((_, idx) => idx)
      .filter((idx) => !chunks[idx].chunkVersions[currentVersion]);

    if (!queue.length) {
      alert(`All chunks at version ${currentVersion} already exist!`);
      return;
    }

    setIsLoading(true);
    setUsedThreads(0);
    // Clear out any old failures
    setFailedChunks([]);

    let newChunkState = [...chunks];
    const newFailed: number[] = [];

    const worker = async () => {
      while (true) {
        let idx: number | undefined;
        if (queue.length > 0) {
          idx = queue.shift();
        }
        if (idx === undefined) {
          return;
        }
        const prevText = newChunkState[idx].chunkVersions[prevVersion];
        if (!prevText) {
          console.error(
            `Chunk #${idx + 1} is missing version ${prevVersion}. Cannot build V${currentVersion}.`
          );
          newFailed.push(idx);
          continue;
        }
        try {
          setUsedThreads((prev) => prev + 1);
          const userInput = prevText;
          const result = await callOpenAIGraphPrompt(
            systemPrompt,
            `PROMPT: ${promptForThisVersion}\n\nCHUNK:\n${userInput}`,
            "gpt-4"
          );
          newChunkState = newChunkState.map((chunk, i) => {
            if (i === idx) {
              return {
                ...chunk,
                chunkVersions: {
                  ...chunk.chunkVersions,
                  [currentVersion]: result,
                },
              };
            }
            return chunk;
          });
          setChunks([...newChunkState]);
        } catch (err) {
          console.error("Error in concurrency worker for chunk #", idx, err);
          newFailed.push(idx);
        } finally {
          setUsedThreads((prev) => prev - 1);
        }
      }
    };

    const tasks: Promise<void>[] = [];
    for (let t = 0; t < maxThreads; t++) {
      tasks.push(worker());
    }

    await Promise.all(tasks);

    setIsLoading(false);
    setFailedChunks(newFailed);

    if (newFailed.length > 0) {
      alert(
        `Version ${currentVersion} partially complete. ${newFailed.length} chunk(s) failed. You can retry them.`
      );
    } else {
      alert(`Version ${currentVersion} created for all chunks!`);
    }
  }

  // "Retry Failed Chunks" concurrency approach
  const handleRetryFailedChunks = async () => {
    if (currentVersion <= 1) {
      alert("Version 1 is base text—no LLM calls. Increase version first.");
      return;
    }

    const prevVersion = currentVersion - 1;
    const promptForThisVersion = versionPrompts[currentVersion] || "";
    if (!promptForThisVersion.trim()) {
      alert(`No prompt set for version ${currentVersion}. Please fill a prompt below first.`);
      return;
    }

    if (failedChunks.length === 0) {
      alert("No failed chunks to retry!");
      return;
    }

    setIsLoading(true);
    setUsedThreads(0);

    let newChunkState = [...chunks];
    const newFailed: number[] = [];
    const queue = [...failedChunks];

    // Clear old failures for a fresh start
    setFailedChunks([]);

    const worker = async () => {
      while (true) {
        let idx: number | undefined;
        if (queue.length > 0) {
          idx = queue.shift();
        }
        if (idx === undefined) {
          return;
        }
        const prevText = newChunkState[idx].chunkVersions[prevVersion];
        if (!prevText) {
          console.error(
            `Chunk #${idx + 1} missing V${prevVersion}. Cannot build V${currentVersion}.`
          );
          newFailed.push(idx);
          continue;
        }

        try {
          setUsedThreads((prev) => prev + 1);
          const userInput = prevText;
          const result = await callOpenAIGraphPrompt(
            systemPrompt,
            `PROMPT: ${promptForThisVersion}\n\nCHUNK:\n${userInput}`,
            "gpt-4"
          );
          newChunkState = newChunkState.map((chunk, i) => {
            if (i === idx) {
              return {
                ...chunk,
                chunkVersions: {
                  ...chunk.chunkVersions,
                  [currentVersion]: result,
                },
              };
            }
            return chunk;
          });
          setChunks([...newChunkState]);
        } catch (err) {
          console.error("Error in concurrency worker for chunk #", idx, err);
          newFailed.push(idx);
        } finally {
          setUsedThreads((prev) => prev - 1);
        }
      }
    };

    const tasks: Promise<void>[] = [];
    for (let t = 0; t < maxThreads; t++) {
      tasks.push(worker());
    }

    await Promise.all(tasks);
    setIsLoading(false);

    setFailedChunks(newFailed);
    if (newFailed.length > 0) {
      alert(`Some chunks (${newFailed.length}) still failed. You may retry again.`);
    } else {
      alert("All previously failed chunks have now completed successfully!");
    }
  };

  // Single-chunk run
  const handleRunPromptForChunk = async (chunkIndex: number) => {
    if (currentVersion <= 1) {
      alert("Version 1 is base text—no LLM calls. Increase version first.");
      return;
    }
    const prevVersion = currentVersion - 1;
    const promptForThisVersion = versionPrompts[currentVersion] || "";
    if (!promptForThisVersion.trim()) {
      alert(`No prompt set for version ${currentVersion}.`);
      return;
    }

    // If chunk already done
    if (chunks[chunkIndex].chunkVersions[currentVersion]) {
      alert(`Chunk #${chunkIndex + 1} is already computed for version ${currentVersion}.`);
      return;
    }

    const prevText = chunks[chunkIndex].chunkVersions[prevVersion];
    if (!prevText) {
      alert(
        `Chunk #${chunkIndex + 1} is missing version ${prevVersion}. Can't build version ${currentVersion}.`
      );
      return;
    }

    setIsLoading(true);
    try {
      const userInput = prevText;
      const result = await callOpenAIGraphPrompt(
        systemPrompt,
        `PROMPT: ${promptForThisVersion}\n\nCHUNK:\n${userInput}`,
        "gpt-4o"
      );
      setChunks((prev) => {
        const newArr = [...prev];
        newArr[chunkIndex] = {
          ...newArr[chunkIndex],
          chunkVersions: {
            ...newArr[chunkIndex].chunkVersions,
            [currentVersion]: result,
          },
        };
        return newArr;
      });
      // If chunkIndex was in failedChunks, remove it now that it succeeded
      setFailedChunks((old) => old.filter((idx) => idx !== chunkIndex));
    } catch (err) {
      console.error("Error running single-chunk LLM:", err);
      // Mark chunkIndex as failed
      setFailedChunks((prev) => (prev.includes(chunkIndex) ? prev : [...prev, chunkIndex]));
    } finally {
      setIsLoading(false);
    }
  };

  // Version navigation
  const handleChangeVersion = (delta: number) => {
    const newVer = Math.max(1, currentVersion + delta);
    setCurrentVersion(newVer);
  };

  // Generate Active Ingredient Report
  const handleGenerateActiveIngredientReport = () => {
    const newNamesFound: string[] = [];
    chunks.forEach((chunk) => {
      const v2output = chunk.chunkVersions[2];
      if (v2output) {
        const names = parseActiveIngredientNames(v2output);
        newNamesFound.push(...names);
      }
    });


    const updated = [...activeIngredientsSoFar, ...newNamesFound];
    setActiveIngredientsSoFar(updated);

    // Download as .txt
    const fileContent = updated.join("\n");
    const blob = new Blob([fileContent], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "active_ingredients_report.txt";
    link.click();
    URL.revokeObjectURL(url);
  };

  // Check if at least one chunk has version 2 text
  const isAnyV2Chunk = chunks.some((c) => c.chunkVersions[2]);

  // Render chunk data
  const renderChunks = () => {
    if (!chunks.length) {
      return <p className="text-sm text-gray-400">No chunks yet...</p>;
    }
    return (
      <div>
        <p className="text-xs mb-2 italic text-gray-200">
          Number of chunks: {chunks.length}
        </p>
        {chunks.map((chunk, idx) => {
          const versionText = chunk.chunkVersions[currentVersion] || "";
          if (!versionText) {
            // Not computed chunk
            return (
              <div
                key={idx}
                className="p-2 mb-2 border border-gray-600 rounded bg-gray-700"
              >
                <p className="text-xs text-gray-400 mb-1">
                  Chunk #{idx + 1}, version {currentVersion} - [Not computed yet]
                </p>
                <button
                  onClick={() => handleRunPromptForChunk(idx)}
                  disabled={isLoading}
                  className="px-3 py-1 bg-blue-500 rounded text-white text-sm disabled:bg-gray-600"
                >
                  {isLoading ? "Running..." : `Compute V${currentVersion}`}
                </button>
              </div>
            );
          }

          // Overlap highlighting
          const length = versionText.length;
          const overlap = rcsChunkOverlap;
          let startHighlight = "";
          let mid = versionText;
          let endHighlight = "";

          if (length > overlap) {
            startHighlight = versionText.slice(0, overlap);
            mid = versionText.slice(overlap);
          }
          if (mid.length > overlap) {
            const endPart = mid.slice(mid.length - overlap);
            const middlePart = mid.slice(0, mid.length - overlap);
            endHighlight = endPart;
            mid = middlePart;
          }

          return (
            <div
              key={idx}
              className="p-2 mb-2 border border-gray-600 rounded bg-gray-700"
            >
              <p className="text-xs text-gray-400 mb-1">
                Chunk #{idx + 1}, version {currentVersion} (length: {versionText.length})
              </p>
              <p className="text-sm whitespace-pre-wrap">
                {startHighlight && <span style={{ color: "red" }}>{startHighlight}</span>}
                {mid}
                {endHighlight && <span style={{ color: "red" }}>{endHighlight}</span>}
              </p>

              <button
                onClick={() => handleRunPromptForChunk(idx)}
                disabled={isLoading}
                className="mt-2 px-3 py-1 bg-blue-500 rounded text-white text-sm disabled:bg-gray-600"
              >
                {isLoading ? "Running..." : "Re-run Prompt on This Chunk"}
              </button>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="h-screen bg-gray-900 text-white">
      {/* CHUNKIFY MODAL */}
      {showChunkifyModal && (
        <div
          className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50"
          onClick={() => setShowChunkifyModal(false)}
        >
          <div
            className="bg-gray-800 p-6 rounded shadow-lg max-w-sm w-full"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-xl font-bold mb-4 text-green-400">Chunkify Settings</h3>
            <label className="block text-xs text-gray-200 mb-1 mt-2">
              Max Chunk Size:
            </label>
            <input
              type="number"
              className="w-full text-black rounded px-2 py-1 mb-2"
              value={rcsChunkSize}
              onChange={(e) => setRcsChunkSize(parseInt(e.target.value) || 1)}
            />

            <label className="block text-xs text-gray-200 mb-1 mt-2">
              Overlap:
            </label>
            <input
              type="number"
              className="w-full text-black rounded px-2 py-1 mb-4"
              value={rcsChunkOverlap}
              onChange={(e) => setRcsChunkOverlap(parseInt(e.target.value) || 0)}
            />

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowChunkifyModal(false)}
                className="px-3 py-1 bg-gray-600 rounded text-white text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleChunkifyConfirm}
                className="px-3 py-1 bg-blue-500 rounded text-white text-sm"
              >
                Split Text
              </button>
            </div>
          </div>
        </div>
      )}

      {/* CONCURRENCY MODAL */}
      {showConcurrencyModal && (
        <div
          className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50"
          onClick={() => setShowConcurrencyModal(false)}
        >
          <div
            className="bg-gray-800 p-6 rounded shadow-lg max-w-sm w-full"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-xl font-bold mb-4 text-blue-400">Concurrency Settings</h3>

            <label className="block text-xs text-gray-200 mb-2">
              Number of Threads (1-20):
            </label>
            <input
              type="number"
              className="w-full text-black rounded px-2 py-1 mb-4"
              value={maxThreads}
              onChange={(e) => {
                let v = parseInt(e.target.value) || 1;
                if (v < 1) v = 1;
                if (v > 20) v = 20;
                setMaxThreads(v);
              }}
            />

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowConcurrencyModal(false)}
                className="px-3 py-1 bg-gray-600 rounded text-white text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleConcurrencyConfirm}
                className="px-3 py-1 bg-blue-500 rounded text-white text-sm"
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 🚀 Activity / Graph modal */}
      {showGraphModal && (
        <GraphModal
          logLines={logLines}
          initialGraphUrl={initialGraphUrl}
          refinedGraphUrl={refinedGraphUrl}
          monteCarloPlotUrl={monteCarloPlotUrl}
          top10Json={top10JsonUrl}
          onClose={handleCloseGraphModal}
        />
      )}

      {/* MAIN LAYOUT */}
      <Split
        className="flex h-full"
        sizes={[50, 50]}
        minSize={300}
        gutterSize={10}
        gutterAlign="center"
        snapOffset={30}
      >
        {/* LEFT: ScribeOCR panel */}
        <div className="bg-gray-800 p-4 shadow-lg flex flex-col">
          <h2 className="text-lg font-bold mb-2">Scribe OCR</h2>
          <p className="text-sm text-gray-300 mb-2">
            Use the embedded Scribe OCR tool to upload/convert your PDF.
            Then download the resulting .txt file and upload it on the right panel.
          </p>
          <iframe
            src="https://scribeocr.com/"
            style={{ flex: 1, border: "none" }}
            title="ScribeOCR"
          />
        </div>

        {/* RIGHT: main control panel */}
        <div className="bg-gray-800 p-4 shadow-lg flex flex-col relative">
          {/* GLOBAL VERSION SELECTOR */}
          <div className="absolute top-2 right-2 flex items-center gap-2 bg-gray-700 p-2 rounded">
            <button
              onClick={() => handleChangeVersion(-1)}
              className="bg-gray-500 px-2 py-1 rounded text-white text-sm"
            >
              -
            </button>
            <span className="text-white text-sm font-bold">V{currentVersion}</span>
            <button
              onClick={() => handleChangeVersion(1)}
              className="bg-gray-500 px-2 py-1 rounded text-white text-sm"
            >
              +
            </button>
          </div>

          {/* Upload & chunkify row */}
          <div className="mb-2 flex items-center gap-2">
            <h2 className="text-lg font-bold text-blue-400">Upload Extracted Text</h2>
            <div className="border-l border-gray-600 h-6" />
            <button
              onClick={() => setShowChunkifyModal(true)}
              className="px-3 py-1 bg-purple-500 rounded text-sm text-white"
            >
              Chunkify
            </button>
            <button
              onClick={handleDropCache}
              className="px-3 py-1 bg-red-500 rounded text-sm text-white"
            >
              Drop Cache
            </button>
            {/* NEW: Retry Failed Chunks button */}
            <button
              onClick={handleRetryFailedChunks}
              disabled={failedChunks.length === 0}
              className="px-3 py-1 bg-blue-600 rounded text-sm text-white disabled:bg-gray-600"
            >
              Retry Failed Chunks
            </button>
          </div>

          {/* File input */}
          <input
            type="file"
            accept=".txt"
            onChange={handleUploadTextFile}
            className="mb-2 text-sm text-gray-200 border border-gray-600 rounded-lg cursor-pointer bg-gray-700 focus:outline-none p-2"
          />

          {/* Word/char stats */}
          {extractedText && (
            <div className="text-xs text-gray-300 mb-2">
              <p>Word Count: {wordCount}</p>
              <p>Character Count: {charCount}</p>
            </div>
          )}

          {/* System prompt */}
          <div className="mb-2">
            <label className="block text-xs text-gray-200 mb-1">
              System Prompt (optional):
            </label>
            <textarea
              className="w-full h-12 p-2 rounded bg-gray-700 text-gray-200 border border-gray-600 text-xs"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            />
          </div>

          {/* Button to trigger handleChunksToGraph function */}
          <button
            onClick={handleChunksToGraph}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded text-sm"
          >
            Generate Graph
          </button>

          {/* Prompt for current version (not needed for V1) */}
          {currentVersion > 1 ? (
            <div className="mb-2">
              <label className="block text-xs text-gray-200 mb-1">
                Prompt for version V{currentVersion}:
              </label>
              <textarea
                className="w-full h-16 p-2 rounded bg-gray-700 text-gray-200 border border-gray-600"
                value={versionPrompts[currentVersion] || ""}
                onChange={(e) =>
                  setVersionPrompts((prev) => ({
                    ...prev,
                    [currentVersion]: e.target.value,
                  }))
                }
                placeholder={`Enter a prompt to transform V${currentVersion - 1} => V${currentVersion}...`}
              />
            </div>
          ) : (
            <p className="text-xs text-gray-400 mb-2">
              Version 1 is the base text from chunkify. No prompt needed.
            </p>
          )}

          {/* "Run All" => concurrency modal (only if version > 1) */}
          {currentVersion > 1 && (
            <button
              onClick={handleOpenConcurrencyModal}
              disabled={!chunks.length || isLoading}
              className="mb-1 px-4 py-2 bg-orange-500 rounded text-white text-sm disabled:bg-gray-600"
            >
              {isLoading
                ? `Building V${currentVersion} for All...`
                : `Run All Chunks => V${currentVersion}`}
            </button>
          )}

          {/* Active Ingredient button only on version 2. */}
          {currentVersion === 2 && (
            <button
              onClick={handleGenerateActiveIngredientReport}
              disabled={!isAnyV2Chunk}
              className={`mb-1 px-4 py-2 rounded text-white text-sm ${
                !isAnyV2Chunk ? "bg-gray-600" : "bg-green-500"
              }`}
            >
              Generate Active Ingredient Report
            </button>
          )}

          {/* Show concurrency usage if we are in concurrency mode */}
          {isLoading && (
            <p className="text-xs text-gray-300">
              Concurrency: {maxThreads} threads, used threads: {usedThreads}
            </p>
          )}

          {/* Chunks display */}
          <div className="flex-1 overflow-auto mt-2">
            <h3 className="text-md font-bold text-white mb-2">
              Chunks - Version {currentVersion}
            </h3>
            {renderChunks()}
          </div>
        </div>
      </Split>
    </div>
  );
};

export default MergedUnstructuredImport;
