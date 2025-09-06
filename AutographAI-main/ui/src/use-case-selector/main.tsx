import React, { useState } from "react";
import { createRoot } from "react-dom/client";

// Import styles
import "@neo4j-ndl/base/lib/neo4j-ds-styles.css";
import "./index.css";

// Components
import FancyBackground from "./background2";
import Background from "./background";
import Hourglass from "./Hourglass";

function App() {
  const [showModal, setShowModal] = useState(false); // State to show/hide modal

  return (
    <div className="relative w-full min-h-screen flex flex-col items-center justify-center bg-black text-white overflow-hidden">
      {/* Background Animations */}
      <Background />
      <FancyBackground />

      {/* Center Content */}
      <div className="relative z-10 flex flex-col items-center p-4">
        {/* Title */}
        <h1 className="font-[cursive] text-6xl mb-6 text-[rgba(255,255,255,0.85)] drop-shadow-lg">
          AutoGraph
        </h1>

        {/* Hourglass */}
        <Hourglass />

        {/* EXPLORE WEB APP - Space Button */}
        <button
          onClick={() => setShowModal(true)}
          className="space-btn px-8 py-4 text-lg font-bold mt-8"
        >
          Explore Web App
        </button>

        {/* Glass Box Modal */}
        {showModal && (
          <div
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2
                      bg-white/10 backdrop-blur-xl p-6 rounded-lg border border-gray-600 shadow-lg
                      w-[300px] md:w-[350px] flex flex-col gap-4 items-center"
          >
            <h2 className="text-2xl font-bold text-white mb-4">Choose Feature</h2>

            <a
              href="use-cases/chat-with-kg/index.html"
              className="neon-button w-full text-center py-2"
            >
              💬 Chat With KG
            </a>
            <a
              href="use-cases/unstructured-import/index.html"
              className="neon-button w-full text-center py-2"
            >
              📂 Create Knowledge Base
            </a>
            <a
              href="use-cases/report-generation/index.html"
              className="neon-button w-full text-center py-2"
            >
              📑 Infringement Detection
            </a>
            <a
              href="http://localhost:7474"
              target="_blank"
              rel="noopener noreferrer"
              className="neon-button w-full text-center py-2"
            >
              ⚡ Go to Neo4j
            </a>

            {/* Close Button */}
            <button
              onClick={() => setShowModal(false)}
              className="mt-4 text-sm text-gray-300 hover:text-white"
            >
              ✖ Close
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// Mount
const container = document.getElementById("root");
const root = createRoot(container!);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
