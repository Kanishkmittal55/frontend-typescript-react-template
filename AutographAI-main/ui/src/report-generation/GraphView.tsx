// src/report-generation/GraphView.tsx
import React from "react";

interface Triplet {
  id: string;
  subject: string;
  predicate: string;
  object: string;
}

interface GraphViewProps {
  triplets: Triplet[];
  // If you need to accept other props, add them here
}

const GraphView: React.FC<GraphViewProps> = ({ triplets }) => {
  // You could use triplets to dynamically populate the graph
  // or just show the iFrame for now
  return (
    <div
      className="w-full"
      style={{
        minHeight: "100vh",
        borderTop: "2px solid #999",
        padding: "1rem",
      }}
    >
      <h2 className="text-lg font-bold mb-2">Knowledge Graph & Controller</h2>

      {/* Graph area (iframe) */}
      <div style={{ width: "100%", height: "500px", border: "1px solid #ccc" }}>
        <iframe
          src="/graph/kg_1.html"
          style={{ width: "100%", height: "100%" }}
          title="Knowledge Graph"
        />
      </div>

      {/* Example: We can show triplets or a Neo4j controller below */}
      <div className="mt-4 p-2 border">
        <p className="font-bold">Neo4j / Graph Controls</p>
        <p>You can add a form here for queries, or display {triplets.length} triplets extracted.</p>
      </div>
    </div>
  );
};

export default GraphView;
