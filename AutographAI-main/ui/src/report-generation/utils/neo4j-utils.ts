// src/utils/neo4j-utils.ts
import neo4j, { Record } from "neo4j-driver";

// 1) Create driver instance
const driver = neo4j.driver(
  "bolt://127.0.0.1:7687",
  neo4j.auth.basic("neo4j", "your12345")
);

export async function fetchSubgraph(subgraphId: string, limitCount = 5) {
    const session = driver.session({ database: "neo4j" });
    try {
      const result = await session.run(
        `
        MATCH (node { subgraphId: $subgraphId })
        WITH collect(node) AS allNodes
        
        // UNWIND to turn it into rows, then reorder randomly and limit
        UNWIND allNodes AS n
        WITH n ORDER BY rand()
        LIMIT 5
        WITH collect(n) AS chosen

        UNWIND chosen AS n
        MATCH (n)-[r]->(m)
        WHERE m IN chosen
        RETURN n, r, m
        `,
        { subgraphId, limitCount }
      );
      return result.records; 
    } finally {
      await session.close();
    }
  }
  

// 3) (Optional) Convert Neo4j Records → Cytoscape elements
//    This example: every node "n" from the query => a Cytoscape "node" 
//    No edges are returned if we never MATCH relationships. 
//    If you need edges, MATCH them or see example expansions below.
// neo4j-utils.ts
export function recordsToCytoscape(records: Record[]) {
    const elements: any[] = [];
    const nodeIds = new Set<string>();
    const edgeIds = new Set<string>();
  
    // -------- Pass 1: Gather all nodes first --------
    for (const record of records) {
      const n = record.get("n");
      const m = record.get("m");
  
      if (n) {
        const nId = n.elementId;
        if (!nodeIds.has(nId)) {
          nodeIds.add(nId);
          elements.push({
            data: {
              id: nId,
              label: n.labels?.[0] || "Node2",
              ...n.properties
            }
          });
        }
      }
  
      if (m) {
        const mId = m.elementId;
        if (!nodeIds.has(mId)) {
          nodeIds.add(mId);
          elements.push({
            data: {
              id: mId,
              label: m.labels?.[0] || "Node2",
              ...m.properties
            }
          });
        }
      }
    }
  
    // -------- Pass 2: Gather all edges --------
    for (const record of records) {
      const r = record.get("r");
      if (r) {
        const rId = r.elementId;
        if (!edgeIds.has(rId)) {
          edgeIds.add(rId);
  
          const n = record.get("n");
          const m = record.get("m");
  
          elements.push({
            data: {
              id: rId,
              source: n.elementId,
              target: m.elementId,
              ...r.properties
            }
          });
        }
      }
    }
  
    return elements;
  }
  