import React, { useRef, useEffect } from "react";

/** Represents an incoming circle traveling from left to the console center. */
interface IncomingCircle {
  text: string;
  radius: number;
  /** Parametric 0..1 for how far along the path it is. */
  t: number;
  /** 0..1 speed factor each frame. */
  speed: number;
  /** The actual control points used for a simple 2D curve to the console center. */
  x1: number; 
  y1: number; 
  cx: number; 
  cy: number; 
  x2: number; 
  y2: number; 
}

/** One word/node in a cluster traveling out to the right. */
interface OutgoingNode {
  text: string;
  radius: number;
  t: number; // param from 0..1
  speed: number;
  // Start point (console center)
  x1: number; 
  y1: number;
  // End point (fan-out region on the right)
  x2: number; 
  y2: number;
}

/** A link between two outgoing nodes. We’ll just store their indexes. */
interface OutgoingLink {
  source: number;
  target: number;
}

/** A cluster of outgoing nodes, traveling together, with links. */
interface OutgoingCluster {
  nodes: OutgoingNode[];
  links: OutgoingLink[];
}

export default function FancyBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // ~~~~~~~~~~~~~~~ CONFIGURABLE PARAMETERS ~~~~~~~~~~~~~~~
    const WORDS_PER_SECOND = 10; // set to 20 if you want it super intense
    // For circle size vs word length:
    let incomingSpeed = 0.00020;
    let baseRadius = 10;        // minimum base radius

    let wordSizeFactor = 1.5;   // each character adds this much radius
    // We’ll attempt to avoid overlap with a small “collisionGap”.
    const collisionGap = 4;     

    // Sizes for the “console” bounding box in the middle
    const consoleW = 400; // smaller console, as requested
    const consoleH = 200;

    // ~~~~~~~~~~~~~~~ RUNTIME STATE ~~~~~~~~~~~~~~~
    let width = 0;
    let height = 0;
    let animationFrameId = 0;

    /** The bounding box for the central console in canvas coords. */
    let consoleBox = { x: 0, y: 0, width: consoleW, height: consoleH };

    /** Collection of all incoming circles currently traveling from left -> console. */
    let incoming: IncomingCircle[] = [];

    /** Outgoing clusters that are traveling from console -> right. */
    let outgoingClusters: OutgoingCluster[] = [];

    // For collision avoidance among newly spawned circles
    function isOverlappingAny(circ: IncomingCircle): boolean {
      // We'll compute the circle’s (x1,y1) start. 
      // Because param t=0 => at (x1,y1). We’ll check against other circles with t=0.
      for (const other of incoming) {
        if (other.t > 0.05) continue; // only worry if the other is near spawn
        // distance between the two spawn points
        const dx = other.x1 - circ.x1;
        const dy = other.y1 - circ.y1;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < other.radius + circ.radius + collisionGap) {
          return true;
        }
      }
      return false;
    }

    // We spawn random words from your list or any set
    const possibleWords = [
      "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
      "Hello", "World", "NodeX", "DataPack", "Random",
      "Spark", "AI", "Token", "Compute", "Engine",
      "Stream", "Graph", "Matrix", "Omega", "Zeta"
    ];

    // ~~~~~~~~~~~~~~~ SPAWN INCOMING CIRCLES ~~~~~~~~~~~~~~~
    function spawnIncoming() {
      // pick random word
      const text = possibleWords[Math.floor(Math.random() * possibleWords.length)];
      const r = baseRadius + wordSizeFactor * text.length;

      // pick random Y start anywhere from 0..height
      // but we’ll do multiple attempts to avoid collision
      const maxAttempts = 10;
      let attempt = 0;
      let placed = false;

      while (attempt < maxAttempts && !placed) {
        const startY = Math.random() * height;
        // We want them to curve to the console center
        // Let’s define the console center:
        const cx = consoleBox.x + consoleBox.width / 2;
        const cy = consoleBox.y + consoleBox.height / 2;

        // We'll define a single control point for a simple quadratic curve
        // For a bit of random “arc,” we can set the control point above or below the midpoint
        const midX = (0 + cx) / 2;
        const midY = (startY + cy) / 2;
        const curveOffset = (Math.random() - 0.5) * 200; 
        // that shifts the control point up or down
        const cpx = midX + 100;  // pull it towards center slightly
        const cpy = midY + curveOffset;

        const newCirc: IncomingCircle = {
          text,
          radius: r,
          t: 0,
          speed: incomingSpeed + Math.random() * 0.003, // param speed
          x1: 0,       // left edge
          y1: startY,
          cx: cpx,     // curve control x
          cy: cpy,     // curve control y
          x2: cx,
          y2: cy,
        };

        // Check overlap with existing newly spawned circles
        if (!isOverlappingAny(newCirc)) {
          incoming.push(newCirc);
          placed = true;
        }
        attempt++;
      }
    }

    // ~~~~~~~~~~~~~~~ CREATE OUTGOING CLUSTER ~~~~~~~~~~~~~~~
    // Once an incoming circle reaches the console, we make a random cluster
    // of 1, 3, 5, or 10 words that are “linked” together, and send it out to the right.
    function spawnOutgoingCluster(baseText: string) {
      // random choice of cluster size
      const possibleSizes = [1, 3, 5, 10];
      const size = possibleSizes[Math.floor(Math.random() * possibleSizes.length)];

      // The cluster’s nodes
      const clusterNodes: OutgoingNode[] = [];
      // The cluster’s links
      const clusterLinks: OutgoingLink[] = [];

      // For random additional words:
      const pickWord = () =>
        possibleWords[Math.floor(Math.random() * possibleWords.length)];

      // console center
      const startX = consoleBox.x + consoleBox.width / 2;
      const startY = consoleBox.y + consoleBox.height / 2;

      // define a trapezoid fan-out region on the right
      // e.g. from x=width*(0.7..1.0), y in some band
      const endXMin = width * 0.85; // go slightly beyond the right edge
      const endXMax = width * 0.75;
      const endYMin = (startY - 200);
      const endYMax = (startY + 200);

      for (let i = 0; i < size; i++) {
        const w = i === 0 ? baseText : pickWord(); // first node is the baseText
        const r = baseRadius + wordSizeFactor * w.length;
        // pick a random end point in the trapezoid area
        const ex = endXMin + Math.random() * (endXMax - endXMin);
        const ey = endYMin + Math.random() * (endYMax - endYMin);

        clusterNodes.push({
          text: w,
          radius: r,
          t: 0,
          speed: 0.001 + Math.random() * 0.002,
          x1: startX,
          y1: startY,
          x2: ex,
          y2: ey,
        });
      }

      // Link them in some random pattern (or a chain)
      // For simplicity, let’s just link them in a chain from node0->node1->node2...
      for (let i = 0; i < size - 1; i++) {
        clusterLinks.push({ source: i, target: i + 1 });
      }

      // Add to the global list
      outgoingClusters.push({
        nodes: clusterNodes,
        links: clusterLinks,
      });
    }

    // ~~~~~~~~~~~~~~~ UPDATE & RENDER LOOP ~~~~~~~~~~~~~~~
    function update() {
      // 1) Move incoming circles
      for (let i = 0; i < incoming.length; i++) {
        const c = incoming[i];
        c.t += c.speed;
        if (c.t >= 1) {
          // Circle has arrived at the console
          // spawn an outgoing cluster
          spawnOutgoingCluster(c.text);
          // remove from array
          incoming.splice(i, 1);
          i--;
        }
      }

      // 2) Move outgoing clusters
      for (let i = 0; i < outgoingClusters.length; i++) {
        const clust = outgoingClusters[i];
        let allDone = true;
        for (const n of clust.nodes) {
          n.t += n.speed;
          if (n.t < 1) allDone = false;
        }
        // If entire cluster is done traveling (all t>=1), remove it
        if (allDone) {
          outgoingClusters.splice(i, 1);
          i--;
        }
      }
    }

    function render() {
       const canvas = canvasRef.current;
       if (!canvas) return;

       const ctx = canvas.getContext("2d");
       if (!ctx) return;

       ctx.clearRect(0, 0, width, height);

      // 1) Draw incoming circles
      for (const c of incoming) {
        // Quadratic bezier from (x1,y1) -> (x2,y2) with control (cx,cy)
        // Param is c.t
        const invT = 1 - c.t;
        const bx = invT * invT * c.x1 + 2 * invT * c.t * c.cx + c.t * c.t * c.x2;
        const by = invT * invT * c.y1 + 2 * invT * c.t * c.cy + c.t * c.t * c.y2;

        // Circle
        ctx.beginPath();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#00FFF0";
        ctx.shadowColor = "#00FFF0";
        ctx.arc(bx, by, c.radius, 0, Math.PI * 2);
        ctx.stroke();

        // Text
        ctx.fillStyle = "white";
        ctx.font = "13px sans-serif";
        const textW = ctx.measureText(c.text).width;
        ctx.fillText(c.text, bx - textW / 2, by + 4);
      }

      // 2) Draw the console box (for reference)
      ctx.save();
      ctx.strokeStyle = "#00ff0088";
      ctx.lineWidth = 2;
    //   ctx.strokeRect(consoleBox.x, consoleBox.y, consoleBox.width, consoleBox.height);
      ctx.restore();

      // 3) Draw outgoing clusters
      for (const clust of outgoingClusters) {
        // first compute each node’s (x,y)
        const positions: { x: number; y: number; r: number; text: string }[] = [];
        for (const n of clust.nodes) {
          const nx = n.x1 + (n.x2 - n.x1) * n.t;
          const ny = n.y1 + (n.y2 - n.y1) * n.t;
          positions.push({ x: nx, y: ny, r: n.radius, text: n.text });
        }

        // draw links
        ctx.strokeStyle = "cyan";
        ctx.lineWidth = 2;
        for (const l of clust.links) {
          const s = positions[l.source];
          const t = positions[l.target];
          ctx.beginPath();
          ctx.moveTo(s.x, s.y);
          ctx.lineTo(t.x, t.y);
          ctx.stroke();
        }

        // draw nodes
        for (const p of positions) {
          ctx.beginPath();
          ctx.fillStyle = "#0f0";
          ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
          ctx.fill();

          // text
          ctx.fillStyle = "black";
          ctx.font = "12px sans-serif";
          const tW = ctx.measureText(p.text).width;
          ctx.fillText(p.text, p.x - tW / 2, p.y + 4);
        }
      }
    }

    function loop() {
      update();
      render();
      animationFrameId = requestAnimationFrame(loop);
    }

    function onResize() {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;

      // Reposition the console
      consoleBox.x = (width - consoleW) / 2;
      consoleBox.y = (height - consoleH) / 2;
    }

    // ~~~~~~ SETUP ~~~~~~
    onResize();
    window.addEventListener("resize", onResize);

    // Main animation loop
    loop();

    // Spawn words at desired rate
    let spawnIntervalId: number | undefined;
    function startSpawning() {
      const intervalMs = 1000 / WORDS_PER_SECOND; 
      spawnIntervalId = window.setInterval(() => {
        spawnIncoming();
      }, intervalMs);
    }
    startSpawning();

    // Cleanup
    return () => {
      window.removeEventListener("resize", onResize);
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (spawnIntervalId) clearInterval(spawnIntervalId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full z-[0] bg-black"
    />
  );
}
