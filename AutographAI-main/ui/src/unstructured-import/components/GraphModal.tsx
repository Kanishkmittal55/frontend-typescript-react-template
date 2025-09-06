import React, { useState, useEffect } from "react";
import Papa from "papaparse";

interface Props {
  logLines: string[];
  initialGraphUrl: string | null;
  refinedGraphUrl: string | null;
  monteCarloPlotUrl: string | null;
  top10Json: string | null;   // ← only remote csv now
  onClose: () => void;
}

type TopKRow = { product_id: string; match_percentage: string };
type ProductInfo = {
  product_id: string;
  name: string;
  url: string;
  image_url?: string;
  ingredients?: string;
};

const PRODUCTS_STATIC_PATH = "/products.csv";   // file you copy to frontend /public/data/

const GraphModal: React.FC<Props> = ({
  logLines,
  initialGraphUrl,
  refinedGraphUrl,
  monteCarloPlotUrl,
  top10Json,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<0 | 1 | 2 | 3>(0);
  const [productsMap, setProductsMap] = useState<Record<string, ProductInfo>>({});
  const [topProducts, setTopProducts] = useState<(TopKRow & ProductInfo)[]>([]);

  /* ---------- load products.csv ONCE from /public/data/ ---------- */
  useEffect(() => {
    fetch(PRODUCTS_STATIC_PATH)
      .then(r => {
        console.log("[products.csv] HTTP", r.status, r.ok ? "✓" : "✗", PRODUCTS_STATIC_PATH);
        return r.text();
      })
      .then(txt =>
        Papa.parse<ProductInfo>(txt, {
          header: true,
          complete: ({ data }) => {
            console.log(`[products.csv] parsed rows: ${data.length}`);   // 👈
            const map: Record<string, ProductInfo> = {};
            data.forEach(p => (map[p.product_id] = p));
            setProductsMap(map);
          },
        }),
      )
      .catch(err => console.error("products.csv load error:", err));
  }, []);  

  
  /* ---------- load Top‑10 JSON whenever its url appears ---------- */
  useEffect(() => {
    if (!top10Json) return;

    console.log("[top‑10] fetching", top10Json);
    fetch(top10Json)
      .then(r => r.json())
      .then((arr: (TopKRow & ProductInfo)[]) => {
        console.log("[top‑10] got", arr.length, "items");
        // join with productsMap just to enrich with image / ingredients
        const joined = arr.map(r => ({
          ...r,
          ...(productsMap[r.product_id] || {})
        }));
        setTopProducts(joined);
      })
      .catch(err => console.error("[top‑10] load failed:", err));
  }, [top10Json, productsMap]);


  const tabs = ["▶ Seed Graph", "▶ Refined Graph", "📦 Top 10", "📈 Monte‑Carlo"];

  return (
    <div className="fixed inset-0 z-50 bg-black bg-opacity-70 flex">
      <div className="bg-gray-900 flex-1 h-full w-full flex flex-col">
        {/* header */}
        <div className="flex justify-between items-center px-4 py-2 border-b border-gray-700">
          <div className="flex space-x-4">
            {tabs.map((t, i) => (
              <button
                key={i}
                onClick={() => setActiveTab(i as 0 | 1 | 2 | 3)}
                className={`px-3 py-1 rounded ${
                  activeTab === i
                    ? "bg-blue-600 text-white"
                    : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                } text-sm`}
              >
                {t}
              </button>
            ))}
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">
            &times;
          </button>
        </div>

        {/* body */}
        <div className="flex flex-1 overflow-hidden">
          {/* logs */}
          <div className="w-1/3 bg-gray-800 p-4 overflow-auto text-xs">
            {logLines.map((l, i) => (
              <pre key={i} className="whitespace-pre-wrap mb-1">
                {l}
              </pre>
            ))}
          </div>

          {/* content */}
          <div className="flex-1 overflow-auto bg-gray-800 p-4">
            {activeTab === 0 &&
              (initialGraphUrl ? (
                <iframe src={initialGraphUrl} className="w-full h-full border-0" />
              ) : (
                <p className="text-gray-400">Waiting for seed graph…</p>
              ))}

            {activeTab === 1 &&
              (refinedGraphUrl ? (
                <iframe src={refinedGraphUrl} className="w-full h-full border-0" />
              ) : (
                <p className="text-gray-400">Waiting for refined graph…</p>
              ))}

            {activeTab === 2 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {topProducts.map((p,i) => (
                  <a
                    key={p.product_id || i}
                    href={p.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex flex-col bg-gray-700 rounded-lg p-4 hover:bg-gray-600"
                  >
                    {p.image_url && (
                      <img
                        src={p.image_url}
                        alt={p.name}
                        className="h-32 w-full object-cover rounded mb-2"
                      />
                    )}
                    <h4 className="text-white font-semibold">{p.name}</h4>
                    <p className="text-green-400">
                      Match: {(() => {
                        const raw = p.match_percentage ?? "";
                        const num = typeof raw === "string" ? parseFloat(raw.replace(/[^0-9.\-]/g, "")) : NaN;
                        console.log(raw)
                        return isNaN(num) ? "N/A" : ` ${num.toFixed(1)}%`;
                      })()}
                    </p>
                    
                  </a>
                ))}
              </div>
            )}

            {activeTab === 3 &&
              (monteCarloPlotUrl ? (
                <img src={monteCarloPlotUrl} className="w-full h-full object-contain" />
              ) : (
                <p className="text-gray-400">Waiting for Monte‑Carlo plot…</p>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GraphModal;
