import React, { useEffect, useState } from "react";
import Papa from "papaparse";

// -------------------------------------------------------------
// Types
// -------------------------------------------------------------
interface Product {
  id: number;
  product_id: string;
  name: string;
  image_url: string;
  brand_name: string;
  brand_id: string;
  sell_price: number;
  list_price: number;
  discount: string;
  discount_value: number;
  color_css?: string;
  url: string;
  ingredients: string;
  batch_id?: string;
  created_at?: string;
  updated_at?: string;
  category: string;
}

interface Patent {
  patentId: string;
  title: string;
  similarityScore?: number;
  colorStatus: "idle" | "red" | "green";
}

interface AnalysisResult {
  patentId: string;
  similarityScore: number;
}

// -------------------------------------------------------------
// Constants
// -------------------------------------------------------------
const ITEMS_PER_PAGE = 100;

// We'll generate 200 patents for demonstration
const initialPatents: Patent[] = Array.from({ length: 200 }, (_, i) => ({
  patentId: `US${(i + 1).toString().padStart(7, "0")}`,
  title: `Patent #${i + 1}`,
  colorStatus: "idle" as const,
}));

// -------------------------------------------------------------
// CSV Parsing Helper
// -------------------------------------------------------------
function parseCsvToProducts(csvText: string): Product[] {
  const results = Papa.parse(csvText, {
    header: true,         // Papa Parse uses first row as keys
    dynamicTyping: true,  // Convert numbers automatically
    skipEmptyLines: true,
  });
  // results.data is an array of objects keyed by your CSV headers
  return results.data as Product[];
}

// -------------------------------------------------------------
// Main Component
// -------------------------------------------------------------
const App: React.FC = () => {
  // Full list of products loaded from CSV
  const [allProducts, setAllProducts] = useState<Product[]>([]);
  // Current page (0-based)
  const [page, setPage] = useState(0);
  // Index on the current page (0-based)
  const [indexOnPage, setIndexOnPage] = useState(0);

  // Currently displayed product (derived from page & indexOnPage)
  const product = React.useMemo(() => {
    const startIndex = page * ITEMS_PER_PAGE;
    const curIndex = startIndex + indexOnPage;
    return allProducts[curIndex] ?? null;
  }, [allProducts, page, indexOnPage]);

  // Total pages
  const totalPages = React.useMemo(() => {
    return Math.ceil(allProducts.length / ITEMS_PER_PAGE);
  }, [allProducts]);

  // Patents
  const [patents, setPatents] = useState<Patent[]>(initialPatents);

  // iFrame toggle
  const [showIframe, setShowIframe] = useState(false);

  // -------------------------------------------------------------
  // On mount: Fetch CSV and parse
  // -------------------------------------------------------------
  useEffect(() => {
    fetch("/products.csv")
      .then((res) => {
        if (!res.ok) throw new Error(`CSV not found: ${res.status}`);
        return res.text();
      })
      .then((csv) => {
        const parsed = parseCsvToProducts(csv);
        setAllProducts(parsed);
        // Reset to page 0, index 0
        setPage(0);
        setIndexOnPage(0);
      })
      .catch((err) => console.error("Failed to load products.csv:", err));
  }, []);

  // If we switch pages or load new data, reset the iframe
  useEffect(() => {
    setShowIframe(false);
  }, [page, indexOnPage]);

  // -------------------------------------------------------------
  // Navigation
  // -------------------------------------------------------------
  const handlePrev = () => {
    if (indexOnPage <= 0) {
      // Move to previous page if possible
      if (page > 0) {
        setPage(page - 1);
        const prevPageCount = Math.min(
          ITEMS_PER_PAGE,
          allProducts.length - (page - 1) * ITEMS_PER_PAGE
        );
        setIndexOnPage(prevPageCount - 1);
      } else {
        // Wrap around to last page
        setPage(totalPages - 1);
        const lastPageCount = allProducts.length % ITEMS_PER_PAGE || ITEMS_PER_PAGE;
        setIndexOnPage(lastPageCount - 1);
      }
    } else {
      setIndexOnPage(indexOnPage - 1);
    }
  };

  const handleNext = () => {
    const startIndex = page * ITEMS_PER_PAGE;
    const pageCount = Math.min(ITEMS_PER_PAGE, allProducts.length - startIndex);

    if (indexOnPage >= pageCount - 1) {
      // Move to next page if available
      if (page < totalPages - 1) {
        setPage(page + 1);
        setIndexOnPage(0);
      } else {
        // Wrap around to first page
        setPage(0);
        setIndexOnPage(0);
      }
    } else {
      setIndexOnPage(indexOnPage + 1);
    }
  };

  // Jump to a different page
  const handleNextPage = () => {
    if (page < totalPages - 1) {
      setPage(page + 1);
      setIndexOnPage(0);
    }
  };

  const handlePrevPage = () => {
    if (page > 0) {
      setPage(page - 1);
      setIndexOnPage(0);
    }
  };

  // -------------------------------------------------------------
  // Analyze Product
  // -------------------------------------------------------------
  const handleAnalyzeProduct = async () => {
    if (!product) return;

    // Reset patents
    setPatents((prev) =>
      prev.map((p) => ({
        ...p,
        similarityScore: undefined,
        colorStatus: "idle",
      }))
    );

    // Example fetch call (replace with real endpoint)
    try {
      const resp = await fetch("/api/analyzeProduct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product }),
      });
      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

      const results = (await resp.json()) as AnalysisResult[];
      results.forEach((res) => {
        setPatents((prev) =>
          prev.map((p) =>
            p.patentId === res.patentId
              ? {
                  ...p,
                  similarityScore: res.similarityScore,
                  colorStatus: res.similarityScore >= 0.8 ? "red" : "green",
                }
              : p
          )
        );
      });
    } catch (err) {
      console.error("Analyze Product failed:", err);
    }
  };

  // -------------------------------------------------------------
  // Render
  // -------------------------------------------------------------
  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Header */}
      <header className="p-4">
        <div className="flex items-center space-x-4">
          {/* “AG” Stylish Logo (inline SVG) */}
          <svg
            width="40"
            height="40"
            viewBox="0 0 48 48"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="text-blue-700"
          >
            <text
              x="50%"
              y="50%"
              textAnchor="middle"
              dy=".35em"
              fill="currentColor"
              fontSize="24"
              fontFamily="serif"
            >
              AG
            </text>
          </svg>

          {/* Dashboard Title */}
          <h1 className="text-xl font-bold text-gray-800">
            Patent Monitoring Dashboard
          </h1>
        </div>
      </header>

      {/* 3-column layout */}
      <div className="flex flex-row flex-1">
        {/* LEFT COLUMN */}
        <div className="w-[43%] flex flex-col p-4">
          <div className="flex-1 bg-purple-50/60 rounded-xl shadow-lg p-4 flex flex-col gap-4">
            {/* 1) Top sub-panel: iFrame area */}
            <div className="flex-1 border border-gray-300 rounded-lg p-2 overflow-hidden">
              {product && product.url && showIframe ? (
                <iframe
                  src={product.url}
                  title="Product Website"
                  className="w-full h-full"
                  sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-500 text-sm text-center">
                  {product?.url
                    ? "Click the product URL (below) to view offering in this space."
                    : "No product URL available"}
                </div>
              )}
            </div>

            {/* Divider line between top and bottom sub-panel */}
            <hr className="border-gray-300" />

            {/* 2) Bottom sub-panel: product details side-by-side */}
            <div className="flex-1 border border-gray-200 rounded-lg p-3">
              {product ? (
                <div className="flex flex-row gap-4 h-full">
                  {/* Photo on the left */}
                  <img
                    src={product.image_url}
                    alt={product.name}
                    className="w-40 h-40 object-contain self-start"
                  />
                  {/* Details on the right */}
                  <div className="flex flex-col flex-1">
                    <h2 className="text-base font-bold mb-1">{product.name}</h2>
                    <p className="text-sm text-gray-600">
                      Brand: {product.brand_name}
                    </p>
                    <p className="text-sm text-gray-600">
                      Price: ${product.sell_price} (List: ${product.list_price})
                    </p>
                    <p className="text-sm text-gray-600">
                      Discount: {product.discount}
                    </p>
                    <p className="text-sm text-gray-600 mb-2">
                      Category: {product.category}
                    </p>

                    {product.url && (
                    <div className="flex items-center gap-2 mb-3">
                      <button
                        onClick={() => setShowIframe((prev) => !prev)}
                        className="text-blue-600 underline text-sm text-left"
                      >
                        {showIframe ? "Close Product Webpage" : "View Product Webpage"}
                      </button>
                      <a
                        href={product.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 underline text-sm"
                      >
                        Go to Website
                      </a>
                    </div>
                  )}


                    {/* Subheading for Ingredients */}
                    <p className="font-semibold text-sm text-gray-700 mb-1">
                      Ingredients:
                    </p>
                    {/* Fixed-size scroll box */}
                    <div className="border border-gray-300 p-2 rounded overflow-auto text-sm text-gray-700 mb-2 h-40">
                      {product.ingredients}
                    </div>

                    {/* Prev / Next buttons (product-level) */}
                    <div className="flex justify-between">
                      <button
                        onClick={handlePrev}
                        className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300 text-sm"
                      >
                        Prev
                      </button>
                      <button
                        onClick={handleNext}
                        className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300 text-sm"
                      >
                        Next
                      </button>
                    </div>

                    {/* Page switcher (small buttons) */}
                    <div className="mt-3 flex items-center gap-2">
                      <button
                        onClick={handlePrevPage}
                        className="px-2 py-1 bg-gray-200 rounded hover:bg-gray-300 text-xs"
                      >
                        ◀ Prev Page
                      </button>
                      <span className="text-xs text-gray-600">
                        Page {page + 1} of {totalPages || 1}
                      </span>
                      <button
                        onClick={handleNextPage}
                        className="px-2 py-1 bg-gray-200 rounded hover:bg-gray-300 text-xs"
                      >
                        Next Page ▶
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-gray-500">Loading product...</p>
              )}
            </div>
          </div>
        </div>

        {/* CENTER COLUMN */}
        <div className="w-[13%]  flex flex-col p-4">
          <div className="flex-1 border-2 border-dashed border-blue-300 rounded-xl shadow-lg bg-yellow-50/60 flex flex-col items-center justify-center gap-4 p-4">
            <button
              onClick={handleAnalyzeProduct}
              className="bg-blue-500 text-white font-semibold px-6 py-3 rounded-full shadow hover:bg-blue-600 transition-colors"
            >
              Analyze Product
            </button>
            {/* More controls in future */}
          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div className="w-[43%] flex flex-col p-4">
          <div className="flex-1 bg-green-50/60 rounded-xl shadow-lg p-4 flex flex-col">
            <h2 className="text-lg font-semibold text-gray-700 mb-3">
              Monitoring ({patents.length}) patents
            </h2>
            <div className="flex-1 border border-gray-300 rounded p-2 overflow-visible flex flex-wrap gap-3">
              {patents.map((patent) => (
                <PatentLight key={patent.patentId} patent={patent} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// -------------------------------------------------------------
// PatentLight Component
// -------------------------------------------------------------
const PatentLight: React.FC<{ patent: Patent }> = ({ patent }) => {
  let bgColor = "bg-gray-400";
  if (patent.colorStatus === "red") bgColor = "bg-red-500";
  if (patent.colorStatus === "green") bgColor = "bg-green-500";

  return (
    <a
      href="/auto-correct-graph"
      className="group relative flex-shrink-0"
      style={{ width: "2rem", height: "2rem" }}
    >
      {/* Circle */}
      <div
        className={`w-full h-full rounded-full ${bgColor} transition-all 
          group-hover:ring-2 group-hover:ring-blue-300 group-hover:rounded-lg`}
      />
      {/* Tooltip */}
      <div
        className="absolute bottom-10 left-1/2 -translate-x-1/2 hidden 
          group-hover:block bg-blue-100 text-xs text-gray-800 
          px-2 py-1 rounded shadow z-10 whitespace-nowrap"
      >
        <div className="font-semibold">{patent.title}</div>
        <div>{patent.patentId}</div>
        {patent.similarityScore !== undefined && (
          <div>Sim: {(patent.similarityScore * 100).toFixed(1)}%</div>
        )}
      </div>
    </a>
  );
};

export default App;
