import os
from typing import Optional
from components.company_report import CompanyReport
import requests
import logging
import networkx as nx
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd

from components.question_proposal_generator import (
    QuestionProposalGenerator,
)
import json
import re
from components.ollama_prompt_creator import run_with_chunk_logging
# ─── back in main.py (add near the top, after imports) ───────────────
from components.ollama_prompt_creator import EMBED_LOCK, Embedder
from components.ollama_prompt_creator import autograph_ai_pipeline
from components.ollama_prompt_creator import product_discovery_workflow
from components.self_attention_chunking_workflow import self_attention_chunking
from components.patent_summary_workflow import workflow_classifier
from components.ollama_summarize_cypher_result import OllamaSummarizeCypherResult
from components.summarize_cypher_result import SummarizeCypherResult
from components.ollamaText2cypher import OllamaText2Cypher
from components.text2cypher import Text2Cypher
from components.unstructured_data_extractor import (
    DataExtractor,
    DataExtractorWithSchema,
)
from driver.neo4j import Neo4jDatabase
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fewshot_examples import get_fewshot_examples
from llm.openai import OpenAIChat
from llm.ollamaapi import OllamaChat
from pydantic import BaseModel
from utils.unstructured_data_utils import save_intermediate_results_to_csv, data_to_cypher
from utils.tokenizers import gpt_tokenizer, llama_tokenizer, regex_tokenizer
from typing import List, Optional ,Union
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import random
import uuid
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path

# NEW ▶ server‑sent‑events response helper
from sse_starlette.sse import EventSourceResponse

LIGHTRAG_URL = os.getenv("LIGHTRAG_URL", "http://lightrag:9621")  
# or "http://localhost:9621" if in dev

class Payload(BaseModel):
    question: str
    api_key: Optional[str]
    model_name: Optional[str]

class chunksPayload(BaseModel):
    chunks: List[str]
    prompt: Optional[str] = None  # Making prompt optional

class ImportPayload(BaseModel):
    input: str
    neo4j_schema: Optional[str]
    api_key: Optional[str]


class questionProposalPayload(BaseModel):
    api_key: Optional[str]


# In‑memory registry of running/completed jobs
JobStatus: Dict[str, Dict[str, Union[str, int, List[str], None]]] = {}
job_status: JobStatus = {}


# Maximum number of records used in the context
HARD_LIMIT_CONTEXT_RECORDS = 10

neo4j_connection = Neo4jDatabase(
    host=os.environ.get("NEO4J_URL", "bolt://kg:7687"), # So we need to put the name of the container in case of docker
    user=os.environ.get("NEO4J_USER", "neo4j"),
    password=os.environ.get("NEO4J_PASS", "your12345"),
    database=os.environ.get("NEO4J_DATABASE", "neo4j"),
)

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

# Initialize LLM modules
openai_api_key = openai_key

# Define FastAPI endpoint
app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="/api"), name="static")


def parse_entries(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []
    x = x.strip()
    # 1) if it looks like JSON, try json.loads
    if x.startswith("[") and x.endswith("]"):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            pass
    # 2) otherwise, split on commas
    #    or just wrap single string in a list
    return [item.strip() for item in x.split(",") if item.strip()]

@app.on_event("startup")
def build_semantic_index():
    df_embed = pd.read_csv("./components/refined_df_latest.csv")
    # you can customise what constitutes “text”
    rows = []
    for _, r in df_embed.iterrows():
        entries = parse_entries(r["split_entries"])
        text = f"{r['representative_member']} " + " ".join(entries)
        rows.append({
            "text":   text,
            "labels": ["refined"],
            "name":   r["representative_member"][:60],
        })
    with EMBED_LOCK:
        Embedder.records.clear()
        Embedder.emb = None
        Embedder.add_records(rows)
    print(f"[semantic] re-indexed {len(rows)} refined rows")
    print(f"[startup] loaded {len(rows)} records into semantic index")

@app.get("/lightrag/chunks")
def get_lightrag_chunks():
    """Proxy to LightRAG /chunks endpoint."""
    try:
        resp = requests.get(f"{LIGHTRAG_URL}/chunks")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LightRAG error: {str(e)}")
    
class ExtractEntitiesPayload(BaseModel):
    chunk_id: str
    custom_prompt: str = None

@app.post("/lightrag/chunks/extract_entities")
def post_extract_entities(payload: ExtractEntitiesPayload):
    """Proxy to LightRAG /chunks/extract_entities endpoint."""
    try:
        data = {
            "chunk_id": payload.chunk_id
        }
        if payload.custom_prompt:
            data["custom_prompt"] = payload.custom_prompt
        resp = requests.post(f"{LIGHTRAG_URL}/chunks/extract_entities", json=data)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LightRAG error: {str(e)}")

@app.get("/lightrag/chunks/{chunk_id}/graph")
def get_chunk_graph(chunk_id: str):
    """Proxy sub-graph retrieval."""
    try:
        url = f"{LIGHTRAG_URL}/chunks/{chunk_id}/graph"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LightRAG error: {str(e)}")


@app.post("/questionProposalsForCurrentDb")
async def questionProposalsForCurrentDb(payload: questionProposalPayload):
    # if not openai_api_key and not payload.api_key:
    #     raise HTTPException(
    #         status_code=422,
    #         detail="Please set OPENAI_API_KEY environment variable or send it as api_key in the request body",
    #     )
    # api_key = openai_api_key if openai_api_key else payload.api_key

    # questionProposalGenerator = QuestionProposalGenerator(
    #     database=neo4j_connection,
    #     llm=OpenAIChat(
    #         openai_api_key=api_key,
    #         model_name="gpt-4o",
    #         max_tokens=512,
    #         temperature=0.8,
    #     ),
    # )

    return "This is the only question you get."


@app.get("/hasapikey")
async def hasApiKey():
    if openai_api_key:
        return JSONResponse(content={
            "output": True,
            "key": openai_api_key
        })
    else:
        return JSONResponse(content={
            "output": False,
            "key": None,
            "message": "No API key detected"
        })


class SemanticSearchPayload(BaseModel):
    q: str
    top_k: int = 10
    n_gram: int = 3   # (UI slider only)


def semantic_search(body: SemanticSearchPayload):
    if not body.q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    with EMBED_LOCK:
        results = Embedder.search(body.q, body.top_k)

    return {"matches": results}


@app.websocket("/text2text")
async def websocket_endpoint(websocket: WebSocket):
    """
    Single handler for semantic + LLM.  Sends back:
      - {"type":"debug", detail:...} at handshake
      - {"type":"start"} when LLM begins
      - {"type":"stream", output: chunk} for each token
      - {"type":"end", output: null} when done
      - {"type":"error", detail:...} on any user or server error
    """
    # helper senders
    async def sendDebugMessage(msg: str):
        await websocket.send_json({"type": "debug", "detail": msg})

    async def sendErrorMessage(msg: str):
        await websocket.send_json({"type": "error", "detail": msg})

    async def onToken(token):
        """
        Called for each streamed chunk from OpenAIChat.generateStreaming.
        token will be a dict with choices[0].delta.content etc.
        """
        delta = token.choices[0].delta  # using modern openai-python Client
        content = getattr(delta, "content", None)
        if content:
            # if finish reason is stop, send an end packet
            if getattr(token.choices[0], "finish_reason", None) == "stop":
                await websocket.send_json({"type": "end", "output": content})
            else:
                await websocket.send_json({"type": "stream", "output": content})

    # accept connection
    await websocket.accept()
    await sendDebugMessage("connected")
    chatHistory = []

    try:
        while True:
            data = await websocket.receive_json()
            print("The Type of data : ", data.get("type"))

            # only one frame type now
            if data.get("type") != "semantic":
                await sendErrorMessage("expecting semantic frame")
                continue

            # print("The Type of data : ", data.get("type"))

            question = (data.get("question") or "").strip()
            await sendDebugMessage("received question: " + question)

            top_k     = int(data.get("top_k", 3))

            if not question:
                await sendErrorMessage("empty question")
                continue

            # 1) semantic search
            with EMBED_LOCK:
                matches = Embedder.search(question, top_k)

            # 2) build prompt
            context_text = "\n".join(
                f"- {m['preview']}  (score {m['score']:.3f})"
                for m in matches
            )

            print("The matches are :", matches)
            full_prompt = (
                f"# Question:\n{question}\n\n"
                f"# Context matches:\n{context_text}\n\n"
                "# Instructions:\n"
                "Use the context above to answer as best as you can."
            )

            # 3) record user turn & notify client we're starting
            chatHistory.append({"role": "user", "content": full_prompt})
            await websocket.send_json({"type": "start"})

            # 4) stream from OpenAI
            llm = OpenAIChat(
                openai_api_key=openai_key,
                model_name="gpt-4o"
            )
            # this will invoke onToken() for each chunk
            await llm.generateStreaming(
                [{"role": "user", "content": full_prompt}],
                onTokenCallback=onToken
            )

            # 5) record assistant turn
            chatHistory.append({"role": "assistant", "content": full_prompt})

            await websocket.send_json({"type": "end"})
            await sendDebugMessage("output done")

    except WebSocketDisconnect:
        print("🔌 client disconnected")
    except Exception as e:
        # any other error
        await sendErrorMessage(f"server error: {e}")
        print("❌", e)
        try:
            await websocket.close()
        except:
            pass

# Initialize the OllamaChat instance
ollama_chat = OllamaChat(model_name="llama3.1", max_tokens=1000, temperature=0.7)

@app.websocket("/ollama/text2text")
async def websocket_endpoint(websocket: WebSocket):
    print("I am here")
    async def send_debug_message(message):
        await websocket.send_json({"type": "debug", "detail": message})

    async def send_error_message(message):
        await websocket.send_json({"type": "error", "detail": message})

    async def onToken(token):
        delta = token["choices"][0]["delta"]
        if "content" not in delta:
            return
        content = delta["content"]
        if token["choices"][0]["finish_reason"] == "stop":
            await websocket.send_json({"type": "end", "output": content})
        else:
            await websocket.send_json({"type": "stream", "output": content})

    async def send_output(content, is_end=False):
        message_type = "end" if is_end else "stream"
        await websocket.send_json({"type": message_type, "output": content})

    await websocket.accept()
    await send_debug_message("connected")
    chat_history = []

    try:
        while True:
            data = await websocket.receive_json()
            print("The recieved data is : ", data)
            if not openai_api_key and not data.get("api_key"):
                raise HTTPException(
                    status_code=422,
                    detail="Please set OPENAI_API_KEY environment variable or send it as api_key in the request body",
                )
            api_key = openai_api_key if openai_api_key else data.get("api_key")

            
            summarize_results = OllamaSummarizeCypherResult()

            print("Summarize Results :", summarize_results)
            # Convert the result to a string
            summarize_results_str = str(summarize_results)

            # If it's a dictionary or JSON-like structure, use json.dumps() for a pretty string

            if isinstance(summarize_results, dict) or isinstance(summarize_results, list):
                summarize_results_str = json.dumps(summarize_results, indent=2)

            # Print the string representation
            print("Summarize Results (String):", summarize_results_str)

            text2cypher = OllamaText2Cypher(
                database=neo4j_connection,
                llm=ollama_chat,
                cypher_examples=get_fewshot_examples(api_key),
            )

            if "type" not in data:
                await websocket.send_json({"error": "missing type"})
                continue

            print("The type of json message received is :" , data["type"])

            if data["type"] == "question":
                try:
                    print("The data type is a question")
                    question = data.get("question")
                    if not question:
                        await send_error_message("Missing question in request.")
                        continue

                    chat_history.append({"role": "user", "content": question})
                    await send_debug_message("received question: " + question)

                    try:
                        results = text2cypher.run(question, chat_history)
                        print("text to cypher results obatined : ", results)
                    except Exception as e:
                        await send_error_message(str(e))
                        continue
                    if results == None:
                        await send_error_message("Could not generate Cypher statement")
                        continue
                    print("Results are not none : ", results["generated_cypher"])
                    await websocket.send_json(
                        {
                            "type": "start",
                        }
                    )
                    print("Reached here")
                    output = await summarize_results.run_async(
                        question,
                        results["output"][:HARD_LIMIT_CONTEXT_RECORDS],
                        callback=onToken,
                    )
                    print("Reached here  : ", output)
                    print("The contents is : ", output)
                    chat_history.append({"role": "system", "content": output})
                    await websocket.send_json(
                        {
                            "type": "end",
                            "output": output,
                            "generated_cypher": results["generated_cypher"],
                        }
                    )

                except Exception as e:
                    await send_error_message(str(e))
                await send_debug_message("output done")

    except WebSocketDisconnect:
        print("disconnected")


# ---------------------------------------------------------------------------
# Process one job and stream progress into JobStatus
# ---------------------------------------------------------------------------
async def process_chunks_and_create_graph(
    job_id: str,
    chunks: List[str],
    prompt: Optional[str]
):

    # ① initialise record
    JobStatus[job_id] = {
        "status":   "Queued",
        "progress": 0,
        "logs":     [],
        "files":    []          
    }

    # ② progress callback used by autograph_ai_pipeline
    def progress_update(
        stage: str,
        pct: int,
        *,
        file_url: str | None = None,          # new
        file_path: str | None = None       # legacy support
    ):
        js = JobStatus[job_id]
        js["status"]   = stage
        js["progress"] = pct
        js.setdefault("logs", []).append(f"{stage} … {pct}%")

        # ---- normalise payload -------------------------------------------
        if file_url:
            url = file_url                    
        elif file_path:
            url = "/static/" + os.path.relpath(file_path, "/api").replace("\\", "/")
        else:
            url = None
        # ------------------------------------------------------------------

        if url:
            js.setdefault("files", []).append(url)
            js["logs"].append(f"✔︎ File ready → {url.rsplit('/',1)[-1]}")

    # ③ run the heavy worker in a thread
    try:
        html_path = await asyncio.to_thread(
            autograph_ai_pipeline,
            chunks,
            prompt,
            "openai",
            progress_cb=progress_update
        )

        JobStatus[job_id].update(
            status="Complete",
            progress=100,
            download_url=f"/downloadGraph/{job_id}?file=0",
            html_path=html_path,
        )
        JobStatus[job_id]["logs"].append("✔︎ Job finished — all files ready")

    except Exception as exc:
        JobStatus[job_id].update(status="Error", progress=100)
        JobStatus[job_id]["logs"].append(f"✖︎ Error: {exc}")
        raise






@app.post("/chunksToKG")
async def chunks_to_kg(payload: chunksPayload, background_tasks: BackgroundTasks):
    """
    Accepts chunks and an optional prompt, and starts a background task
    to process them and create a graph. Returns a job ID to track progress.
    """
    if not payload.chunks:
        raise HTTPException(status_code=400, detail="'chunks' list is empty")

    job_id = uuid.uuid4().hex[:8]
    JobStatus[job_id] = {"status": "Queued", "progress": 0, "logs": []}

    background_tasks.add_task(process_chunks_and_create_graph, job_id, payload.chunks, payload.prompt)
    return {"job_id": job_id, "message": "Graph creation started"}

@app.get("/jobStatus/{job_id}")
async def job_status_endpoint(job_id: str):
    if job_id not in JobStatus:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return JobStatus[job_id]

# —— NEW: live log stream ——
@app.get("/jobLogs/{job_id}")
async def job_logs(job_id: str):
    if job_id not in JobStatus:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    async def event_generator():
        last_log = last_file = 0
        while True:
            js = JobStatus[job_id]

            # ① send new log lines
            logs = js.get("logs", [])
            while last_log < len(logs):
                yield {"event": "log", "data": logs[last_log]}
                last_log += 1

            # ② send new files
            files = js.get("files", [])
            while last_file < len(files):
                # fname = os.path.basename(files[last_file])
                # url   = f"/downloadGraph/{job_id}?file={last_file}"
                url   = files[last_file]                   # "/static/src/Seed-Graph.html"
                fname = url.rsplit("/", 1)[-1]
                payload = json.dumps({"name": fname, "url": url})
                yield {"event": "file", "data": payload}
                last_file += 1

            # ③ exit when finished and everything delivered
            if js["status"] in {"Complete", "Error"} \
               and last_log >= len(logs) and last_file >= len(files):
                yield {"event": "end", "data": js['status']}
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())

@app.get("/downloadGraph/{job_id}")
async def download_graph(job_id: str, file: int = 0):
    data = JobStatus.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    try:
        file_path = data["files"][file]        # may raise IndexError
    except (KeyError, IndexError):
        raise HTTPException(status_code=404, detail="File index invalid")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=500, detail="File missing on server")

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=os.path.basename(file_path),
    )


@app.post("/data2cypher")
async def root(payload: ImportPayload):
    """
    Takes an input and created a Cypher query
    """
    if not openai_api_key and not payload.api_key:
        raise HTTPException(
            status_code=422,
            detail="Please set OPENAI_API_KEY environment variable or send it as api_key in the request body",
        )
    api_key = openai_api_key if openai_api_key else payload.api_key

    try:
        result = ""

        intermediate_results = []
        
        # Initialize LLM 
        llm = OpenAIChat(
            openai_api_key=api_key, model_name="gpt-4o", max_tokens=4000
        )

        # Step 1 : Run Extraction
        if not payload.neo4j_schema:
            extractor = DataExtractor(llm=llm)
            result, chunks = extractor.run_with_chunk_logging(data=payload.input)
        else:
            extractor = DataExtractorWithSchema(llm=llm)
            result = extractor.run(schema=payload.neo4j_schema, data=payload.input)

        
        # Log Extraction result
        print("Extracted result: " + str(result))

        # Log extraction result
        intermediate_results.append({
            "stage": "Extraction",
            "chunks": chunks,
            "data": result
        })

        # Step 2: Data Disambiguation
        # disambiguation = DataDisambiguation(llm=llm)
        # disambiguation_result = disambiguation.run(result)

        # print("Disambiguation result " + str(disambiguation_result))

        # # Log disambiguation result
        # intermediate_results.append({
        #     "stage": "Disambiguation",
        #     "chunks": [],
        #     "data": disambiguation_result
        # })

        # # Step 3: Convert Disambiguated Data to Cypher
        # cypher_script = data_to_cypher(disambiguation_result)

        # # Log Cypher Conversion
        # intermediate_results.append({
        #     "stage": "Cypher Conversion",
        #     "chunks": [],
        #     "data": cypher_script
        # })

        # # Step 4: Save Intermediate Results to CSV
        # save_intermediate_results_to_csv(intermediate_results)

        # return {"data": disambiguation_result}

    except Exception as e:
        print(e)
        return f"Error: {e}"
    
dummy_product_1 = {
    "website_name": "YesStyle",
    "properties": {
        "product_name": "Brightening Body Milk",
        "description": "Brightening Body Milk by ZHUBEN, designed for moisturizing and brightening skin. Includes a rich blend of natural oils and extracts.",
        "functional_roles": {
            "moisturizer": [
                {"chemical": "glycerin", "weight": "null"},
                {"chemical": "butyrospermum parkii (shea) butter", "weight": "null"},
                {"chemical": "olea europaea (olive) fruit oil", "weight": "null"},
                {"chemical": "camellia japonica seed oil", "weight": "null"},
                {"chemical": "caprylic/capric triglyceride", "weight": "null"}
            ],
            "brightener": [
                {"chemical": "niacinamide", "weight": "null"},
                {"chemical": "astragalus membranaceus root extract", "weight": "null"},
                {"chemical": "paeonia albiflora root extract", "weight": "null"}
            ],
            "soothing_agent": [
                {"chemical": "bisabolol", "weight": "null"},
                {"chemical": "calophyllum inophyllum seed oil", "weight": "null"},
                {"chemical": "dipotassium glycyrrhizate", "weight": "null"}
            ],
            "antioxidant": [
                {"chemical": "scutellaria baicalensis root extract", "weight": "null"},
                {"chemical": "bletilla striata root extract", "weight": "null"}
            ]
        }
    },
    "type": "beauty-body-moisturizers"
}

dummy_product_2 = {
    "website_name": "YesStyle",
    "type": "beauty-cheeks",
    "properties": {
        "product_name": "3 In 1 Highlight Blush Contour Palette - 3 Types",
        "description": "HANDAIYAN 3 In 1 Highlight Blush Contour Palette",
        "functional_roles": {
            "highlighter": [
                { "chemical": "mica", "weight": "null" },
                { "chemical": "CI 77941", "weight": "null" }
            ],
            "blush": [
                { "chemical": "magnesium stearate", "weight": "null" },
                { "chemical": "talc", "weight": "null" }
            ],
            "preservatives": [
                { "chemical": "methylparaben", "weight": "null" },
                { "chemical": "propylparaben", "weight": "null" }
            ],
            "binding agents": [
                { "chemical": "dimethicone", "weight": "null" },
                { "chemical": "ethylhexyl palmitate", "weight": "null" }
            ],
            "colorants": [
                { "chemical": "CI 77942", "weight": "null" },
                { "chemical": "CI 77499", "weight": "null" }
            ]
        }
    }
}

dummy_product_3 = {
    "website_name": "Trigaine Official Store",
    "type": "hair-care",
    "properties": {
        "product_name": "Trigaine Caffeine Shampoo",
        "description": "A cleansing formulation designed to clean and condition hair, featuring a combination of surfactants, moisturizers, and conditioning agents to provide a thorough cleanse while maintaining hair and scalp health. Ideal for daily use.",
        "functional_roles": {
            "surfactant": [
                { "chemical": "Sodium Lauryl Ether Sulphate", "weight": "variable" },
                { "chemical": "Cocamidopropyl Betaine", "weight": "variable" },
                { "chemical": "Coco Mono Ethanolamide", "weight": "variable" }
            ],
            "conditioning agent": [
                { "chemical": "Dimethicone", "weight": "variable" },
                { "chemical": "Guar Gum", "weight": "variable" }
            ],
            "moisturizer": [
                { "chemical": "Shea Butter", "weight": "variable" },
                { "chemical": "Aloe Vera Juice", "weight": "variable" },
                { "chemical": "Panthenol", "weight": "variable" },
                { "chemical": "Jojoba Seed Oil", "weight": "variable" }
            ],
            "pH adjuster": [
                { "chemical": "Citric Acid", "weight": "variable" }
            ],
            "electrolyte": [
                { "chemical": "Sodium Chloride", "weight": "variable" }
            ],
            "thickener": [
                { "chemical": "Ethylene Glycol Distearate", "weight": "variable" },
                { "chemical": "PEG-150 Distearate", "weight": "variable" }
            ],
            "antimicrobial agent": [
                { "chemical": "Phenoxyethanol", "weight": "variable" }
            ],
            "miscellaneous": [
                { "chemical": "Fragrance", "weight": "variable" }
            ]
        }
    }
}


### The Main Autograph flow in the final pipeline till now.
@app.post("/api/make_product_report")
async def root(payload: ImportPayload):
    """
    Takes an input and creates a Cypher query and functional role object
    """
    if not openai_api_key and not payload.api_key:
        raise HTTPException(
            status_code=403,
            detail="Please set OPENAI_API_KEY environment variable or send it as api_key in the request body",
        )
    api_key = openai_api_key if openai_api_key else payload.api_key

    try:        
        finalized_information = await product_discovery_workflow(data=payload.input, provider="openai")
        print("The payload input is : ", payload.input)
       
        # Log Extraction Result
        print("Finalized Information:", finalized_information)

        # Build JSON object for the frontend
        response_object = {
            "type": "cosmetic_product_patent",
            "patent_no": finalized_information.get("patent_no",""),
            "inventor_names": finalized_information.get("inventor_names",""),
            "cpcc_codes": finalized_information.get("cpcc_codes",""),
            "assignee":finalized_information.get("assignee_information",""),
            "task_type": "product discover workflow",
            "properties": {
                "product_name": finalized_information.get("product_name", ""),
                "description":finalized_information.get("description", ""),
                "functional_roles": finalized_information.get("functional_roles", [])
            }
        }

        # Save the response_object to a JSON file using patent_no as the filename
        patent_no = response_object["patent_no"].replace(" ", "_").replace("/", "_")  # Replace unsafe characters
        filename = f"{patent_no}.json"
        folder_path = "./backup"  # Save all backups in a folder called "backup"
        
        os.makedirs(folder_path, exist_ok=True)  # Ensure the folder exists
        filepath = os.path.join(folder_path, filename)
        
        with open(filepath, "w") as json_file:
            json.dump(response_object, json_file, indent=4)
        
        print(f"Backup saved: {filepath}")

        # # Save the analysed information in the knowledge graph database 
        db = Neo4jDatabase(host="bolt://kg:7688", user="neo4j", password="your12345", read_only=False)
        db.insert_patent_data(response_object)

        # # Dummy ingridient insert- 
        # db.insert_real_world_product(dummy_product_1)

        return ""

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error: {e}")


@app.post("/api/detail_document_summary")
async def root(payload: ImportPayload):
    """
    Takes an input and creates a summary of the document provided.
    """
    try:        
        finalized_information = await workflow_classifier(data=payload.input, provider="openai")
       
        # Log Extraction Result
        print("Finalized Information:", finalized_information)

        # Build JSON object for the frontend
        response_object = {
           "sections": finalized_information
        }

        return response_object

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error: {e}")



@app.post("/api/load_product_data")
async def root(payload: ImportPayload):
    """
    Takes an input and creates a Cypher query and functional role object
    """
    try:        
    
        db = Neo4jDatabase(host="bolt://kg:7688", user="neo4j", password="your12345", read_only=False)

        # Dummy ingredient insert- 
        db.insert_real_world_product(dummy_product_3)

        return ""

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    

@app.post("/ollama/data2cypher")
async def root(payload: ImportPayload):
    """
    Takes an input and created a Cypher query
    """
    if not openai_api_key and not payload.api_key:
        raise HTTPException(
            status_code=403,
            detail="Please set OPENAI_API_KEY environment variable or send it as api_key in the request body",
        )
    api_key = openai_api_key if openai_api_key else payload.api_key
    
    try:
        result = ""

        intermediate_results = []

        # Initialize LLM 
        llm = OpenAIChat(
            openai_api_key=api_key, model_name="gpt-4o", max_tokens=16384
        )

        # Step 1 : Run Extraction
        if not payload.neo4j_schema:
            result, chunks = await run_with_chunk_logging(data=payload.input, provider="openai")
        else:
            extractor = DataExtractorWithSchema(llm=ollama_chat)
            result = extractor.run(schema=payload.neo4j_schema, data=payload.input)

        # Log Extraction result
        print("Extracted result: " + str(result))

        # Log extraction result
        intermediate_results.append({
            "stage": "Extraction",
            "chunks": chunks,
        })

        # Step 2: Data Disambiguation
        # disambiguation = DataDisambiguation(llm=llm)
        # disambiguation_result = disambiguation.run(result)
        # chunks = disambiguation_result.get("chunks", [])

        # print("Disambiguation result " + str(disambiguation_result))

        # # Log disambiguation result
        # intermediate_results.append({
        #     "stage": "Disambiguation",
        #     "chunks": chunks ,
        # })

        # # Step 3: Convert Disambiguated Data to Cypher
        # cypher_script = data_to_cypher(disambiguation_result)

        # # Log Cypher Conversion
        # intermediate_results.append({
        #     "stage": "Cypher Conversion",
        #     "chunks": [],
        #     "data": cypher_script
        # })

        # print("The Cypher script is : ", cypher_script)

        # Step 4: Save Intermediate Results to CSV
        save_intermediate_results_to_csv(intermediate_results)

        return {"data": ""}

    except Exception as e:
        print(e)
        return f"Error: {e}"


class companyReportPayload(BaseModel):
    company: str
    api_key: Optional[str]


# This endpoint is database specific and only works with the Demo database.
@app.post("/companyReport")
async def companyInformation(payload: companyReportPayload):
    api_key = openai_api_key if openai_api_key else payload.api_key
    if not openai_api_key and not payload.api_key:
        raise HTTPException(
            status_code=422,
            detail="Please set OPENAI_API_KEY environment variable or send it as api_key in the request body",
        )
    api_key = openai_api_key if openai_api_key else payload.api_key

    llm = OpenAIChat(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo-16k-0613",
        max_tokens=512,
    )
    print("Running company report for " + payload.company)
    company_report = CompanyReport(neo4j_connection, payload.company, llm)
    result = company_report.run()

    return JSONResponse(content={"output": result})


# Text to token , take the text and model name being used - 

# Payload Definition
class TokenizerPayload(BaseModel):
    text: str  # Input text to tokenize
    model: str  # Model name (e.g., gpt-3.5-turbo, llama, regex)

@app.post("/tokenize")
async def tokenize_text(payload: TokenizerPayload):
    """
    Endpoint to calculate token count for input text based on the model name.
    """
    try:
        # Extract text and model name from payload
        text = payload.text
        model = payload.model.lower()

        if "gpt" in model:
            token_count = gpt_tokenizer(text, model)
        elif "llama" in model:
            token_count = llama_tokenizer(text)
        elif "regex" in model:
            token_count = regex_tokenizer(text)
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported model. Use 'gpt', 'llama', or 'regex'."
            )
        # Return the token count
        return JSONResponse(content={"token_count": token_count})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define the payload structure for the Ollama endpoint
class OllamaPayload(BaseModel):
    prompt: str  # User's prompt
    model: Optional[str] = "llama3.2"  # Default model


@app.post("/ollama/chat")
async def ollama_completion(payload: OllamaPayload):
    """
    Endpoint to interact with the Ollama Chat API
    """
    OLLAMA_URI = "http://host.docker.internal:11434/api/chat"

    print("The Received prompt is :" , payload.prompt)

    # Adjust model and system message based on condition
    model = payload.model
    
    if payload.model == "auto":
        model = "llama3.1"
    try:
        response = requests.post(
            OLLAMA_URI,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": payload.prompt
                    },
                ],
                "stream": False
            },
        )
        logging.debug(f"Raw response: {response.text}")  # Log the raw response
        response.raise_for_status()
        return {"generated_text": response.json()}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")
    except ValueError as e:  # Handle JSON decoding errors
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON received from Ollama: {response.text}",
        )
    
def extract_patent_no(prompt: str) -> str:
    """
    Extract the patent number from the prompt using regex.
    """
    patent_no_match = re.search(r"\bUS\s?\d{4,}\s?[A-Z]?\d*\b", prompt)
    return patent_no_match.group(0) if patent_no_match else None
    

def process_prompt_and_query(prompt: str, flag: str = None) -> str:
    """
    Process the user prompt, identify keywords, and prepare the Cypher query.
    """
    patent_no = extract_patent_no(prompt)
    print("The patent no is : ", patent_no)

    if flag == "detailed" or ("creator mode : run detailed query" in prompt.lower()):
        query = f"""
         MATCH (patent:patent {{aa_patent_no: '{patent_no}'}})-[:PROTECTS]->(product:product),
         (product)-[:OF]->(role:functional_role),
         (product)-[r:CONTAINS]->(chemical:chemical)
         WITH patent, 
            product, 
            COLLECT(DISTINCT role.name) AS roles, 
            COLLECT(DISTINCT chemical.name) AS chemicals, 
            COLLECT(DISTINCT r.weight) AS weights, 
            COLLECT(DISTINCT r.functional_role) AS functional_roles
         RETURN 'Patent Type: cosmetic_product_patent\n' +
            'Patent No: ' + patent.aa_patent_no + '\n' +
            'Product Name: ' + product.aa_product_name + '\n' +
            'Functional Roles: ' + apoc.text.join(roles, ', ') + '\n' +
            'Chemicals: ' + apoc.text.join(chemicals, ', ') + '\n' +
            'Weights: ' + apoc.text.join(weights, ', ') + '\n' +
            'Roles: ' + apoc.text.join(functional_roles, ', ') AS context
        """

        print('The query sent is :', query)
    else:
        query = """
        MATCH (head:patents {d_type: "patents"})-[:HAS]->(patent:patent)
        RETURN {
            type: "patent_overview",
            patent_no: patent.aa_patent_no,
            inventors: patent.inventor_names,
            cpcc_codes: patent.cpcc_codes,
            assignee: patent.the_assignee
        } AS result
        """
    return query

def get_website_offered_products_query() -> str:
    query = """
    MATCH (website:website)-[:OFFERS]->(product:product),
          (product)-[:OF]->(role:functional_role),
          (product)-[:CONTAINS]->(chemical:chemical)
    WITH website, 
         product, 
         COLLECT(DISTINCT role.name) AS functional_roles, 
         COLLECT(DISTINCT chemical.name) AS chemicals, 
         COLLECT(DISTINCT chemical.weight) AS weights
    RETURN 'Website: ' + website.name + '\n' +
           'Product Name: ' + product.aa_product_name + '\n' +
           'Description: ' + product.description + '\n' +
           'Functional Roles: ' + apoc.text.join(functional_roles, ', ') + '\n' +
           'Chemicals: ' + apoc.text.join(chemicals, ', ') + '\n' +
           'Weights: ' + apoc.text.join(weights, ', ') AS context
    ORDER BY website.name, product.aa_product_name
    """
    return query


import os
from dotenv import load_dotenv
from pathlib import Path

# Get the root directory dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # Moves up 3 levels to reach root/

# Load the environment variables from the .env file in the root directory
load_dotenv(ROOT_DIR / ".env")

# Now retrieve the API key as a string
api_key = openai_api_key


llm = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", max_tokens=4096
)

# Helper Functions - 
async def openai_generate(prompt: str) -> str:
    """
    Sends a prompt to the OpenAI endpoint and returns the raw output.

    :param prompt: The input prompt to send to the OpenAI API.
    :return: The raw output generated by the OpenAI API.
    """
    messages = [
        {"role": "user", "content": prompt}  # Construct the message payload
    ]
    print(f"Sending request to OpenAI endpoint with messages: {messages}")
    
    # Assuming `llm.generate()` is correctly configured to accept the messages
    output = llm.generate(messages)
    print("The output is:", output)
    return output


@app.post("/ollama/cyphered")
async def ollama_completion(payload: OllamaPayload):
    OLLAMA_URI = "http://host.docker.internal:11434/api/chat"
    prompt = payload.prompt
    model = payload.model if payload.model != "auto" else "llama3.1"
    
    try:
        # Step 1: Determine if the query is about patents or websites
        website_flag_prompt = f"""
        Given the question: '{payload.prompt}', determine whether the user is asking for website-offered product information or patent-related data. Follow these rules:

        1. If the question mentions **websites, products, offerings, availability**, or anything related to real-world products or their details:
           - **Return True** for website product data.
           - Look for terms like "website," "products," "offerings," "availability," or similar phrases.

        2. If the question is about patents, such as **status, validity, ownership, chemical information, or composition analysis**, and does not mention website-related terms:
           - **Return False** for patent data.

        Your response must strictly be:
        - True if the query requires website product information.
        - False if the query is about patents.

        Do not explain your answer.
        """

        print("Website Flag Prompt is:", website_flag_prompt)

        # Request to determine if the query is about websites or patents for ollama
        website_flag_request = {
            "model": model,
            "messages": [
                {"role": "user", "content": website_flag_prompt}
            ],
            "stream": False
        }

        website_flag_response = requests.post(OLLAMA_URI, json=website_flag_request)
        print("The response for website flag prompt:", website_flag_response.json())
        website_flag_response.raise_for_status()

        # Extract the website/patent flag
        llama_website_flag = (
            website_flag_response.json()
            .get("message", {})
            .get("content", "false")
            .strip()
            .lower()
        )

        print("The llama website flag is:", llama_website_flag)

        # Comaparing the flag generated by Llama and openai
        website_flag = await openai_generate(website_flag_prompt)
        print("The openai website flag is:", website_flag)

        query = None


        if website_flag == "false" or website_flag == "False":
            print("I reached here")
            patent_flag_prompt = f"""
                Given the question: '{payload.prompt}', determine whether to run a detailed or less detailed query. Follow these strict rules:

                1. If the question requests **any chemical information**, **ingredient details**, **composition analysis**, or anything related to the **substances, materials, or chemicals** mentioned in the patent or product:
                - **Return True** for a detailed query.
                - Do not overthink; just check for any hints or synonyms related to chemicals, ingredients, or composition. If present, choose detailed query.

                2. If the question **only asks for general document information** about the patent, such as its **status, validity, or ownership**, and there is **no mention of chemicals, ingredients, or compositions**, then:
                - **Return False** for a less detailed query.

                3. Be mindful of synonyms and alternate phrases like "analysis", "substances," "components," "materials," or "formulations" that imply chemical or ingredient details.

                Your response must strictly be:
                - **True** if the query requires detailed chemical or ingredient-related information.
                - **False** if the query strictly asks for general patent document information without mentioning chemical or composition details.

                Dont explain your answer.
                """


            print("Flag prompt is :", patent_flag_prompt)
            
            llm_flag_request = {
                "model": payload.model if payload.model != "auto" else "llama3.1",
                "messages": [
                    {"role": "user", "content": patent_flag_prompt}
                ],
                "stream": False
            }

            llm_response = requests.post(OLLAMA_URI, json=llm_flag_request)
            print("The reponse for flag prmpt :", llm_response.json())
            llm_response.raise_for_status()
            # Extracting the flag from the correct key in the JSON response
            llama_llm_flag = (
                llm_response.json()
                .get("message", {})
                .get("content", "false")
                .strip()
                .lower()
            )

            print("The final flag is:", llama_llm_flag)

            llm_flag = await openai_generate(patent_flag_prompt)
            print("The open ai flag is:", llm_flag)

            # Process prompt and query
            query = process_prompt_and_query(payload.prompt, flag=("detailed" if (llm_flag == "true" or llm_flag == "True") else "less detailed"))

        elif website_flag == "true" or website_flag == "True":
            query = get_website_offered_products_query()
        
        # Execute query and get data
        # Execute query and get data
        try:
            neo4j_result = neo4j_connection.query(query)
            print("The neo 4j results are :", neo4j_result)
        except Exception as e:
            # Capture Neo4j query execution errors
            raise HTTPException(status_code=500, detail=f"Neo4j query execution failed: {str(e)}")
        
        if not neo4j_result:
            raise HTTPException(status_code=404, detail="No relevant data found in the database.")
        
        # Prepare context for Llama
        context = f"The following information about patents has been retrieved:\n{json.dumps(neo4j_result, indent=2)}"
        llama_prompt = f"""Context:\n{context}\n\n "Input Question":{prompt} You are a smart assistant and a summarizing tool, if the users question want you to search for the full patent ingredient and composition analysis , then act accordingly, and if the user just wants information about the patent document use the context to provide relevant information, the context will generally contain more than what's required , it is you job and responsibility to smartly filter out whats required and whats not required according to the Input Question. Dont any other suggestions, notes or comment, no json format , output the results in the form of lists and paragraphs properly formatted for use."""

        # Call Llama API
        response = requests.post(
            OLLAMA_URI,
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": llama_prompt},
                ],
                "stream": False
            },
        )
        response.raise_for_status()
        return {"generated_text": response.json()}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/openai/cyphered")
async def ollama_completion(payload: OllamaPayload):
    prompt = payload.prompt
    model = payload.model if payload.model != "auto" else "llama3.1"

    print(f"Received prompt: {prompt}")
    print(f"Using model: {model}")
    
    try:
        # Step 1: Determine the query type
        similarity_flag_prompt = f"""
        Given the question: '{payload.prompt}', determine if the query is about similarity analysis of a website product with individual patent only and only if a patent number is provided. Follow these rules:

        1. If the question explicitly mentions similarity, and if a patent number has been provided comparison, or matching between a website product and other patents, return True.
        2. Otherwise, return False.

        Your response must strictly be:
        - True if the query is about product similarity or comparison.
        - False otherwise.
        """
        print("Similarity Flag Prompt is:", similarity_flag_prompt)

        # Call OpenAI to determine similarity flag
        similarity_flag = await openai_generate(similarity_flag_prompt)
        print("The OpenAI similarity flag is:", similarity_flag)

        query = None

        if similarity_flag.lower() == "true":
            patent_no = extract_patent_no(payload.prompt)  # Assuming patent_no is provided in the payload

            # Query 1: Extract patent data
            patent_query = f"""
            MATCH (patent:patent {{aa_patent_no: '{patent_no}'}})-[:PROTECTS]->(product:product),
                  (product)-[:OF]->(role:functional_role),
                  (product)-[r:CONTAINS]->(chemical:chemical)
            WITH patent, 
                 product, 
                 COLLECT(DISTINCT role.name) AS roles, 
                 COLLECT(DISTINCT chemical.name) AS chemicals, 
                 COLLECT(DISTINCT r.weight) AS weights, 
                 COLLECT(DISTINCT r.functional_role) AS functional_roles
            RETURN 'Patent Type: cosmetic_product_patent\n' +
                   'Patent No: ' + patent.aa_patent_no + '\n' +
                   'Product Name: ' + product.aa_product_name + '\n' +
                   'Functional Roles: ' + apoc.text.join(roles, ', ') + '\n' +
                   'Chemicals: ' + apoc.text.join(chemicals, ', ') + '\n' +
                   'Weights: ' + apoc.text.join(weights, ', ') + '\n' +
                   'Roles: ' + apoc.text.join(functional_roles, ', ') AS context
            """

            print("Executing patent query:", patent_query)
            patent_result = neo4j_connection.query(patent_query)
            print("Patent query result:", patent_result)

            if not patent_result:
                raise HTTPException(status_code=404, detail="No patent data found for the given patent number.")

            # Query 2: Extract website data
            website_query = f"""
            MATCH (website:website)-[:OFFERS]->(product:product),
                  (product)-[:OF]->(role:functional_role),
                  (product)-[:CONTAINS]->(chemical:chemical)
            WITH website, 
                 product, 
                 COLLECT(DISTINCT role.name) AS functional_roles, 
                 COLLECT(DISTINCT chemical.name) AS chemicals, 
                 COLLECT(DISTINCT chemical.weight) AS weights
            RETURN 'Website: ' + website.name + '\n' +
                   'Product Name: ' + product.aa_product_name + '\n' +
                   'Description: ' + product.description + '\n' +
                   'Functional Roles: ' + apoc.text.join(functional_roles, ', ') + '\n' +
                   'Chemicals: ' + apoc.text.join(chemicals, ', ') + '\n' +
                   'Weights: ' + apoc.text.join(weights, ', ') AS context
            ORDER BY website.name, product.aa_product_name
            """

            print("Executing website query:", website_query)
            website_result = neo4j_connection.query(website_query)
            print("Website query result:", website_result)

            if not website_result:
                raise HTTPException(status_code=404, detail="No website data found.")

            # Concatenate results
            combined_context = f"Patent Data:\n{json.dumps(patent_result, indent=2)}\n\nWebsite Data:\n{json.dumps(website_result, indent=2)}"

            # Prepare final prompt for Llama
            llama_prompt = f"""Context:\n{combined_context}\n\n"Input Question": {prompt}\nUsing the above data, analyze the similarity between the input patent data and all website products in the context. Highlight key similarities and differences in terms of chemicals, functional roles, and product descriptions. Provide a clear and concise response.Also necessarily provide a similarity score in percentage out of 10 at the end. The user might have demands so check the Input Question properly for instructions if any. """
            print("Generated Llama Prompt:", llama_prompt)

            # Call OpenAI for similarity analysis
            response = await openai_generate(llama_prompt)
            print("Llama similarity response:", response)
            return response

        else:
            # Handle standard patent and website queries
            website_flag_prompt = f"""
            Given the question: '{payload.prompt}', determine whether the user is asking for website-offered product information or patent-related data. Follow these rules:

            1. If the question mentions **websites, products, offerings, availability**, or anything related to real-world products or their details:
               - **Return True** for website product data.
               - Look for terms like "website," "products," "offerings," "availability," or similar phrases.

            2. If the question is about patents, such as **status, validity, ownership, chemical information, or composition analysis**, and does not mention website-related terms:
               - **Return False** for patent data.

            Your response must strictly be:
            - True if the query requires website product information.
            - False if the query is about patents.
            """
            print("Website Flag Prompt is:", website_flag_prompt)
            website_flag = await openai_generate(website_flag_prompt)
            print("The OpenAI website flag is:", website_flag)

            if website_flag.lower() == "false":
                patent_flag_prompt = f"""
                    Given the question: '{payload.prompt}', determine whether to run a detailed or less detailed query. Follow these strict rules:

                    1. If the question requests **any chemical information**, **ingredient details**, **composition analysis**, or anything related to the **substances, materials, or chemicals** mentioned in the patent or product:
                    - **Return True** for a detailed query.

                    2. If the question **only asks for general document information** about the patent, such as its **status, validity, or ownership**, and there is **no mention of chemicals, ingredients, or compositions**, then:
                    - **Return False** for a less detailed query.
                    """
                print("Patent Flag Prompt is:", patent_flag_prompt)
                llm_flag = await openai_generate(patent_flag_prompt)
                print("The OpenAI patent flag is:", llm_flag)

                query = process_prompt_and_query(payload.prompt, flag=("detailed" if llm_flag.lower() == "true" else "less detailed"))

            elif website_flag.lower() == "true":
                query = get_website_offered_products_query()

            try:
                print("Executing query:", query)
                neo4j_result = neo4j_connection.query(query)
                print("The Neo4j results are:", neo4j_result)
            except Exception as e:
                print(f"Error during Neo4j query execution: {e}")
                raise HTTPException(status_code=500, detail=f"Neo4j query execution failed: {str(e)}")

            if not neo4j_result:
                print("Error: No relevant data found.")
                raise HTTPException(status_code=404, detail="No relevant data found in the database.")

            # Prepare context for Llama
            context = f"The following information has been retrieved:\n{json.dumps(neo4j_result, indent=2)}"
            llama_prompt = f"""Context:\n{context}\n\n"Input Question": {prompt} Provide a detailed response based on the retrieved information. Filter unnecessary details and summarize effectively."""
            print("Generated Llama Prompt:", llama_prompt)

            response = await openai_generate(llama_prompt)
            print("Llama response:", response)
            return response

    except HTTPException as e:
        print(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# A pipeline for creating "Precision graph for Active Ingredient Analysis (AIS)"
    

@app.post("/")

@app.post("/companyReport/list")
async def companyReportList():
    company_data = neo4j_connection.query(
        "MATCH (n:Organization) WITH n WHERE rand() < 0.01 return n.name LIMIT 5",
    )

    return JSONResponse(content={"output": [x["n.name"] for x in company_data]})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    return {"status": "ok"}

class GraphMLExportRequest(BaseModel):
    neo4j_url: str            # e.g., "bolt://localhost:7687"
    user: str                 # e.g., "neo4j"
    password: str             # e.g., "your12345"
    database_name: str        # e.g., "neo4j"
    filename: str = "export"  # optional filename prefix for the graphml file


@app.post("/exportGraphML")
def export_graphml(payload: GraphMLExportRequest):
    """
    Connects to a user-specified Neo4j database, exports *all* nodes and relationships
    as a GraphML file, and saves the file in /graphs/ at the root of the API.
    """
    # 1) Build a separate Neo4jDatabase instance with the user-provided details
    custom_neo4j_connection = Neo4jDatabase(
        host=payload.neo4j_url,
        user=payload.user,
        password=payload.password,
        database=payload.database_name
    )

    # 2) Query all nodes and relationships from the specified database
    #    (These queries fetch all nodes and their labels/properties, and all relationships.)
    nodes_result = custom_neo4j_connection.query("""
        MATCH (n)
        RETURN id(n) AS node_id, labels(n) AS labels, n AS properties
    """)

    rels_result = custom_neo4j_connection.query("""
        MATCH (n)-[r]->(m)
        RETURN id(r) AS rel_id, type(r) AS rel_type, id(n) AS start_id, id(m) AS end_id, r AS properties
    """)

    # 3) Construct a NetworkX graph (directed or undirected depending on your use case)
    G = nx.MultiDiGraph()

    # 3a) Add nodes
    for record in nodes_result:
        node_id = record["node_id"]
        labels = record["labels"]
        props = dict(record["properties"])  # Convert Neo4j property object to a normal dict

        # The node label(s) can be combined or you can keep them in 'props'
        # Add them so you can see them in the GraphML later:
        props["_labels"] = labels
        # Add node to the graph
        G.add_node(node_id, **props)

    # 3b) Add relationships
    for record in rels_result:
        rel_id = record["rel_id"]
        rel_type = record["rel_type"]
        start_id = record["start_id"]
        end_id = record["end_id"]
        props = dict(record["properties"])
        
        # Also store the relationship type
        props["_type"] = rel_type
        # Add edge to the graph
        G.add_edge(start_id, end_id, key=rel_id, **props)

    # 4) Generate the filename and path for saving
    #    We'll timestamp it so we don't overwrite existing files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{payload.filename}_{timestamp}.graphml"
    graphs_dir = os.path.join(os.getcwd(), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)  # Make sure /graphs/ folder exists
    file_path = os.path.join(graphs_dir, filename)

    # 5) Use NetworkX to write out the GraphML
    nx.write_graphml(G, file_path)

    # 6) Return some status information
    return {
        "message": "GraphML export successful",
        "graphml_file_path": file_path,
        "nodes_count": G.number_of_nodes(),
        "edges_count": G.number_of_edges()
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=int(os.environ.get("PORT", 7860)), host="0.0.0.0")
