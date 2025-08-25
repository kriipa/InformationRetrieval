# main.py
from fastapi import FastAPI, Query, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import json
import math
from crawler.crawler import run_scrape

from engine.search_engine import SimpleSearchEngine 

# --- Global Variables & Lifespan Management ---
templates = Jinja2Templates(directory="templates")
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once when the server starts.
    It loads the data and builds the search engine.
    """
    print("🚀 Server starting up...")
    
    # Load the publication data
    with open('./data/publications_data.json', 'r') as f:
        publication_data = json.load(f)
    
    # Initialize and build the simple search engine
    engine = SimpleSearchEngine(data=publication_data)
    engine.build() # This will load or create 'inverted_index.json'
    
    # Store the engine instance so our endpoints can use it
    ml_models["search_engine"] = engine
    
    print("✅ Search engine loaded successfully!")
    yield
    
    # This code runs when the server shuts down
    ml_models.clear()
    print("🔥 Server shutting down...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Simple Publication Search",
    description="A web interface for a simple inverted index search.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Web Page Endpoints ---
@app.get("/search")
async def web_search(request: Request, q: str = Query(None), page: int = Query(1), per_page: int = Query(10)):
    """
    Serves the HTML search page.
    If a query 'q' is provided, it performs the search and displays the results.
    Supports pagination via 'page' and 'per_page'.
    """
    results = None
    total_pages = 1

    if q:
        engine = ml_models["search_engine"]

        # Try to use engine pagination API if available
        try:
            maybe = engine.search(query=q, page=page, per_page=per_page)
            # Accept either a list or a dict with 'results' and optional 'total_pages'
            if isinstance(maybe, dict):
                results = maybe.get("results", [])
                total_pages = maybe.get("total_pages", max(1, math.ceil(len(results) / per_page) if results else 1))
            else:
                # assume list
                all_results = list(maybe)
                total_pages = max(1, math.ceil(len(all_results) / per_page))
                # clamp page
                page = max(1, min(page, total_pages))
                start = (page - 1) * per_page
                results = all_results[start:start + per_page]
        except TypeError:
            # Engine doesn't accept page/per_page — paginate locally
            all_results = engine.search(query=q)
            if not isinstance(all_results, list):
                # If engine returns non-list, try to coerce
                all_results = list(all_results) if all_results is not None else []
            total_pages = max(1, math.ceil(len(all_results) / per_page) if all_results else 1)
            page = max(1, min(page, total_pages))
            start = (page - 1) * per_page
            results = all_results[start:start + per_page]

    # Ensure template always receives these keys to avoid Jinja undefined errors
    context = {
        "request": request,
        "query": q,
        "results": results,
        "page": page,
        "total_pages": total_pages,
    }

    return templates.TemplateResponse("index.html", context)

@app.get("/")
async def read_root(request: Request):
    """
    Serves the main home page.
    """
    context = {
        "request": request,
        "query": None,
        "results": None,
        "page": 1,
        "total_pages": 1,
    }
    return templates.TemplateResponse("index.html", context)

@app.post("/trigger-scrape")
def trigger_scrape(background_tasks: BackgroundTasks):
    def scrape_and_index():
        run_scrape()
        with open('./data/publications_data.json', 'r') as f:
            publication_data = json.load(f)
        engine = SimpleSearchEngine(data=publication_data)
        engine.build()
        ml_models["search_engine"] = engine
        print("✅ Scrape and re-index complete!")

    background_tasks.add_task(scrape_and_index)
    return {"status": "Scrape and re-index started"}