from fastapi import FastAPI, Query, Request, BackgroundTasks, Body
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import json
import math
from crawler.crawler import run_scrape

from engine.search_engine import SimpleSearchEngine 
from engine.classifier_model import get_or_train_classifier

templates = Jinja2Templates(directory="templates")
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once when the server starts.
    It loads the data and builds the search engine.
    """
    print("Server starting up...")

    with open('./data/publications_data.json', 'r') as f:
        publication_data = json.load(f)

    engine = SimpleSearchEngine(data=publication_data)
    engine.build()

    ml_models["search_engine"] = engine

    print("Search engine loaded successfully!")

    try:
        classifier = get_or_train_classifier(data_path='./data/')
        ml_models["classifier"] = classifier
        print("Classifier model loaded successfully!")
    except Exception as e:
        print(f"Error loading classifier model: {e}")

    yield

    ml_models.clear()
    print("Server shutting down...")

app = FastAPI(
    title="Simple Publication Search",
    description="A web interface for a simple inverted index search.",
    version="1.0.0",
    lifespan=lifespan
)

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

        try:
            maybe = engine.search(query=q, page=page, per_page=per_page)
            if isinstance(maybe, dict):
                results = maybe.get("results", [])
                total_pages = maybe.get("total_pages", max(1, math.ceil(len(results) / per_page) if results else 1))
            else:
                all_results = list(maybe)
                total_pages = max(1, math.ceil(len(all_results) / per_page))
                page = max(1, min(page, total_pages))
                start = (page - 1) * per_page
                results = all_results[start:start + per_page]
        except TypeError:
            all_results = engine.search(query=q)
            if not isinstance(all_results, list):
                all_results = list(all_results) if all_results is not None else []
            total_pages = max(1, math.ceil(len(all_results) / per_page) if all_results else 1)
            page = max(1, min(page, total_pages))
            start = (page - 1) * per_page
            results = all_results[start:start + per_page]

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
    """Serves the main home page."""
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
        print("Scrape and re-index complete!")

    background_tasks.add_task(scrape_and_index)
    return {"status": "Scrape and re-index started"}

@app.post("/classify")
async def classify_text(text: str = Body(..., embed=True)):
    """
    Classifies the input text and returns the predicted category with confidence scores.
    Expects JSON: { "text": "your text here" }
    """
    classifier = ml_models.get("classifier")
    if classifier is None:
        return {"error": "Classifier model not loaded."}

    predicted_category = classifier.predict([text])[0]
    prediction_probabilities = classifier.predict_proba([text])[0]
    class_names = classifier.classes_
    probabilities = {class_names[i]: round(float(prediction_probabilities[i]), 4)
                    for i in range(len(class_names))}

    return {
        "text": text,
        "predicted_category": predicted_category,
        "prediction_scores": probabilities
    }
