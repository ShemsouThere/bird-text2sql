"""FastAPI server for Text-to-SQL inference."""
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rich.console import Console

from scripts.utils import load_config, setup_logging

console = Console()

# Request/response models
class PredictRequest(BaseModel):
    question: str
    db_id: str
    evidence: str = ""

class PredictResponse(BaseModel):
    sql: str
    candidates: List[str]
    selected_method: str
    time_seconds: float

class BatchRequest(BaseModel):
    questions: List[PredictRequest]

class BatchResponse(BaseModel):
    results: List[PredictResponse]
    total_time_seconds: float

class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_loaded: bool
    total_predictions: int
    average_time_seconds: float

# Global state
app = FastAPI(title="Bird Text-to-SQL", version="1.0.0")
pipeline = None
stats = {"total_predictions": 0, "total_time": 0.0}
config = None
logger = None


@app.on_event("startup")
async def startup():
    """Load the pipeline once at startup."""
    global pipeline, config, logger

    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    preset_path = os.environ.get("PRESET_PATH", None)
    config = load_config(config_path, preset_path)
    logger = setup_logging(config["training"]["log_dir"], "serve")

    logger.info("Loading Text-to-SQL pipeline...")
    try:
        from inference.pipeline import Text2SQLPipeline
        pipeline = Text2SQLPipeline(config)
        pipeline.load_model()
        logger.info("Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        # Pipeline stays None - health check will report not loaded


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    avg_time = stats["total_time"] / max(stats["total_predictions"], 1)
    model_name = config["model"]["name"] if config else "unknown"
    return HealthResponse(
        status="healthy" if pipeline is not None else "degraded",
        model_name=model_name,
        model_loaded=pipeline is not None,
        total_predictions=stats["total_predictions"],
        average_time_seconds=round(avg_time, 3),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict SQL for a single question."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        result = pipeline.predict(
            question=request.question,
            db_id=request.db_id,
            evidence=request.evidence,
        )
        elapsed = time.time() - start

        stats["total_predictions"] += 1
        stats["total_time"] += elapsed

        if logger:
            logger.info(f"Prediction for db={request.db_id} took {elapsed:.2f}s")

        return PredictResponse(
            sql=result.get("sql", ""),
            candidates=result.get("candidates", []),
            selected_method=result.get("selected_method", "unknown"),
            time_seconds=round(elapsed, 3),
        )
    except Exception as e:
        if logger:
            logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    """Predict SQL for a batch of questions."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        questions = [
            {"question": q.question, "db_id": q.db_id, "evidence": q.evidence}
            for q in request.questions
        ]
        results = pipeline.predict_batch(questions)
        elapsed = time.time() - start

        stats["total_predictions"] += len(results)
        stats["total_time"] += elapsed

        if logger:
            logger.info(f"Batch of {len(results)} predictions took {elapsed:.2f}s")

        responses = []
        for r in results:
            timing = r.get("timing", {})
            responses.append(PredictResponse(
                sql=r.get("sql", ""),
                candidates=r.get("candidates", []),
                selected_method=r.get("selected_method", "unknown"),
                time_seconds=r.get("time_seconds", round(timing.get("total", 0.0), 3)),
            ))

        return BatchResponse(
            results=responses,
            total_time_seconds=round(elapsed, 3),
        )
    except Exception as e:
        if logger:
            logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Text-to-SQL API Server")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    os.environ["CONFIG_PATH"] = args.config
    if args.preset:
        os.environ["PRESET_PATH"] = args.preset

    cfg = load_config(args.config, args.preset)
    host = args.host or cfg.get("serve", {}).get("host", "0.0.0.0")
    port = args.port or cfg.get("serve", {}).get("port", 8000)
    log_level = cfg.get("serve", {}).get("log_level", "info")

    console.print(f"[bold green]Starting server on {host}:{port}[/bold green]")
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
