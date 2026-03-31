"""FastAPI application for UDIAgent.

This is the reference server implementation that wraps the ``udiagent``
library as a configurable microservice.  Configuration is read from
environment variables (via ``ServerConfig.from_env()``).

Run with::

    uv run fastapi dev src/udiagent/server/app.py --port 8007
"""

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from udiagent.agent import UDIAgent
from udiagent.messages import split_tool_calls
from udiagent.orchestrator import Orchestrator
from udiagent.structured_functions import export_registry_json
from udiagent.server.config import ServerConfig
from udiagent.server.auth import make_verify_jwt
from udiagent.server.models import (
    YACCompletionRequest,
    YACBenchmarkCompletionRequest,
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

# --- Logging setup ---
_log_dir = Path(__file__).resolve().parent.parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler(
            _log_dir / "udi_agent.log", maxBytes=5_000_000, backupCount=3
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# --- Config ---
config = ServerConfig.from_env()

# --- Agent & Orchestrator ---
agent = UDIAgent(
    model_name=config.udi_model_name,
    gpt_model_name=config.gpt_model_name,
    vllm_server_url=config.vllm_server_url,
    vllm_server_port=config.vllm_server_port,
    tokenizer_name=config.udi_tokenizer_name,
    use_vis_pipeline=config.use_vis_pipeline,
    openai_api_key=config.openai_api_key,
)

orchestrator = Orchestrator(
    agent=agent,
    use_vis_pipeline=config.use_vis_pipeline,
)

# --- FastAPI app ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

verify_jwt = make_verify_jwt(
    config.jwt_secret_key,
    config.jwt_algorithm,
    config.insecure_dev_mode,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def read_root():
    return {
        "service": "UDIAgent API",
        "status": "running",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API status and info"},
        ],
    }


@app.post("/v1/yac/completions")
def yac_completions(
    request: YACCompletionRequest,
    token_payload: dict = Depends(verify_jwt),
    x_openai_key: str | None = Header(None, alias="X-OpenAI-Key"),
):
    logger.info("Received /v1/yac/completions request: %s", request)

    result = orchestrator.run(
        messages=request.messages,
        data_schema=request.dataSchema,
        data_domains=request.dataDomains,
        openai_api_key=x_openai_key,
    )
    logger.info("orchestrator_choice: %s", result.orchestrator_choice)
    logger.info("tool_calls: %s", result.tool_calls)
    return result.tool_calls


@app.post("/v1/yac/benchmark")
def yac_benchmark(
    request: YACBenchmarkCompletionRequest,
    token_payload: dict = Depends(verify_jwt),
    x_openai_key: str | None = Header(None, alias="X-OpenAI-Key"),
):
    use_pipeline = (
        request.use_pipeline
        if request.use_pipeline is not None
        else config.use_vis_pipeline
    )

    if request.orchestrator_choice is not None:
        # Legacy path: explicit orchestrator_choice override for A/B testing
        tool_calls = orchestrator.run_legacy(
            messages=request.messages,
            data_schema=request.dataSchema,
            data_domains=request.dataDomains,
            calls_to_make=request.orchestrator_choice,
            use_pipeline=use_pipeline,
            openai_api_key=x_openai_key,
        )
        calls_to_make = request.orchestrator_choice
    else:
        # New path: tool-calling orchestration
        result = orchestrator.run(
            messages=request.messages,
            data_schema=request.dataSchema,
            data_domains=request.dataDomains,
            openai_api_key=x_openai_key,
        )
        tool_calls = result.tool_calls
        calls_to_make = result.orchestrator_choice

    return {"tool_calls": tool_calls, "orchestrator_choice": calls_to_make}


@app.get("/v1/yac/examples")
def yac_examples():
    examples_path = "./data/example_prompts.json"
    if not os.path.exists(examples_path):
        return JSONResponse(
            content={"error": f"File {examples_path} not found."}, status_code=404
        )
    with open(examples_path, "r") as f:
        data = json.load(f)
    prompts = [item["input"]["messages"][0]["content"] for item in data]
    return JSONResponse(content=prompts)


@app.get("/v1/yac/structured_functions")
def yac_structured_functions():
    """Return the structured function registry for frontend consumption."""
    return JSONResponse(content=export_registry_json())


@app.get("/v1/yac/benchmark_analysis")
def yac_benchmark_analysis():
    result_filename = "./out/benchmark_analysis.json"
    if not os.path.exists(result_filename):
        return JSONResponse(
            content={"error": f"File {result_filename} not found."}, status_code=404
        )

    with open(result_filename, "r") as f:
        data = json.load(f)

    return JSONResponse(content=data)
