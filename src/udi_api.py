import os
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from udi_agent import UDIAgent

# from src.udi_agent import UDIAgent
from fastapi.middleware.cors import CORSMiddleware
import copy

from jose import jwt, JWTError
from dotenv import load_dotenv

from vis_generate import (
    generate_vis_spec,
    load_grammar,
    load_skills,
    _render_template,
    simplify_data_domains,
)
from structured_functions import (
    validate_structured_text,
    segment_structured_text,
    get_function_signatures,
    export_registry_json,
)

# --- Logging setup ---
_log_dir = Path(__file__).resolve().parent.parent / "logs"
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

load_dotenv()  # automatically loads from .env

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
MODEL_NAME = os.getenv("UDI_MODEL_NAME")
TOKENIZER_NAME = os.getenv("UDI_TOKENIZER_NAME", MODEL_NAME)
INSECURE_DEV_MODE = int(os.getenv("INSECURE_DEV_MODE", "0")) == 1
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost")
VLLM_SERVER_PORT = int(os.getenv("VLLM_SERVER_PORT", "55001"))

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

GPT_MODEL_NAME = os.getenv("GPT_MODEL_NAME", "gpt-5.4")
USE_VIS_PIPELINE = int(os.getenv("USE_VIS_PIPELINE", "0")) == 1

# init agent
agent = UDIAgent(
    model_name=MODEL_NAME,
    gpt_model_name=GPT_MODEL_NAME,
    vllm_server_url=VLLM_SERVER_URL,
    vllm_server_port=VLLM_SERVER_PORT,
    tokenizer_name=TOKENIZER_NAME,
    use_vis_pipeline=USE_VIS_PIPELINE,
)


if USE_VIS_PIPELINE:
    _pipeline_grammar = load_grammar("udi")

_skills = load_skills()


def parse_schema_from_dict(raw):
    """Parse a data schema dict into the structure expected by structured_functions.

    Similar to generate_tools.parse_schema but works with an in-memory dict
    instead of a file path.
    """
    entities = {}
    for resource in raw.get("resources", []):
        name = resource["name"]
        row_count = resource.get("udi:row_count", 0)
        fields = {}
        for field in resource.get("schema", {}).get("fields", []):
            cardinality = field.get("udi:cardinality", 0)
            if cardinality == 0:
                continue
            fields[field["name"]] = {
                "type": field.get("udi:data_type", ""),
                "cardinality": cardinality,
            }
        entities[name] = {"row_count": row_count, "fields": fields}
    return {"entities": entities}


# ---------------------------------------------------------------------------
# Top-level tool definitions (declarative registry)
# ---------------------------------------------------------------------------

ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Rebuff",
            "description": (
                "Use this tool when the user's request CANNOT be fulfilled by any "
                "other available tool. For example: requests about topics unrelated "
                "to the loaded data, requests for unsupported chart types, or "
                "requests that require capabilities the system does not have."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The original user query that cannot be fulfilled.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "A brief explanation of why the request cannot be fulfilled.",
                    },
                },
                "required": ["user_request", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ClarifyVariable",
            "description": (
                "Use this tool when the user's request references an ambiguous variable "
                "that could match multiple fields or entities in the dataset. For example, "
                "'age' might match 'age_value' in donors or 'sample_age' in samples. "
                "Returns a clarification request with candidate variables for the user "
                "to choose from."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "A natural-language explanation of what is ambiguous and why clarification is needed.",
                    },
                    "ambiguous_variables": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query_term": {
                                    "type": "string",
                                    "description": "The term from the user's request that is ambiguous.",
                                },
                                "candidates": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "field_name": {
                                                "type": "string",
                                                "description": "The actual field name in the schema.",
                                            },
                                            "entity": {
                                                "type": "string",
                                                "description": "The dataset entity this field belongs to.",
                                            },
                                        },
                                        "required": ["field_name", "entity"],
                                        "additionalProperties": False,
                                    },
                                    "description": "Candidate fields that could match the ambiguous term.",
                                },
                            },
                            "required": ["query_term", "candidates"],
                            "additionalProperties": False,
                        },
                        "description": "List of ambiguous variables with their candidate matches.",
                    },
                },
                "required": ["message", "ambiguous_variables"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "FreeTextExplain",
            "description": (
                "Use this tool when the user asks an informational question — about "
                "available functionality, datasets, or general system capabilities — "
                "that does NOT require generating a visualization or filtering data. "
                "For example: 'what can I do?', 'what data is available?', 'how many "
                "tables are there?', 'what types of charts can you make?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The original informational question from the user.",
                    },
                    "response_type": {
                        "type": "string",
                        "enum": ["capabilities", "data_summary", "general"],
                        "description": (
                            "The category of explanation needed: 'capabilities' for what the "
                            "system can do, 'data_summary' for information about loaded datasets, "
                            "'general' for other informational questions."
                        ),
                    },
                },
                "required": ["user_request", "response_type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "CreateVisualization",
            "description": (
                "Create a data visualization. Supports: bar charts (vertical/horizontal, "
                "with count/min/max/avg/median/sum aggregations), stacked and grouped bar "
                "charts, scatterplots, heatmaps, histograms, CDF line charts, pie/donut "
                "charts, dot strips, density curves, and data tables. Can visualize a "
                "single entity or join two related entities. The specific visualization "
                "type will be automatically selected based on the data and request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A brief natural-language description of what visualization to create.",
                    },
                    "title": {
                        "type": "string",
                        "description": "A short, informative title for the chart (e.g. 'Donor Count by Sex', 'Height vs Weight').",
                    },
                },
                "required": ["description", "title"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "FilterData",
            "description": (
                "Filter the dataset to a subset of rows. Use for categorical filters "
                "(e.g. filter to Female donors) or numeric range filters (e.g. filter "
                "to age > 50). Call this tool multiple times for multiple filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The entity (table) to filter.",
                    },
                    "field": {
                        "type": "string",
                        "description": "The field to filter on.",
                    },
                    "filterType": {
                        "type": "string",
                        "enum": ["point", "interval"],
                        "description": "Type of filter: 'point' for categorical values, 'interval' for numeric ranges.",
                    },
                    "intervalRange": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number", "description": "Minimum value."},
                            "max": {"type": "number", "description": "Maximum value."},
                        },
                        "required": ["min", "max"],
                        "additionalProperties": False,
                        "description": "Range for interval filters. Required when filterType is 'interval'.",
                    },
                    "pointValues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Values to filter for. Required when filterType is 'point'.",
                    },
                },
                "required": ["entity", "field", "filterType"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# JWT verification
# ---------------------------------------------------------------------------


def verify_jwt(authorization: str = Header(...)):
    if INSECURE_DEV_MODE:
        # skip verification in dev mode
        return {"dev_mode": True}
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization[len("Bearer ") :]  # strip "Bearer "

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Optionally, you can return payload info here if needed
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@app.get("/")
def read_root():
    return {
        "service": "UDIAgent API",
        "status": "running",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API status and info"},
        ],
    }


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    tools: list[dict]


class YACCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    dataSchema: str
    dataDomains: str


class YACBenchmarkCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    dataSchema: str
    dataDomains: str
    orchestrator_choice: str | None = None
    use_pipeline: bool | None = None  # override USE_VIS_PIPELINE per-request


# ---------------------------------------------------------------------------
# Tool dispatch handlers
# ---------------------------------------------------------------------------


def _handle_create_visualization(
    tool_args: dict, request, use_pipeline: bool, openai_api_key: str | None = None
):
    """Dispatch handler for CreateVisualization tool calls.

    Uses the tool call's ``description`` argument as the user message so that
    each call generates a distinct visualization, even when the orchestrator
    returns multiple CreateVisualization tool calls for one user turn.
    """
    # Build a focused request whose messages end with the specific description
    focused = copy.deepcopy(request)
    description = tool_args.get("description", "")
    if description:
        focused.messages = [
            msg for msg in focused.messages if msg.get("role") != "user"
        ] + [{"role": "user", "content": description}]

    if use_pipeline:
        result = function_call_render_visualization_pipeline(
            focused, openai_api_key=openai_api_key
        )
    else:
        result = function_call_render_visualization_legacy(
            focused, openai_api_key=openai_api_key
        )

    title = tool_args.get("title", "")
    if title:
        result["arguments"]["title"] = title

    return result


def _handle_rebuff(
    tool_args: dict, request, use_pipeline: bool, openai_api_key: str | None = None
):
    """Dispatch handler for Rebuff tool calls.

    Generates a rebuff response with available capabilities derived from
    the current tool set.
    """
    # Derive available capabilities from ORCHESTRATOR_TOOLS (excluding Rebuff itself)
    available_capabilities = [
        f"{t['function']['name']}: {t['function']['description']}"
        for t in ORCHESTRATOR_TOOLS
        if t["function"]["name"] != "Rebuff"
    ]

    rebuff_skill = _skills.get("rebuff")
    if rebuff_skill:
        rendered = _render_template(
            rebuff_skill.instructions,
            {
                "user_request": tool_args.get("user_request", ""),
                "reason": tool_args.get("reason", ""),
                "available_tools": "\n".join(
                    f"- {cap}" for cap in available_capabilities
                ),
            },
        )
        gpt_client = agent._get_gpt_client(openai_api_key)
        resp = gpt_client.chat.completions.create(
            model=agent.gpt_model_name,
            messages=[
                {"role": "system", "content": rendered},
                {
                    "role": "user",
                    "content": tool_args.get("user_request", ""),
                },
            ],
            temperature=0.0,
            max_completion_tokens=1024,
        )
        try:
            response_data = json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, IndexError):
            response_data = {
                "message": f"Sorry, I cannot fulfill this request. {tool_args.get('reason', '')}",
                "suggestions": [],
            }
    else:
        response_data = {
            "message": f"Sorry, I cannot fulfill this request. {tool_args.get('reason', '')}",
            "suggestions": [],
        }

    return {
        "name": "Rebuff",
        "arguments": response_data,
    }


def _handle_free_text_explain(
    tool_args: dict, request, use_pipeline: bool, openai_api_key: str | None = None
):
    """Dispatch handler for FreeTextExplain tool calls.

    Generates a structured text response using the skill prompt, dynamically
    injecting available tools, data schema, and structured function signatures.
    The response may contain {function_name(args)} references that are validated
    and optionally resolved server-side.
    """
    # Derive available capabilities from ORCHESTRATOR_TOOLS
    available_tools = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}"
        for t in ORCHESTRATOR_TOOLS
        if t["function"]["name"] not in ("Rebuff", "FreeTextExplain")
    )

    # Simplify data schema for LLM consumption
    data_schema_simple = simplify_data_domains(request.dataDomains)

    explain_skill = _skills.get("free_text_explain")
    if explain_skill:
        rendered = _render_template(
            explain_skill.instructions,
            {
                "user_request": tool_args.get("user_request", ""),
                "response_type": tool_args.get("response_type", "general"),
                "available_tools": available_tools,
                "data_schema": data_schema_simple,
                "structured_functions": get_function_signatures(),
            },
        )
        gpt_client = agent._get_gpt_client(openai_api_key)
        resp = gpt_client.chat.completions.create(
            model=agent.gpt_model_name,
            messages=[
                {"role": "system", "content": rendered},
                {
                    "role": "user",
                    "content": tool_args.get("user_request", ""),
                },
            ],
            temperature=0.0,
            max_completion_tokens=1024,
        )
        text_response = resp.choices[0].message.content
    else:
        text_response = "I can help you explore and visualize data. Try asking for a specific chart or data summary."

    # Validate structured function references
    validation_errors = validate_structured_text(text_response)

    # Segment text into mixed list of plain strings and structured element objects
    if not validation_errors:
        try:
            schema_dict = (
                json.loads(request.dataSchema)
                if isinstance(request.dataSchema, str)
                else request.dataSchema
            )
            schema_parsed = parse_schema_from_dict(schema_dict)
            text_segments, has_structured = segment_structured_text(
                text_response, schema_parsed
            )
        except Exception:
            text_segments = [text_response]
            has_structured = False
    else:
        text_segments = [text_response]
        has_structured = False

    return {
        "name": "FreeTextExplain",
        "arguments": {
            "response_type": tool_args.get("response_type", "general"),
            "text": text_segments,
            "has_structured_elements": has_structured,
        },
    }


def _handle_clarify_variable(
    tool_args: dict, request, use_pipeline: bool, openai_api_key: str | None = None
):
    """Dispatch handler for ClarifyVariable tool calls.

    Enriches the LLM-provided candidates with data_type and description
    looked up from the data schema, so the LLM only needs to provide
    field_name and entity.
    """
    # Parse the data schema to look up field metadata
    try:
        schema_raw = (
            json.loads(request.dataSchema)
            if isinstance(request.dataSchema, str)
            else request.dataSchema
        )
    except (json.JSONDecodeError, TypeError):
        schema_raw = {}

    # Build a lookup: (entity, field_name) -> {data_type, description}
    field_meta = {}
    for resource in schema_raw.get("resources", []):
        entity_name = resource.get("name", "")
        for field in resource.get("schema", {}).get("fields", []):
            fname = field.get("name", "")
            field_meta[(entity_name, fname)] = {
                "data_type": field.get("udi:data_type", "unknown"),
                "description": field.get("description", "").strip(),
            }

    # Enrich each candidate with data_type and description from the schema
    ambiguous_variables = tool_args.get("ambiguous_variables", [])
    for var in ambiguous_variables:
        for candidate in var.get("candidates", []):
            key = (candidate.get("entity", ""), candidate.get("field_name", ""))
            meta = field_meta.get(key, {})
            candidate["data_type"] = meta.get("data_type", "unknown")
            candidate["description"] = meta.get("description", "")

    return {
        "name": "ClarifyVariable",
        "arguments": {
            "message": tool_args.get("message", ""),
            "ambiguous_variables": ambiguous_variables,
        },
    }


def _handle_filter_data(
    tool_args: dict, request, use_pipeline: bool, openai_api_key: str | None = None
):
    """Dispatch handler for FilterData tool calls.

    Builds a single FilterData result from the tool call arguments.
    """
    # Build the filter structure from the tool call args
    filter_obj = {
        "filterType": tool_args["filterType"],
        "intervalRange": tool_args.get("intervalRange", {"min": 0, "max": 0}),
        "pointValues": tool_args.get("pointValues", [""]),
    }
    return {
        "name": "FilterData",
        "arguments": {
            "entity": tool_args["entity"],
            "field": tool_args["field"],
            "filter": filter_obj,
        },
    }


# Dispatch dict: tool name -> handler function
TOOL_DISPATCH = {
    "Rebuff": _handle_rebuff,
    "ClarifyVariable": _handle_clarify_variable,
    "FreeTextExplain": _handle_free_text_explain,
    "CreateVisualization": _handle_create_visualization,
    "FilterData": _handle_filter_data,
}


# ---------------------------------------------------------------------------
# Orchestrator: tool-calling based routing
# ---------------------------------------------------------------------------


def orchestrate_tool_calls(
    request: YACCompletionRequest,
    use_pipeline: bool = USE_VIS_PIPELINE,
    openai_api_key: str | None = None,
):
    """Use LLM tool calling to determine and execute the right actions.

    The LLM is given CreateVisualization and FilterData tools and can
    return one or more tool calls per request. Each tool call is dispatched
    to its handler, and results are collected into a flat list.
    """
    messages = normalize_tool_calls(copy.deepcopy(request.messages))

    orchestrate_skill = _skills["orchestrate"]
    rendered = _render_template(
        orchestrate_skill.instructions,
        {"data_domains": simplify_data_domains(request.dataDomains)},
    )
    messages.insert(0, {"role": "system", "content": rendered})

    # Call LLM with tool definitions
    gpt_client = agent._get_gpt_client(openai_api_key)
    resp = gpt_client.chat.completions.create(
        model=agent.gpt_model_name,
        messages=messages,
        tools=ORCHESTRATOR_TOOLS,
        tool_choice="required",
        temperature=0.0,
        max_completion_tokens=1024,
    )

    choice = resp.choices[0]
    if not choice.message.tool_calls:
        return [], "render-visualization"

    # Dispatch each tool call
    tool_calls = []
    has_vis = False
    has_filter = False
    has_rebuff = False
    has_clarify = False
    has_explain = False

    for tc in choice.message.tool_calls:
        tool_name = tc.function.name
        tool_args = json.loads(tc.function.arguments)

        handler = TOOL_DISPATCH.get(tool_name)
        if handler is None:
            print(f"Unknown tool: {tool_name}, skipping")
            continue

        result = handler(
            tool_args, request, use_pipeline, openai_api_key=openai_api_key
        )
        tool_calls.append(result)

        if tool_name == "CreateVisualization":
            has_vis = True
        elif tool_name == "FilterData":
            has_filter = True
        elif tool_name == "Rebuff":
            has_rebuff = True
        elif tool_name == "ClarifyVariable":
            has_clarify = True
        elif tool_name == "FreeTextExplain":
            has_explain = True

    # Derive orchestrator_choice for backward compatibility
    if has_explain:
        orchestrator_choice = "explain"
    elif has_clarify:
        orchestrator_choice = "clarify-variable"
    elif has_rebuff:
        orchestrator_choice = "rebuff"
    elif has_vis and has_filter:
        orchestrator_choice = "both"
    elif has_filter:
        orchestrator_choice = "get-subset-of-data"
    else:
        orchestrator_choice = "render-visualization"

    return tool_calls, orchestrator_choice


# ---------------------------------------------------------------------------
# Legacy orchestrator (for benchmark override)
# ---------------------------------------------------------------------------


def _run_legacy_orchestration(
    request, calls_to_make, use_pipeline, openai_api_key: str | None = None
):
    """Run legacy if/else orchestration for backward-compatible benchmark overrides."""
    tool_calls = []
    if calls_to_make in ("both", "get-subset-of-data"):
        tool_calls.extend(function_call_filter(request, openai_api_key=openai_api_key))
    if calls_to_make in ("both", "render-visualization"):
        if use_pipeline:
            tool_calls.append(
                function_call_render_visualization_pipeline(
                    request, openai_api_key=openai_api_key
                )
            )
        else:
            tool_calls.append(
                function_call_render_visualization_legacy(
                    request, openai_api_key=openai_api_key
                )
            )
    return tool_calls


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/yac/completions")
def yac_completions(
    request: YACCompletionRequest,
    token_payload: dict = Depends(verify_jwt),
    x_openai_key: str | None = Header(None, alias="X-OpenAI-Key"),
):
    split_tool_calls(request)
    tool_calls, orchestrator_choice = orchestrate_tool_calls(
        request, openai_api_key=x_openai_key
    )
    logger.info("orchestrator_choice: %s", orchestrator_choice)
    logger.info("tool_calls: %s", tool_calls)
    return tool_calls


@app.post("/v1/yac/benchmark")
def yac_benchmark(
    request: YACBenchmarkCompletionRequest,
    token_payload: dict = Depends(verify_jwt),
    x_openai_key: str | None = Header(None, alias="X-OpenAI-Key"),
):
    split_tool_calls(request)
    use_pipeline = (
        request.use_pipeline if request.use_pipeline is not None else USE_VIS_PIPELINE
    )

    if request.orchestrator_choice is not None:
        # Legacy path: explicit orchestrator_choice override for A/B testing
        calls_to_make = request.orchestrator_choice
        tool_calls = _run_legacy_orchestration(
            request, calls_to_make, use_pipeline, openai_api_key=x_openai_key
        )
    else:
        # New path: tool-calling orchestration
        tool_calls, calls_to_make = orchestrate_tool_calls(
            request, use_pipeline, openai_api_key=x_openai_key
        )

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
def yac_benchmark_nalysis():
    RESULT_FILENAME = "./out/benchmark_analysis.json"
    if not os.path.exists(RESULT_FILENAME):
        return JSONResponse(
            content={"error": f"File {RESULT_FILENAME} not found."}, status_code=404
        )

    with open(RESULT_FILENAME, "r") as f:
        data = json.load(f)

    return JSONResponse(content=data)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def split_tool_calls(request: YACCompletionRequest):
    # for each message in the request if there are multiple tool calls, split them into separate messages.
    # Note: this was needed because jinja template used cannot handle multiple tool calls in a single message.
    new_messages = []
    for message in request.messages:
        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
            if isinstance(tool_calls, list) and len(tool_calls) > 1:
                # split into multiple messages
                for i, tool_call in enumerate(tool_calls):
                    new_message = message.copy()
                    new_message["tool_calls"] = [tool_call]
                    new_messages.append(new_message)
            else:
                # single tool_call (or not a list) — keep as-is
                new_messages.append(message)
        else:
            new_messages.append(message)

    request.messages = new_messages
    return


def normalize_tool_calls(messages: list[dict]):
    """Reformat tool_calls into valid OpenAI function-calling format and
    inject synthetic tool-response messages where missing.

    The frontend echoes back dispatched results as flat dicts
    (e.g. {"name": "RenderVisualization", "arguments": {...}}).
    The OpenAI API requires each tool_call entry to have "id",
    "type": "function", and arguments as a JSON string.  It also
    requires a corresponding "tool" role message for every tool_call_id.
    """
    result = []
    for idx, message in enumerate(messages):
        if "tool_calls" not in message:
            result.append(message)
            continue

        # Normalize the tool_calls on this assistant message
        normalized = []
        for i, tc in enumerate(message.get("tool_calls", [])):
            if not isinstance(tc, dict):
                continue
            # Already in OpenAI format
            if "type" in tc and "function" in tc and "id" in tc:
                normalized.append(tc)
                continue
            # Flat / dispatched format → convert
            fn = tc.get("function", {})
            if fn:
                name = fn.get("name", "")
                args = fn.get("arguments", {})
            else:
                name = tc.get("name", "")
                args = tc.get("arguments", {})
            normalized.append(
                {
                    "id": f"call_{idx}_{i}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args)
                        if not isinstance(args, str)
                        else args,
                    },
                }
            )
        message["tool_calls"] = normalized
        # Ensure content is not None (OpenAI requires it)
        if not message.get("content"):
            message["content"] = ""
        result.append(message)

        # Check if the next message(s) already contain tool responses
        # for these IDs; if not, inject synthetic ones.
        next_idx = idx + 1
        existing_ids = set()
        while next_idx < len(messages):
            nxt = messages[next_idx]
            if nxt.get("role") == "tool":
                existing_ids.add(nxt.get("tool_call_id"))
                next_idx += 1
            else:
                break

        for tc in normalized:
            if tc["id"] not in existing_ids:
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "OK",
                    }
                )

    return result


def strip_tool_calls(messages: list[dict]):
    """Remove tool_calls from messages (used by legacy code paths)."""
    for message in messages:
        if "tool_calls" in message:
            del message["tool_calls"]
    return messages


# ---------------------------------------------------------------------------
# Filter handler (legacy, used by benchmark override path)
# ---------------------------------------------------------------------------


def function_call_filter(
    request: YACCompletionRequest, openai_api_key: str | None = None
):
    messages = copy.deepcopy(
        request.messages
    )  # make a copy to avoid mutating the original
    strip_tool_calls(messages)
    interstitialMessage = {
        "role": "system",
        "content": f"You are a helpful assistant that will explore, and analyze datasets. The following defines the available dataset entities, fields, and their domains:\n{request.dataDomains}\nRight now you need to filter the data based on the users request.",
    }
    messages.insert(len(messages) - 1, interstitialMessage)

    response = agent.gpt_completions_guided_json(
        messages=messages,
        openai_api_key=openai_api_key,
        json_schema=json.dumps(
            {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "properties": {
                            "filterType": {"enum": ["point", "interval"]},
                            "intervalRange": {
                                "type": "object",
                                "properties": {
                                    "min": {
                                        "type": "number",
                                        "description": "The minimum for the filter.",
                                    },
                                    "max": {
                                        "type": "number",
                                        "description": "The maximum for the filter.",
                                    },
                                },
                                "required": ["min", "max"],
                                "additionalProperties": False,
                            },
                            "pointValues": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "description": "The values to filter for categorical fields.",
                            },
                        },
                        "required": ["filterType", "intervalRange", "pointValues"],
                        "additionalProperties": False,
                    },
                    "entity": {
                        "type": "string",
                        "description": "The entity to filter based on the current dataset schema.",
                    },
                    "field": {
                        "type": "string",
                        "description": "The field to filter. Must be a quantitative field from the selected entity.",
                    },
                },
                "required": ["entity", "field", "filter"],
                "additionalProperties": False,
            }
        ),
    )
    logger.debug("filter response: %s", response)
    tool_calls = [{"name": "FilterData", "arguments": args} for args in response]
    return tool_calls


# ---------------------------------------------------------------------------
# Visualization handlers
# ---------------------------------------------------------------------------


def function_call_render_visualization_legacy(
    request: YACCompletionRequest, openai_api_key: str | None = None
):
    f = open("./src/UDIGrammarSchema.json", "r")
    udi_grammar_dict = json.load(f)
    f.close()
    f = open("./src/UDIGrammarSchema_spec_string.json", "r")
    udi_grammar_string = f.read()
    f.close()
    messages = copy.deepcopy(
        request.messages
    )  # make a copy to avoid mutating the original
    firstMessage = {
        "role": "system",
        "content": f"You are a helpful assistant that will explore, and analyze datasets with visualizations. The following defines the available datasets:\n{request.dataSchema}\nTypically, your actions will use the provided functions. You have access to the following functions.",
    }
    messages.insert(0, firstMessage)
    response = agent.completions_guided_json(
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "RenderVisualization",
                    "description": "Render a visualization with a provided visualization grammar of graphics style specification.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spec": udi_grammar_dict,
                        },
                        "required": ["spec"],
                    },
                },
            }
        ],
        json_schema=udi_grammar_string,
    )
    return json.loads(response.choices[0].text)


def function_call_render_visualization_pipeline(
    request: YACCompletionRequest, openai_api_key: str | None = None
):
    messages = copy.deepcopy(request.messages)
    strip_tool_calls(messages)
    result = generate_vis_spec(
        agent=agent,
        messages=messages,
        data_schema=request.dataSchema,
        grammar=_pipeline_grammar,
        openai_api_key=openai_api_key,
    )
    return {
        "name": "RenderVisualization",
        "arguments": {"spec": result["spec"]},
        "meta": result.get("meta"),
    }


def function_call_render_visualization(
    request: YACCompletionRequest, openai_api_key: str | None = None
):
    if USE_VIS_PIPELINE:
        return function_call_render_visualization_pipeline(
            request, openai_api_key=openai_api_key
        )
    return function_call_render_visualization_legacy(
        request, openai_api_key=openai_api_key
    )


class UDICompletionRequest(BaseModel):
    model: str
    messages: list[dict]
