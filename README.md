# UDIAgent

LLM-powered data visualization orchestration library for the Universal Discovery Interface (UDI).

UDIAgent orchestrates GPT-4 and a fine-tuned Llama model to generate data visualization specs from natural language queries. It can be used as a **standalone Python library** or deployed as a **FastAPI microservice**.

## Installation

```bash
# Core library only
pip install udiagent

# With the reference FastAPI server
pip install udiagent[server]

# With LangFuse observability
pip install udiagent[langfuse]

# With benchmarking tools
pip install udiagent[benchmark]

# Everything
pip install udiagent[all]
```

For local development with `uv`:

```bash
uv sync --extra server --extra langfuse --extra test   # server + dev
```

## Library Usage

```python
from udiagent import UDIAgent, Orchestrator

# Initialize the agent with explicit configuration (no environment variables)
agent = UDIAgent(
    model_name="HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B",
    gpt_model_name="gpt-4.1",
    openai_api_key="sk-...",
    vllm_server_url="http://localhost",
    vllm_server_port=55001,
)

# Create an orchestrator
orch = Orchestrator(agent, use_vis_pipeline=True)

# Run a query
result = orch.run(
    messages=[{"role": "user", "content": "Show me a bar chart of donors by sex"}],
    data_schema='{"resources": [...]}',
    data_domains='[{"entity": "donors", "field": "sex", ...}]',
)

# result.tool_calls — list of tool call dicts (e.g. RenderVisualization, FilterData)
# result.orchestrator_choice — "render-visualization", "both", "explain", etc.
```

### Key Classes

| Class | Description |
|---|---|
| `UDIAgent` | LLM client abstraction for OpenAI and vLLM backends |
| `Orchestrator` | Routes user requests to visualization, filter, explanation, and clarification handlers |
| `OrchestratorResult` | Dataclass with `tool_calls` and `orchestrator_choice` |

### Utility Functions

| Function | Description |
|---|---|
| `load_grammar()` | Load the UDI Grammar JSON schema (bundled with the package) |
| `load_skills()` | Load skill prompt templates (bundled with the package) |
| `generate_vis_spec()` | Generate a visualization spec using the skills pipeline |
| `run_vis_pipeline()` | Run the 4-stage modular pipeline (plan/retrieve/generate/validate) |
| `simplify_data_domains()` | Simplify data domains JSON into compact LLM-friendly text |
| `parse_schema_from_dict()` | Parse a data schema dict into structured format |

## Server Usage

The `udiagent.server` subpackage provides a reference FastAPI application that wraps the library as a configurable microservice. It reads configuration from environment variables.

### Running the Server for Local Development

```bash
# Development
uv run fastapi dev src/udiagent/server/app.py --port 8007

# Production
uv run fastapi run src/udiagent/server/app.py --port 8007
```

The fine-tuned model is served separately via vLLM:

```bash
vllm serve HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B --port 55001 --host 127.0.0.1
```

### Server Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | No | — | OpenAI API key. If not set, must be provided per-request via `X-OpenAI-Key` header. |
| `UDI_MODEL_NAME` | Yes | — | Fine-tuned model name (e.g. `HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B`) |
| `GPT_MODEL_NAME` | No | `gpt-5.4` | OpenAI model for orchestration |
| `UDI_TOKENIZER_NAME` | No | `UDI_MODEL_NAME` | Tokenizer name (defaults to model name) |
| `VLLM_SERVER_URL` | No | `http://localhost` | vLLM server hostname |
| `VLLM_SERVER_PORT` | No | `55001` | vLLM server port |
| `USE_VIS_PIPELINE` | No | `0` | Set to `1` to enable the multi-stage visualization pipeline |
| `JWT_SECRET_KEY` | Yes* | — | JWT signing key (*not required if `INSECURE_DEV_MODE=1`) |
| `JWT_ALGORITHM` | No | `HS256` | JWT algorithm |
| `INSECURE_DEV_MODE` | No | `0` | Set to `1` to skip JWT verification (development only) |
| `LANGFUSE_SECRET_KEY` | No | — | LangFuse observability secret key |
| `LANGFUSE_PUBLIC_KEY` | No | — | LangFuse observability public key |
| `LANGFUSE_BASE_URL` | No | — | LangFuse instance URL |

### Server Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status and info |
| `/v1/yac/completions` | POST | Main orchestrator — routes user requests to tools |
| `/v1/yac/benchmark` | POST | Benchmark variant with optional orchestrator override |
| `/v1/yac/examples` | GET | Example prompts from `data/example_prompts.json` |
| `/v1/yac/structured_functions` | GET | Structured function registry |
| `/v1/yac/benchmark_analysis` | GET | Latest benchmark analysis results |

### Docker

```bash
docker build -t udiagent .
docker run -p 80:80 --env-file .env udiagent
```

## Architecture

### Orchestration Flow

```
User query
  → Orchestrator.run()
    → GPT with ORCHESTRATOR_TOOLS (5 tools: CreateVisualization, FilterData,
      FreeTextExplain, ClarifyVariable, Rebuff)
    → Dispatch each tool call to its handler
    → Return OrchestratorResult(tool_calls, orchestrator_choice)
```

### Visualization Pipeline (when `use_vis_pipeline=True`)

```
Stage 1: PLAN     — GPT identifies mark type, fields, data tasks
Stage 2: RETRIEVE — Few-shot example retrieval via embedding RAG + tag boost
Stage 3: GENERATE — Full UDI Grammar spec generation (GPT or vLLM)
Stage 4: VALIDATE — JSON schema validation with repair-retry loop
```

### Design Principles

- **Stateless** — All context travels in message history; no server-side session state
- **Pluggable LLM backends** — OpenAI for orchestration, vLLM for fine-tuned generation
- **Skills as Markdown** — Prompt templates live in `.md` files with YAML frontmatter
- **Per-request key override** — Supports both default and per-request OpenAI API keys


## Regenerating Template Visualizations and Tool Definitions

The vis pipeline uses two generated artifacts:
- `src/udiagent/data/skills/template_visualizations.json` — template visualization specs
- `src/udiagent/generated_vis_tools.py` — typed OpenAI function-calling tool definitions

To regenerate both in one step:

```bash
uv pip install -e ".[codegen]"
uv run python scripts/regenerate_vis_tools.py
```

By default this uses `data/data_domains/hubmap_data_schema.json` as the schema. To use a different schema:

```bash
uv run python scripts/regenerate_vis_tools.py --schema data/data_domains/SenNet_domains.json
```

## Benchmarking

### Step 0: Start the API server

```bash
uv run fastapi dev src/udiagent/server/app.py --port 8007 &
```

### Step 1: Run tiny benchmark (1 example)

```bash
uv run python -m udiagent.benchmark.runner --no-orchestrator --path ./data/benchmark_dqvis/tiny.jsonl
```

### Step 2: Run small benchmark (100 examples)

```bash
uv run python -m udiagent.benchmark.runner --no-orchestrator --path ./data/benchmark_dqvis/small.jsonl --workers 5
```

Resume a failed run:

```bash
uv run python -m udiagent.benchmark.runner --path ./data/benchmark_dqvis/small.jsonl --workers 5 --resume ./out/<TIMESTAMP>/benchmark_results.json
```

## License

MIT
