"""
Modular Visualization Generation Pipeline.

Decomposes visualization spec generation into stages:
  1. PLAN    — identify mark type, fields, data tasks (optional, can be skipped)
  2. RETRIEVE — fetch top-k few-shot examples via embedding RAG + tag boost
  3. GENERATE — produce a full grammar spec via configurable LLM backend
  4. VALIDATE — JSON schema validation with repair-retry loop
"""

import json
import math
from pathlib import Path

import jsonschema

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_PIPELINE_CONFIG = {
    "plan": {
        "backend": "gpt",       # "gpt" | "vllm" | "skip"
        "temperature": 0.0,
    },
    "retrieve": {
        "backend": "embedding",  # "embedding" | "tags_only"
        "top_k": 3,
        "tag_boost": True,       # boost embedding scores for tag matches
    },
    "generate": {
        "backend": "vllm",      # "gpt" | "vllm"
        "temperature": 0.0,
        "n": 1,
    },
    "validate": {
        "max_retries": 3,
        "validators": ["json_schema"],  # later: add "vision"
    },
}

# ---------------------------------------------------------------------------
# Plan schema — used for structured output in Stage 1
# ---------------------------------------------------------------------------

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "mark_type": {
            "type": "string",
            "enum": ["bar", "line", "point", "area", "arc", "rect", "text", "geometry", "row"],
            "description": "The primary mark type for the visualization.",
        },
        "primary_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The key data fields to use in the visualization.",
        },
        "data_tasks": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "comparison", "distribution", "correlation", "trend",
                    "proportion", "ranking", "aggregation", "filtering",
                ],
            },
            "description": "The analytical tasks the visualization addresses.",
        },
        "needs_aggregation": {
            "type": "boolean",
            "description": "Whether the data needs aggregation (groupby/rollup).",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief reasoning for the chosen approach.",
        },
    },
    "required": ["mark_type", "primary_fields", "data_tasks", "needs_aggregation", "reasoning"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Grammar loading
# ---------------------------------------------------------------------------


def load_grammar(grammar_name, base_path="./src"):
    """Load a grammar definition by name.

    Returns {"schema_dict": ..., "schema_string": ..., "system_prompt": ...}
    """
    base = Path(base_path)
    if grammar_name == "udi":
        with open(base / "UDIGrammarSchema.json") as f:
            schema_dict = json.load(f)
        with open(base / "UDIGrammarSchema_spec_string.json") as f:
            schema_string = f.read()
        system_prompt = (
            "You are a helpful assistant that creates data visualizations using "
            "the UDI Grammar specification. Generate a valid UDI Grammar JSON spec "
            "based on the user's request and the provided data schema."
        )
        return {
            "schema_dict": schema_dict,
            "schema_string": schema_string,
            "system_prompt": system_prompt,
        }
    else:
        raise ValueError(f"Unknown grammar: {grammar_name}")

# ---------------------------------------------------------------------------
# Example spec loading
# ---------------------------------------------------------------------------


def load_example_specs(path, embed_fn=None):
    """Load example specs and optionally pre-compute embeddings.

    Returns {"examples": [...], "embeddings": [...] or None}
    """
    p = Path(path)
    if not p.exists():
        return {"examples": [], "embeddings": None}

    with open(p) as f:
        examples = json.load(f)

    if not examples:
        return {"examples": examples, "embeddings": None}

    embeddings = None
    if embed_fn is not None:
        texts = [
            f"{ex['metadata']['question']} {ex['metadata'].get('description', '')}"
            for ex in examples
        ]
        embeddings = embed_fn(texts)

    return {"examples": examples, "embeddings": embeddings}

# ---------------------------------------------------------------------------
# Stage 1: Plan
# ---------------------------------------------------------------------------


def stage_plan(agent, messages, data_schema, data_domains, config):
    """Produce a structured plan dict (mark type, fields, tasks, etc.).

    Returns plan dict or None if skipped.
    """
    backend = config.get("backend", "skip")
    if backend == "skip":
        return None

    plan_messages = list(messages) + [
        {
            "role": "system",
            "content": (
                "Analyze the user's visualization request. Based on the data schema and "
                "domains below, determine the best mark type, primary fields, analytical "
                "data tasks, and whether aggregation is needed.\n\n"
                f"Data Schema:\n{data_schema}\n\n"
                f"Data Domains:\n{data_domains}\n\n"
                "Respond with a structured JSON plan."
            ),
        },
    ]

    if backend == "gpt":
        results = agent.gpt_completions_guided_json(
            messages=plan_messages,
            json_schema=json.dumps(PLAN_SCHEMA),
        )
        return results[0] if results else None
    elif backend == "vllm":
        resp = agent.completions_guided_json(
            messages=plan_messages,
            tools=[],
            json_schema=json.dumps(PLAN_SCHEMA),
        )
        return json.loads(resp.choices[0].text)
    else:
        raise ValueError(f"Unknown plan backend: {backend}")

# ---------------------------------------------------------------------------
# Stage 2: Retrieve Examples
# ---------------------------------------------------------------------------


def _cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tag_score(example, plan):
    """Compute a tag-boost score based on plan overlap with example tags."""
    if plan is None:
        return 0.0
    score = 0.0
    tags = example.get("tags", {})
    # mark type match
    if plan.get("mark_type") in tags.get("mark_types", []):
        score += 0.15
    # data task overlap
    plan_tasks = set(plan.get("data_tasks", []))
    example_tasks = set(tags.get("data_tasks", []))
    overlap = plan_tasks & example_tasks
    score += 0.1 * len(overlap)
    return score


def stage_retrieve(query_text, plan, example_store, embed_fn, config):
    """Retrieve top-k relevant example specs.

    Uses embedding similarity + tag boost when embed_fn is available,
    falls back to pure tag matching otherwise.

    Returns list of example dicts.
    """
    if example_store is None or not example_store.get("examples"):
        return []

    examples = example_store["examples"]
    embeddings = example_store.get("embeddings")
    top_k = config.get("top_k", 3)
    use_tag_boost = config.get("tag_boost", True)
    backend = config.get("backend", "embedding")

    scored = []

    if backend == "embedding" and embed_fn is not None and embeddings is not None:
        # Embed the query
        query_embedding = embed_fn([query_text])[0]
        for i, ex in enumerate(examples):
            sim = _cosine_similarity(query_embedding, embeddings[i])
            tag_boost = _tag_score(ex, plan) if use_tag_boost else 0.0
            scored.append((sim + tag_boost, ex))
    else:
        # tags-only fallback
        for ex in examples:
            tag_boost = _tag_score(ex, plan)
            scored.append((tag_boost, ex))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:top_k]]

# ---------------------------------------------------------------------------
# Stage 3: Generate Spec
# ---------------------------------------------------------------------------


def _build_generate_messages(messages, data_schema, grammar, plan, examples):
    """Assemble the prompt messages for spec generation."""
    system_parts = [grammar["system_prompt"]]

    system_parts.append(
        f"The following defines the available datasets:\n{data_schema}"
    )

    if plan is not None:
        plan_summary = (
            f"Visualization plan: mark={plan.get('mark_type')}, "
            f"fields={plan.get('primary_fields')}, "
            f"tasks={plan.get('data_tasks')}, "
            f"aggregation={'yes' if plan.get('needs_aggregation') else 'no'}. "
            f"Reasoning: {plan.get('reasoning', '')}"
        )
        system_parts.append(plan_summary)

    if examples:
        example_strs = []
        for ex in examples:
            q = ex["metadata"]["question"]
            spec_json = json.dumps(ex["spec"], separators=(",", ":"))
            example_strs.append(f"Q: {q}\nSpec: {spec_json}")
        system_parts.append(
            "Here are some example specifications for reference:\n\n"
            + "\n\n".join(example_strs)
        )

    system_content = "\n\n".join(system_parts)

    gen_messages = [{"role": "system", "content": system_content}]
    # Append the conversation messages
    gen_messages.extend(messages)
    return gen_messages


def stage_generate(agent, messages, data_schema, grammar, plan, examples, config):
    """Generate a visualization spec string.

    Returns raw spec string (JSON).
    """
    gen_messages = _build_generate_messages(messages, data_schema, grammar, plan, examples)
    backend = config.get("backend", "vllm")
    n = config.get("n", 1)

    if backend == "gpt":
        results = agent.gpt_completions_guided_json(
            messages=gen_messages,
            json_schema=grammar["schema_string"],
            n=n,
        )
        # GPT returns parsed dicts; the wrapper schema has name/arguments/spec
        # We need to extract the spec from the response
        if results:
            result = results[0]
            # The schema_string wraps as RenderVisualizationWrapper
            if "arguments" in result and "spec" in result["arguments"]:
                return result["arguments"]["spec"]
            return json.dumps(result)
        return "{}"
    elif backend == "vllm":
        resp = agent.completions_guided_json(
            messages=gen_messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "RenderVisualization",
                        "description": "Render a visualization with a provided visualization grammar of graphics style specification.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "spec": grammar["schema_dict"],
                            },
                            "required": ["spec"],
                        },
                    },
                }
            ],
            json_schema=grammar["schema_string"],
            n=n,
        )
        return resp.choices[0].text
    else:
        raise ValueError(f"Unknown generate backend: {backend}")

# ---------------------------------------------------------------------------
# Stage 4: Validate + Repair
# ---------------------------------------------------------------------------


def _validate_json_schema(spec_dict, grammar):
    """Validate spec_dict against the grammar's JSON schema.

    Returns list of error strings (empty if valid).
    """
    schema = grammar["schema_dict"]
    errors = []
    try:
        jsonschema.validate(instance=spec_dict, schema=schema)
    except jsonschema.ValidationError as e:
        errors.append(str(e.message))
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e.message}")
    return errors


def _build_repair_messages(messages, data_schema, grammar, plan, examples,
                           failed_spec_str, errors):
    """Build a repair prompt with the failed spec and error details."""
    gen_messages = _build_generate_messages(messages, data_schema, grammar, plan, examples)

    repair_content = (
        f"The previous attempt produced an invalid specification.\n\n"
        f"Failed spec:\n{failed_spec_str}\n\n"
        f"Validation errors:\n" + "\n".join(f"- {e}" for e in errors) + "\n\n"
        f"Please fix the specification to be valid according to the grammar schema."
    )
    gen_messages.append({"role": "user", "content": repair_content})
    return gen_messages


def stage_validate(agent, spec_str, messages, data_schema, grammar, plan, examples,
                   config, generate_config):
    """Validate the spec and attempt repair if invalid.

    Returns (spec_dict, valid, errors).
    """
    max_retries = config.get("max_retries", 3)
    validators = config.get("validators", ["json_schema"])

    for attempt in range(max_retries + 1):
        # Parse JSON
        try:
            spec_dict = json.loads(spec_str) if isinstance(spec_str, str) else spec_str
        except json.JSONDecodeError as e:
            errors = [f"JSON parse error: {e}"]
            if attempt < max_retries:
                spec_str = _repair(
                    agent, spec_str, errors, messages, data_schema,
                    grammar, plan, examples, generate_config,
                )
                continue
            return spec_str, False, errors

        # Run validators
        all_errors = []
        if "json_schema" in validators:
            all_errors.extend(_validate_json_schema(spec_dict, grammar))

        if not all_errors:
            return spec_dict, True, []

        # Retry via repair
        if attempt < max_retries:
            spec_str = _repair(
                agent, spec_str if isinstance(spec_str, str) else json.dumps(spec_dict),
                all_errors, messages, data_schema, grammar, plan, examples,
                generate_config,
            )
        else:
            return spec_dict, False, all_errors

    # Should not reach here, but just in case
    return spec_str, False, ["Max retries exceeded"]


def _repair(agent, failed_spec_str, errors, messages, data_schema, grammar,
            plan, examples, generate_config):
    """Re-generate spec with repair context."""
    repair_messages = _build_repair_messages(
        messages, data_schema, grammar, plan, examples, failed_spec_str, errors,
    )
    backend = generate_config.get("backend", "vllm")

    if backend == "gpt":
        results = agent.gpt_completions_guided_json(
            messages=repair_messages,
            json_schema=grammar["schema_string"],
        )
        if results:
            result = results[0]
            if "arguments" in result and "spec" in result["arguments"]:
                return result["arguments"]["spec"]
            return json.dumps(result)
        return "{}"
    elif backend == "vllm":
        resp = agent.completions_guided_json(
            messages=repair_messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "RenderVisualization",
                        "description": "Render a visualization with a provided visualization grammar of graphics style specification.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "spec": grammar["schema_dict"],
                            },
                            "required": ["spec"],
                        },
                    },
                }
            ],
            json_schema=grammar["schema_string"],
        )
        return resp.choices[0].text
    else:
        raise ValueError(f"Unknown generate backend: {backend}")

# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run_vis_pipeline(agent, messages, data_schema, data_domains, grammar,
                     example_store=None, embed_fn=None, config=None):
    """Run the full visualization generation pipeline.

    Args:
        agent: UDIAgent instance (has vllm + gpt methods)
        messages: chat history
        data_schema: JSON string of dataset schema
        data_domains: JSON string of data domains
        grammar: dict from load_grammar() — schema_dict, schema_string, system_prompt
        example_store: dict from load_example_specs() — {"examples": [...], "embeddings": [...]}
        embed_fn: callable(list[str]) -> list[list[float]], or None for tags-only fallback
        config: pipeline config dict

    Returns: {"spec": dict, "valid": bool, "errors": list, "plan": dict|None}
    """
    if config is None:
        config = DEFAULT_PIPELINE_CONFIG

    # Stage 1: Plan
    plan = stage_plan(agent, messages, data_schema, data_domains, config["plan"])

    # Stage 2: Retrieve examples
    # Extract query text from the last user message
    query_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            query_text = msg.get("content", "")
            break

    examples = stage_retrieve(
        query_text, plan, example_store, embed_fn, config["retrieve"],
    )

    # Stage 3: Generate
    spec_str = stage_generate(
        agent, messages, data_schema, grammar, plan, examples, config["generate"],
    )

    # Stage 4: Validate + Repair
    spec_result, valid, errors = stage_validate(
        agent, spec_str, messages, data_schema, grammar, plan, examples,
        config["validate"], config["generate"],
    )

    return {"spec": spec_result, "valid": valid, "errors": errors, "plan": plan}
