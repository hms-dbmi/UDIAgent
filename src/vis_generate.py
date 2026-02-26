"""
Visualization spec generation with self-correction.

Generates a UDI Grammar spec via LLM, validates against the JSON schema,
and re-prompts with error feedback if validation fails.
"""

import json
from pathlib import Path

import jsonschema


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
# Internal helpers
# ---------------------------------------------------------------------------

def _call_llm(agent, messages, grammar, config, backend):
    """Call the LLM and return the raw spec string."""
    if backend == "gpt":
        results = agent.gpt_completions_guided_json(
            messages=messages,
            json_schema=grammar["schema_string"],
            n=config.get("n", 1),
        )
        if results:
            result = results[0]
            if "arguments" in result and "spec" in result["arguments"]:
                return json.dumps(result["arguments"]["spec"])
            return json.dumps(result)
        return "{}"
    elif backend == "vllm":
        resp = agent.completions_guided_json(
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
                                "spec": grammar["schema_dict"],
                            },
                            "required": ["spec"],
                        },
                    },
                }
            ],
            json_schema=grammar["schema_string"],
            n=config.get("n", 1),
        )
        return resp.choices[0].text
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _parse_and_validate(spec_str, schema_dict):
    """Parse JSON string and validate against schema.

    Returns (spec_dict | None, errors list).
    """
    try:
        spec_dict = json.loads(spec_str) if isinstance(spec_str, str) else spec_str
    except json.JSONDecodeError as e:
        return None, [f"JSON parse error: {e}"]

    errors = []
    try:
        jsonschema.validate(instance=spec_dict, schema=schema_dict)
    except jsonschema.ValidationError as e:
        errors.append(str(e.message))
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e.message}")

    return spec_dict, errors


# ---------------------------------------------------------------------------
# Generation with self-correction
# ---------------------------------------------------------------------------

def generate_vis_spec(agent, messages, data_schema, grammar, config=None):
    """Generate a visualization spec, retrying with error feedback on validation failure.

    Args:
        agent: UDIAgent instance
        messages: chat history (list of dicts with role/content)
        data_schema: JSON string describing available datasets
        grammar: dict from load_grammar()
        config: optional dict with keys:
            backend: "gpt" | "vllm" (default "gpt")
            n: int (default 1)
            max_corrections: int (default 2) — number of retry attempts on
                validation failure. Set to 0 to disable correction.

    Returns: {"spec": dict|str, "valid": bool, "errors": list, "corrections": int}
    """
    if config is None:
        config = {}
    backend = config.get("backend", "gpt")
    max_corrections = config.get("max_corrections", 2)

    # Build prompt
    system_content = (
        f"{grammar['system_prompt']}\n\n"
        f"The following defines the available datasets:\n{data_schema}"
    )
    gen_messages = [{"role": "system", "content": system_content}] + list(messages)

    # Initial call
    spec_str = _call_llm(agent, gen_messages, grammar, config, backend)
    spec_dict, errors = _parse_and_validate(spec_str, grammar["schema_dict"])

    # Correction loop
    corrections = 0
    while errors and corrections < max_corrections:
        feedback_content = spec_str if isinstance(spec_str, str) else json.dumps(spec_str)
        gen_messages.append({"role": "assistant", "content": feedback_content})
        correction_msg = (
            "The output failed schema validation against the UDI Grammar specification.\n\n"
            "Here is an example of a valid UDI Grammar spec:\n"
            '{"source": [{"name": "sales", "source": "./data/sales.csv"}], '
            '"transformation": [{"groupby": ["region"]}, '
            '{"rollup": {"total": {"op": "sum", "field": "amount"}}}], '
            '"representation": {"mark": "bar", '
            '"mapping": [{"encoding": "x", "field": "region", "type": "nominal"}, '
            '{"encoding": "y", "field": "total", "type": "quantitative"}]}}\n\n'
            "Key rules:\n"
            '- source: array of {"name": string, "source": string}\n'
            '- transformation: each item uses the operation name as key, e.g. {"groupby": [...]}, '
            '{"rollup": {...}}, {"join": {"on": [...]}, "in": [...], "out": string}\n'
            '- representation: {"mark": string, "mapping": array of '
            '{"encoding": string, "field": string, "type": string}}\n\n'
            f"Validation errors: {'; '.join(errors)}\n\n"
            "Please regenerate the spec as valid UDI Grammar JSON."
        )
        gen_messages.append({"role": "user", "content": correction_msg})

        spec_str = _call_llm(agent, gen_messages, grammar, config, backend)
        spec_dict, errors = _parse_and_validate(spec_str, grammar["schema_dict"])
        corrections += 1

    return {
        "spec": spec_dict if spec_dict is not None else spec_str,
        "valid": len(errors) == 0,
        "errors": errors,
        "corrections": corrections,
    }
