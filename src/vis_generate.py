"""
Single-shot visualization spec generation.

One API call: build prompt with data context → LLM → parse JSON result.
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
# Single-shot generation
# ---------------------------------------------------------------------------

def generate_vis_spec(agent, messages, data_schema, grammar, config=None):
    """Generate a visualization spec in a single API call.

    Args:
        agent: UDIAgent instance
        messages: chat history (list of dicts with role/content)
        data_schema: JSON string describing available datasets
        grammar: dict from load_grammar()
        config: optional dict with keys:
            backend: "gpt" | "vllm" (default "vllm")
            temperature: float (default 0.0)
            n: int (default 1)

    Returns: {"spec": dict|str, "valid": bool, "errors": list}
    """
    if config is None:
        config = {}
    backend = config.get("backend", "gpt")

    # Build prompt
    system_content = (
        f"{grammar['system_prompt']}\n\n"
        f"The following defines the available datasets:\n{data_schema}"
    )
    gen_messages = [{"role": "system", "content": system_content}] + list(messages)

    # Call LLM
    if backend == "gpt":
        results = agent.gpt_completions_guided_json(
            messages=gen_messages,
            json_schema=grammar["schema_string"],
            n=config.get("n", 1),
        )
        if results:
            result = results[0]
            if "arguments" in result and "spec" in result["arguments"]:
                spec_str = json.dumps(result["arguments"]["spec"])
            else:
                spec_str = json.dumps(result)
        else:
            spec_str = "{}"
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
            n=config.get("n", 1),
        )
        spec_str = resp.choices[0].text
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Parse & validate
    try:
        spec_dict = json.loads(spec_str) if isinstance(spec_str, str) else spec_str
    except json.JSONDecodeError as e:
        return {"spec": spec_str, "valid": False, "errors": [f"JSON parse error: {e}"]}

    errors = []
    try:
        jsonschema.validate(instance=spec_dict, schema=grammar["schema_dict"])
    except jsonschema.ValidationError as e:
        errors.append(str(e.message))
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e.message}")

    return {"spec": spec_dict, "valid": len(errors) == 0, "errors": errors}
