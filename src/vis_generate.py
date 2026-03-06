"""
Markdown-driven skills infrastructure for visualization spec generation.

Skills are .md files on disk (YAML frontmatter + LLM instructions).
A code-driven executor runs a plan (ordered list of skill names),
calling the LLM with each skill's instructions and passing a shared
context between them.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jsonschema


# ---------------------------------------------------------------------------
# Skill data model
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A skill loaded from a markdown file."""

    name: str
    description: str
    instructions: str


# ---------------------------------------------------------------------------
# Skill loader
# ---------------------------------------------------------------------------


def _parse_frontmatter(text):
    """Parse YAML frontmatter from a markdown string.

    Returns (metadata dict, body string).
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}, text

    body = text[match.end() :]
    metadata = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return metadata, body


def _render_template(instructions, variables):
    """Replace {{key}} placeholders in instructions with values from variables.

    Supports including arbitrary textual data in skill prompts.
    Unknown placeholders are left as-is.
    """

    def replacer(m):
        key = m.group(1).strip()
        if key in variables:
            return str(variables[key])
        return m.group(0)

    return re.sub(r"\{\{(.+?)\}\}", replacer, instructions)


def load_skills(skills_dir="./src/skills"):
    """Load all skill .md files from a directory.

    Returns dict mapping skill name -> Skill instance.
    """
    skills = {}
    skills_path = Path(skills_dir)
    if not skills_path.is_dir():
        return skills

    for md_file in sorted(skills_path.glob("*.md")):
        text = md_file.read_text()
        metadata, body = _parse_frontmatter(text)
        name = metadata.get("name", md_file.stem)
        description = metadata.get("description", "")
        skills[name] = Skill(name=name, description=description, instructions=body)

    return skills


# ---------------------------------------------------------------------------
# Few-shot example loading
# ---------------------------------------------------------------------------

_examples_cache: dict[str, Optional[str]] = {}


def _load_examples(
    examples_path: str = "./src/skills/template_visualizations.json",
) -> str:
    """Load few-shot examples from a JSON file and format them for prompt injection.

    Each example is formatted as a query/spec pair. Results are cached by path.
    Returns empty string if file doesn't exist or is empty.
    """
    if examples_path in _examples_cache:
        return _examples_cache[examples_path]

    path = Path(examples_path)
    if not path.exists():
        _examples_cache[examples_path] = ""
        return ""

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        _examples_cache[examples_path] = ""
        return ""

    if not data:
        _examples_cache[examples_path] = ""
        return ""

    lines = []
    for i, ex in enumerate(data, 1):
        query_templates = ex.get("query_templates", ex.get("query_template", ""))
        if isinstance(query_templates, list):
            query = "; ".join(query_templates)
        else:
            query = query_templates
        spec = ex.get("spec_template", "")
        if not query or not spec:
            continue
        lines.append(f"**Example {i}** (type: {ex.get('chart_type', 'unknown')})")
        lines.append(f"- Query: {query}")
        desc = ex.get("description", "")
        if desc:
            lines.append(f"- Description: {desc}")
        design = ex.get("design_considerations", "")
        if design:
            lines.append(f"- Design: {design}")
        tasks = ex.get("tasks", "")
        if tasks:
            lines.append(f"- Tasks: {tasks}")
        lines.append(f"- Spec: {spec}")
        lines.append("")

    result = "\n".join(lines)
    _examples_cache[examples_path] = result
    return result


# ---------------------------------------------------------------------------
# Grammar loading (unchanged — config/setup, not a skill)
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
# LLM call helpers
# ---------------------------------------------------------------------------


def _call_llm_with_tools(agent, messages, tools, config):
    """Call the LLM with function-calling tools. Returns (tool_name, arguments) or None."""
    try:
        resp = agent.gpt_model.chat.completions.create(
            model=agent.gpt_model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1024,
        )
        choice = resp.choices[0]
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            return tc.function.name, json.loads(tc.function.arguments)
    except Exception:
        pass
    return None


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
# Skill executor
# ---------------------------------------------------------------------------


def _resolve_placeholder(tag, bindings, schema):
    """Resolve a single <tag> placeholder using bindings and schema."""
    # Entity URL: E.url, E1.url, E2.url
    if tag.endswith(".url"):
        entity_key = tag[:-4]
        entity_name = bindings.get(entity_key, "")
        return schema.get("entities", {}).get(entity_name, {}).get("url", "")

    # Relationship join keys: E1.r.E2.id.from, E1.r.E2.id.to
    if ".r." in tag and ".id." in tag:
        parts = tag.split(".")
        e1_name = bindings.get(parts[0], "")
        e2_name = bindings.get(parts[2], "")
        direction = parts[4]  # "from" or "to"
        for rel in schema.get("relationships", []):
            if rel["from_entity"] == e1_name and rel["to_entity"] == e2_name:
                return rel["from_field"] if direction == "from" else rel["to_field"]
            if rel["from_entity"] == e2_name and rel["to_entity"] == e1_name:
                return rel["to_field"] if direction == "from" else rel["from_field"]
        return ""

    # Strip type suffix: F:n -> F, E1.F:q -> E1.F
    base = tag.split(":")[0] if ":" in tag else tag

    return bindings.get(base, "")


def instantiate_template(spec_template, bindings, schema):
    """Resolve all <placeholder> tags in a spec template.

    Args:
        spec_template: Template string with <E>, <F:n>, <E.url>, etc.
        bindings: Maps abstract names to real names, e.g. {"E": "donors", "F": "sex"}
        schema: Dict with "entities" (name -> {"url": ...}) and "relationships".

    Returns: Parsed spec dict.
    """
    spec = spec_template
    while True:
        match = re.search(r"<([^>]+)>", spec)
        if not match:
            break
        resolved = _resolve_placeholder(match.group(1), bindings, schema)
        spec = spec.replace(match.group(0), resolved, 1)
    return json.loads(spec)


def _extract_xy_placeholders(spec_template):
    """Extract placeholder names used in x and y encodings from a spec template.

    Returns dict like {"x": "F2", "y": "F1"} with the first placeholder-based
    field found for each encoding. Non-placeholder fields (e.g. "count <E>") are ignored.
    """
    result = {}
    try:
        spec = json.loads(spec_template)
    except (json.JSONDecodeError, TypeError):
        return result

    rep = spec.get("representation", {})
    reps = rep if isinstance(rep, list) else [rep]
    for r in reps:
        mappings = r.get("mapping", [])
        if isinstance(mappings, dict):
            mappings = [mappings]
        for m in mappings:
            enc = m.get("encoding")
            field = m.get("field", "")
            if enc in ("x", "y") and enc not in result:
                # Only consider fields that are a single placeholder like "<F1>"
                match = re.fullmatch(r'<([^>]+)>', field)
                if match:
                    result[enc] = match.group(1)
    return result


def validate_bindings(spec_template, bindings, schema):
    """Validate tool bindings against the schema before template instantiation.

    Returns list of error strings (empty = valid).
    """
    errors = []
    entities = schema.get("entities", {})
    entity_names = list(entities.keys())

    # Collect entity bindings (E, E1, E2)
    entity_bindings = {}
    for key, val in bindings.items():
        if key in ("E", "E1", "E2"):
            entity_bindings[key] = val

    # Check entities exist
    for key, name in entity_bindings.items():
        if name not in entities:
            errors.append(f"Entity '{name}' not found. Available: {', '.join(entity_names)}")

    if errors:
        return errors

    # Check join entities are different
    if "E1" in entity_bindings and "E2" in entity_bindings:
        if entity_bindings["E1"] == entity_bindings["E2"]:
            errors.append(f"entity1 and entity2 cannot be the same ('{entity_bindings['E1']}')")
            return errors

        # Check relationship exists
        e1, e2 = entity_bindings["E1"], entity_bindings["E2"]
        has_rel = any(
            (r["from_entity"] == e1 and r["to_entity"] == e2) or
            (r["from_entity"] == e2 and r["to_entity"] == e1)
            for r in schema.get("relationships", [])
        )
        if not has_rel:
            errors.append(f"No relationship between '{e1}' and '{e2}'")

    # Check that x and y encodings don't resolve to the same field
    xy_placeholders = _extract_xy_placeholders(spec_template)
    if xy_placeholders.get("x") and xy_placeholders.get("y"):
        x_binding = xy_placeholders["x"].split(":")[0]  # strip type suffix
        y_binding = xy_placeholders["y"].split(":")[0]
        x_val = bindings.get(x_binding)
        y_val = bindings.get(y_binding)
        if x_val and y_val and x_val == y_val:
            errors.append(
                f"x and y encodings must use different fields: "
                f"both '{x_binding}' and '{y_binding}' are set to '{x_val}'"
            )

    # Extract placeholder type requirements from spec_template
    placeholder_types = {}
    for match in re.finditer(r'<([^>]+)>', spec_template):
        ph = match.group(1)
        # Determine the binding key: strip type suffix, e.g. "F:n" -> "F", "E1.F:q" -> "E1.F"
        base = ph.split(":")[0] if ":" in ph else ph
        field_type = None
        if ":n" in ph:
            field_type = "nominal"
        elif ":q" in ph and ":q|o|n" not in ph:
            field_type = "quantitative"
        elif ":o" in ph:
            field_type = "ordinal"
        if field_type and base not in placeholder_types:
            placeholder_types[base] = field_type

    # Check fields exist on entities and types match
    for key, field_name in bindings.items():
        if key in ("E", "E1", "E2"):
            continue

        # Determine which entity this field belongs to
        if key.startswith("E1."):
            entity_name = entity_bindings.get("E1")
        elif key.startswith("E2."):
            entity_name = entity_bindings.get("E2")
        else:
            entity_name = entity_bindings.get("E")

        if not entity_name or entity_name not in entities:
            continue

        entity_fields = entities[entity_name].get("fields", {})

        if field_name not in entity_fields:
            available_by_type = {}
            for fn, ft in entity_fields.items():
                available_by_type.setdefault(ft, []).append(fn)
            avail_str = "; ".join(f"{t}: {', '.join(fs)}" for t, fs in available_by_type.items())
            errors.append(
                f"Field '{field_name}' not found on entity '{entity_name}'. "
                f"Available fields — {avail_str}"
            )
            continue

        # Check field type matches template requirement
        expected_type = placeholder_types.get(key)
        if expected_type:
            actual_type = entity_fields[field_name]
            if actual_type != expected_type:
                # List available fields of the expected type
                matching = [fn for fn, ft in entity_fields.items() if ft == expected_type]
                errors.append(
                    f"Field '{field_name}' is {actual_type} but template requires {expected_type}. "
                    f"Available {expected_type} fields: {', '.join(matching)}"
                )

    return errors


def _load_generated_tools():
    """Load generated tool data. Returns (tool_defs, tool_dispatch, templates, schema) or None."""
    try:
        from generated_vis_tools import TOOL_DEFS, TOOL_DISPATCH, TEMPLATES, SCHEMA
        return TOOL_DEFS, TOOL_DISPATCH, TEMPLATES, SCHEMA
    except ImportError:
        return None


def _execute_generate(skill, context):
    """Execute the generate skill: try function-calling tools first, fall back to LLM."""
    agent = context["agent"]
    grammar = context["grammar"]
    config = context["config"]
    data_schema = context["data_schema"]
    data_schema_simple = simplify_data_schema(data_schema)
    backend = config.get("backend", "gpt")

    # --- Primary path: function-calling with generated tools ---
    generated = _load_generated_tools()
    if generated is not None and backend == "gpt":
        tool_defs, tool_dispatch, templates, tool_schema = generated

        # Build system prompt with data schema context
        system_msg = (
            "You are a data visualization assistant. The user wants a visualization "
            "from the available datasets. Select the most appropriate visualization "
            "tool and provide the correct arguments.\n\n"
            f"## Available Datasets\n\n{data_schema_simple}"
        )
        tool_messages = [{"role": "system", "content": system_msg}] + list(context["messages"])

        result = _call_llm_with_tools(agent, tool_messages, tool_defs, config)
        for _attempt in range(2):  # initial + one retry
            if result is None:
                break
            tool_name, tool_args = result
            dispatch = tool_dispatch.get(tool_name)
            if dispatch is None:
                break

            template_idx, param_map = dispatch
            bindings = {param_map[k]: v for k, v in tool_args.items() if k in param_map}
            validation_errors = validate_bindings(templates[template_idx], bindings, tool_schema)

            if validation_errors:
                if _attempt == 0:
                    # Retry once with error hints
                    hint = "The previous tool call had errors:\n" + "\n".join(f"- {e}" for e in validation_errors)
                    retry_messages = tool_messages + [
                        {"role": "assistant", "content": f"Tool call: {tool_name}({json.dumps(tool_args)})"},
                        {"role": "user", "content": hint + "\n\nPlease select a corrected tool call."},
                    ]
                    result = _call_llm_with_tools(agent, retry_messages, tool_defs, config)
                    continue
                else:
                    break  # Still invalid after retry, fall through

            try:
                spec_dict = instantiate_template(templates[template_idx], bindings, tool_schema)
                spec_str = json.dumps(spec_dict)
                context["spec_str"] = spec_str
                context["gen_messages"] = tool_messages
                context["tool_used"] = tool_name
                context["tool_args"] = tool_args
                context["validation_retries"] = _attempt
                return context
            except Exception:
                break  # Fall through to LLM generation

    # --- Fallback: single-shot LLM generation ---
    examples_path = config.get(
        "examples_path", "./src/skills/template_visualizations.json"
    )
    examples = _load_examples(examples_path)

    rendered = _render_template(
        skill.instructions,
        {
            "data_schema": data_schema_simple,
            "examples": examples,
        },
    )

    gen_messages = [{"role": "system", "content": rendered}] + list(context["messages"])

    spec_str = _call_llm(agent, gen_messages, grammar, config, backend)
    context["spec_str"] = spec_str
    context["gen_messages"] = gen_messages
    context["tool_used"] = None
    context["tool_args"] = None
    return context


def simplify_data_schema(data_schema):
    """Simplify the data schema for better LLM consumption.
        - Convert from json to yaml.
        - resolve file path.
        - remove empty tables, extra information, etc.
    Format should be:
        tables:
        - name: table1
          description: description of table1 (optional)
          path: resolved file path of table1 [udi:path + resources[i].path]
          rows: row count
          columns:
            - name: column1
              description: description of column1 (optional)
              type: data type of column1 [udi:data_type]
    """
    try:
        schema = (
            json.loads(data_schema) if isinstance(data_schema, str) else data_schema
        )
    except (json.JSONDecodeError, TypeError):
        return data_schema

    base_path = schema.get("udi:path", "./")
    lines = ["tables:"]

    for resource in schema.get("resources", []):
        row_count = resource.get("udi:row_count", 0)
        if row_count == 0:
            continue

        name = resource.get("name", "")
        description = resource.get("description", "")
        path = base_path + resource.get("path", "")

        lines.append(f"  - name: {name}")
        lines.append(f"    path: {path}")
        if description:
            lines.append(f"    description: {description}")
        # lines.append(f"    rows: {row_count}")

        fields = resource.get("schema", {}).get("fields", [])
        columns = []
        for field in fields:
            if field.get("udi:cardinality", 0) == 0:
                continue
            col_name = field.get("name", "")
            col_type = field.get("udi:data_type")
            desc = field.get("description", "").strip()
            col_lines = [f"        - name: {col_name}"]
            col_lines.append(f"          type: {col_type}")
            if col_type == "nominal" or col_type == "ordinal":
                cardinality = field.get("udi:cardinality", 0)
                col_lines.append(f"          unique_values: {cardinality}")
            if desc:
                col_lines.append(f"          description: {desc}")
            columns.append("\n".join(col_lines))

        if columns:
            lines.append("    columns:")
            lines.extend(columns)

    return "\n".join(lines)


def _execute_validate(skill, context):
    """Execute the validate skill: parse, validate, and correct via LLM."""
    agent = context["agent"]
    grammar = context["grammar"]
    config = context["config"]
    backend = config.get("backend", "gpt")
    max_corrections = config.get("max_corrections", 0)
    spec_str = context.get("spec_str", "{}")
    gen_messages = context.get("gen_messages", list(context["messages"]))

    spec_dict, errors = _parse_and_validate(spec_str, grammar["schema_dict"])

    # Load few-shot examples for validation context
    examples_path = config.get(
        "examples_path", "./src/skills/template_visualizations.json"
    )
    examples = _load_examples(examples_path)

    corrections = 0
    while errors and corrections < max_corrections:
        # Render the validate skill with current errors as context
        rendered = _render_template(
            skill.instructions,
            {
                "spec_str": spec_str
                if isinstance(spec_str, str)
                else json.dumps(spec_str),
                "errors": "; ".join(errors),
                "examples": examples,
            },
        )

        feedback_content = (
            spec_str if isinstance(spec_str, str) else json.dumps(spec_str)
        )
        gen_messages.append({"role": "assistant", "content": feedback_content})
        gen_messages.append({"role": "user", "content": rendered})

        spec_str = _call_llm(agent, gen_messages, grammar, config, backend)
        spec_dict, errors = _parse_and_validate(spec_str, grammar["schema_dict"])
        corrections += 1

    context["spec_str"] = spec_str
    context["spec_dict"] = spec_dict
    context["valid"] = len(errors) == 0
    context["errors"] = errors
    context["corrections"] = corrections
    return context


# Map skill names to executor functions.
# New skills need an entry here only if they require custom execution logic
# beyond a simple LLM call.
_SKILL_EXECUTORS = {
    "generate": _execute_generate,
    "validate": _execute_validate,
}


def run_skills(plan, context, registry):
    """Execute skills in plan order, threading context through each.

    Args:
        plan: list of skill name strings
        context: shared state dict
        registry: dict mapping skill names to Skill instances

    Returns: the final context dict
    """
    for skill_name in plan:
        if skill_name not in registry:
            raise ValueError(f"Unknown skill: {skill_name}")
        skill = registry[skill_name]

        executor_fn = _SKILL_EXECUTORS.get(skill_name)
        if executor_fn is not None:
            context = executor_fn(skill, context)
        else:
            # Default executor: render skill instructions as system prompt,
            # call LLM, store raw result. Works for simple skills that just
            # need an LLM response without custom parsing.
            rendered = _render_template(skill.instructions, context)
            messages = [{"role": "system", "content": rendered}] + list(
                context["messages"]
            )
            spec_str = _call_llm(
                context["agent"],
                messages,
                context["grammar"],
                context["config"],
                context["config"].get("backend", "gpt"),
            )
            context["spec_str"] = spec_str

    return context


# ---------------------------------------------------------------------------
# Public API (backwards compatible)
# ---------------------------------------------------------------------------


def generate_vis_spec(agent, messages, data_schema, grammar, config=None):
    """Generate a visualization spec using the skills pipeline.

    Args:
        agent: UDIAgent instance
        messages: chat history (list of dicts with role/content)
        data_schema: JSON string describing available datasets
        grammar: dict from load_grammar()
        config: optional dict with keys:
            backend: "gpt" | "vllm" (default "gpt")
            n: int (default 1)
            max_corrections: int (default 2)

    Returns: {"spec": dict|str, "valid": bool, "errors": list, "corrections": int}
    """
    if config is None:
        config = {}

    # Load skills from disk
    registry = load_skills()

    # Build shared context
    context = {
        "agent": agent,
        "messages": messages,
        "data_schema": data_schema,
        "grammar": grammar,
        "config": config,
        "spec_str": "{}",
        "spec_dict": None,
        "valid": False,
        "errors": [],
        "corrections": 0,
    }

    # Execute the default plan
    plan = ["generate", "validate"]
    context = run_skills(plan, context, registry)

    spec = context["spec_dict"] if context["spec_dict"] is not None else context["spec_str"]
    if not isinstance(spec, str):
        spec = json.dumps(spec)

    return {
        "spec": spec,
        "valid": context["valid"],
        "errors": context["errors"],
        "corrections": context["corrections"],
        "meta": {
            "tool_used": context.get("tool_used"),
            "tool_args": context.get("tool_args"),
            "validation_retries": context.get("validation_retries", 0),
        },
    }
