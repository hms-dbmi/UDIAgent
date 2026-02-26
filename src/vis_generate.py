"""
Markdown-driven skills infrastructure for visualization spec generation.

Skills are .md files on disk (YAML frontmatter + LLM instructions).
A code-driven executor runs a plan (ordered list of skill names),
calling the LLM with each skill's instructions and passing a shared
context between them.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

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

    body = text[match.end():]
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

def _execute_generate(skill, context):
    """Execute the generate skill: single-shot LLM call."""
    agent = context["agent"]
    grammar = context["grammar"]
    config = context["config"]
    backend = config.get("backend", "gpt")

    # Render skill instructions with context variables
    rendered = _render_template(skill.instructions, {
        "data_schema": context["data_schema"],
    })

    # Build messages: skill instructions as system prompt + user conversation
    gen_messages = [{"role": "system", "content": rendered}] + list(context["messages"])

    spec_str = _call_llm(agent, gen_messages, grammar, config, backend)
    context["spec_str"] = spec_str
    context["gen_messages"] = gen_messages
    return context


def _execute_validate(skill, context):
    """Execute the validate skill: parse, validate, and correct via LLM."""
    agent = context["agent"]
    grammar = context["grammar"]
    config = context["config"]
    backend = config.get("backend", "gpt")
    max_corrections = config.get("max_corrections", 2)
    spec_str = context.get("spec_str", "{}")
    gen_messages = context.get("gen_messages", list(context["messages"]))

    spec_dict, errors = _parse_and_validate(spec_str, grammar["schema_dict"])

    corrections = 0
    while errors and corrections < max_corrections:
        # Render the validate skill with current errors as context
        rendered = _render_template(skill.instructions, {
            "spec_str": spec_str if isinstance(spec_str, str) else json.dumps(spec_str),
            "errors": "; ".join(errors),
        })

        feedback_content = spec_str if isinstance(spec_str, str) else json.dumps(spec_str)
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
            messages = [{"role": "system", "content": rendered}] + list(context["messages"])
            spec_str = _call_llm(
                context["agent"], messages, context["grammar"],
                context["config"], context["config"].get("backend", "gpt"),
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

    return {
        "spec": context["spec_dict"] if context["spec_dict"] is not None else context["spec_str"],
        "valid": context["valid"],
        "errors": context["errors"],
        "corrections": context["corrections"],
    }
