"""
Meta codegen script: reads template visualizations and emits typed OpenAI
function-calling tool definitions. The output is dataset-agnostic — tool
parameters, template spec strings, and dispatch maps all use placeholder
names (``<E>``, ``<F>``, ``<E1.F1>``, ...). The per-request data schema is
substituted into the templates at runtime by ``udiagent.vis_generate``.

Usage:
    python -m udiagent.generate_tools \
        --templates src/udiagent/data/skills/template_visualizations.json \
        --output src/udiagent/generated_vis_tools.py
"""

import argparse
import json
import pprint
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Template analysis
# ---------------------------------------------------------------------------

def _extract_placeholders(template_str: str) -> set[str]:
    """Extract all <placeholder> names from a template string."""
    return set(re.findall(r'<([^>]+)>', template_str))


def _derive_tool_name(template: dict, index: int) -> str:
    """Derive a meaningful tool name from chart_type + description keywords."""
    chart_type = template.get("chart_type", "chart").lower()
    desc = template.get("description", "").lower()

    suffixes = []

    # Detect join/cross-entity
    if "join" in desc or "related entity" in desc:
        suffixes.append("join")

    # Detect aggregation
    agg_keywords = ["count", "average", "mean", "median", "minimum", "maximum",
                     "total", "sum", "frequency", "proportion", "percentage"]
    for kw in agg_keywords:
        if kw in desc:
            suffixes.append({"minimum": "min", "maximum": "max", "average": "avg",
                             "mean": "avg", "total": "sum", "frequency": "freq",
                             "proportion": "proportion", "percentage": "pct"}.get(kw, kw))
            break

    # Detect layout/style modifiers
    if "horizontal" in desc:
        suffixes.append("horiz")
    elif "vertical" in desc:
        suffixes.append("vert")
    if "stacked" in desc:
        suffixes.append("stacked")
    if "grouped" in desc or "side-by-side" in desc:
        suffixes.append("grouped")
    if "normalized" in desc:
        suffixes.append("normalized")
    if "color" in desc or "colored" in desc:
        suffixes.append("by_color")
    if "cumulative" in desc or "cdf" in desc:
        suffixes.append("cdf")
    if "density" in desc or "kde" in desc:
        suffixes.append("density")
    if "distribution" in desc and "cdf" not in suffixes and "density" not in suffixes:
        suffixes.append("distribution")
    if "ranked" in desc or "rank" in desc:
        suffixes.append("ranked")
    if "sorted" in desc or "ordered" in desc:
        suffixes.append("sorted")
    if "raw data" in desc or "raw" in desc:
        suffixes.append("raw")
    if "null" in desc:
        suffixes.append("null")
    if "non-null" in desc:
        suffixes.append("nonnull")
    if "min and max" in desc or "min/max" in desc:
        suffixes.append("range")
    if "distinct" in desc:
        suffixes.append("distinct")
    if "most frequent" in desc:
        suffixes.append("mode")

    suffix = "_".join(suffixes) if suffixes else "basic"
    name = f"vis_{index:03d}_{chart_type}_{suffix}"
    return re.sub(r'[^a-z0-9_]', '', name)


def _build_tool_description(template: dict) -> str:
    """Build a rich description from template metadata."""
    parts = []
    if template.get("chart_type"):
        parts.append(f"[{template['chart_type']}]")
    if template.get("description"):
        parts.append(template["description"])
    if template.get("design_considerations"):
        parts.append(f"Design: {template['design_considerations']}")
    if template.get("tasks"):
        parts.append(f"Tasks: {template['tasks']}")
    query_templates = template.get("query_templates", [])
    if isinstance(query_templates, str):
        query_templates = [query_templates]
    if query_templates:
        parts.append(f"Query patterns: {'; '.join(query_templates)}")
    return " ".join(parts)


def _get_field_type_for_placeholder(placeholder: str) -> str | None:
    """:n -> nominal, :q -> quantitative, :o -> ordinal, :q|o|n -> any"""
    if ":n" in placeholder:
        return "nominal"
    elif ":q" in placeholder and ":q|o|n" not in placeholder:
        return "quantitative"
    elif ":o" in placeholder:
        return "ordinal"
    return None


def _extract_encoding_info(spec_template: str) -> dict[str, dict]:
    """Extract encoding roles and declared types for each placeholder from a spec template.

    Parses the spec_template JSON and walks the representation mappings to find
    which visual encoding (x, y, color, theta, etc.) each placeholder is used in,
    and what data type the encoding declares.

    Returns: dict mapping placeholder base (e.g. "F1", "E2.F") to
             {"encodings": ["x", ...], "declared_type": "nominal" | "quantitative" | None}
    """
    info: dict[str, dict] = {}
    try:
        spec = json.loads(spec_template)
    except (json.JSONDecodeError, TypeError):
        return info

    rep = spec.get("representation", {})
    reps = rep if isinstance(rep, list) else [rep]
    for r in reps:
        mappings = r.get("mapping", [])
        if isinstance(mappings, dict):
            mappings = [mappings]
        for m in mappings:
            encoding = m.get("encoding", "")
            field = m.get("field", "")
            declared_type = m.get("type")  # "nominal", "quantitative", "ordinal"
            # Match fields that are a single placeholder like "<F1>" or "<E2.F>"
            match = re.fullmatch(r'<([^>]+)>', field)
            if match and encoding:
                ph = match.group(1)
                base = ph.split(":")[0] if ":" in ph else ph
                if base not in info:
                    info[base] = {"encodings": [], "declared_type": None}
                if encoding not in info[base]["encodings"]:
                    info[base]["encodings"].append(encoding)
                if declared_type and info[base]["declared_type"] is None:
                    info[base]["declared_type"] = declared_type
    return info


_ENCODING_LABELS = {
    "x": "x-axis",
    "y": "y-axis",
    "color": "color",
    "theta": "angle/size",
    "radius": "radius",
    "radius2": "outer radius",
    "opacity": "opacity",
    "size": "size",
    "text": "text label",
    "xOffset": "x-axis sub-group",
    "yOffset": "y-axis sub-group",
}


def _build_field_description(field_type: str | None, encoding_info: dict | None) -> str:
    """Build a descriptive string for a field parameter.

    Args:
        field_type: Type from placeholder suffix (:n, :q, :o) or None.
        encoding_info: {"encodings": [...], "declared_type": str|None} from spec template.
    """
    # Prefer placeholder suffix type, fall back to declared type from encoding
    resolved_type = field_type
    if not resolved_type and encoding_info:
        resolved_type = encoding_info.get("declared_type")
    type_str = resolved_type or "any type"

    encodings = encoding_info.get("encodings", []) if encoding_info else []
    if encodings:
        labels = [_ENCODING_LABELS.get(e, e) for e in encodings]
        return f"{type_str} field, encodes {', '.join(labels)}."
    return f"{type_str} field."


# ---------------------------------------------------------------------------
# Tool generation (single entity templates)
# ---------------------------------------------------------------------------

def _generate_single_entity_tool(
    template: dict, index: int
) -> tuple[dict, dict] | None:
    """Generate tool definition + param map for a single-entity template.

    The output is dataset-agnostic: constraint-based pruning (which would
    require a schema) is left to runtime validation in ``vis_generate``.
    """
    spec_template = template.get("spec_template", "")
    placeholders = _extract_placeholders(spec_template)

    tool_name = _derive_tool_name(template, index)
    description = _build_tool_description(template)
    encoding_info = _extract_encoding_info(spec_template)

    properties = {
        "entity": {"type": "string", "description": "The data entity (table) to visualize."},
    }
    required = ["entity"]
    param_map = {"entity": "E"}

    # Determine field parameters from placeholders
    seen = set()
    for ph in sorted(placeholders):
        if ph in ("E", "E.url"):
            continue
        m = re.match(r'(F\d*)', ph)
        if not m:
            continue
        base = m.group(1)  # F, F1, F2, F3
        param_name = {"F": "field", "F1": "field1", "F2": "field2", "F3": "field3"}.get(base)
        if not param_name or param_name in seen:
            continue
        seen.add(param_name)

        field_type = _get_field_type_for_placeholder(ph)
        properties[param_name] = {
            "type": "string",
            "description": _build_field_description(field_type, encoding_info.get(base)),
        }
        required.append(param_name)
        param_map[param_name] = base

    tool_def = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description[:1024],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
    return tool_def, param_map


# ---------------------------------------------------------------------------
# Tool generation (join/two-entity templates)
# ---------------------------------------------------------------------------

def _generate_join_entity_tool(
    template: dict, index: int
) -> tuple[dict, dict] | None:
    """Generate tool definition + param map for a two-entity join template.

    Dataset-agnostic — relationship/constraint checks are performed at
    runtime by ``vis_generate.validate_bindings``.
    """
    spec_template = template.get("spec_template", "")
    placeholders = _extract_placeholders(spec_template)

    tool_name = _derive_tool_name(template, index)
    description = _build_tool_description(template)
    encoding_info = _extract_encoding_info(spec_template)

    properties = {
        "entity1": {"type": "string", "description": "The primary data entity (table)."},
        "entity2": {"type": "string", "description": "The secondary data entity (table) to join with."},
    }
    required = ["entity1", "entity2"]
    param_map = {"entity1": "E1", "entity2": "E2"}

    seen = set()
    for ph in sorted(placeholders):
        if ph in ("E1", "E1.url", "E2", "E2.url", "E1.r.E2.id.from", "E1.r.E2.id.to"):
            continue

        if ph.startswith("E1.F"):
            param_name = "entity1_field"
            m = re.match(r'E1\.(F\d*)', ph)
            base = "E1." + m.group(1) if m else "E1.F"
        elif ph.startswith("E2.F"):
            param_name = "entity2_field"
            m = re.match(r'E2\.(F\d*)', ph)
            base = "E2." + m.group(1) if m else "E2.F"
        else:
            continue

        if param_name in seen:
            continue
        seen.add(param_name)

        field_type = _get_field_type_for_placeholder(ph)
        properties[param_name] = {
            "type": "string",
            "description": _build_field_description(field_type, encoding_info.get(base)),
        }
        required.append(param_name)
        param_map[param_name] = base

    tool_def = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description[:1024],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
    return tool_def, param_map


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(templates_path: str, output_path: str):
    """Generate the typed vis tools module (dataset-agnostic)."""
    with open(templates_path) as f:
        templates = json.load(f)

    tool_defs = []
    spec_templates = []
    tool_dispatch = {}
    tool_name_set = {}

    for i, template in enumerate(templates):
        spec_template = template.get("spec_template", "")
        placeholders = _extract_placeholders(spec_template)
        is_join = "E1" in placeholders or "E2" in placeholders

        if is_join:
            result = _generate_join_entity_tool(template, i)
        else:
            result = _generate_single_entity_tool(template, i)

        if result is None:
            continue

        tool_def, param_map = result
        tool_name = tool_def["function"]["name"]

        # Handle duplicate names
        if tool_name in tool_name_set:
            tool_name = f"{tool_name}_{i}"
            tool_def["function"]["name"] = tool_name
        tool_name_set[tool_name] = i

        template_idx = len(spec_templates)
        spec_templates.append(spec_template)
        tool_defs.append(tool_def)
        tool_dispatch[tool_name] = (template_idx, param_map)

    output = [
        '"""',
        'Auto-generated visualization tool definitions.',
        '',
        f'Generated from: {Path(templates_path).resolve().relative_to(Path.cwd())}',
        f'Tools: {len(tool_defs)}',
        '',
        'Dataset-agnostic — tool parameters, template spec strings, and the',
        'dispatch map use placeholder names (<E>, <F>, <E1.F1>, ...). The',
        'per-request data schema is substituted into templates at runtime by',
        'udiagent.vis_generate.',
        '',
        'DO NOT EDIT — regenerate with: python -m udiagent.generate_tools',
        '"""',
        '',
        '',
        '# Spec template strings (indexed by position)',
        f'TEMPLATES = {pprint.pformat(spec_templates, width=120)}',
        '',
        '',
        '# OpenAI function-calling tool definitions',
        f'TOOL_DEFS = {pprint.pformat(tool_defs, width=120)}',
        '',
        '',
        '# Dispatch: tool name -> (template_index, param_to_binding_map)',
        f'TOOL_DISPATCH = {pprint.pformat(tool_dispatch, width=120)}',
        '',
    ]

    Path(output_path).write_text("\n".join(output))
    print(f"Generated {len(tool_defs)} tools -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dataset-agnostic typed visualization tool definitions",
    )
    parser.add_argument(
        "--templates",
        default="src/udiagent/data/skills/template_visualizations.json",
        help="Path to template visualizations JSON",
    )
    parser.add_argument(
        "--output",
        default="src/udiagent/generated_vis_tools.py",
        help="Output Python module path",
    )
    args = parser.parse_args()
    generate(args.templates, args.output)
