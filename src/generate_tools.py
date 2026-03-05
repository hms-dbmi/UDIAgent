"""
Meta codegen script: reads template visualizations + a data schema and generates
typed OpenAI function-calling tool definitions and Python spec-builder functions.

Usage:
    python src/generate_tools.py \
        --templates src/skills/template_visualizations.json \
        --schema data/data_domains/hubmap_data_schema.json \
        --output src/generated_vis_tools.py
"""

import argparse
import json
import pprint
import re
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------

def parse_schema(schema_path: str) -> dict:
    """Parse a UDI data schema into a structured representation.

    Returns:
        {
            "base_path": str,
            "entities": {
                "name": {
                    "path": str,
                    "url": str,  # base_path + path
                    "row_count": int,
                    "fields": {
                        "field_name": {
                            "type": str,  # "nominal", "quantitative", "ordinal", "temporal"
                            "cardinality": int,
                        }
                    },
                    "nominal_fields": [...],
                    "quantitative_fields": [...],
                },
            },
            "relationships": [
                {
                    "from_entity": str,
                    "to_entity": str,
                    "from_field": str,
                    "to_field": str,
                    "from_cardinality": str,  # "one" or "many"
                    "to_cardinality": str,
                }
            ]
        }
    """
    with open(schema_path) as f:
        raw = json.load(f)

    base_path = raw.get("udi:path", "./")
    entities = {}
    relationships = []

    for resource in raw.get("resources", []):
        name = resource["name"]
        row_count = resource.get("udi:row_count", 0)
        path = resource.get("path", "")
        url = base_path + path

        fields = {}
        nominal_fields = []
        quantitative_fields = []

        for field in resource.get("schema", {}).get("fields", []):
            cardinality = field.get("udi:cardinality", 0)
            if cardinality == 0:
                continue
            dtype = field.get("udi:data_type", "")
            fields[field["name"]] = {
                "type": dtype,
                "cardinality": cardinality,
            }
            if dtype == "nominal":
                nominal_fields.append(field["name"])
            elif dtype == "quantitative":
                quantitative_fields.append(field["name"])

        entities[name] = {
            "path": path,
            "url": url,
            "row_count": row_count,
            "fields": fields,
            "nominal_fields": nominal_fields,
            "quantitative_fields": quantitative_fields,
        }

        for fk in resource.get("schema", {}).get("foreignKeys", []):
            card = fk.get("udi:cardinality", {})
            relationships.append({
                "from_entity": name,
                "to_entity": fk["reference"]["resource"],
                "from_field": fk["fields"][0],
                "to_field": fk["reference"]["fields"][0],
                "from_cardinality": card.get("from", "many"),
                "to_cardinality": card.get("to", "one"),
            })

    return {
        "base_path": base_path,
        "entities": entities,
        "relationships": relationships,
    }


# ---------------------------------------------------------------------------
# Constraint evaluation
# ---------------------------------------------------------------------------

def _eval_field_constraints(constraints: list[str], entity_info: dict, field_name: str, prefix: str = "F") -> bool:
    """Check if a field satisfies the constraints for a given prefix (F, F1, F2, etc.).

    Only evaluates cardinality constraints like 'F.c <= 4', 'F.c > 1', etc.
    Skips relationship constraints and cross-field constraints.
    """
    field_info = entity_info["fields"].get(field_name)
    if not field_info:
        return False
    cardinality = field_info["cardinality"]
    entity_count = entity_info["row_count"]

    for constraint in constraints:
        # Match constraints for this prefix: e.g., "F.c <= 4" or "F1.c > 10"
        # Also handle "F.c * 2 < E.c" style
        c = constraint.strip()

        # F.c * 2 < E.c  ->  cardinality * 2 < entity_count
        m = re.match(rf'^{re.escape(prefix)}\.c\s*\*\s*(\d+)\s*<\s*E\d?\.c$', c)
        if m:
            multiplier = int(m.group(1))
            if not (cardinality * multiplier < entity_count):
                return False
            continue

        # Simple F.c comparisons: F.c <= 4, F.c > 1, F.c < 25, etc.
        m = re.match(rf'^{re.escape(prefix)}\.c\s*(<=|>=|<|>|==)\s*(\d+)$', c)
        if m:
            op, val = m.group(1), int(m.group(2))
            if op == '<=' and not (cardinality <= val):
                return False
            if op == '>=' and not (cardinality >= val):
                return False
            if op == '<' and not (cardinality < val):
                return False
            if op == '>' and not (cardinality > val):
                return False
            if op == '==' and not (cardinality == val):
                return False
            continue

        # Skip constraints we can't evaluate statically (cross-field, relationship, etc.)

    return True


def _eval_entity_constraints(constraints: list[str], entity_info: dict, prefix: str = "E") -> bool:
    """Check if an entity satisfies constraints like 'E.c > 0'."""
    row_count = entity_info["row_count"]
    for constraint in constraints:
        c = constraint.strip()
        m = re.match(rf'^{re.escape(prefix)}\.c\s*(<=|>=|<|>|==)\s*(\d+)$', c)
        if m:
            op, val = m.group(1), int(m.group(2))
            if op == '<=' and not (row_count <= val):
                return False
            if op == '>=' and not (row_count >= val):
                return False
            if op == '<' and not (row_count < val):
                return False
            if op == '>' and not (row_count > val):
                return False
            if op == '==' and not (row_count == val):
                return False
    return True


# ---------------------------------------------------------------------------
# Template analysis
# ---------------------------------------------------------------------------

def _extract_placeholders(template_str: str) -> set[str]:
    """Extract all <placeholder> names from a template string."""
    return set(re.findall(r'<([^>]+)>', template_str))


def _derive_tool_name(template: dict, index: int) -> str:
    """Derive a tool function name from chart_type + description."""
    desc = template.get("description", "")
    chart = template.get("chart_type", "chart")

    # Clean description into snake_case words
    words = re.sub(r'[^a-zA-Z0-9\s]', '', desc).lower().split()
    # Take first 6 words max for the name
    name_part = "_".join(words[:6])
    if not name_part:
        name_part = chart

    # Ensure uniqueness with index
    return f"vis_{index:03d}_{re.sub(r'[^a-z0-9_]', '', name_part)}"


def _build_tool_description(template: dict) -> str:
    """Build a rich description from template metadata."""
    parts = []
    if template.get("description"):
        parts.append(template["description"])
    if template.get("design_considerations"):
        parts.append(f"Design: {template['design_considerations']}")
    if template.get("tasks"):
        parts.append(f"Tasks: {template['tasks']}")
    if template.get("query_template"):
        parts.append(f"Query pattern: {template['query_template']}")
    return " ".join(parts)


def _get_field_type_for_placeholder(placeholder: str) -> str | None:
    """Determine required field type from placeholder suffix.

    :n -> nominal, :q -> quantitative, :o -> ordinal, :q|o|n -> any
    """
    if ":n" in placeholder:
        return "nominal"
    elif ":q" in placeholder and ":q|o|n" not in placeholder:
        return "quantitative"
    elif ":o" in placeholder:
        return "ordinal"
    return None  # any type


# ---------------------------------------------------------------------------
# Tool generation (single entity templates)
# ---------------------------------------------------------------------------

def _generate_single_entity_tool(
    template: dict, index: int, schema: dict
) -> tuple[dict, str] | None:
    """Generate tool definition + spec-builder code for a single-entity template.

    Returns (tool_def, python_code) or None if no valid combinations exist.
    """
    constraints = template.get("constraints", [])
    spec_template = template.get("spec_template", "")
    placeholders = _extract_placeholders(spec_template)

    tool_name = _derive_tool_name(template, index)
    description = _build_tool_description(template)

    # Determine which entities are valid
    valid_entities = []
    for ename, einfo in schema["entities"].items():
        if _eval_entity_constraints(constraints, einfo, "E"):
            valid_entities.append(ename)
    if not valid_entities:
        return None

    # Build parameters
    properties = {
        "entity": {
            "type": "string",
            "description": "The data entity (table) to visualize.",
            "enum": valid_entities,
        }
    }
    required = ["entity"]

    # Determine field parameters from placeholders
    field_params = {}
    for ph in sorted(placeholders):
        if ph in ("E", "E.url"):
            continue
        # Map placeholder to parameter name
        if ph == "F" or ph == "F:n" or ph == "F:q" or ph == "F:q|o|n":
            param_name = "field"
        elif ph.startswith("F1") or ph == "F1:n" or ph == "F1:q":
            param_name = "field1"
        elif ph.startswith("F2") or ph == "F2:n" or ph == "F2:q":
            param_name = "field2"
        elif ph.startswith("F3") or ph == "F3:n" or ph == "F3:q":
            param_name = "field3"
        else:
            continue

        if param_name in field_params:
            continue

        field_type = _get_field_type_for_placeholder(ph)
        # Determine constraint prefix (F, F1, F2, F3)
        constraint_prefix = re.match(r'(F\d*)', ph).group(1)

        # Collect valid fields across all valid entities
        valid_fields_per_entity = {}
        for ename in valid_entities:
            einfo = schema["entities"][ename]
            valid = []
            for fname, finfo in einfo["fields"].items():
                if field_type and finfo["type"] != field_type:
                    continue
                if _eval_field_constraints(constraints, einfo, fname, constraint_prefix):
                    valid.append(fname)
            valid_fields_per_entity[ename] = valid

        # Union of all valid fields for enum
        all_valid = sorted(set(f for fields in valid_fields_per_entity.values() for f in fields))
        if not all_valid:
            return None

        field_params[param_name] = {
            "type": "string",
            "description": f"Field name ({field_type or 'any type'}) from the entity.",
            "enum": all_valid,
        }
        required.append(param_name)

    properties.update(field_params)

    tool_def = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description[:1024],  # OpenAI limit
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }

    # Generate Python spec-builder function
    param_list = ", ".join(required)
    code = _generate_spec_builder_code(tool_name, param_list, spec_template, template, is_join=False)

    return tool_def, code


# ---------------------------------------------------------------------------
# Tool generation (join/two-entity templates)
# ---------------------------------------------------------------------------

def _generate_join_entity_tool(
    template: dict, index: int, schema: dict
) -> tuple[dict, str] | None:
    """Generate tool for templates involving two entities with a join."""
    constraints = template.get("constraints", [])
    spec_template = template.get("spec_template", "")
    placeholders = _extract_placeholders(spec_template)

    tool_name = _derive_tool_name(template, index)
    description = _build_tool_description(template)

    # Find valid entity pairs from relationships
    valid_pairs = []
    for rel in schema["relationships"]:
        e1_info = schema["entities"].get(rel["from_entity"])
        e2_info = schema["entities"].get(rel["to_entity"])
        if not e1_info or not e2_info:
            continue
        if _eval_entity_constraints(constraints, e1_info, "E1") and \
           _eval_entity_constraints(constraints, e2_info, "E2"):
            # Check relationship cardinality constraints
            ok = True
            for c in constraints:
                if "E1.r.E2.c.to == 'one'" in c and rel["to_cardinality"] != "one":
                    ok = False
                if "E1.r.E2.c.from == 'many'" in c and rel["from_cardinality"] != "many":
                    ok = False
            if ok:
                valid_pairs.append((rel["from_entity"], rel["to_entity"], rel))
    # Also check reverse direction
    for rel in schema["relationships"]:
        e1_info = schema["entities"].get(rel["to_entity"])
        e2_info = schema["entities"].get(rel["from_entity"])
        if not e1_info or not e2_info:
            continue
        if _eval_entity_constraints(constraints, e1_info, "E1") and \
           _eval_entity_constraints(constraints, e2_info, "E2"):
            ok = True
            for c in constraints:
                if "E1.r.E2.c.to == 'one'" in c and rel["from_cardinality"] != "one":
                    ok = False
                if "E1.r.E2.c.from == 'many'" in c and rel["to_cardinality"] != "many":
                    ok = False
            if ok:
                rev_rel = {
                    "from_entity": rel["to_entity"],
                    "to_entity": rel["from_entity"],
                    "from_field": rel["to_field"],
                    "to_field": rel["from_field"],
                    "from_cardinality": rel["to_cardinality"],
                    "to_cardinality": rel["from_cardinality"],
                }
                valid_pairs.append((rel["to_entity"], rel["from_entity"], rev_rel))

    if not valid_pairs:
        return None

    valid_e1 = sorted(set(p[0] for p in valid_pairs))
    valid_e2 = sorted(set(p[1] for p in valid_pairs))

    properties = {
        "entity1": {
            "type": "string",
            "description": "The primary data entity (table).",
            "enum": valid_e1,
        },
        "entity2": {
            "type": "string",
            "description": "The secondary data entity (table) to join with.",
            "enum": valid_e2,
        },
    }
    required = ["entity1", "entity2"]

    # Field parameters for join templates
    for ph in sorted(placeholders):
        if ph in ("E1", "E1.url", "E2", "E2.url", "E1.r.E2.id.from", "E1.r.E2.id.to"):
            continue

        if ph.startswith("E1.F") or ph.startswith("E1.F1"):
            param_name = "entity1_field"
            entity_list = valid_e1
            constraint_prefix = re.match(r'E1\.(F\d*)', ph).group(1)
        elif ph.startswith("E2.F") or ph.startswith("E2.F2"):
            param_name = "entity2_field"
            entity_list = valid_e2
            constraint_prefix = re.match(r'E2\.(F\d*)', ph).group(1)
        else:
            continue

        if param_name in properties:
            continue

        field_type = _get_field_type_for_placeholder(ph)
        all_valid = set()
        for ename in entity_list:
            einfo = schema["entities"][ename]
            for fname, finfo in einfo["fields"].items():
                if field_type and finfo["type"] != field_type:
                    continue
                if _eval_field_constraints(constraints, einfo, fname, constraint_prefix):
                    all_valid.add(fname)

        if not all_valid:
            return None

        properties[param_name] = {
            "type": "string",
            "description": f"Field name ({field_type or 'any type'}) from the entity.",
            "enum": sorted(all_valid),
        }
        required.append(param_name)

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

    param_list = ", ".join(required)
    code = _generate_spec_builder_code(tool_name, param_list, spec_template, template, is_join=True)

    return tool_def, code


# ---------------------------------------------------------------------------
# Spec builder code generation
# ---------------------------------------------------------------------------

def _generate_spec_builder_code(
    tool_name: str, param_list: str, spec_template: str, template: dict, is_join: bool
) -> str:
    """Generate Python function code that builds a spec from arguments."""
    # Escape the spec template for embedding in code
    spec_template_escaped = spec_template.replace("\\", "\\\\").replace('"', '\\"')

    if is_join:
        body = f'''def {tool_name}({param_list}, _schema=None):
    """Build spec: {template.get("description", "")}"""
    _schema = _schema or {{}}
    e1_info = _schema.get("entities", {{}}).get(entity1, {{}})
    e2_info = _schema.get("entities", {{}}).get(entity2, {{}})
    # Find join keys from relationships
    join_from, join_to = "", ""
    for rel in _schema.get("relationships", []):
        if rel["from_entity"] == entity1 and rel["to_entity"] == entity2:
            join_from, join_to = rel["from_field"], rel["to_field"]
            break
        if rel["from_entity"] == entity2 and rel["to_entity"] == entity1:
            join_from, join_to = rel["to_field"], rel["from_field"]
            break
    spec_str = "{spec_template_escaped}"
    spec_str = spec_str.replace("<E1>", entity1).replace("<E1.url>", e1_info.get("url", ""))
    spec_str = spec_str.replace("<E2>", entity2).replace("<E2.url>", e2_info.get("url", ""))
    spec_str = spec_str.replace("<E1.r.E2.id.from>", join_from)
    spec_str = spec_str.replace("<E1.r.E2.id.to>", join_to)'''
        # Add field replacements if present
        if "entity1_field" in param_list:
            body += '''
    spec_str = spec_str.replace("<E1.F>", entity1_field).replace("<E1.F1>", entity1_field)
    spec_str = spec_str.replace("<E1.F:n>", entity1_field).replace("<E1.F:q>", entity1_field)
    spec_str = spec_str.replace("<E1.F1:n>", entity1_field).replace("<E1.F1:q>", entity1_field)'''
        if "entity2_field" in param_list:
            body += '''
    spec_str = spec_str.replace("<E2.F>", entity2_field).replace("<E2.F2>", entity2_field)
    spec_str = spec_str.replace("<E2.F:n>", entity2_field).replace("<E2.F:q>", entity2_field)
    spec_str = spec_str.replace("<E2.F2:n>", entity2_field).replace("<E2.F2:q>", entity2_field)'''
        body += '''
    return json.loads(spec_str)
'''
    else:
        body = f'''def {tool_name}({param_list}, _schema=None):
    """Build spec: {template.get("description", "")}"""
    _schema = _schema or {{}}
    e_info = _schema.get("entities", {{}}).get(entity, {{}})
    spec_str = "{spec_template_escaped}"
    spec_str = spec_str.replace("<E>", entity).replace("<E.url>", e_info.get("url", ""))'''
        if "field" in param_list:
            body += '''
    spec_str = spec_str.replace("<F>", field).replace("<F:n>", field).replace("<F:q>", field).replace("<F:q|o|n>", field)'''
        if "field1" in param_list:
            body += '''
    spec_str = spec_str.replace("<F1>", field1).replace("<F1:n>", field1).replace("<F1:q>", field1)'''
        if "field2" in param_list:
            body += '''
    spec_str = spec_str.replace("<F2>", field2).replace("<F2:n>", field2).replace("<F2:q>", field2)'''
        if "field3" in param_list:
            body += '''
    spec_str = spec_str.replace("<F3>", field3).replace("<F3:n>", field3).replace("<F3:q>", field3)'''
        body += '''
    return json.loads(spec_str)
'''

    return body


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(templates_path: str, schema_path: str, output_path: str):
    """Generate the typed vis tools module."""
    with open(templates_path) as f:
        templates = json.load(f)

    schema = parse_schema(schema_path)

    tool_defs = []
    builder_codes = []
    tool_name_to_index = {}

    for i, template in enumerate(templates):
        spec_template = template.get("spec_template", "")
        placeholders = _extract_placeholders(spec_template)

        # Determine if this is a join template (has E1, E2) or single entity (has E)
        is_join = "E1" in placeholders or "E2" in placeholders

        if is_join:
            result = _generate_join_entity_tool(template, i, schema)
        else:
            result = _generate_single_entity_tool(template, i, schema)

        if result is None:
            continue

        tool_def, code = result
        tool_name = tool_def["function"]["name"]

        # Handle duplicate names
        if tool_name in tool_name_to_index:
            tool_name = f"{tool_name}_{i}"
            tool_def["function"]["name"] = tool_name
            code = code.replace(f"def {tool_def['function']['name'][:-len(f'_{i}')]}", f"def {tool_name}", 1)

        tool_name_to_index[tool_name] = i
        tool_defs.append(tool_def)
        builder_codes.append(code)

    # Write the output module
    output = [
        '"""',
        'Auto-generated visualization tool definitions and spec builders.',
        '',
        f'Generated from: {templates_path}',
        f'Schema: {schema_path}',
        f'Tools: {len(tool_defs)}',
        '',
        'DO NOT EDIT — regenerate with: python src/generate_tools.py',
        '"""',
        '',
        'import json',
        '',
        '',
        '# ---------------------------------------------------------------------------',
        '# Schema metadata (used by spec builders at runtime)',
        '# ---------------------------------------------------------------------------',
        '',
        f'SCHEMA = {pprint.pformat({"entities": {name: {"url": info["url"]} for name, info in schema["entities"].items()}, "relationships": schema["relationships"]}, width=120)}',
        '',
        '',
        '# ---------------------------------------------------------------------------',
        '# Spec builder functions',
        '# ---------------------------------------------------------------------------',
        '',
    ]

    for code in builder_codes:
        output.append('')
        output.append(code)

    # Generate TOOL_DEFS list
    output.append('')
    output.append('# ---------------------------------------------------------------------------')
    output.append('# OpenAI function-calling tool definitions')
    output.append('# ---------------------------------------------------------------------------')
    output.append('')
    output.append(f'TOOL_DEFS = {pprint.pformat(tool_defs, width=120)}')
    output.append('')

    # Generate dispatch map
    output.append('')
    output.append('# ---------------------------------------------------------------------------')
    output.append('# Dispatch: tool name -> builder function')
    output.append('# ---------------------------------------------------------------------------')
    output.append('')
    output.append('TOOL_BUILDERS = {')
    for td in tool_defs:
        name = td["function"]["name"]
        output.append(f'    "{name}": {name},')
    output.append('}')
    output.append('')

    Path(output_path).write_text("\n".join(output))
    print(f"Generated {len(tool_defs)} tools -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate typed visualization tools from templates + schema")
    parser.add_argument("--templates", default="src/skills/template_visualizations.json", help="Path to template visualizations JSON")
    parser.add_argument("--schema", default="data/data_domains/hubmap_data_schema.json", help="Path to data schema JSON")
    parser.add_argument("--output", default="src/generated_vis_tools.py", help="Output Python module path")
    args = parser.parse_args()
    generate(args.templates, args.schema, args.output)
