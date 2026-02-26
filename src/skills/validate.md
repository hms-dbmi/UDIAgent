---
name: validate
description: Validate and correct a UDI Grammar spec against the schema
---

# Validate and Fix UDI Grammar Spec

The following spec failed schema validation against the UDI Grammar specification.

## Failed Spec

{{spec_str}}

## Validation Errors

{{errors}}

## UDI Grammar Format

The output must be a valid UDI Grammar JSON object with these top-level keys:

- **source**: array of data sources, each with `"name"` (string) and `"source"` (string, CSV path)
- **transformation** (optional): array of data operations. Each operation uses the operation name as the key:
  - `{"groupby": ["field1", "field2"]}`
  - `{"rollup": {"new_field": {"op": "count|sum|mean|min|max|median", "field": "source_field"}}}`
  - `{"join": {"on": ["left_key", "right_key"]}, "in": ["left_table", "right_table"], "out": "joined_name"}`
  - `{"filter": "expression"}`
  - `{"orderby": [{"field": "name", "order": "ascending|descending"}]}`
  - `{"derive": {"new_field": "expression"}}`
  - `{"binby": {"field": "name", "step": number}}`
- **representation**: visualization specification with:
  - `"mark"`: one of `"bar"`, `"line"`, `"point"`, `"area"`, `"arc"`, `"rect"`, `"text"`, `"geometry"`
  - `"mapping"`: array of field mappings, each with `"encoding"` (e.g. `"x"`, `"y"`, `"color"`), `"field"` (string), and `"type"` (`"quantitative"`, `"nominal"`, `"ordinal"`, `"temporal"`)

## Example of a Valid Spec

```json
{"source": [{"name": "sales", "source": "./data/sales.csv"}], "transformation": [{"groupby": ["region"]}, {"rollup": {"total": {"op": "sum", "field": "amount"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "region", "type": "nominal"}, {"encoding": "y", "field": "total", "type": "quantitative"}]}}
```

Fix the spec so it conforms to the UDI Grammar schema. Respond with only the corrected JSON spec. Do not include any explanation or markdown formatting.
