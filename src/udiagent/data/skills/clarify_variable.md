---
name: clarify_variable
description: Generate a clarification request when the user's query is ambiguous
---

# Clarify

The user's request is ambiguous. Generate a clarification response.

## User Request

{{user_request}}

## Ambiguous Terms

{{ambiguous_terms}}

## Data Schema

{{data_schema}}

## Instructions

There are two kinds of clarification:

- **variable** — the ambiguity is about which schema field the user means. Candidates MUST reference actual `(entity, field_name)` pairs from the data schema. Use the bare field name (e.g. `sex`), never a qualified name (e.g. `donor.sex`).
- **general** — the ambiguity is not about a schema field (for example, asking which chart type the user wants). Candidates can be free-form option labels, and `entity` may be an empty string.

Respond with a JSON object containing exactly three keys:

1. **"clarification_type"**: either `"variable"` or `"general"`, per the distinction above.
2. **"message"**: A polite natural-language explanation of what is ambiguous and why clarification is needed.
3. **"ambiguous_variables"**: An array of objects, one per ambiguous term. Each object must have:
   - `"query_term"`: The term from the user's request that is ambiguous
   - `"candidates"`: An array of candidate matches, each with only:
     - `"field_name"`: For variable clarifications, the schema field name. For general clarifications, the option label.
     - `"entity"`: For variable clarifications, which dataset entity (table) the field belongs to. For general clarifications, an empty string.
   Do NOT include data_type or description — those are added automatically from the schema for variable clarifications.

Respond with only the JSON object. Do not include any explanation or markdown formatting.
