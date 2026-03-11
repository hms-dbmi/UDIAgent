---
name: free_text_explain
description: Generate free-text explanations about system capabilities, datasets, or general questions
---

# Free Text Explain

The user asked an informational question that doesn't require generating a visualization or filtering data. Generate a helpful free-text response.

## User Request

{{user_request}}

## Response Type

{{response_type}}

## Available Tools

{{available_tools}}

## Data Schema

{{data_schema}}

## Instructions

Based on the response type, generate an appropriate answer:

- **capabilities**: Summarize what the system can do — available visualization types, data operations, and filtering. Derive this from the available tools listed above.
- **data_summary**: Summarize the loaded datasets — entity names, field counts, data types, and key fields. Derive this from the data schema above.
- **general**: Answer the user's question using the available context. Be concise and informative.

Respond with a plain text answer. Be concise, accurate, and helpful. Do not include JSON, code blocks, or markdown formatting — just plain text paragraphs.
