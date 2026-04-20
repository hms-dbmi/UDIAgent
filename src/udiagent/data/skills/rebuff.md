---
name: rebuff
description: Generate a rebuff response when the user's request cannot be fulfilled by any available tool
---

# Rebuff Response

The user made a request that cannot be fulfilled by any available tool. Generate a helpful rebuff response.

## User Request

{{user_request}}

## Reason

{{reason}}

## Instructions

Respond with a JSON object containing exactly one key:

1. **"message"**: A polite, clear statement explaining that this specific request is not currently supported. Be specific about *why* it cannot be done.

Respond with only the JSON object. Do not include any explanation or markdown formatting.
