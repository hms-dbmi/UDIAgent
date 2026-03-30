"""Optional dependency handling."""

import logging

logger = logging.getLogger(__name__)


def get_openai_class():
    """Return the OpenAI client class, preferring langfuse-wrapped if available."""
    try:
        from langfuse.openai import OpenAI

        return OpenAI
    except ImportError:
        from openai import OpenAI

        return OpenAI
