"""Unified grammar and skill loading with importlib.resources support."""

import importlib.resources
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    """A skill loaded from a markdown file."""

    name: str
    description: str
    instructions: str


def _package_data_path() -> Path:
    """Return the path to the bundled data directory."""
    return importlib.resources.files("udiagent") / "data"


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


def load_skills(skills_dir: Optional[str | Path] = None) -> dict[str, Skill]:
    """Load all skill .md files from a directory.

    If *skills_dir* is ``None``, uses the bundled package data.
    Returns dict mapping skill name -> Skill instance.
    """
    if skills_dir is None:
        skills_path = _package_data_path() / "skills"
    else:
        skills_path = Path(skills_dir)

    if not skills_path.is_dir():
        return {}

    skills: dict[str, Skill] = {}
    for md_file in sorted(skills_path.glob("*.md")):
        text = md_file.read_text()
        metadata, body = _parse_frontmatter(text)
        name = metadata.get("name", md_file.stem)
        description = metadata.get("description", "")
        skills[name] = Skill(name=name, description=description, instructions=body)

    return skills


def load_grammar(grammar_name: str = "udi", base_path: Optional[str | Path] = None) -> dict:
    """Load a grammar definition by name.

    If *base_path* is ``None``, uses the bundled package data.
    Returns {"schema_dict": ..., "schema_string": ..., "system_prompt": ...}
    """
    if base_path is None:
        base = _package_data_path()
    else:
        base = Path(base_path)

    if grammar_name == "udi":
        schema_dict = json.loads((base / "UDIGrammarSchema.json").read_text())
        schema_string = (base / "UDIGrammarSchema_spec_string.json").read_text()
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
