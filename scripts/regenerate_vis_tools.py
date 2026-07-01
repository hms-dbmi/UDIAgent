"""
Regenerate template visualizations and typed tool definitions in one step.

Usage:
    # Line-item set (default)
    python scripts/regenerate_vis_tools.py
    python scripts/regenerate_vis_tools.py --schema data/data_domains/hubmap_data_schema.json

    # Data-cube set
    python scripts/regenerate_vis_tools.py --template-set cube
"""

import argparse
import subprocess
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_pkg_skills = _repo_root / "src" / "udiagent" / "data" / "skills"
_generate_tools = _repo_root / "src" / "udiagent" / "generate_tools.py"

# Per-template-set defaults: which generator builds the templates, where the
# templates and generated tools live, and the schema they are typed against.
_TEMPLATE_SETS = {
    "line_item": {
        "generator": _repo_root / "scripts" / "template_viz_generation.py",
        "templates": _pkg_skills / "template_visualizations.json",
        "schema": _repo_root / "data" / "data_domains" / "hubmap_data_schema.json",
        "tools": _repo_root / "src" / "udiagent" / "generated_vis_tools.py",
    },
    "cube": {
        "generator": _repo_root / "scripts" / "template_viz_generation_cube.py",
        "templates": _pkg_skills / "template_visualizations_cube.json",
        "schema": _repo_root / "data" / "data_domains" / "encounter_cube_schema.json",
        "tools": _repo_root / "src" / "udiagent" / "generated_vis_tools_cube.py",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate template visualizations and typed tool definitions.",
    )
    parser.add_argument(
        "--template-set",
        choices=sorted(_TEMPLATE_SETS),
        default="line_item",
        help="Which template set to regenerate (default: line_item).",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to data schema JSON (default: the schema for the chosen template set).",
    )
    parser.add_argument(
        "--templates-output",
        type=str,
        default=None,
        help="Output path for template visualizations JSON.",
    )
    parser.add_argument(
        "--tools-output",
        type=str,
        default=None,
        help="Output path for the generated tools module.",
    )
    args = parser.parse_args()

    cfg = _TEMPLATE_SETS[args.template_set]
    schema = args.schema or str(cfg["schema"])
    templates_output = args.templates_output or str(cfg["templates"])
    tools_output = args.tools_output or str(cfg["tools"])

    # Step 1: Generate template visualizations
    print(f"=== Step 1: Generating '{args.template_set}' template visualizations ===")
    result = subprocess.run(
        [
            sys.executable,
            str(cfg["generator"]),
            "-o", templates_output,
        ],
        cwd=str(_repo_root),
    )
    if result.returncode != 0:
        print("ERROR: template generation failed", file=sys.stderr)
        sys.exit(1)

    # Step 2: Generate typed tool definitions
    print("\n=== Step 2: Generating typed tool definitions ===")
    result = subprocess.run(
        [
            sys.executable,
            str(_generate_tools),
            "--templates", templates_output,
            "--schema", schema,
            "--output", tools_output,
        ],
        cwd=str(_repo_root),
    )
    if result.returncode != 0:
        print("ERROR: generate_tools.py failed", file=sys.stderr)
        sys.exit(1)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
