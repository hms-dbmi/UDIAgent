"""Generate the *data-cube* template visualization set.

Unlike ``template_viz_generation.py`` (which targets tidy, line-item tables and
re-aggregates with groupby/rollup/count), this generator targets a **pre-aggregated
powerset cube**: one measure column (e.g. ``cnt``) plus several dimension columns,
where every row is a marginal/crosstab and *empty* dimension columns mean the row is
aggregated over that dimension.

The core idea: to read the count for a set of "active" dimensions, you do NOT
re-aggregate. You **filter to the marginal rows** — the active dimensions non-null and
every other dimension null — then map the measure directly. Proportional charts do this
marginal filter first, then compute proportions on top.

Chart types that require per-record data (scatterplot, grouped_scatter, dot,
grouped_dot, histogram, CDF/KDE/density) are impossible on a cube and are omitted.

Specs are emitted as plain dicts (no ``udi_grammar_py`` dependency) and every spec is
validated against ``UDIGrammarSchema.json`` before being written.

Usage:
    python scripts/template_viz_generation_cube.py \
        --schema data/data_domains/encounter_cube_schema.json \
        -o src/udiagent/data/skills/template_visualizations_cube.json
"""

import argparse
import json
from pathlib import Path

import jsonschema

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SCHEMA = _REPO_ROOT / "data" / "data_domains" / "encounter_cube_schema.json"
_DEFAULT_OUT = (
    _REPO_ROOT / "src" / "udiagent" / "data" / "skills" / "template_visualizations_cube.json"
)
_GRAMMAR = _REPO_ROOT / "src" / "udiagent" / "data" / "UDIGrammarSchema.json"


# ---------------------------------------------------------------------------
# Cube description (loaded from the UDI schema)
# ---------------------------------------------------------------------------


def load_cube(schema_path):
    """Read measure, dimensions and per-field metadata from a UDI cube schema."""
    raw = json.loads(Path(schema_path).read_text())
    resource = raw["resources"][0]
    fields = {f["name"]: f for f in resource["schema"]["fields"]}
    measures = resource.get("udi:measures")
    dims = resource.get("udi:dimensions")
    if not measures or not dims:
        # Fall back: a single quantitative field is the measure, the rest are dims.
        measures = [n for n, f in fields.items() if f.get("udi:data_type") == "quantitative"][:1]
        dims = [n for n in fields if n not in measures]
    return {
        "entity": resource["name"],
        "measure": measures[0],
        "dims": list(dims),
        "fields": fields,
    }


# Human-friendly names for query templates / descriptions.
DISPLAY = {
    "cnt": "encounter count",
    "period_start_month": "month",
    "class_display": "encounter class",
    "age_at_visit": "age",
    "gender": "gender",
    "race_display": "race",
    "ethnicity_display": "ethnicity",
}


def display(field):
    return DISPLAY.get(field, field.replace("_", " "))


# ---------------------------------------------------------------------------
# Spec building blocks
# ---------------------------------------------------------------------------


def source(cube):
    return {"name": "<E>", "source": "<E.url>"}


def marginal_filter(cube, active):
    """Filter expression selecting the marginal rows for the *active* dimensions.

    Active dimensions must be non-null and every other dimension null, which uniquely
    selects the pre-aggregated rows at that grouping level in a powerset cube.
    """
    active = list(active)
    parts = []
    for d in cube["dims"]:
        if d in active:
            parts.append(f"d['{d}'] != null")
        else:
            parts.append(f"d['{d}'] == null")
    return " && ".join(parts)


def enc_type(cube, field):
    """Map a cube field's data type to a UDI grammar encoding type.

    The grammar only allows quantitative / ordinal / nominal, so temporal
    dimensions are encoded as ordinal (ordered categories).
    """
    dt = cube["fields"].get(field, {}).get("udi:data_type", "nominal")
    if dt == "quantitative":
        return "quantitative"
    if dt == "temporal":
        return "ordinal"
    return "nominal"


def data_type(cube, field):
    """Return the cube field's declared udi:data_type (quantitative/temporal/nominal/ordinal)."""
    return cube["fields"].get(field, {}).get("udi:data_type", "nominal")


def cardinality(cube, field):
    return cube["fields"].get(field, {}).get("udi:cardinality", 0)


# ---------------------------------------------------------------------------
# Complexity + row assembly (mirrors template_viz_generation.py conventions)
# ---------------------------------------------------------------------------


def _key_count(node):
    if isinstance(node, dict):
        return sum(_key_count(v) for v in node.values())
    if isinstance(node, list):
        return sum(_key_count(v) for v in node)
    return 1


def make_row(query_templates, spec, chart_type, task_types, description,
             design_considerations, tasks):
    n = _key_count(spec)
    if n <= 12:
        complexity = "simple"
    elif n <= 24:
        complexity = "medium"
    elif n <= 36:
        complexity = "complex"
    else:
        complexity = "extra complex"
    return {
        "query_templates": query_templates,
        "spec_template": json.dumps(spec),
        "creation_method": "template",
        "chart_type": chart_type,
        "chart_complexity": complexity,
        "spec_key_count": n,
        "task_types": task_types,
        "description": description,
        "design_considerations": design_considerations,
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# Template families
# ---------------------------------------------------------------------------


def bar_by_dim(cube, dim):
    """Bar chart of the measure by a single dimension (its L1 marginal)."""
    dtype = enc_type(cube, dim)
    horizontal = dtype == "nominal" and cardinality(cube, dim) > 4
    dim_enc = {"encoding": "y" if horizontal else "x", "field": dim, "type": dtype}
    meas_enc = {
        "encoding": "x" if horizontal else "y",
        "field": cube["measure"],
        "type": "quantitative",
    }
    spec = {
        "source": source(cube),
        "transformation": [{"filter": marginal_filter(cube, [dim])}],
        "representation": {"mark": "bar", "mapping": [dim_enc, meas_enc]},
    }
    orient = "horizontal" if horizontal else "vertical"
    return make_row(
        query_templates=[
            f"How many encounters are there by {display(dim)}?",
            f"Make a bar chart of {display(cube['measure'])} by {display(dim)}.",
        ],
        spec=spec,
        chart_type="barchart",
        task_types=["Compute_Derived_Value", "Determine_Range"],
        description=(
            f"Shows the pre-aggregated {display(cube['measure'])} for each "
            f"{display(dim)} as a {orient} bar chart."
        ),
        design_considerations=(
            f"Reads the cube's {display(dim)} marginal by filtering to rows where "
            f"{dim} is present and all other dimensions are empty; the measure is "
            f"mapped directly with no re-aggregation. {orient.capitalize()} "
            f"orientation chosen from category count."
        ),
        tasks="Compare counts across categories; identify the most or least common category.",
    )


def line_over_time(cube, dim):
    """Line chart of the measure over a temporal dimension (its L1 marginal)."""
    spec = {
        "source": source(cube),
        "transformation": [
            {"filter": marginal_filter(cube, [dim])},
            {"orderby": {"field": dim, "order": "asc"}},
        ],
        "representation": {
            "mark": "line",
            "mapping": [
                {"encoding": "x", "field": dim, "type": enc_type(cube, dim)},
                {"encoding": "y", "field": cube["measure"], "type": "quantitative"},
            ],
        },
    }
    return make_row(
        query_templates=[
            f"How does the {display(cube['measure'])} change over {display(dim)}?",
            f"Make a line chart of {display(cube['measure'])} over {display(dim)}.",
        ],
        spec=spec,
        chart_type="line",
        task_types=["Characterize_Distribution", "Determine_Range"],
        description=(
            f"Shows the pre-aggregated {display(cube['measure'])} over "
            f"{display(dim)} as a line chart."
        ),
        design_considerations=(
            f"Filters to the {display(dim)} marginal (all other dimensions empty), "
            f"orders the axis ascending, and maps the measure directly. The month is "
            f"encoded as an ordered (ordinal) axis."
        ),
        tasks="Identify trends over time; spot peaks, troughs, and seasonality.",
    )


def circular_by_dim(cube, dim, donut=False):
    """Pie or donut of the measure by a single dimension (its L1 marginal)."""
    mapping = [
        {"encoding": "theta", "field": cube["measure"], "type": "quantitative"},
        {"encoding": "color", "field": dim, "type": "nominal"},
    ]
    rep = {"mark": "arc", "mapping": mapping}
    if donut:
        mapping.append({"encoding": "radius", "value": 60})
        mapping.append({"encoding": "radius2", "value": 80})
    kind = "donut" if donut else "pie"
    spec = {
        "source": source(cube),
        "transformation": [{"filter": marginal_filter(cube, [dim])}],
        "representation": rep,
    }
    return make_row(
        query_templates=[f"Make a {kind} chart of {display(cube['measure'])} by {display(dim)}."],
        spec=spec,
        chart_type="circular",
        task_types=["Compute_Derived_Value", "Determine_Range"],
        description=(
            f"Shows the proportional {display(cube['measure'])} for each "
            f"{display(dim)} as a {kind} chart."
        ),
        design_considerations=(
            f"Filters to the {display(dim)} marginal and maps the measure to angle; "
            f"the renderer normalizes each slice against the total. Suitable for few "
            f"categories."
        ),
        tasks="Assess part-to-whole proportions; identify the dominant category.",
    )


def stacked_bar(cube, axis, sub):
    """Vertical stacked bar of the measure by two dimensions (their L2 marginal)."""
    spec = {
        "source": source(cube),
        "transformation": [{"filter": marginal_filter(cube, [axis, sub])}],
        "representation": {
            "mark": "bar",
            "mapping": [
                {"encoding": "x", "field": axis, "type": "nominal"},
                {"encoding": "y", "field": cube["measure"], "type": "quantitative"},
                {"encoding": "color", "field": sub, "type": "nominal"},
            ],
        },
    }
    return make_row(
        query_templates=[
            f"How many encounters are there by {display(axis)} and {display(sub)}?",
            f"Make a stacked bar chart of {display(axis)} and {display(sub)}.",
        ],
        spec=spec,
        chart_type="stacked_bar",
        task_types=["Compute_Derived_Value"],
        description=(
            f"Shows the pre-aggregated {display(cube['measure'])} by {display(axis)} "
            f"and {display(sub)} as a vertical stacked bar chart."
        ),
        design_considerations=(
            f"Filters to the two-dimension marginal ({axis} and {sub} present, all "
            f"other dimensions empty) and maps the measure directly. Color encodes the "
            f"sub-group; prefer the field with fewer categories for color."
        ),
        tasks="Compare group compositions across categories; identify dominant sub-groups.",
    )


def grouped_bar(cube, axis, sub):
    """Grouped (side-by-side) bar of the measure by two dimensions (L2 marginal)."""
    spec = {
        "source": source(cube),
        "transformation": [{"filter": marginal_filter(cube, [axis, sub])}],
        "representation": {
            "mark": "bar",
            "mapping": [
                {"encoding": "x", "field": axis, "type": "nominal"},
                {"encoding": "y", "field": cube["measure"], "type": "quantitative"},
                {"encoding": "xOffset", "field": sub, "type": "nominal"},
                {"encoding": "color", "field": sub, "type": "nominal"},
            ],
        },
    }
    return make_row(
        query_templates=[f"What is the {display(cube['measure'])} of {display(sub)} for each {display(axis)}?"],
        spec=spec,
        chart_type="stacked_bar",
        task_types=["Compute_Derived_Value"],
        description=(
            f"Shows the pre-aggregated {display(cube['measure'])} by {display(axis)} "
            f"and {display(sub)} as a grouped (side-by-side) bar chart."
        ),
        design_considerations=(
            f"Filters to the two-dimension marginal and uses xOffset for side-by-side "
            f"grouping, enabling direct comparison of {display(sub)} within each "
            f"{display(axis)}."
        ),
        tasks="Directly compare sub-group counts within and across categories.",
    )


def normalized_bar(cube, axis, sub):
    """Proportional (normalized) stacked bar: marginal filter first, then proportion."""
    measure = cube["measure"]
    spec = {
        "source": source(cube),
        "transformation": [
            {"filter": marginal_filter(cube, [axis, sub])},
            {"groupby": axis, "out": "groupTotals"},
            {"rollup": {"axis_total": {"op": "sum", "field": measure}}},
            {"groupby": [sub, axis], "in": "<E>"},
            {"rollup": {"cell_total": {"op": "sum", "field": measure}}},
            {"join": {"on": axis}, "in": ["<E>", "groupTotals"], "out": "datasets"},
            {"derive": {"proportion": "d['cell_total'] / d['axis_total']"}},
        ],
        "representation": {
            "mark": "bar",
            "mapping": [
                {"encoding": "x", "field": axis, "type": "nominal"},
                {"encoding": "y", "field": "proportion", "type": "quantitative"},
                {"encoding": "color", "field": sub, "type": "nominal"},
            ],
        },
    }
    return make_row(
        query_templates=[f"What is the proportion of {display(sub)} for each {display(axis)}?"],
        spec=spec,
        chart_type="stacked_bar",
        task_types=["Compute_Derived_Value"],
        description=(
            f"Shows the relative proportion of {display(sub)} within each "
            f"{display(axis)} as a normalized stacked bar chart."
        ),
        design_considerations=(
            f"First filters to the two-dimension marginal, then sums the measure per "
            f"{display(axis)} and divides each cell by its group total to obtain "
            f"proportions. Color is preferably the field with fewer categories."
        ),
        tasks="Compare relative proportions across categories; identify dominant sub-groups.",
    )


def heatmap(cube, xdim, ydim):
    """Heatmap of the measure across two dimensions (their L2 marginal)."""
    measure = cube["measure"]
    spec = {
        "source": source(cube),
        "transformation": [
            {"filter": marginal_filter(cube, [xdim, ydim])},
            {"derive": {"udi_internal_percentile": f"d['{measure}'] / max(d['{measure}'])"}},
            {"derive": {
                "udi_internal_text_color_threshold":
                    "d.udi_internal_percentile > .5 ? 'large' : 'small'"
            }},
        ],
        "representation": [
            {"mark": "rect", "mapping": [
                {"encoding": "color", "field": measure, "type": "quantitative"},
                {"encoding": "y", "field": ydim, "type": "nominal"},
                {"encoding": "x", "field": xdim, "type": "nominal"},
            ]},
            {"mark": "text", "mapping": [
                {"encoding": "text", "field": measure, "type": "quantitative"},
                {"encoding": "y", "field": ydim, "type": "nominal"},
                {"encoding": "x", "field": xdim, "type": "nominal"},
                {"encoding": "color", "field": "udi_internal_text_color_threshold",
                 "type": "nominal", "domain": ["large", "small"],
                 "range": ["white", "black"], "omitLegend": True},
            ]},
        ],
    }
    return make_row(
        query_templates=[
            f"Are there clusters in {display(cube['measure'])} across {display(xdim)} and {display(ydim)}?",
            f"Make a heatmap of {display(xdim)} and {display(ydim)}.",
        ],
        spec=spec,
        chart_type="heatmap",
        task_types=["Cluster", "Compute_Derived_Value", "Correlate"],
        description=(
            f"Shows the pre-aggregated {display(cube['measure'])} for each combination "
            f"of {display(xdim)} and {display(ydim)} as a labeled heatmap."
        ),
        design_considerations=(
            f"Filters to the two-dimension marginal and maps the measure to cell color; "
            f"overlaid text shows exact values with contrast-aware color. The field with "
            f"more categories is preferably on the y-axis."
        ),
        tasks="Identify clusters or patterns across two dimensions; compare counts across combinations.",
    )


def total_table(cube):
    """Single-row table with the grand total (the all-empty L0 marginal)."""
    spec = {
        "source": source(cube),
        "transformation": [{"filter": marginal_filter(cube, [])}],
        "representation": {
            "mark": "row",
            "mapping": [{"encoding": "text", "field": cube["measure"],
                         "mark": "text", "type": "nominal"}],
        },
    }
    return make_row(
        query_templates=[
            "How many encounters are there in total?",
            f"What is the total {display(cube['measure'])}?",
        ],
        spec=spec,
        chart_type="table",
        task_types=["Retrieve_Value", "Compute_Derived_Value"],
        description=f"Shows the grand-total {display(cube['measure'])} as a single-row table.",
        design_considerations=(
            "Reads the grand-total row directly by filtering to the marginal where every "
            "dimension is empty; no aggregation is performed."
        ),
        tasks="Retrieve the overall total.",
    )


def count_table_by_dim(cube, dim):
    """Table listing each category with its measure and an in-cell bar."""
    spec = {
        "source": source(cube),
        "transformation": [
            {"filter": marginal_filter(cube, [dim])},
            {"orderby": {"field": cube["measure"], "order": "desc"}},
        ],
        "representation": {
            "mark": "row",
            "mapping": [
                {"encoding": "text", "field": dim, "mark": "text", "type": "nominal"},
                {"encoding": "x", "field": cube["measure"], "mark": "bar",
                 "type": "quantitative", "range": {"min": 0.1, "max": 1}},
            ],
        },
    }
    return make_row(
        query_templates=[
            f"List the {display(cube['measure'])} for each {display(dim)}.",
            f"What is the range of {display(dim)} values?",
        ],
        spec=spec,
        chart_type="table",
        task_types=["Determine_Range", "Sort", "Retrieve_Value"],
        description=(
            f"Lists each {display(dim)} with its pre-aggregated {display(cube['measure'])} "
            f"as a sorted table with in-cell bars."
        ),
        design_considerations=(
            f"Filters to the {display(dim)} marginal, orders by the measure, and adds "
            f"in-cell bars for visual comparison."
        ),
        tasks="Determine the distinct values of a dimension; compare category counts.",
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def generate(cube):
    nominal_dims = [d for d in cube["dims"] if data_type(cube, d) in ("nominal", "ordinal")]
    temporal_dims = [d for d in cube["dims"] if data_type(cube, d) == "temporal"]
    quant_dims = [d for d in cube["dims"] if data_type(cube, d) == "quantitative"]

    rows = []

    # Single-dimension bar charts (nominal + quantitative dims).
    for d in nominal_dims + quant_dims:
        rows.append(bar_by_dim(cube, d))

    # Line charts over temporal dimensions.
    for d in temporal_dims:
        rows.append(line_over_time(cube, d))

    # Pie + donut for a representative low-cardinality nominal dimension.
    if nominal_dims:
        rep = nominal_dims[0]
        rows.append(circular_by_dim(cube, rep, donut=False))
        rows.append(circular_by_dim(cube, rep, donut=True))

    # Two-dimension charts across representative nominal pairs.
    pairs = []
    for i in range(len(nominal_dims)):
        for j in range(len(nominal_dims)):
            if i == j:
                continue
            pairs.append((nominal_dims[i], nominal_dims[j]))
    # Keep a representative, bounded set: each axis dim paired with the next one.
    rep_pairs = []
    for i in range(len(nominal_dims) - 1):
        rep_pairs.append((nominal_dims[i], nominal_dims[i + 1]))
    for axis, sub in rep_pairs:
        rows.append(stacked_bar(cube, axis, sub))
        rows.append(grouped_bar(cube, axis, sub))
        rows.append(normalized_bar(cube, axis, sub))
        rows.append(heatmap(cube, axis, sub))

    # Tables.
    rows.append(total_table(cube))
    for d in nominal_dims:
        rows.append(count_table_by_dim(cube, d))

    return rows


def validate_specs(rows, grammar_path):
    schema = json.loads(Path(grammar_path).read_text())
    for i, row in enumerate(rows):
        spec = json.loads(row["spec_template"])
        try:
            jsonschema.validate(instance=spec, schema=schema)
        except jsonschema.ValidationError as e:
            raise SystemExit(
                f"Spec {i} ({row['chart_type']}) failed grammar validation: {e.message}"
            )


def main():
    parser = argparse.ArgumentParser(description="Generate data-cube visualization templates.")
    parser.add_argument("--schema", default=str(_DEFAULT_SCHEMA), help="Path to the cube UDI schema JSON.")
    parser.add_argument("-o", "--output", default=str(_DEFAULT_OUT), help="Output template JSON path.")
    parser.add_argument("--grammar", default=str(_GRAMMAR), help="Path to UDIGrammarSchema.json.")
    args = parser.parse_args()

    cube = load_cube(args.schema)
    rows = generate(cube)
    validate_specs(rows, args.grammar)

    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")

    from collections import Counter
    print(f"Generated {len(rows)} data-cube visualization templates.")
    print("Chart types:", dict(Counter(r["chart_type"] for r in rows)))
    print(f"Validated against grammar: {args.grammar}")
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
