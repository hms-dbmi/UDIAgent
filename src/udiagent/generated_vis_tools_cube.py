"""
Auto-generated visualization tool definitions.

Generated from: src/udiagent/data/skills/template_visualizations_cube.json
Schema: data/data_domains/encounter_cube_schema.json
Tools: 25

DO NOT EDIT — regenerate with: python src/generate_tools.py
"""


# Schema metadata (entity URLs and relationships)
SCHEMA = {'entities': {'encounter_counts': {'fields': {'age_at_visit': {'cardinality': 97, 'type': 'quantitative'},
                                              'class_display': {'cardinality': 5, 'type': 'nominal'},
                                              'cnt': {'cardinality': 1327, 'type': 'quantitative'},
                                              'ethnicity_display': {'cardinality': 2, 'type': 'nominal'},
                                              'gender': {'cardinality': 2, 'type': 'nominal'},
                                              'period_start_month': {'cardinality': 881, 'type': 'temporal'},
                                              'race_display': {'cardinality': 5, 'type': 'nominal'}},
                                   'url': './data/example_data_cube/core__count_encounter_month.csv'}},
 'relationships': []}


# Spec template strings (indexed by position)
TEMPLATES = ['{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] != null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "y", "field": '
 '"class_display", "type": "nominal"}, {"encoding": "x", "field": "cnt", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] != null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"gender", "type": "nominal"}, {"encoding": "y", "field": "cnt", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] != null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "y", "field": '
 '"race_display", "type": "nominal"}, {"encoding": "x", "field": "cnt", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] != null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"ethnicity_display", "type": "nominal"}, {"encoding": "y", "field": "cnt", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] != null && d['gender'] == null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"age_at_visit", "type": "quantitative"}, {"encoding": "y", "field": "cnt", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] != null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}, {"orderby": {"field": "period_start_month", "order": "asc"}}], "representation": '
 '{"mark": "line", "mapping": [{"encoding": "x", "field": "period_start_month", "type": "ordinal"}, {"encoding": "y", '
 '"field": "cnt", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] != null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "arc", "mapping": [{"encoding": "theta", "field": '
 '"cnt", "type": "quantitative"}, {"encoding": "color", "field": "class_display", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] != null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "arc", "mapping": [{"encoding": "theta", "field": '
 '"cnt", "type": "quantitative"}, {"encoding": "color", "field": "class_display", "type": "nominal"}, {"encoding": '
 '"radius", "value": 60}, {"encoding": "radius2", "value": 80}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] != null && d['age_at_visit'] == null && d['gender'] != null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"class_display", "type": "nominal"}, {"encoding": "y", "field": "cnt", "type": "quantitative"}, {"encoding": '
 '"color", "field": "gender", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] != null && d['age_at_visit'] == null && d['gender'] != null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"class_display", "type": "nominal"}, {"encoding": "y", "field": "cnt", "type": "quantitative"}, {"encoding": '
 '"xOffset", "field": "gender", "type": "nominal"}, {"encoding": "color", "field": "gender", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] != null && d['age_at_visit'] == null && d['gender'] != null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}, {"groupby": "class_display", "out": "groupTotals"}, {"rollup": {"axis_total": '
 '{"op": "sum", "field": "cnt"}}}, {"groupby": ["gender", "class_display"], "in": "<E>"}, {"rollup": {"cell_total": '
 '{"op": "sum", "field": "cnt"}}}, {"join": {"on": "class_display"}, "in": ["<E>", "groupTotals"], "out": "datasets"}, '
 '{"derive": {"proportion": "d[\'cell_total\'] / d[\'axis_total\']"}}], "representation": {"mark": "bar", "mapping": '
 '[{"encoding": "x", "field": "class_display", "type": "nominal"}, {"encoding": "y", "field": "proportion", "type": '
 '"quantitative"}, {"encoding": "color", "field": "gender", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] != null && d['age_at_visit'] == null && d['gender'] != null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}, {"derive": {"udi_internal_percentile": "d[\'cnt\'] / max(d[\'cnt\'])"}}, '
 '{"derive": {"udi_internal_text_color_threshold": "d.udi_internal_percentile > .5 ? \'large\' : \'small\'"}}], '
 '"representation": [{"mark": "rect", "mapping": [{"encoding": "color", "field": "cnt", "type": "quantitative"}, '
 '{"encoding": "y", "field": "gender", "type": "nominal"}, {"encoding": "x", "field": "class_display", "type": '
 '"nominal"}]}, {"mark": "text", "mapping": [{"encoding": "text", "field": "cnt", "type": "quantitative"}, '
 '{"encoding": "y", "field": "gender", "type": "nominal"}, {"encoding": "x", "field": "class_display", "type": '
 '"nominal"}, {"encoding": "color", "field": "udi_internal_text_color_threshold", "type": "nominal", "domain": '
 '["large", "small"], "range": ["white", "black"], "omitLegend": true}]}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] != null && d['race_display'] != null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"gender", "type": "nominal"}, {"encoding": "y", "field": "cnt", "type": "quantitative"}, {"encoding": "color", '
 '"field": "race_display", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] != null && d['race_display'] != null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"gender", "type": "nominal"}, {"encoding": "y", "field": "cnt", "type": "quantitative"}, {"encoding": "xOffset", '
 '"field": "race_display", "type": "nominal"}, {"encoding": "color", "field": "race_display", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] != null && d['race_display'] != null && "
 'd[\'ethnicity_display\'] == null"}, {"groupby": "gender", "out": "groupTotals"}, {"rollup": {"axis_total": {"op": '
 '"sum", "field": "cnt"}}}, {"groupby": ["race_display", "gender"], "in": "<E>"}, {"rollup": {"cell_total": {"op": '
 '"sum", "field": "cnt"}}}, {"join": {"on": "gender"}, "in": ["<E>", "groupTotals"], "out": "datasets"}, {"derive": '
 '{"proportion": "d[\'cell_total\'] / d[\'axis_total\']"}}], "representation": {"mark": "bar", "mapping": '
 '[{"encoding": "x", "field": "gender", "type": "nominal"}, {"encoding": "y", "field": "proportion", "type": '
 '"quantitative"}, {"encoding": "color", "field": "race_display", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] != null && d['race_display'] != null && "
 'd[\'ethnicity_display\'] == null"}, {"derive": {"udi_internal_percentile": "d[\'cnt\'] / max(d[\'cnt\'])"}}, '
 '{"derive": {"udi_internal_text_color_threshold": "d.udi_internal_percentile > .5 ? \'large\' : \'small\'"}}], '
 '"representation": [{"mark": "rect", "mapping": [{"encoding": "color", "field": "cnt", "type": "quantitative"}, '
 '{"encoding": "y", "field": "race_display", "type": "nominal"}, {"encoding": "x", "field": "gender", "type": '
 '"nominal"}]}, {"mark": "text", "mapping": [{"encoding": "text", "field": "cnt", "type": "quantitative"}, '
 '{"encoding": "y", "field": "race_display", "type": "nominal"}, {"encoding": "x", "field": "gender", "type": '
 '"nominal"}, {"encoding": "color", "field": "udi_internal_text_color_threshold", "type": "nominal", "domain": '
 '["large", "small"], "range": ["white", "black"], "omitLegend": true}]}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] != null && "
 'd[\'ethnicity_display\'] != null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"race_display", "type": "nominal"}, {"encoding": "y", "field": "cnt", "type": "quantitative"}, {"encoding": "color", '
 '"field": "ethnicity_display", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] != null && "
 'd[\'ethnicity_display\'] != null"}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"race_display", "type": "nominal"}, {"encoding": "y", "field": "cnt", "type": "quantitative"}, {"encoding": '
 '"xOffset", "field": "ethnicity_display", "type": "nominal"}, {"encoding": "color", "field": "ethnicity_display", '
 '"type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] != null && "
 'd[\'ethnicity_display\'] != null"}, {"groupby": "race_display", "out": "groupTotals"}, {"rollup": {"axis_total": '
 '{"op": "sum", "field": "cnt"}}}, {"groupby": ["ethnicity_display", "race_display"], "in": "<E>"}, {"rollup": '
 '{"cell_total": {"op": "sum", "field": "cnt"}}}, {"join": {"on": "race_display"}, "in": ["<E>", "groupTotals"], '
 '"out": "datasets"}, {"derive": {"proportion": "d[\'cell_total\'] / d[\'axis_total\']"}}], "representation": {"mark": '
 '"bar", "mapping": [{"encoding": "x", "field": "race_display", "type": "nominal"}, {"encoding": "y", "field": '
 '"proportion", "type": "quantitative"}, {"encoding": "color", "field": "ethnicity_display", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] != null && "
 'd[\'ethnicity_display\'] != null"}, {"derive": {"udi_internal_percentile": "d[\'cnt\'] / max(d[\'cnt\'])"}}, '
 '{"derive": {"udi_internal_text_color_threshold": "d.udi_internal_percentile > .5 ? \'large\' : \'small\'"}}], '
 '"representation": [{"mark": "rect", "mapping": [{"encoding": "color", "field": "cnt", "type": "quantitative"}, '
 '{"encoding": "y", "field": "ethnicity_display", "type": "nominal"}, {"encoding": "x", "field": "race_display", '
 '"type": "nominal"}]}, {"mark": "text", "mapping": [{"encoding": "text", "field": "cnt", "type": "quantitative"}, '
 '{"encoding": "y", "field": "ethnicity_display", "type": "nominal"}, {"encoding": "x", "field": "race_display", '
 '"type": "nominal"}, {"encoding": "color", "field": "udi_internal_text_color_threshold", "type": "nominal", "domain": '
 '["large", "small"], "range": ["white", "black"], "omitLegend": true}]}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}], "representation": {"mark": "row", "mapping": [{"encoding": "text", "field": '
 '"cnt", "mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] != null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}, {"orderby": {"field": "cnt", "order": "desc"}}], "representation": {"mark": '
 '"row", "mapping": [{"encoding": "text", "field": "class_display", "mark": "text", "type": "nominal"}, {"encoding": '
 '"x", "field": "cnt", "mark": "bar", "type": "quantitative", "range": {"min": 0.1, "max": 1}}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] != null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] == null"}, {"orderby": {"field": "cnt", "order": "desc"}}], "representation": {"mark": '
 '"row", "mapping": [{"encoding": "text", "field": "gender", "mark": "text", "type": "nominal"}, {"encoding": "x", '
 '"field": "cnt", "mark": "bar", "type": "quantitative", "range": {"min": 0.1, "max": 1}}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] != null && "
 'd[\'ethnicity_display\'] == null"}, {"orderby": {"field": "cnt", "order": "desc"}}], "representation": {"mark": '
 '"row", "mapping": [{"encoding": "text", "field": "race_display", "mark": "text", "type": "nominal"}, {"encoding": '
 '"x", "field": "cnt", "mark": "bar", "type": "quantitative", "range": {"min": 0.1, "max": 1}}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'period_start_month\'] == null && '
 "d['class_display'] == null && d['age_at_visit'] == null && d['gender'] == null && d['race_display'] == null && "
 'd[\'ethnicity_display\'] != null"}, {"orderby": {"field": "cnt", "order": "desc"}}], "representation": {"mark": '
 '"row", "mapping": [{"encoding": "text", "field": "ethnicity_display", "mark": "text", "type": "nominal"}, '
 '{"encoding": "x", "field": "cnt", "mark": "bar", "type": "quantitative", "range": {"min": 0.1, "max": 1}}]}}']


# OpenAI function-calling tool definitions
TOOL_DEFS = [{'function': {'description': '[barchart] Shows the pre-aggregated encounter count for each encounter class as a '
                              "horizontal bar chart. Design: Reads the cube's encounter class marginal by filtering to "
                              'rows where class_display is present and all other dimensions are empty; the measure is '
                              'mapped directly with no re-aggregation. Horizontal orientation chosen from category '
                              'count. Tasks: Compare counts across categories; identify the most or least common '
                              'category. Query patterns: How many encounters are there by encounter class?; Make a bar '
                              'chart of encounter count by encounter class.',
               'name': 'vis_000_barchart_count_horiz',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Shows the pre-aggregated encounter count for each gender as a vertical bar '
                              "chart. Design: Reads the cube's gender marginal by filtering to rows where gender is "
                              'present and all other dimensions are empty; the measure is mapped directly with no '
                              're-aggregation. Vertical orientation chosen from category count. Tasks: Compare counts '
                              'across categories; identify the most or least common category. Query patterns: How many '
                              'encounters are there by gender?; Make a bar chart of encounter count by gender.',
               'name': 'vis_001_barchart_count_vert',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Shows the pre-aggregated encounter count for each race as a horizontal bar '
                              "chart. Design: Reads the cube's race marginal by filtering to rows where race_display "
                              'is present and all other dimensions are empty; the measure is mapped directly with no '
                              're-aggregation. Horizontal orientation chosen from category count. Tasks: Compare '
                              'counts across categories; identify the most or least common category. Query patterns: '
                              'How many encounters are there by race?; Make a bar chart of encounter count by race.',
               'name': 'vis_002_barchart_count_horiz',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Shows the pre-aggregated encounter count for each ethnicity as a vertical '
                              "bar chart. Design: Reads the cube's ethnicity marginal by filtering to rows where "
                              'ethnicity_display is present and all other dimensions are empty; the measure is mapped '
                              'directly with no re-aggregation. Vertical orientation chosen from category count. '
                              'Tasks: Compare counts across categories; identify the most or least common category. '
                              'Query patterns: How many encounters are there by ethnicity?; Make a bar chart of '
                              'encounter count by ethnicity.',
               'name': 'vis_003_barchart_count_vert',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[barchart] Shows the pre-aggregated encounter count for each age as a vertical bar '
                              "chart. Design: Reads the cube's age marginal by filtering to rows where age_at_visit is "
                              'present and all other dimensions are empty; the measure is mapped directly with no '
                              're-aggregation. Vertical orientation chosen from category count. Tasks: Compare counts '
                              'across categories; identify the most or least common category. Query patterns: How many '
                              'encounters are there by age?; Make a bar chart of encounter count by age.',
               'name': 'vis_004_barchart_count_vert',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[line] Shows the pre-aggregated encounter count over month as a line chart. Design: '
                              'Filters to the month marginal (all other dimensions empty), orders the axis ascending, '
                              'and maps the measure directly. The month is encoded as an ordered (ordinal) axis. '
                              'Tasks: Identify trends over time; spot peaks, troughs, and seasonality. Query patterns: '
                              'How does the encounter count change over month?; Make a line chart of encounter count '
                              'over month.',
               'name': 'vis_005_line_count',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[circular] Shows the proportional encounter count for each encounter class as a pie '
                              'chart. Design: Filters to the encounter class marginal and maps the measure to angle; '
                              'the renderer normalizes each slice against the total. Suitable for few categories. '
                              'Tasks: Assess part-to-whole proportions; identify the dominant category. Query '
                              'patterns: Make a pie chart of encounter count by encounter class.',
               'name': 'vis_006_circular_count',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[circular] Shows the proportional encounter count for each encounter class as a donut '
                              'chart. Design: Filters to the encounter class marginal and maps the measure to angle; '
                              'the renderer normalizes each slice against the total. Suitable for few categories. '
                              'Tasks: Assess part-to-whole proportions; identify the dominant category. Query '
                              'patterns: Make a donut chart of encounter count by encounter class.',
               'name': 'vis_007_circular_count',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the pre-aggregated encounter count by encounter class and gender as '
                              'a vertical stacked bar chart. Design: Filters to the two-dimension marginal '
                              '(class_display and gender present, all other dimensions empty) and maps the measure '
                              'directly. Color encodes the sub-group; prefer the field with fewer categories for '
                              'color. Tasks: Compare group compositions across categories; identify dominant '
                              'sub-groups. Query patterns: How many encounters are there by encounter class and '
                              'gender?; Make a stacked bar chart of encounter class and gender.',
               'name': 'vis_008_stacked_bar_count_vert_stacked',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the pre-aggregated encounter count by encounter class and gender as '
                              'a grouped (side-by-side) bar chart. Design: Filters to the two-dimension marginal and '
                              'uses xOffset for side-by-side grouping, enabling direct comparison of gender within '
                              'each encounter class. Tasks: Directly compare sub-group counts within and across '
                              'categories. Query patterns: What is the encounter count of gender for each encounter '
                              'class?',
               'name': 'vis_009_stacked_bar_count_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the relative proportion of gender within each encounter class as a '
                              'normalized stacked bar chart. Design: First filters to the two-dimension marginal, then '
                              'sums the measure per encounter class and divides each cell by its group total to obtain '
                              'proportions. Color is preferably the field with fewer categories. Tasks: Compare '
                              'relative proportions across categories; identify dominant sub-groups. Query patterns: '
                              'What is the proportion of gender for each encounter class?',
               'name': 'vis_010_stacked_bar_count_stacked_normalized',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[heatmap] Shows the pre-aggregated encounter count for each combination of encounter '
                              'class and gender as a labeled heatmap. Design: Filters to the two-dimension marginal '
                              'and maps the measure to cell color; overlaid text shows exact values with '
                              'contrast-aware color. The field with more categories is preferably on the y-axis. '
                              'Tasks: Identify clusters or patterns across two dimensions; compare counts across '
                              'combinations. Query patterns: Are there clusters in encounter count across encounter '
                              'class and gender?; Make a heatmap of encounter class and gender.',
               'name': 'vis_011_heatmap_count',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the pre-aggregated encounter count by gender and race as a vertical '
                              'stacked bar chart. Design: Filters to the two-dimension marginal (gender and '
                              'race_display present, all other dimensions empty) and maps the measure directly. Color '
                              'encodes the sub-group; prefer the field with fewer categories for color. Tasks: Compare '
                              'group compositions across categories; identify dominant sub-groups. Query patterns: How '
                              'many encounters are there by gender and race?; Make a stacked bar chart of gender and '
                              'race.',
               'name': 'vis_012_stacked_bar_count_vert_stacked',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the pre-aggregated encounter count by gender and race as a grouped '
                              '(side-by-side) bar chart. Design: Filters to the two-dimension marginal and uses '
                              'xOffset for side-by-side grouping, enabling direct comparison of race within each '
                              'gender. Tasks: Directly compare sub-group counts within and across categories. Query '
                              'patterns: What is the encounter count of race for each gender?',
               'name': 'vis_013_stacked_bar_count_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the relative proportion of race within each gender as a normalized '
                              'stacked bar chart. Design: First filters to the two-dimension marginal, then sums the '
                              'measure per gender and divides each cell by its group total to obtain proportions. '
                              'Color is preferably the field with fewer categories. Tasks: Compare relative '
                              'proportions across categories; identify dominant sub-groups. Query patterns: What is '
                              'the proportion of race for each gender?',
               'name': 'vis_014_stacked_bar_proportion_stacked_normalized',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[heatmap] Shows the pre-aggregated encounter count for each combination of gender and '
                              'race as a labeled heatmap. Design: Filters to the two-dimension marginal and maps the '
                              'measure to cell color; overlaid text shows exact values with contrast-aware color. The '
                              'field with more categories is preferably on the y-axis. Tasks: Identify clusters or '
                              'patterns across two dimensions; compare counts across combinations. Query patterns: Are '
                              'there clusters in encounter count across gender and race?; Make a heatmap of gender and '
                              'race.',
               'name': 'vis_015_heatmap_count',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the pre-aggregated encounter count by race and ethnicity as a '
                              'vertical stacked bar chart. Design: Filters to the two-dimension marginal (race_display '
                              'and ethnicity_display present, all other dimensions empty) and maps the measure '
                              'directly. Color encodes the sub-group; prefer the field with fewer categories for '
                              'color. Tasks: Compare group compositions across categories; identify dominant '
                              'sub-groups. Query patterns: How many encounters are there by race and ethnicity?; Make '
                              'a stacked bar chart of race and ethnicity.',
               'name': 'vis_016_stacked_bar_count_vert_stacked',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the pre-aggregated encounter count by race and ethnicity as a '
                              'grouped (side-by-side) bar chart. Design: Filters to the two-dimension marginal and '
                              'uses xOffset for side-by-side grouping, enabling direct comparison of ethnicity within '
                              'each race. Tasks: Directly compare sub-group counts within and across categories. Query '
                              'patterns: What is the encounter count of ethnicity for each race?',
               'name': 'vis_017_stacked_bar_count_grouped',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[stacked_bar] Shows the relative proportion of ethnicity within each race as a '
                              'normalized stacked bar chart. Design: First filters to the two-dimension marginal, then '
                              'sums the measure per race and divides each cell by its group total to obtain '
                              'proportions. Color is preferably the field with fewer categories. Tasks: Compare '
                              'relative proportions across categories; identify dominant sub-groups. Query patterns: '
                              'What is the proportion of ethnicity for each race?',
               'name': 'vis_018_stacked_bar_proportion_stacked_normalized',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[heatmap] Shows the pre-aggregated encounter count for each combination of race and '
                              'ethnicity as a labeled heatmap. Design: Filters to the two-dimension marginal and maps '
                              'the measure to cell color; overlaid text shows exact values with contrast-aware color. '
                              'The field with more categories is preferably on the y-axis. Tasks: Identify clusters or '
                              'patterns across two dimensions; compare counts across combinations. Query patterns: Are '
                              'there clusters in encounter count across race and ethnicity?; Make a heatmap of race '
                              'and ethnicity.',
               'name': 'vis_019_heatmap_count',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Shows the grand-total encounter count as a single-row table. Design: Reads the '
                              'grand-total row directly by filtering to the marginal where every dimension is empty; '
                              'no aggregation is performed. Tasks: Retrieve the overall total. Query patterns: How '
                              'many encounters are there in total?; What is the total encounter count?',
               'name': 'vis_020_table_count',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Lists each encounter class with its pre-aggregated encounter count as a sorted '
                              'table with in-cell bars. Design: Filters to the encounter class marginal, orders by the '
                              'measure, and adds in-cell bars for visual comparison. Tasks: Determine the distinct '
                              'values of a dimension; compare category counts. Query patterns: List the encounter '
                              'count for each encounter class.; What is the range of encounter class values?',
               'name': 'vis_021_table_count_sorted',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Lists each gender with its pre-aggregated encounter count as a sorted table '
                              'with in-cell bars. Design: Filters to the gender marginal, orders by the measure, and '
                              'adds in-cell bars for visual comparison. Tasks: Determine the distinct values of a '
                              'dimension; compare category counts. Query patterns: List the encounter count for each '
                              'gender.; What is the range of gender values?',
               'name': 'vis_022_table_count_sorted',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Lists each race with its pre-aggregated encounter count as a sorted table with '
                              'in-cell bars. Design: Filters to the race marginal, orders by the measure, and adds '
                              'in-cell bars for visual comparison. Tasks: Determine the distinct values of a '
                              'dimension; compare category counts. Query patterns: List the encounter count for each '
                              'race.; What is the range of race values?',
               'name': 'vis_023_table_count_sorted',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': '[table] Lists each ethnicity with its pre-aggregated encounter count as a sorted table '
                              'with in-cell bars. Design: Filters to the ethnicity marginal, orders by the measure, '
                              'and adds in-cell bars for visual comparison. Tasks: Determine the distinct values of a '
                              'dimension; compare category counts. Query patterns: List the encounter count for each '
                              'ethnicity.; What is the range of ethnicity values?',
               'name': 'vis_024_table_count_sorted',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'}]


# Dispatch: tool name -> (template_index, param_to_binding_map)
TOOL_DISPATCH = {'vis_000_barchart_count_horiz': (0, {'entity': 'E'}),
 'vis_001_barchart_count_vert': (1, {'entity': 'E'}),
 'vis_002_barchart_count_horiz': (2, {'entity': 'E'}),
 'vis_003_barchart_count_vert': (3, {'entity': 'E'}),
 'vis_004_barchart_count_vert': (4, {'entity': 'E'}),
 'vis_005_line_count': (5, {'entity': 'E'}),
 'vis_006_circular_count': (6, {'entity': 'E'}),
 'vis_007_circular_count': (7, {'entity': 'E'}),
 'vis_008_stacked_bar_count_vert_stacked': (8, {'entity': 'E'}),
 'vis_009_stacked_bar_count_grouped': (9, {'entity': 'E'}),
 'vis_010_stacked_bar_count_stacked_normalized': (10, {'entity': 'E'}),
 'vis_011_heatmap_count': (11, {'entity': 'E'}),
 'vis_012_stacked_bar_count_vert_stacked': (12, {'entity': 'E'}),
 'vis_013_stacked_bar_count_grouped': (13, {'entity': 'E'}),
 'vis_014_stacked_bar_proportion_stacked_normalized': (14, {'entity': 'E'}),
 'vis_015_heatmap_count': (15, {'entity': 'E'}),
 'vis_016_stacked_bar_count_vert_stacked': (16, {'entity': 'E'}),
 'vis_017_stacked_bar_count_grouped': (17, {'entity': 'E'}),
 'vis_018_stacked_bar_proportion_stacked_normalized': (18, {'entity': 'E'}),
 'vis_019_heatmap_count': (19, {'entity': 'E'}),
 'vis_020_table_count': (20, {'entity': 'E'}),
 'vis_021_table_count_sorted': (21, {'entity': 'E'}),
 'vis_022_table_count_sorted': (22, {'entity': 'E'}),
 'vis_023_table_count_sorted': (23, {'entity': 'E'}),
 'vis_024_table_count_sorted': (24, {'entity': 'E'})}
