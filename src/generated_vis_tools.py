"""
Auto-generated visualization tool definitions.

Generated from: src/skills/template_visualizations.json
Schema: data/data_domains/hubmap_data_schema.json
Tools: 64

DO NOT EDIT — regenerate with: python src/generate_tools.py
"""


# Schema metadata (entity URLs and relationships)
SCHEMA = {'entities': {'datasets': {'url': './data/hubmap_2025-05-05/datasets.csv'},
              'donors': {'url': './data/hubmap_2025-05-05/donors.csv'},
              'samples': {'url': './data/hubmap_2025-05-05/samples.csv'}},
 'relationships': [{'from_cardinality': 'many',
                    'from_entity': 'samples',
                    'from_field': 'donor.hubmap_id',
                    'to_cardinality': 'one',
                    'to_entity': 'donors',
                    'to_field': 'hubmap_id'},
                   {'from_cardinality': 'many',
                    'from_entity': 'datasets',
                    'from_field': 'donor.hubmap_id',
                    'to_cardinality': 'one',
                    'to_entity': 'donors',
                    'to_field': 'hubmap_id'},
                   {'from_cardinality': 'many',
                    'from_entity': 'datasets',
                    'from_field': 'donor.hubmap_id',
                    'to_cardinality': 'many',
                    'to_entity': 'samples',
                    'to_field': 'donor.hubmap_id'}]}


# Spec template strings (indexed by position)
TEMPLATES = ['{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F>"}, {"rollup": {"<E> count": '
 '{"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F>", "type": '
 '"nominal"}, {"encoding": "y", "field": "<E> count", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F>"}, {"rollup": {"<E> count": '
 '{"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<E> count", "type": '
 '"quantitative"}, {"encoding": "y", "field": "<F>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F>"}, {"rollup": {"<E> count": '
 '{"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F>", "type": '
 '"nominal"}, {"encoding": "y", "field": "<E> count", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F>"}, {"rollup": {"<E> count": '
 '{"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<E> count", "type": '
 '"quantitative"}, {"encoding": "y", "field": "<F>", "type": "nominal"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": "<E2.F>"}, {"rollup": {"<E1> count": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": '
 '[{"encoding": "x", "field": "<E2.F>", "type": "nominal"}, {"encoding": "y", "field": "<E1> count", "type": '
 '"quantitative"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": "<E2.F>"}, {"rollup": {"<E1> count": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": '
 '[{"encoding": "x", "field": "<E1> count", "type": "quantitative"}, {"encoding": "y", "field": "<E2.F>", "type": '
 '"nominal"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": ["<E2.F2>", "<E1.F1>"]}, {"rollup": {"count <E1>": {"op": "count"}}}], "representation": {"mark": "bar", '
 '"mapping": [{"encoding": "y", "field": "count <E1>", "type": "quantitative"}, {"encoding": "color", "field": '
 '"<E2.F2>", "type": "nominal"}, {"encoding": "x", "field": "<E1.F1>", "type": "nominal"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": ["<E2.F2>", "<E1.F1>"]}, {"rollup": {"count <E1>": {"op": "count"}}}], "representation": {"mark": "bar", '
 '"mapping": [{"encoding": "x", "field": "count <E1>", "type": "quantitative"}, {"encoding": "color", "field": '
 '"<E1.F1>", "type": "nominal"}, {"encoding": "y", "field": "<E2.F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F2>", "<F1>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "y", "field": "count '
 '<E>", "type": "quantitative"}, {"encoding": "color", "field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": '
 '"<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "count '
 '<E>", "type": "quantitative"}, {"encoding": "color", "field": "<F1>", "type": "nominal"}, {"encoding": "y", "field": '
 '"<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "y", "field": "count '
 '<E>", "type": "quantitative"}, {"encoding": "xOffset", "field": "<F1>", "type": "nominal"}, {"encoding": "color", '
 '"field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "count '
 '<E>", "type": "quantitative"}, {"encoding": "yOffset", "field": "<F1>", "type": "nominal"}, {"encoding": "color", '
 '"field": "<F1>", "type": "nominal"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "count '
 '<E>", "type": "quantitative"}, {"encoding": "color", "field": "<F1>", "type": "nominal"}, {"encoding": "y", "field": '
 '"<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>", "out": "groupCounts"}, '
 '{"rollup": {"<F2>_count": {"op": "count"}}}, {"groupby": ["<F1>", "<F2>"], "in": "<E>"}, {"rollup": '
 '{"<F1>_and_<F2>_count": {"op": "count"}}}, {"join": {"on": "<F2>"}, "in": ["<E>", "groupCounts"], "out": '
 '"datasets"}, {"derive": {"frequency": "d[\'<F1>_and_<F2>_count\'] / d[\'<F2>_count\']"}}], "representation": '
 '{"mark": "bar", "mapping": [{"encoding": "y", "field": "frequency", "type": "quantitative"}, {"encoding": "color", '
 '"field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>", "out": "groupCounts"}, '
 '{"rollup": {"<F2>_count": {"op": "count"}}}, {"groupby": ["<F1>", "<F2>"], "in": "<E>"}, {"rollup": '
 '{"<F1>_and_<F2>_count": {"op": "count"}}}, {"join": {"on": "<F2>"}, "in": ["<E>", "groupCounts"], "out": '
 '"datasets"}, {"derive": {"frequency": "d[\'<F1>_and_<F2>_count\'] / d[\'<F2>_count\']"}}], "representation": '
 '{"mark": "bar", "mapping": [{"encoding": "x", "field": "frequency", "type": "quantitative"}, {"encoding": "color", '
 '"field": "<F1>", "type": "nominal"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"minimum <F1>": '
 '{"op": "min", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "minimum '
 '<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"minimum <F1>": '
 '{"op": "min", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F2>", '
 '"type": "nominal"}, {"encoding": "y", "field": "minimum <F1>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"maximum <F1>": '
 '{"op": "max", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "maximum '
 '<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"maximum <F1>": '
 '{"op": "max", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F2>", '
 '"type": "nominal"}, {"encoding": "y", "field": "maximum <F1>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"average <F1>": '
 '{"op": "mean", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"average <F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"average <F1>": '
 '{"op": "mean", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F2>", '
 '"type": "nominal"}, {"encoding": "y", "field": "average <F1>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"median <F1>": '
 '{"op": "median", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"median <F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"median <F1>": '
 '{"op": "median", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": '
 '"<F2>", "type": "nominal"}, {"encoding": "y", "field": "median <F1>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"total <F1>": '
 '{"op": "sum", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "total '
 '<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F2>"}, {"rollup": {"total <F1>": '
 '{"op": "sum", "field": "<F1>"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F2>", '
 '"type": "nominal"}, {"encoding": "y", "field": "total <F1>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "representation": {"mark": "point", "mapping": [{"encoding": "x", '
 '"field": "<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "representation": {"mark": "point", "mapping": [{"encoding": "x", '
 '"field": "<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "<F1>", '
 '"type": "nominal"}, {"encoding": "y", "field": "count", "type": "quantitative"}, {"encoding": "color", "field": '
 '"<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F1>", "<F2>"]}, {"rollup": '
 '{"count": {"op": "count"}}}], "representation": {"mark": "bar", "mapping": [{"encoding": "x", "field": "count", '
 '"type": "quantitative"}, {"encoding": "y", "field": "<F1>", "type": "nominal"}, {"encoding": "color", "field": '
 '"<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F>"}, {"rollup": {"frequency": '
 '{"op": "frequency"}}}], "representation": {"mark": "arc", "mapping": [{"encoding": "theta", "field": "frequency", '
 '"type": "quantitative"}, {"encoding": "color", "field": "<F>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": "<F>"}, {"rollup": {"frequency": '
 '{"op": "frequency"}}}], "representation": {"mark": "arc", "mapping": [{"encoding": "theta", "field": "frequency", '
 '"type": "quantitative"}, {"encoding": "color", "field": "<F>", "type": "nominal"}, {"encoding": "radius", "value": '
 '60}, {"encoding": "radius2", "value": 80}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"rollup": {"<E> Records": {"op": "count"}}}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}]}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}]}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": "<E1.r.E2.id.from>"}, {"rollup": {"<E1> count": {"op": "count"}}}, {"orderby": {"field": "<E1> count", '
 '"order": "desc"}}, {"derive": {"rank": "rank()"}}, {"derive": {"most frequent": "d.rank == 1 ? \'yes\' : \'no\'"}}], '
 '"representation": [{"mark": "row", "mapping": [{"encoding": "x", "field": "<E1> count", "mark": "bar", "type": '
 '"quantitative", "domain": {"min": 0}}, {"encoding": "color", "column": "<E1> count", "mark": "bar", "field": "most '
 'frequent", "type": "nominal", "domain": ["yes", "no"], "range": ["#FFA500", "#c6cfd8"]}]}, {"mark": "row", '
 '"mapping": {"encoding": "text", "field": "*", "mark": "text", "type": "nominal"}}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"orderby": '
 '{"field": "<F>", "order": "desc"}}, {"derive": {"largest": "rank() == 1 ? \'largest\' : \'not\'"}}], '
 '"representation": {"mark": "row", "mapping": [{"encoding": "x", "field": "<F>", "mark": "bar", "type": '
 '"quantitative"}, {"encoding": "color", "column": "<F>", "mark": "bar", "field": "largest", "type": "nominal", '
 '"domain": ["largest", "not"], "range": ["#FFA500", "c6cfd8"]}, {"encoding": "text", "field": "*", "mark": "text", '
 '"type": "nominal"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": "<E1.r.E2.id.from>"}, {"rollup": {"Largest <E1.F>": {"op": "max", "field": "<E1.F>"}}}, {"filter": '
 '"d[\'Largest <E1.F>\'] != null"}, {"orderby": {"field": "Largest <E1.F>", "order": "desc"}}, {"derive": {"rank": '
 '"rank()"}}, {"derive": {"largest": "d.rank == 1 ? \'yes\' : \'no\'"}}], "representation": {"mark": "row", "mapping": '
 '[{"encoding": "x", "field": "Largest <E1.F>", "mark": "bar", "type": "quantitative"}, {"encoding": "color", '
 '"column": "Largest <E1.F>", "mark": "bar", "field": "largest", "type": "nominal", "domain": ["yes", "no"], "range": '
 '["#FFA500", "#c6cfd8"]}, {"encoding": "text", "field": "*", "mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"orderby": '
 '{"field": "<F>", "order": "asc"}}, {"derive": {"smallest": "rank() == 1 ? \'smallest\' : \'not\'"}}], '
 '"representation": {"mark": "row", "mapping": [{"encoding": "color", "column": "<F>", "mark": "rect", "orderby": '
 '"<F>", "field": "smallest", "type": "nominal", "domain": ["smallest", "not"], "range": ["#ffdb9a", "white"]}, '
 '{"encoding": "text", "field": "*", "mark": "text", "type": "nominal"}]}}',
 '{"source": [{"name": "<E1>", "source": "<E1.url>"}, {"name": "<E2>", "source": "<E2.url>"}], "transformation": '
 '[{"join": {"on": ["<E1.r.E2.id.from>", "<E1.r.E2.id.to>"]}, "in": ["<E1>", "<E2>"], "out": "<E1>__<E2>"}, '
 '{"groupby": "<E1.r.E2.id.from>"}, {"rollup": {"Smallest <E1.F>": {"op": "min", "field": "<E1.F>"}}}, {"filter": '
 '"d[\'Smallest <E1.F>\'] != null"}, {"orderby": {"field": "Smallest <E1.F>", "order": "asc"}}, {"derive": {"rank": '
 '"rank()"}}, {"derive": {"smallest": "d.rank == 1 ? \'yes\' : \'no\'"}}], "representation": {"mark": "row", '
 '"mapping": [{"encoding": "color", "column": "Smallest <E1.F>", "mark": "bar", "orderby": "Smallest <E1.F>", "field": '
 '"smallest", "type": "nominal", "domain": ["yes", "no"], "range": ["#ffdb9a", "white"]}, {"encoding": "text", '
 '"field": "*", "mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"orderby": '
 '{"field": "<F>", "order": "asc"}}], "representation": {"mark": "row", "mapping": [{"encoding": "x", "column": "<F>", '
 '"mark": "bar", "field": "<F>", "type": "quantitative", "range": {"min": 0.2, "max": 1}}, {"encoding": "text", '
 '"field": "*", "mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"rollup": '
 '{"<F> min": {"op": "min", "field": "<F>"}, "<F> max": {"op": "max", "field": "<F>"}}}], "representation": {"mark": '
 '"row", "mapping": [{"encoding": "text", "field": "<F> min", "mark": "text", "type": "nominal"}, {"encoding": "text", '
 '"field": "<F> max", "mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"groupby": '
 '"<F>"}, {"rollup": {"count": {"op": "count"}}}], "representation": {"mark": "row", "mapping": [{"encoding": "text", '
 '"field": "<F>", "mark": "text", "type": "nominal"}, {"encoding": "x", "field": "count", "mark": "bar", "type": '
 '"quantitative", "range": {"min": 0.1, "max": 1}}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F1>\'] != null"}, {"groupby": '
 '"<F2>"}, {"rollup": {"<F1> min": {"op": "min", "field": "<F1>"}, "<F1> max": {"op": "max", "field": "<F1>"}}}, '
 '{"derive": {"range": "d[\'<F1> max\'] - d[\'<F1> min\']"}}, {"orderby": {"field": "range", "order": "desc"}}], '
 '"representation": {"mark": "row", "mapping": [{"encoding": "text", "field": "<F2>", "mark": "text", "type": '
 '"nominal"}, {"encoding": "text", "field": "<F1> min", "mark": "text", "type": "nominal"}, {"encoding": "x", '
 '"column": "range", "mark": "bar", "field": "<F1> min", "type": "quantitative", "domain": {"numberFields": ["<F1> '
 'min", "<F1> max"]}}, {"encoding": "x2", "column": "range", "mark": "bar", "field": "<F1> max", "type": '
 '"quantitative", "domain": {"numberFields": ["<F1> min", "<F1> max"]}}, {"encoding": "text", "field": "<F1> max", '
 '"mark": "text", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\']"}, {"groupby": "<F>"}, '
 '{"rollup": {"count": {"op": "count"}}}, {"orderby": {"field": "count", "order": "desc"}}, {"derive": {"rank": '
 '"rank()"}}, {"derive": {"most frequent": "d.rank == 1 ? \'yes\' : \'no\'"}}], "representation": {"mark": "row", '
 '"mapping": [{"encoding": "color", "column": "<F>", "mark": "bar", "orderby": "<F>", "field": "most frequent", '
 '"type": "nominal", "domain": ["yes", "no"], "range": ["#ffdb9a", "white"]}, {"encoding": "text", "field": "<F>", '
 '"mark": "text", "type": "nominal"}, {"encoding": "x", "field": "count", "mark": "bar", "type": "quantitative", '
 '"domain": {"min": 0}}, {"encoding": "color", "column": "count", "mark": "bar", "field": "most frequent", "type": '
 '"nominal", "domain": ["yes", "no"], "range": ["#FFA500", "#c6cfd8"]}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"orderby": '
 '{"field": "<F>", "order": "asc"}}, {"derive": {"total": "count()"}}, {"derive": {"percentile": {"rolling": '
 '{"expression": "count() / d.total"}}}}], "representation": {"mark": "line", "mapping": [{"encoding": "x", "field": '
 '"<F>", "type": "quantitative"}, {"encoding": "y", "field": "percentile", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"orderby": '
 '{"field": "<F>", "order": "asc"}}, {"derive": {"total": "count()"}}, {"derive": {"percentile": {"rolling": '
 '{"expression": "count() / d.total"}}}}], "representation": {"mark": "line", "mapping": [{"encoding": "x", "field": '
 '"<F>", "type": "quantitative"}, {"encoding": "y", "field": "percentile", "type": "quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F1>\'] != null"}, {"orderby": '
 '{"field": "<F1>", "order": "asc"}}, {"groupby": "<F2>"}, {"derive": {"total": "count()"}}, {"derive": {"percentile": '
 '{"rolling": {"expression": "count() / d.total"}}}}], "representation": {"mark": "line", "mapping": [{"encoding": '
 '"x", "field": "<F1>", "type": "quantitative"}, {"encoding": "y", "field": "percentile", "type": "quantitative"}, '
 '{"encoding": "color", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F1>\'] != null"}, {"orderby": '
 '{"field": "<F1>", "order": "asc"}}, {"groupby": "<F2>"}, {"derive": {"total": "count()"}}, {"derive": {"percentile": '
 '{"rolling": {"expression": "count() / d.total"}}}}], "representation": {"mark": "line", "mapping": [{"encoding": '
 '"x", "field": "<F1>", "type": "quantitative"}, {"encoding": "y", "field": "percentile", "type": "quantitative"}, '
 '{"encoding": "color", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F2>", "<F1>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}, {"derive": {"udi_internal_percentile": "d[\'count <E>\'] / max(d[\'count '
 '<E>\'])"}}, {"derive": {"udi_internal_text_color_threshold": "d.udi_internal_percentile > .5 ? \'large\' : '
 '\'small\'"}}], "representation": [{"mark": "rect", "mapping": [{"encoding": "color", "field": "count <E>", "type": '
 '"quantitative"}, {"encoding": "y", "field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": "<F2>", "type": '
 '"nominal"}]}, {"mark": "text", "mapping": [{"encoding": "text", "field": "count <E>", "type": "quantitative"}, '
 '{"encoding": "y", "field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": "<F2>", "type": "nominal"}, '
 '{"encoding": "color", "field": "udi_internal_text_color_threshold", "type": "nominal", "domain": ["large", "small"], '
 '"range": ["white", "black"], "omitLegend": true}]}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F2>", "<F1>"]}, {"rollup": '
 '{"count <E>": {"op": "count"}}}, {"derive": {"udi_internal_percentile": "d[\'count <E>\'] / max(d[\'count '
 '<E>\'])"}}, {"derive": {"udi_internal_text_color_threshold": "d.udi_internal_percentile > .5 ? \'large\' : '
 '\'small\'"}}], "representation": [{"mark": "rect", "mapping": [{"encoding": "color", "field": "count <E>", "type": '
 '"quantitative"}, {"encoding": "y", "field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": "<F2>", "type": '
 '"nominal"}]}, {"mark": "text", "mapping": [{"encoding": "text", "field": "count <E>", "type": "quantitative"}, '
 '{"encoding": "y", "field": "<F1>", "type": "nominal"}, {"encoding": "x", "field": "<F2>", "type": "nominal"}, '
 '{"encoding": "color", "field": "udi_internal_text_color_threshold", "type": "nominal", "domain": ["large", "small"], '
 '"range": ["white", "black"], "omitLegend": true}]}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"groupby": ["<F3>", "<F2>"]}, {"rollup": '
 '{"average <F1>": {"op": "mean", "field": "<F1>"}}}], "representation": {"mark": "rect", "mapping": [{"encoding": '
 '"color", "field": "average <F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}, '
 '{"encoding": "x", "field": "<F3>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "representation": {"mark": "point", "mapping": [{"encoding": "x", '
 '"field": "<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "quantitative"}, {"encoding": '
 '"color", "field": "<F3>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"binby": '
 '{"field": "<F>", "output": {"bin_start": "start", "bin_end": "end"}}}, {"rollup": {"count": {"op": "count"}}}], '
 '"representation": {"mark": "rect", "mapping": [{"encoding": "x", "field": "start", "type": "quantitative"}, '
 '{"encoding": "x2", "field": "end", "type": "quantitative"}, {"encoding": "y", "field": "count", "type": '
 '"quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"binby": '
 '{"field": "<F>", "output": {"bin_start": "start", "bin_end": "end"}}}, {"rollup": {"count": {"op": "count"}}}], '
 '"representation": {"mark": "rect", "mapping": [{"encoding": "x", "field": "start", "type": "quantitative"}, '
 '{"encoding": "x2", "field": "end", "type": "quantitative"}, {"encoding": "y", "field": "count", "type": '
 '"quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F>\'] != null"}, {"kde": '
 '{"field": "<F>", "output": {"sample": "<F>", "density": "density"}}}], "representation": {"mark": "area", "mapping": '
 '[{"encoding": "x", "field": "<F>", "type": "quantitative"}, {"encoding": "y", "field": "density", "type": '
 '"quantitative"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "representation": {"mark": "point", "mapping": {"encoding": "x", '
 '"field": "<F>", "type": "quantitative"}}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"filter": "d[\'<F1>\'] != null"}, {"groupby": '
 '"<F2>"}, {"kde": {"field": "<F1>", "output": {"sample": "<F1>", "density": "density"}}}], "representation": '
 '[{"mark": "area", "mapping": [{"encoding": "x", "field": "<F1>", "type": "quantitative"}, {"encoding": "color", '
 '"field": "<F2>", "type": "nominal"}, {"encoding": "y", "field": "density", "type": "quantitative"}, {"encoding": '
 '"opacity", "value": 0.25}]}, {"mark": "line", "mapping": [{"encoding": "x", "field": "<F1>", "type": '
 '"quantitative"}, {"encoding": "color", "field": "<F2>", "type": "nominal"}, {"encoding": "y", "field": "density", '
 '"type": "quantitative"}]}]}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "representation": {"mark": "point", "mapping": [{"encoding": "x", '
 '"field": "<F1>", "type": "quantitative"}, {"encoding": "y", "field": "<F2>", "type": "nominal"}, {"encoding": '
 '"color", "field": "<F2>", "type": "nominal"}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"derive": {"<E> Count": "count()"}}, {"filter": '
 '"d[\'<F>\'] != null"}, {"rollup": {"Valid <F> Count": {"op": "count"}, "<E> Count": {"op": "median", "field": "<E> '
 'Count"}}}, {"derive": {"Valid <F> %": "d[\'Valid <F> Count\'] / d[\'<E> Count\']"}}], "representation": {"mark": '
 '"row", "mapping": [{"encoding": "text", "field": "Valid <F> Count", "mark": "text", "type": "nominal"}, {"encoding": '
 '"text", "field": "<E> Count", "mark": "text", "type": "nominal"}, {"encoding": "x", "field": "Valid <F> %", "mark": '
 '"bar", "type": "quantitative", "domain": {"min": 0, "max": 1}}, {"encoding": "y", "field": "Valid <F> %", "mark": '
 '"line", "type": "quantitative", "range": {"min": 0.5, "max": 0.5}}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"derive": {"<E> Count": "count()"}}, {"filter": '
 '"d[\'<F>\'] != null"}, {"rollup": {"Valid <F> Count": {"op": "count"}, "<E> Count": {"op": "median", "field": "<E> '
 'Count"}}}, {"derive": {"Valid <F> %": "d[\'Valid <F> Count\'] / d[\'<E> Count\']"}}], "representation": {"mark": '
 '"row", "mapping": [{"encoding": "text", "field": "Valid <F> Count", "mark": "text", "type": "nominal"}, {"encoding": '
 '"text", "field": "<E> Count", "mark": "text", "type": "nominal"}, {"encoding": "x", "field": "Valid <F> %", "mark": '
 '"bar", "type": "quantitative", "domain": {"min": 0, "max": 1}}, {"encoding": "y", "field": "Valid <F> %", "mark": '
 '"line", "type": "quantitative", "range": {"min": 0.5, "max": 0.5}}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"derive": {"<E> Count": "count()"}}, {"filter": '
 '"d[\'<F>\'] != null"}, {"rollup": {"Valid <F> Count": {"op": "count"}, "<E> Count": {"op": "median", "field": "<E> '
 'Count"}}}, {"derive": {"Null <F> Count": "d[\'<E> Count\'] - d[\'Valid <F> Count\']", "Null <F> %": "1 - d[\'Valid '
 '<F> Count\'] / d[\'<E> Count\']"}}], "representation": {"mark": "row", "mapping": [{"encoding": "text", "field": '
 '"Null <F> Count", "mark": "text", "type": "nominal"}, {"encoding": "text", "field": "<E> Count", "mark": "text", '
 '"type": "nominal"}, {"encoding": "x", "field": "Null <F> %", "mark": "bar", "type": "quantitative", "domain": '
 '{"min": 0, "max": 1}}, {"encoding": "y", "field": "Null <F> %", "mark": "line", "type": "quantitative", "range": '
 '{"min": 0.5, "max": 0.5}}]}}',
 '{"source": {"name": "<E>", "source": "<E.url>"}, "transformation": [{"derive": {"<E> Count": "count()"}}, {"filter": '
 '"d[\'<F>\'] != null"}, {"rollup": {"Valid <F> Count": {"op": "count"}, "<E> Count": {"op": "median", "field": "<E> '
 'Count"}}}, {"derive": {"Null <F> Count": "d[\'<E> Count\'] - d[\'Valid <F> Count\']", "Null <F> %": "1 - d[\'Valid '
 '<F> Count\'] / d[\'<E> Count\']"}}], "representation": {"mark": "row", "mapping": [{"encoding": "text", "field": '
 '"Null <F> Count", "mark": "text", "type": "nominal"}, {"encoding": "text", "field": "<E> Count", "mark": "text", '
 '"type": "nominal"}, {"encoding": "x", "field": "Null <F> %", "mark": "bar", "type": "quantitative", "domain": '
 '{"min": 0, "max": 1}}, {"encoding": "y", "field": "Null <F> %", "mark": "line", "type": "quantitative", "range": '
 '{"min": 0.5, "max": 0.5}}]}}']


# OpenAI function-calling tool definitions
TOOL_DEFS = [{'function': {'description': 'Counts entities grouped by a nominal field, displayed as a vertical bar chart. Design: '
                              'Vertical orientation chosen because category count is small (<=4), keeping x-axis '
                              'labels readable. Tasks: Compare counts across categories; identify the most or least '
                              'common category. Query pattern: How many <E> are there, grouped by <F:n>?',
               'name': 'vis_000_counts_entities_grouped_by_a_nominal',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Counts entities grouped by a nominal field, displayed as a horizontal bar chart. '
                              'Design: Horizontal orientation chosen because category count is high (>4), allowing '
                              'longer labels on the y-axis. Tasks: Compare counts across categories; identify the most '
                              'or least common category. Query pattern: How many <E> are there, grouped by <F:n>?',
               'name': 'vis_001_counts_entities_grouped_by_a_nominal',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a vertical bar chart counting entities by a nominal field. Design: Vertical '
                              'orientation for small category counts (<=4). Count aggregation applied automatically. '
                              'Tasks: Compare counts across categories; identify the most or least common category; '
                              'assess the range of counts. Query pattern: Make a bar chart of <E> <F:n>.',
               'name': 'vis_002_creates_a_vertical_bar_chart_counting',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a horizontal bar chart counting entities by a nominal field. Design: Horizontal '
                              'orientation for higher category counts (>4), improving label readability. Tasks: '
                              'Compare counts across categories; identify the most or least common category; assess '
                              'the range of counts. Query pattern: Make a bar chart of <E> <F:n>.',
               'name': 'vis_003_creates_a_horizontal_bar_chart_counting',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Joins two entities and counts records grouped by a field from the related entity, '
                              'displayed as a vertical bar chart. Design: Cross-entity join groups by a field not '
                              'native to the counted entity. Vertical orientation for small category counts (<=4). '
                              'Requires a many-to-one relationship. Tasks: Compare counts across categories from a '
                              'related entity; discover cross-entity frequency patterns. Query pattern: How many <E1> '
                              'are there, grouped by <E2.F:n>?',
               'name': 'vis_004_joins_two_entities_and_counts_records',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'},
                                             'entity2_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity2_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Joins two entities and counts records grouped by a field from the related entity, '
                              'displayed as a horizontal bar chart. Design: Cross-entity join with horizontal '
                              'orientation for higher category counts (>4). Requires a many-to-one relationship. '
                              'Tasks: Compare counts across categories from a related entity; discover cross-entity '
                              'frequency patterns. Query pattern: How many <E1> are there, grouped by <E2.F:n>?',
               'name': 'vis_005_joins_two_entities_and_counts_records',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'},
                                             'entity2_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity2_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Joins two entities and produces a vertical stacked bar chart of counts grouped by two '
                              'nominal fields. Design: Stacked bars show part-to-whole composition within each '
                              'category. Vertical layout for small category counts (<=4). Color encodes the secondary '
                              'grouping field from the related entity. Tasks: Compare group compositions across '
                              'categories; identify dominant sub-groups within each bar. Query pattern: How many <E1> '
                              'are there, grouped by <E1.F1:n> and <E2.F2:n>?',
               'name': 'vis_006_joins_two_entities_and_produces_a',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity1_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'},
                                             'entity2_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity1_field', 'entity2_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Joins two entities and produces a horizontal stacked bar chart of counts grouped by two '
                              'nominal fields. Design: Horizontal orientation for higher category counts (>4). Color '
                              'encodes the primary grouping field. Cross-entity join required. Tasks: Compare group '
                              'compositions across categories; identify dominant sub-groups within each bar. Query '
                              'pattern: How many <E1> are there, grouped by <E1.F1:n> and <E2.F2:n>?',
               'name': 'vis_007_joins_two_entities_and_produces_a',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity1_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'},
                                             'entity2_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity1_field', 'entity2_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Counts entities grouped by two nominal fields, displayed as a vertical stacked bar '
                              'chart. Design: Vertical stacked layout for small category counts (<=4). Color encodes '
                              'the sub-group field; x-axis shows the primary grouping. Tasks: Compare group '
                              'compositions across categories; identify dominant sub-groups within each bar. Query '
                              'pattern: How many <E> are there, grouped by <F1:n> and <F2:n>?',
               'name': 'vis_008_counts_entities_grouped_by_two_nominal',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Counts entities grouped by two nominal fields, displayed as a horizontal stacked bar '
                              'chart. Design: Horizontal stacked layout for higher category counts (>4). Color encodes '
                              'the sub-group field. Tasks: Compare group compositions across categories; identify '
                              'dominant sub-groups within each bar. Query pattern: How many <E> are there, grouped by '
                              '<F1:n> and <F2:n>?',
               'name': 'vis_009_counts_entities_grouped_by_two_nominal',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Counts entities grouped by two nominal fields, displayed as a grouped (side-by-side) '
                              'vertical bar chart. Design: Uses xOffset for side-by-side grouping, allowing direct '
                              'comparison between sub-groups. Suitable for small category counts (<=4). Tasks: '
                              'Directly compare sub-group counts within and across categories. Query pattern: What is '
                              'the count of <F1:n> for each <F2:n>?',
               'name': 'vis_010_counts_entities_grouped_by_two_nominal',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Counts entities grouped by two nominal fields, displayed as a grouped (side-by-side) '
                              'horizontal bar chart. Design: Uses yOffset for side-by-side grouping in horizontal '
                              'orientation. Chosen when at least one field has more than 4 categories. Tasks: Directly '
                              'compare sub-group counts within and across categories. Query pattern: What is the count '
                              'of <F1:n> for each <F2:n>?',
               'name': 'vis_011_counts_entities_grouped_by_two_nominal',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Counts entities grouped by two nominal fields, displayed as a horizontal stacked bar '
                              'chart. Design: Horizontal stacked layout for higher category counts (>4). Color encodes '
                              'the sub-group; stacking shows part-to-whole within each bar. Allows up to 10 sub-group '
                              'categories. Tasks: Compare group compositions across categories; identify dominant '
                              'sub-groups within each bar. Query pattern: What is the count of <F1:n> for each <F2:n>?',
               'name': 'vis_012_counts_entities_grouped_by_two_nominal',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Shows the frequency (proportion) of one nominal field within each category of another, '
                              'as a vertical normalized bar chart. Design: Normalization computes proportions per '
                              'group, enabling fair comparison across groups of different sizes. Vertical layout for '
                              'small category counts (<=4). Tasks: Compare relative proportions across categories; '
                              'identify which sub-groups dominate in each group. Query pattern: What is the frequency '
                              'of <F1:n> for each <F2:n>?',
               'name': 'vis_013_shows_the_frequency_proportion_of_one',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Shows the frequency (proportion) of one nominal field within each category of another, '
                              'as a horizontal normalized bar chart. Design: Normalization for proportional '
                              'comparison. Horizontal layout for higher category counts (>4). Tasks: Compare relative '
                              'proportions across categories; identify which sub-groups dominate in each group. Query '
                              'pattern: What is the frequency of <F1:n> for each <F2:n>?',
               'name': 'vis_014_shows_the_frequency_proportion_of_one',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the minimum of a quantitative field for each category, displayed as a '
                              'horizontal bar chart. Design: Horizontal orientation for many categories (>4). Bar '
                              'length encodes the minimum aggregate value for easy comparison. Tasks: Compare the '
                              'minimum value across categories; identify which group has the highest or lowest '
                              'minimum. Query pattern: What is the minimum <F1:q> for each <F2:n>?',
               'name': 'vis_015_computes_the_minimum_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the minimum of a quantitative field for each category, displayed as a vertical '
                              'bar chart. Design: Vertical orientation for few categories (<=4). Bar height encodes '
                              'the minimum aggregate value. Tasks: Compare the minimum value across categories; '
                              'identify which group has the highest or lowest minimum. Query pattern: What is the '
                              'minimum <F1:q> for each <F2:n>?',
               'name': 'vis_016_computes_the_minimum_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the maximum of a quantitative field for each category, displayed as a '
                              'horizontal bar chart. Design: Horizontal orientation for many categories (>4). Bar '
                              'length encodes the maximum aggregate value for easy comparison. Tasks: Compare the '
                              'maximum value across categories; identify which group has the highest or lowest '
                              'maximum. Query pattern: What is the maximum <F1:q> for each <F2:n>?',
               'name': 'vis_017_computes_the_maximum_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the maximum of a quantitative field for each category, displayed as a vertical '
                              'bar chart. Design: Vertical orientation for few categories (<=4). Bar height encodes '
                              'the maximum aggregate value. Tasks: Compare the maximum value across categories; '
                              'identify which group has the highest or lowest maximum. Query pattern: What is the '
                              'maximum <F1:q> for each <F2:n>?',
               'name': 'vis_018_computes_the_maximum_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the average of a quantitative field for each category, displayed as a '
                              'horizontal bar chart. Design: Horizontal orientation for many categories (>4). Bar '
                              'length encodes the average aggregate value for easy comparison. Tasks: Compare the '
                              'average value across categories; identify which group has the highest or lowest '
                              'average. Query pattern: What is the average <F1:q> for each <F2:n>?',
               'name': 'vis_019_computes_the_average_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the average of a quantitative field for each category, displayed as a vertical '
                              'bar chart. Design: Vertical orientation for few categories (<=4). Bar height encodes '
                              'the average aggregate value. Tasks: Compare the average value across categories; '
                              'identify which group has the highest or lowest average. Query pattern: What is the '
                              'average <F1:q> for each <F2:n>?',
               'name': 'vis_020_computes_the_average_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the median of a quantitative field for each category, displayed as a '
                              'horizontal bar chart. Design: Horizontal orientation for many categories (>4). Bar '
                              'length encodes the median aggregate value for easy comparison. Tasks: Compare the '
                              'median value across categories; identify which group has the highest or lowest median. '
                              'Query pattern: What is the median <F1:q> for each <F2:n>?',
               'name': 'vis_021_computes_the_median_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the median of a quantitative field for each category, displayed as a vertical '
                              'bar chart. Design: Vertical orientation for few categories (<=4). Bar height encodes '
                              'the median aggregate value. Tasks: Compare the median value across categories; identify '
                              'which group has the highest or lowest median. Query pattern: What is the median <F1:q> '
                              'for each <F2:n>?',
               'name': 'vis_022_computes_the_median_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the total of a quantitative field for each category, displayed as a horizontal '
                              'bar chart. Design: Horizontal orientation for many categories (>4). Bar length encodes '
                              'the total aggregate value for easy comparison. Tasks: Compare the total value across '
                              'categories; identify which group has the highest or lowest total. Query pattern: What '
                              'is the total <F1:q> for each <F2:n>?',
               'name': 'vis_023_computes_the_total_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the total of a quantitative field for each category, displayed as a vertical '
                              'bar chart. Design: Vertical orientation for few categories (<=4). Bar height encodes '
                              'the total aggregate value. Tasks: Compare the total value across categories; identify '
                              'which group has the highest or lowest total. Query pattern: What is the total <F1:q> '
                              'for each <F2:n>?',
               'name': 'vis_024_computes_the_total_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Plots two quantitative fields as a scatterplot to explore their relationship. Design: '
                              'Point marks on two quantitative axes reveal correlations, clusters, and outliers. Data '
                              'size capped at 100k rows for rendering performance. Tasks: Assess correlation between '
                              'two variables; identify outliers and clusters. Query pattern: Is there a correlation '
                              'between <F1:q> and <F2:q>?',
               'name': 'vis_025_plots_two_quantitative_fields_as_a',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a scatterplot of two quantitative fields. Design: Point marks on quantitative x '
                              'and y axes. Data size capped at 100k rows for performance. Tasks: Assess correlation; '
                              'identify clusters, outliers, extremes, and the range of both variables. Query pattern: '
                              'Make a scatterplot of <F1:q> and <F2:q>?',
               'name': 'vis_026_creates_a_scatterplot_of_two_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a vertical stacked bar chart of counts grouped by two nominal fields. Design: '
                              'Vertical stacked layout for small primary category counts (<=4). Color encodes the '
                              'secondary field. Tasks: Compare group compositions across categories; assess the '
                              'overall range of counts. Query pattern: Make a stacked bar chart of <F1:n> and <F2:n>?',
               'name': 'vis_027_creates_a_vertical_stacked_bar_chart',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a horizontal stacked bar chart of counts grouped by two nominal fields. Design: '
                              'Horizontal stacked layout for higher primary category counts (>4). Color encodes the '
                              'secondary field. Tasks: Compare group compositions across categories; assess the '
                              'overall range of counts. Query pattern: Make a stacked bar chart of <F1:n> and <F2:n>?',
               'name': 'vis_028_creates_a_horizontal_stacked_bar_chart',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a pie chart showing the frequency distribution of a nominal field. Design: Arc '
                              'marks with theta encoding map frequency to angle. Suitable for fields with few '
                              'categories (<8) where part-to-whole perception is the goal. Tasks: Assess part-to-whole '
                              'proportions; identify the dominant category. Query pattern: Make a pie chart of <F:n>?',
               'name': 'vis_029_creates_a_pie_chart_showing_the',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a donut chart showing the frequency distribution of a nominal field. Design: '
                              'Donut variant with inner/outer radius creates a hollow center that can improve label '
                              'readability. Suitable for few categories (<8). Tasks: Assess part-to-whole proportions; '
                              'identify the dominant category. Query pattern: Make a donut chart of <F:n>?',
               'name': 'vis_030_creates_a_donut_chart_showing_the',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Counts the total number of records in an entity and displays the result as a single-row '
                              'table. Design: Simple rollup with no visual encoding beyond the count value. Useful as '
                              'a quick data quality or size check. Tasks: Retrieve the total record count for an '
                              'entity. Query pattern: How many <E> records are there?',
               'name': 'vis_031_counts_the_total_number_of_records',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Displays the raw data for an entity as a table for exploration. Design: No aggregation '
                              'or transformation applied; shows the underlying data as-is. Tasks: Explore the raw '
                              'data; understand field values and ranges. Query pattern: What does the <E> data look '
                              'like?',
               'name': 'vis_032_displays_the_raw_data_for_an',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a table displaying the raw data for an entity. Design: No aggregation or '
                              'transformation; presents the full dataset for manual inspection. Tasks: Explore raw '
                              'data; retrieve specific values; identify anomalies and extremes. Query pattern: Make a '
                              'table of <E>?',
               'name': 'vis_033_creates_a_table_displaying_the_raw',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'}},
                              'required': ['entity'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Joins two related entities and displays the combined data as a table. Design: '
                              'Cross-entity join enriches the view by combining fields from two related entities. '
                              'Requires a valid foreign-key relationship. Tasks: Explore combined data from two '
                              'related entities; understand field values across the join. Query pattern: What does the '
                              'combined data of <E1> and <E2> look like?',
               'name': 'vis_034_joins_two_related_entities_and_displays',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'}},
                              'required': ['entity1', 'entity2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a table that joins and displays the combined data of two entities. Design: '
                              'Cross-entity join via foreign key. Presents the full combined dataset for manual '
                              'inspection. Tasks: Explore joined data; retrieve specific values; identify anomalies '
                              'and extremes. Query pattern: Make a table that combines <E1> and <E2>.',
               'name': 'vis_035_creates_a_table_that_joins_and',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'}},
                              'required': ['entity1', 'entity2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Finds which related entity record has the highest count of associated records, '
                              'displayed as a ranked table with bar indicators. Design: Groups by foreign key, counts, '
                              'ranks, and highlights the top record with color encoding. Bar marks on the count column '
                              'provide visual comparison. Tasks: Identify the record with the most associated '
                              'entities; compare counts across records. Query pattern: What <E2> has the most <E1>?',
               'name': 'vis_036_finds_which_related_entity_record_has',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'}},
                              'required': ['entity1', 'entity2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Finds the record with the largest value in a quantitative field, displayed as a ranked '
                              'table with bar indicators. Design: Sorts descending by the target field, derives a '
                              'rank, and highlights the top record with color. Bar marks provide visual magnitude '
                              'comparison. Tasks: Identify the record with the largest value; compare values across '
                              'records. Query pattern: What Record in <E> has the largest <F:q>?',
               'name': 'vis_037_finds_the_record_with_the_largest',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Joins two entities, computes the maximum of a quantitative field per group, and ranks '
                              'the results in a table with bar indicators. Design: Cross-entity join followed by '
                              'group-level max aggregation. Highlights the top record with color encoding. Useful when '
                              'the extremum requires aggregation across a relationship. Tasks: Identify which related '
                              'record has the largest aggregated value; compare across groups. Query pattern: What '
                              'Record in <E2> has the largest <E1> <E1.F:q>?',
               'name': 'vis_038_joins_two_entities_computes_the_maximum',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity1_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity1_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Finds the record with the smallest value in a quantitative field, displayed as a ranked '
                              'table with conditional formatting. Design: Sorts ascending by the target field, derives '
                              'a rank, and highlights the top record with background color. Uses rect mark for '
                              'row-level highlighting. Tasks: Identify the record with the smallest value; compare '
                              'values across records. Query pattern: What Record in <E> has the smallest <F:q>?',
               'name': 'vis_039_finds_the_record_with_the_smallest',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Joins two entities, computes the minimum of a quantitative field per group, and ranks '
                              'the results in a table with conditional formatting. Design: Cross-entity join followed '
                              'by group-level min aggregation. Highlights the top record with background color via '
                              'rect mark. Tasks: Identify which related record has the smallest aggregated value; '
                              'compare across groups. Query pattern: What Record in <E2> has the smallest <E1> '
                              '<E1.F:q>?',
               'name': 'vis_040_joins_two_entities_computes_the_minimum',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity1': {'description': 'The primary data entity (table).',
                                                         'type': 'string'},
                                             'entity1_field': {'description': 'Field name (any type) from the entity.',
                                                               'type': 'string'},
                                             'entity2': {'description': 'The secondary data entity (table) to join '
                                                                        'with.',
                                                         'type': 'string'}},
                              'required': ['entity1', 'entity2', 'entity1_field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Sorts entity records by a quantitative field and displays the result as an ordered '
                              'table with in-cell bar marks. Design: Ordered by the quantitative field with nulls '
                              'filtered out. In-cell bar marks provide visual comparison of magnitude alongside the '
                              'text values. Tasks: View records in sorted order; compare relative magnitudes. Query '
                              'pattern: Order the <E> by <F:q>?',
               'name': 'vis_041_sorts_entity_records_by_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the minimum and maximum of a quantitative field and displays them as a '
                              'single-row table. Design: Simple rollup of min and max. Filters out nulls before '
                              'aggregation for accuracy. Tasks: Determine the range of a quantitative field. Query '
                              'pattern: What is the range of <E> <F:q> values?',
               'name': 'vis_042_computes_the_minimum_and_maximum_of',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Lists all distinct values of a nominal field with their counts, displayed as a table '
                              'with in-cell bar marks. Design: Groups by the nominal field and counts occurrences. '
                              'In-cell bars provide visual frequency comparison. Limits to fields with fewer than 50 '
                              'categories. Tasks: Determine the range (distinct values) of a nominal field; compare '
                              'category frequencies. Query pattern: What is the range of <E> <F:n> values?',
               'name': 'vis_043_lists_all_distinct_values_of_a',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the min and max of a quantitative field for each category of a nominal field, '
                              'displayed as a table with range bar marks. Design: Groups by nominal field, computes '
                              'min/max and derived range, then orders by range descending. Uses x/x2 encoding to show '
                              'the span between min and max values. Tasks: Compare the spread of a quantitative field '
                              'across categories; identify which group has the widest or narrowest range. Query '
                              'pattern: What is the range of <E> <F1:q> values for every <F2:n>?',
               'name': 'vis_044_computes_the_min_and_max_of',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Finds the most frequent value of a nominal field, displayed as a ranked table with bar '
                              'marks and conditional formatting. Design: Groups by nominal field, counts, ranks, and '
                              'highlights the top value. Combines bar marks for count comparison and background color '
                              'for emphasis. Tasks: Identify the most frequent category; compare frequencies across '
                              'all categories. Query pattern: What is the most frequent <F:n>?',
               'name': 'vis_045_finds_the_most_frequent_value_of',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Shows the cumulative distribution function (CDF) of a quantitative field as a line '
                              'chart. Design: Sorts by value, computes rolling percentile, and draws a line. The CDF '
                              'reveals the full distribution shape including median, quartiles, and tails. Tasks: '
                              'Characterize the distribution of a variable; identify median, quartiles, and '
                              'concentration of values. Query pattern: What is the cumulative distribution of <F:q>?',
               'name': 'vis_046_shows_the_cumulative_distribution_function_cdf',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a CDF (cumulative distribution function) plot of a quantitative field. Design: '
                              'Sorts values, computes cumulative percentile, and renders as a line chart. Tasks: '
                              'Characterize the distribution of a variable; identify median, quartiles, and '
                              'concentration of values. Query pattern: Make a CDF plot of <F:q>.',
               'name': 'vis_047_creates_a_cdf_cumulative_distribution_function',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Shows the cumulative distribution of a quantitative field for each category of a '
                              'nominal field, with separate lines per group. Design: Groups by nominal field before '
                              'computing per-group CDF. Color encodes group identity. Limited to fewer than 5 groups '
                              'for readability. Tasks: Compare distributions across groups; identify which groups have '
                              'higher or lower concentrations of values. Query pattern: What is the cumulative '
                              'distribution of <F1:q> for each <F2:n>?',
               'name': 'vis_048_shows_the_cumulative_distribution_of_a',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a CDF plot with a separate line for each category of a nominal field. Design: '
                              'Per-group CDF computation with color encoding to distinguish groups. Limited to fewer '
                              'than 5 groups. Tasks: Compare distributions across groups; identify which groups have '
                              'higher or lower concentrations of values. Query pattern: Make a CDF plot of <F1:q> with '
                              'a line for each <F2:n>.',
               'name': 'vis_049_creates_a_cdf_plot_with_a',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Displays the count of entities for each combination of two nominal fields as a heatmap '
                              'with labeled cells. Design: Rect marks with quantitative color encoding show density. '
                              'Overlaid text marks display exact counts. Text color adapts based on cell intensity for '
                              'readability. Tasks: Identify clusters or patterns in the co-occurrence of two fields; '
                              'find the most and least common combinations. Query pattern: Are there any clusters with '
                              'respect to <E> counts of <F1:n> and <F2:n>?',
               'name': 'vis_050_displays_the_count_of_entities_for',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a heatmap showing entity counts for each combination of two nominal fields. '
                              'Design: Rect marks with color encoding and text labels. Text color adapts to background '
                              'intensity. Both axes limited to 30 or fewer categories. Tasks: Identify clusters and '
                              'patterns; compare counts across combinations; find correlations between two fields. '
                              'Query pattern: Make a heatmap of <E> <F1:n> and <F2:n>.',
               'name': 'vis_051_creates_a_heatmap_showing_entity_counts',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Displays the average of a quantitative field for each combination of two nominal fields '
                              'as a heatmap. Design: Uses three fields: a quantitative measure aggregated by average, '
                              'and two nominal axes. Color encodes the aggregate value. Requires overlapping fields '
                              'across all three. Tasks: Identify patterns in the average value across two categorical '
                              'dimensions; find combinations with extreme values. Query pattern: What is the average '
                              '<F1:q> for each <F2:n> and <F3:n>?',
               'name': 'vis_052_displays_the_average_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field3': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2', 'field3'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Plots two quantitative fields as a scatterplot with points colored by a nominal field '
                              'to reveal group-level clusters. Design: Adds color encoding to a standard scatterplot '
                              'to separate groups visually. Limited to fewer than 8 color categories for perceptual '
                              'clarity. Tasks: Identify clusters that separate by group; assess whether the '
                              'relationship between two quantitative fields differs across groups. Query pattern: Are '
                              'there clusters of <E> <F1:q> and <F2:q> values across different <F3:n> groups?',
               'name': 'vis_053_plots_two_quantitative_fields_as_a',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field3': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2', 'field3'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Shows the distribution of a quantitative field as a histogram with automatically '
                              'computed bins. Design: Uses binby to create equal-width bins. Rect marks span from bin '
                              'start to bin end on x, with count on y. Requires high cardinality (>250) to ensure '
                              'meaningful binning. Tasks: Characterize the shape of a distribution; identify modes, '
                              'skewness, and gaps. Query pattern: What is the distribution of <F:q>?',
               'name': 'vis_054_shows_the_distribution_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Creates a histogram of a quantitative field with automatically computed bins. Design: '
                              'Uses binby for automatic bin computation. Rect marks show bin ranges and counts. Lower '
                              'cardinality threshold (>5) than the question variant. Tasks: Characterize the shape of '
                              'a distribution; identify modes, skewness, and gaps. Query pattern: Make a histogram of '
                              '<F:q>?',
               'name': 'vis_055_creates_a_histogram_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Shows the distribution of a quantitative field as a smooth density curve (KDE) rendered '
                              'as an area chart. Design: Kernel density estimation produces a smooth curve. Area mark '
                              'fills below the density line. Used for moderate cardinality (50-250) where a smooth '
                              'estimate is more informative than binning. Tasks: Characterize the shape of a '
                              'distribution; identify modes and overall density patterns. Query pattern: What is the '
                              'distribution of <F:q>?',
               'name': 'vis_056_shows_the_distribution_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Shows the distribution of a quantitative field as individual points along a single '
                              'axis. Design: Point marks on a single quantitative x-axis. Best for small datasets (50 '
                              'or fewer values) where individual observations are meaningful and overplotting is '
                              'minimal. Tasks: Characterize the distribution; identify individual values, clusters, '
                              'and outliers. Query pattern: What is the distribution of <F:q>?',
               'name': 'vis_057_shows_the_distribution_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Compares the distribution of a quantitative field across categories using overlapping '
                              'density curves (KDE) with area and line marks. Design: Per-group KDE with '
                              'semi-transparent area fills and line outlines. Color encodes group identity. Limited to '
                              'fewer than 4 groups to avoid excessive overlap. Opacity set to 0.25 for layering. '
                              'Tasks: Compare distribution shapes across groups; identify shifts in central tendency '
                              'or spread. Query pattern: Is the distribution of <F1:q> similar for each <F2:n>?',
               'name': 'vis_058_compares_the_distribution_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Compares the distribution of a quantitative field across categories using dot strips, '
                              'with one row per category. Design: Points plotted on a quantitative x-axis with nominal '
                              'y-axis for group separation. Color reinforces group identity. Best for small datasets '
                              '(50 or fewer values per group). Tasks: Compare distributions across groups; identify '
                              'clusters and outliers within each group. Query pattern: Is the distribution of <F1:q> '
                              'similar for each <F2:n>?',
               'name': 'vis_059_compares_the_distribution_of_a_quantitative',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field1': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'},
                                             'field2': {'description': 'Field name (any type) from the entity.',
                                                        'type': 'string'}},
                              'required': ['entity', 'field1', 'field2'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Counts the number of records with a non-null value in a specified field, shown '
                              'alongside the total with a percentage bar. Design: Derives total count before '
                              'filtering, then filters to non-null and counts valid records. Percentage bar and 50%% '
                              'reference line provide visual context for data completeness. Tasks: Assess data '
                              'completeness for a field; determine how many records have valid values. Query pattern: '
                              'How many <E> records have a non-null <F:q|o|n>?',
               'name': 'vis_060_counts_the_number_of_records_with',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the percentage of records with a non-null value in a specified field, shown '
                              'with a percentage bar. Design: Same computation as non-null count but framed as a '
                              'percentage question. Percentage bar with 50%% reference line aids interpretation. '
                              'Tasks: Assess data completeness as a proportion; determine what fraction of records '
                              'have valid values. Query pattern: What percentage of <E> records have a non-null '
                              '<F:q|o|n>?',
               'name': 'vis_061_computes_the_percentage_of_records_with',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Counts the number of records with a null value in a specified field, shown alongside '
                              'the total with a percentage bar. Design: Derives null count as total minus valid count. '
                              'Percentage bar shows the null proportion with a 50%% reference line. Tasks: Assess data '
                              'quality; determine how many records are missing a value. Query pattern: How many <E> '
                              'records have a null <F:q|o|n>?',
               'name': 'vis_062_counts_the_number_of_records_with',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Computes the percentage of records with a null value in a specified field, shown with a '
                              'percentage bar. Design: Same computation as null count but framed as a percentage. '
                              'Percentage bar visualizes the null proportion. Tasks: Assess data quality as a '
                              'proportion; determine what fraction of records are missing a value. Query pattern: What '
                              'percentage of <E> records have a null <F:q|o|n>?',
               'name': 'vis_063_computes_the_percentage_of_records_with',
               'parameters': {'additionalProperties': False,
                              'properties': {'entity': {'description': 'The data entity (table) to visualize.',
                                                        'type': 'string'},
                                             'field': {'description': 'Field name (any type) from the entity.',
                                                       'type': 'string'}},
                              'required': ['entity', 'field'],
                              'type': 'object'}},
  'type': 'function'}]


# Dispatch: tool name -> (template_index, param_to_binding_map)
TOOL_DISPATCH = {'vis_000_counts_entities_grouped_by_a_nominal': (0, {'entity': 'E', 'field': 'F'}),
 'vis_001_counts_entities_grouped_by_a_nominal': (1, {'entity': 'E', 'field': 'F'}),
 'vis_002_creates_a_vertical_bar_chart_counting': (2, {'entity': 'E', 'field': 'F'}),
 'vis_003_creates_a_horizontal_bar_chart_counting': (3, {'entity': 'E', 'field': 'F'}),
 'vis_004_joins_two_entities_and_counts_records': (4, {'entity1': 'E1', 'entity2': 'E2', 'entity2_field': 'E2.F'}),
 'vis_005_joins_two_entities_and_counts_records': (5, {'entity1': 'E1', 'entity2': 'E2', 'entity2_field': 'E2.F'}),
 'vis_006_joins_two_entities_and_produces_a': (6,
                                               {'entity1': 'E1',
                                                'entity1_field': 'E1.F1',
                                                'entity2': 'E2',
                                                'entity2_field': 'E2.F2'}),
 'vis_007_joins_two_entities_and_produces_a': (7,
                                               {'entity1': 'E1',
                                                'entity1_field': 'E1.F1',
                                                'entity2': 'E2',
                                                'entity2_field': 'E2.F2'}),
 'vis_008_counts_entities_grouped_by_two_nominal': (8, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_009_counts_entities_grouped_by_two_nominal': (9, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_010_counts_entities_grouped_by_two_nominal': (10, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_011_counts_entities_grouped_by_two_nominal': (11, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_012_counts_entities_grouped_by_two_nominal': (12, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_013_shows_the_frequency_proportion_of_one': (13, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_014_shows_the_frequency_proportion_of_one': (14, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_015_computes_the_minimum_of_a_quantitative': (15, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_016_computes_the_minimum_of_a_quantitative': (16, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_017_computes_the_maximum_of_a_quantitative': (17, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_018_computes_the_maximum_of_a_quantitative': (18, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_019_computes_the_average_of_a_quantitative': (19, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_020_computes_the_average_of_a_quantitative': (20, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_021_computes_the_median_of_a_quantitative': (21, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_022_computes_the_median_of_a_quantitative': (22, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_023_computes_the_total_of_a_quantitative': (23, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_024_computes_the_total_of_a_quantitative': (24, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_025_plots_two_quantitative_fields_as_a': (25, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_026_creates_a_scatterplot_of_two_quantitative': (26, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_027_creates_a_vertical_stacked_bar_chart': (27, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_028_creates_a_horizontal_stacked_bar_chart': (28, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_029_creates_a_pie_chart_showing_the': (29, {'entity': 'E', 'field': 'F'}),
 'vis_030_creates_a_donut_chart_showing_the': (30, {'entity': 'E', 'field': 'F'}),
 'vis_031_counts_the_total_number_of_records': (31, {'entity': 'E'}),
 'vis_032_displays_the_raw_data_for_an': (32, {'entity': 'E'}),
 'vis_033_creates_a_table_displaying_the_raw': (33, {'entity': 'E'}),
 'vis_034_joins_two_related_entities_and_displays': (34, {'entity1': 'E1', 'entity2': 'E2'}),
 'vis_035_creates_a_table_that_joins_and': (35, {'entity1': 'E1', 'entity2': 'E2'}),
 'vis_036_finds_which_related_entity_record_has': (36, {'entity1': 'E1', 'entity2': 'E2'}),
 'vis_037_finds_the_record_with_the_largest': (37, {'entity': 'E', 'field': 'F'}),
 'vis_038_joins_two_entities_computes_the_maximum': (38, {'entity1': 'E1', 'entity1_field': 'E1.F', 'entity2': 'E2'}),
 'vis_039_finds_the_record_with_the_smallest': (39, {'entity': 'E', 'field': 'F'}),
 'vis_040_joins_two_entities_computes_the_minimum': (40, {'entity1': 'E1', 'entity1_field': 'E1.F', 'entity2': 'E2'}),
 'vis_041_sorts_entity_records_by_a_quantitative': (41, {'entity': 'E', 'field': 'F'}),
 'vis_042_computes_the_minimum_and_maximum_of': (42, {'entity': 'E', 'field': 'F'}),
 'vis_043_lists_all_distinct_values_of_a': (43, {'entity': 'E', 'field': 'F'}),
 'vis_044_computes_the_min_and_max_of': (44, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_045_finds_the_most_frequent_value_of': (45, {'entity': 'E', 'field': 'F'}),
 'vis_046_shows_the_cumulative_distribution_function_cdf': (46, {'entity': 'E', 'field': 'F'}),
 'vis_047_creates_a_cdf_cumulative_distribution_function': (47, {'entity': 'E', 'field': 'F'}),
 'vis_048_shows_the_cumulative_distribution_of_a': (48, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_049_creates_a_cdf_plot_with_a': (49, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_050_displays_the_count_of_entities_for': (50, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_051_creates_a_heatmap_showing_entity_counts': (51, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_052_displays_the_average_of_a_quantitative': (52,
                                                    {'entity': 'E', 'field1': 'F1', 'field2': 'F2', 'field3': 'F3'}),
 'vis_053_plots_two_quantitative_fields_as_a': (53, {'entity': 'E', 'field1': 'F1', 'field2': 'F2', 'field3': 'F3'}),
 'vis_054_shows_the_distribution_of_a_quantitative': (54, {'entity': 'E', 'field': 'F'}),
 'vis_055_creates_a_histogram_of_a_quantitative': (55, {'entity': 'E', 'field': 'F'}),
 'vis_056_shows_the_distribution_of_a_quantitative': (56, {'entity': 'E', 'field': 'F'}),
 'vis_057_shows_the_distribution_of_a_quantitative': (57, {'entity': 'E', 'field': 'F'}),
 'vis_058_compares_the_distribution_of_a_quantitative': (58, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_059_compares_the_distribution_of_a_quantitative': (59, {'entity': 'E', 'field1': 'F1', 'field2': 'F2'}),
 'vis_060_counts_the_number_of_records_with': (60, {'entity': 'E', 'field': 'F'}),
 'vis_061_computes_the_percentage_of_records_with': (61, {'entity': 'E', 'field': 'F'}),
 'vis_062_counts_the_number_of_records_with': (62, {'entity': 'E', 'field': 'F'}),
 'vis_063_computes_the_percentage_of_records_with': (63, {'entity': 'E', 'field': 'F'})}
