'''
This file is for benchmarking the performance of the UDI Agent. 
At a high level it will open a benchmark.json file that includes a list of inputs and expected outputs.
It will then run the UDI Agent on each input and compare the output to the expected output
and record the types of errors made.

It also includes the option to run with different information ablated, such as field descriptions.
'''
import json
import requests
import argparse
import sys
from jsonschema import validate, ValidationError

PORT = 8000
RESULT_FILENAME = "./out/benchmark_results.json"
ANALYSIS_FILENAME = "./out/benchmark_analysis.json"

def run_benchmark(benchmark_file):
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)

    results = collect_results(benchmark_data)
    analysis = analyze_results(results)
    return

def save_data_to_file(data, path):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data successfully saved to {path}")
    except Exception as e:
        print(f"Failed to save data to {path}: {e}")

def collect_results(benchmark_data):
    for index, item in enumerate(benchmark_data):
        print("Processing item:", index + 1, "of", len(benchmark_data))
        input, expected = item["input"], item["expected"]
        output = fetch_agent_output(input)
        item["output"] = output
    save_data_to_file(benchmark_data, RESULT_FILENAME)
    return benchmark_data

def fetch_agent_output(input):
    # server = f"http://localhost/v1"
    server = f"http://127.0.0.1:{PORT}/v1"
    try:
        response = requests.post(
            f"{server}/yac/benchmark",
            headers={"Content-Type": "application/json"},
            data=json.dumps(input)
        )

        if not response.ok:
            raise Exception(f"HTTP error! status: {response.status_code}, message: {response.text}")

        data = response.json()
        return data

    except Exception as e:
        print("Error querying LLM:", e)
        return None
    return

def analyze_results(results_data):
    for item in results_data:
        input, expected, output = item["input"], item["expected"], item["output"]
        data_domains = input.get("dataDomains", "[]")
        data_domains = json.loads(data_domains)
        rubric_results = check_rubric(expected, output, data_domains)
        item["rubric"] = rubric_results
    save_data_to_file(results_data, ANALYSIS_FILENAME)
    return results_data

def update_rubric(rubric, key, expected, output, pass_value=None):
    if pass_value is None:
        pass_value = expected == output
    rubric[key] = {
        "expected": expected,
        "output": output,
        "pass": pass_value
    }

def check_rubric(expected, output, data_domains):
    rubric = {}

    # orchestrator makes correct decision (filter/vis/both)
    output_orchestrator_choice = output.get('orchestrator_choice', None)
    expected_orchestrator_choice = expected.get('orchestrator_choice', None)
    orchestrator_choice_correct = output_orchestrator_choice == expected_orchestrator_choice
    update_rubric(rubric, 'orchestrator_choice', expected_orchestrator_choice, output_orchestrator_choice)

    # FILTER RUBRIC
    if (expected_orchestrator_choice in ['get-subset-of-data', 'both']) and output_orchestrator_choice in ['get-subset-of-data', 'both']:
        check_filter_rubric(rubric, expected, output, data_domains)
      
    # VIS RUBRIC
    if (expected_orchestrator_choice in ['render-visualization', 'both']) and output_orchestrator_choice in ['render-visualization', 'both']:
        check_vis_rubric(rubric, expected, output, data_domains)

    return rubric

def check_filter_rubric(rubric, expected, output, data_domains):
    expected_tool_calls = expected.get("tool_calls", [])
    output_tool_calls = output.get("tool_calls", [])

    expected_filter_call_args = [call for call in expected_tool_calls if call['name'] == "FilterData"][0]['arguments']
    output_filter_call_args = [call for call in output_tool_calls if call['name'] == "FilterData"][0]['arguments']

    # correct type of filter (point vs range)
    expected_filter_type = expected_filter_call_args['filter']['filterType']
    output_filter_type = output_filter_call_args['filter']['filterType']
    update_rubric(rubric, 'filter_type', expected_filter_type, output_filter_type)

    # correct entity
    output_filter_entity = output_filter_call_args['entity']
    update_rubric(rubric, 'filter_entity_correct', expected_filter_call_args['entity'], output_filter_entity)

    # valid entity (it exists in the data schema)
    valid_entity_options = { x['entity'] for x in data_domains }
    passed = output_filter_entity in valid_entity_options
    update_rubric(rubric, 'filter_entity_valid', json.dumps(list(valid_entity_options)), output_filter_entity, passed)

    # correct field
    output_filter_field = output_filter_call_args['field']
    update_rubric(rubric, 'filter_field_correct', expected_filter_call_args['field'], output_filter_field)
    
    # valid field (it exists in the data schema)
    valid_field_options = { x['field'] for x in data_domains }
    passed = output_filter_field in valid_field_options
    update_rubric(rubric, 'filter_field_valid', json.dumps(list(valid_field_options)), output_filter_field, passed)
    
    # interval correct (for range filters)
    if expected_filter_type == "interval":
        expected_interval = expected_filter_call_args['filter']['intervalRange']
        output_interval = output_filter_call_args['filter']['intervalRange']
        update_rubric(rubric, 'filter_interval_min', expected_interval['min'], output_interval['min'])
        update_rubric(rubric, 'filter_interval_max', expected_interval['max'], output_interval['max'])

    # point correct (for point filters)
    if expected_filter_type == "point":
        expected_points = expected_filter_call_args['filter']['pointValues']
        output_points = output_filter_call_args['filter']['pointValues']
        passed = set(expected_points) == set(output_points)
        update_rubric(rubric, 'filter_point_value', json.dumps(list(expected_points)), json.dumps(list(output_points)), passed)

    # point valid (for point filters), the point values exist in the field domain
    if expected_filter_type == "point":
        valid_points = [x['domain']['values'] for x in data_domains if x['entity'] == output_filter_entity and x['field'] == output_filter_field][0]
        passed = all(point in valid_points for point in output_points)
        update_rubric(rubric, 'filter_point_value_valid', json.dumps(list(valid_points)), json.dumps(list(output_points)), passed)

    return rubric

def check_vis_rubric(rubric, expected, output, data_domains):
    expected_tool_calls = expected.get("tool_calls", [])
    output_tool_calls = output.get("tool_calls", [])

    expected_vis_spec = [call for call in expected_tool_calls if call['name'] == "RenderVisualization"][0]['arguments']["spec"]
    expected_vis_spec_json = json.loads(expected_vis_spec)
    output_vis_spec = [call for call in output_tool_calls if call['name'] == "RenderVisualization"][0]['arguments']["spec"]
    
    # generates specification that is a valid json
    try:
        output_vis_spec_json = json.loads(output_vis_spec)
        valid_json = True
    except:
        valid_json = False
    update_rubric(rubric, 'vis_spec_valid_json', expected_vis_spec, output_vis_spec, valid_json)

    if not valid_json:
        # worthless, stop here
        return rubric

    # exact match of vis spec
    spec_match = expected_vis_spec_json == output_vis_spec_json
    update_rubric(rubric, 'vis_spec_exact_match', expected_vis_spec, output_vis_spec, spec_match)

    if spec_match:
        # perfect, stop here
        return rubric

    # Generates specification that adheres to to udi grammar
    f = open('./src/UDIGrammarSchema.json', 'r')
    udi_grammar_dict = json.load(f)
    f.close()
    try:
        valid_udi_spec = True
        validate(instance=output_vis_spec_json, schema=udi_grammar_dict)
    except ValidationError as e:
        valid_udi_spec = False
    update_rubric(rubric, 'vis_spec_adhere_grammar', expected_vis_spec, output_vis_spec, valid_udi_spec)
    
    # TODO: ideas for more checks
    # selects valid entities and fields
    # selects correct entities and fields
    # check different parts of spec (source, transform, mark, encoding, etc.)
    return rubric

if __name__ == "__main__":
    # run_benchmark('./data/benchmark.json')
    parser = argparse.ArgumentParser(description="Run UDI Agent benchmark or analysis.")
    parser.add_argument(
        "-p", "--path",
        default="./data/benchmark.json",
        help="Path to benchmark JSON file (default: ./data/benchmark.json)"
    )
    parser.add_argument(
        "-a", "--action",
        choices=["full", "collect", "analyze"],
        default="full",
        help="Action to perform: 'full' (run benchmark then analyze), 'collect' (run benchmark only), or 'analyze' (run analysis only)."
    )
    # parser.add_argument(
    #     "--out",
    #     default="./data/benchmark_results.json",
    #     help="Output path to write collected results (defaults: ./data/benchmark_results.json)."
    # )

    args = parser.parse_args()

    if args.action == "full":
        # run_benchmark will run collection and analysis internally
        run_benchmark(args.path)
        sys.exit(0)

    if args.action == "collect":
        try:
            with open(args.path, "r") as f:
                benchmark_data = json.load(f)
        except Exception as e:
            print(f"Failed to read benchmark file {args.path}: {e}")
            sys.exit(1)
        collect_results(benchmark_data)

        sys.exit(0)

    if args.action == "analyze":
        try:
            with open(args.path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read benchmark file {args.path}: {e}")
            sys.exit(1)

        analyze_results(data)
        sys.exit(0)
