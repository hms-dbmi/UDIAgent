"""
This file is for benchmarking the performance of the UDI Agent.
At a high level it will open a benchmark.json file that includes a list of inputs and expected outputs.
It will then run the UDI Agent on each input and compare the output to the expected output
and record the types of errors made.

It also includes the option to run with different information ablated, such as field descriptions.
"""

from datetime import datetime
import json
import requests
import argparse
import sys
import uuid
from jsonschema import validate, ValidationError
import copy
import os

PORT = 8007
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULT_FILENAME = "./out/" + timestamp + "/benchmark_results.json"
ANALYSIS_FILENAME = "./out/" + timestamp + "/benchmark_analysis.json"

run_id = str(uuid.uuid4())


def run_benchmark(benchmark_file, no_orchestrator=False):
    with open(benchmark_file, "r") as f:
        benchmark_data = json.load(f)

    results = collect_results(benchmark_data, no_orchestrator=no_orchestrator)
    analysis = analyze_results(results)
    return


def save_data_to_file(data, path):
    # ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data successfully saved to {path}")
    except Exception as e:
        print(f"Failed to save data to {path}: {e}")


def collect_results(benchmark_data, no_orchestrator=False):
    # get free text description of benchmark run from user input
    description = input("Enter a description for this benchmark run: ")

    for index, item in enumerate(benchmark_data):
        print("Processing item:", index + 1, "of", len(benchmark_data))
        item_input, expected = item["input"], item["expected"]
        output = fetch_agent_output(
            item_input, expected, no_orchestrator=no_orchestrator
        )
        item["output"] = output

    benchmark_results = {
        "metadata": {
            "run_id": run_id,
            "timestamp": timestamp,
            "description": description,
            "no_orchestrator": no_orchestrator,
        },
        "results": benchmark_data,
    }

    save_data_to_file(benchmark_results, RESULT_FILENAME)
    return benchmark_results


def fetch_agent_output(input, expected, no_orchestrator=False):
    # server = f"http://localhost/v1"
    server = f"http://127.0.0.1:{PORT}/v1"
    try:
        payload = copy.deepcopy(input)
        if no_orchestrator:
            payload["orchestrator_choice"] = expected["orchestrator_choice"]

        data = json.dumps(payload)

        response = requests.post(
            f"{server}/yac/benchmark",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer fake_token",
            },
            data=data,
        )

        if not response.ok:
            raise Exception(
                f"HTTP error! status: {response.status_code}, message: {response.text}"
            )

        data = response.json()
        return data

    except Exception as e:
        print("Error querying LLM:", e)
        return None


def analyze_results(results_data):
    for item in results_data["results"]:
        input, expected, output = item["input"], item["expected"], item["output"]
        data_domains = input.get("dataDomains", "[]")
        data_domains = json.loads(data_domains)
        rubric_results = check_rubric(expected, output, data_domains)
        item["rubric"] = rubric_results
        item["score"] = calculate_item_score(rubric_results)

    overall_scores = calculate_overall_scores(results_data["results"])

    analysis = {
        "metadata": results_data["metadata"],
        "scores": overall_scores,
        "results": results_data["results"],
    }

    save_data_to_file(
        analysis,
        ANALYSIS_FILENAME,
    )
    return analysis


def update_rubric(rubric, key, expected, output, group, points, pass_value=None):
    if pass_value is None:
        pass_value = expected == output
    rubric[key] = {
        "expected": expected,
        "output": output,
        "group": group,
        "points": points,
        "pass": pass_value,
    }
    return pass_value


def check_rubric(expected, output, data_domains):
    rubric = {}

    # orchestrator makes correct decision (filter/vis/both)
    output_orchestrator_choice = output.get("orchestrator_choice", None)
    expected_orchestrator_choice = expected.get("orchestrator_choice", None)
    update_rubric(
        rubric,
        "orchestrator_choice",
        expected_orchestrator_choice,
        output_orchestrator_choice,
        "orchestrator",
        100,
    )

    # FILTER RUBRIC
    if (
        expected_orchestrator_choice in ["get-subset-of-data", "both"]
    ) and output_orchestrator_choice in ["get-subset-of-data", "both"]:
        check_filter_rubric(rubric, expected, output, data_domains)

    # VIS RUBRIC
    if (
        expected_orchestrator_choice in ["render-visualization", "both"]
    ) and output_orchestrator_choice in ["render-visualization", "both"]:
        check_vis_rubric(rubric, expected, output, data_domains)

    return rubric


def check_filter_rubric(rubric, expected, output, data_domains):
    expected_tool_calls = expected.get("tool_calls", [])
    output_tool_calls = output.get("tool_calls", [])

    expected_filter_call_args = [
        call for call in expected_tool_calls if call["name"] == "FilterData"
    ][0]["arguments"]
    output_filter_call_args = [
        call for call in output_tool_calls if call["name"] == "FilterData"
    ][0]["arguments"]

    # correct type of filter (point vs range): 40 points
    expected_filter_type = expected_filter_call_args["filter"]["filterType"]
    output_filter_type = output_filter_call_args["filter"]["filterType"]
    update_rubric(
        rubric, "filter_type", expected_filter_type, output_filter_type, "filter", 40
    )

    # correct entity: 20 points
    output_filter_entity = output_filter_call_args["entity"]
    correct_entity = update_rubric(
        rubric,
        "filter_entity_correct",
        expected_filter_call_args["entity"],
        output_filter_entity,
        "filter",
        20,
    )

    # valid entity (it exists in the data schema): 5 points
    if not correct_entity:
        valid_entity_options = {x["entity"] for x in data_domains}
        passed = output_filter_entity in valid_entity_options
        update_rubric(
            rubric,
            "filter_entity_valid",
            json.dumps(list(valid_entity_options)),
            output_filter_entity,
            "filter",
            5,
            passed,
        )

    # correct field: 20 points
    output_filter_field = output_filter_call_args["field"]
    correct_field = update_rubric(
        rubric,
        "filter_field_correct",
        expected_filter_call_args["field"],
        output_filter_field,
        "filter",
        20,
    )

    # valid field (it exists in the data schema): 5 points
    if not correct_field:
        valid_field_options = {x["field"] for x in data_domains}
        passed = output_filter_field in valid_field_options
        update_rubric(
            rubric,
            "filter_field_valid",
            json.dumps(list(valid_field_options)),
            output_filter_field,
            "filter",
            5,
            passed,
        )

    # interval correct (for range filters): 20 points
    if expected_filter_type == "interval":
        expected_interval = expected_filter_call_args["filter"]["intervalRange"]
        output_interval = output_filter_call_args["filter"]["intervalRange"]
        update_rubric(
            rubric,
            "filter_interval_min",
            expected_interval["min"],
            output_interval["min"],
            "filter",
            10,
        )
        update_rubric(
            rubric,
            "filter_interval_max",
            expected_interval["max"],
            output_interval["max"],
            "filter",
            10,
        )

    # point correct (for point filters): 20 points
    point_values_correct = False
    if expected_filter_type == "point":
        expected_points = expected_filter_call_args["filter"]["pointValues"]
        output_points = output_filter_call_args["filter"]["pointValues"]
        point_values_correct = set(expected_points) == set(output_points)
        update_rubric(
            rubric,
            "filter_point_value",
            json.dumps(list(expected_points)),
            json.dumps(list(output_points)),
            "filter",
            20,
            point_values_correct,
        )

    # point valid (for point filters), the point values exist in the field domain: 5 points
    if expected_filter_type == "point" and not point_values_correct:
        valid_points = [
            x["domain"]["values"]
            for x in data_domains
            if x["entity"] == output_filter_entity and x["field"] == output_filter_field
        ][0]
        passed = all(point in valid_points for point in output_points)
        update_rubric(
            rubric,
            "filter_point_value_valid",
            json.dumps(list(valid_points)),
            json.dumps(list(output_points)),
            "filter",
            5,
            passed,
        )

    return rubric


def check_vis_rubric(rubric, expected, output, data_domains):
    expected_tool_calls = expected.get("tool_calls", [])
    output_tool_calls = output.get("tool_calls", [])

    expected_vis_spec = [
        call for call in expected_tool_calls if call["name"] == "RenderVisualization"
    ][0]["arguments"]["spec"]
    expected_vis_spec_json = json.loads(expected_vis_spec)
    output_vis_spec = [
        call for call in output_tool_calls if call["name"] == "RenderVisualization"
    ][0]["arguments"]["spec"]

    # generates specification that is a valid json
    try:
        output_vis_spec_json = json.loads(output_vis_spec)
        valid_json = True
    except:
        valid_json = False

    if not valid_json:
        # worthless, stop here, update rubric with failed test. (only include passed test if the vis spec isn't 100% correct)
        update_rubric(
            rubric,
            "vis_spec_valid_json",
            expected_vis_spec,
            output_vis_spec,
            "vis",
            10,
            valid_json,
        )
        return rubric

    # exact match of vis spec: 100 points
    spec_match = expected_vis_spec_json == output_vis_spec_json
    update_rubric(
        rubric,
        "vis_spec_exact_match",
        expected_vis_spec,
        output_vis_spec,
        "vis",
        100,
        spec_match,
    )

    if spec_match:
        # perfect, stop here
        return rubric

    # valid json but not exact match: 10 points
    update_rubric(
        rubric,
        "vis_spec_valid_json",
        expected_vis_spec,
        output_vis_spec,
        "vis",
        10,
        valid_json,
    )

    # Generates specification that adheres to to udi grammar: 10 points
    f = open("./src/UDIGrammarSchema.json", "r")
    udi_grammar_dict = json.load(f)
    f.close()
    try:
        valid_udi_spec = True
        validate(instance=output_vis_spec_json, schema=udi_grammar_dict)
    except ValidationError as e:
        valid_udi_spec = False
    update_rubric(
        rubric,
        "vis_spec_adhere_grammar",
        expected_vis_spec,
        output_vis_spec,
        "vis",
        10,
        valid_udi_spec,
    )

    # the source part of the spec is correct: 25 points
    expected_source = expected_vis_spec_json.get("source", None)
    output_source = output_vis_spec_json.get("source", None)
    update_rubric(
        rubric, "vis_spec_source_correct", expected_source, output_source, "vis", 25
    )

    # the transformation part of the spec is correct: 25 points
    expected_transformation = expected_vis_spec_json.get("transformation", None)
    output_transformation = output_vis_spec_json.get("transformation", None)
    update_rubric(
        rubric,
        "vis_spec_transformation_correct",
        expected_transformation,
        output_transformation,
        "vis",
        25,
    )

    # the representation part of the spec is correct: 25 points
    expected_representation = expected_vis_spec_json.get("representation", None)
    output_representation = output_vis_spec_json.get("representation", None)
    update_rubric(
        rubric,
        "vis_spec_representation_correct",
        expected_representation,
        output_representation,
        "vis",
        25,
    )

    return rubric


def calculate_item_score(rubric):
    """
    For each rubric group calculate the total points.
    The group score is the points divided by 100.
    The overall score is the average of the group scores.

    :param rubric: a dictionary of rubric items. Each item has a group (string), points (int), and pass (bool).
    :return: a dictionary of group results (score and total points) and an overall score.
    """
    scores = {}
    group_scores = {}
    for item in rubric.values():
        group = item["group"]
        points = item["points"]
        passed = item["pass"]

        if group not in group_scores:
            group_scores[group] = {"points": 0}

        if passed:
            group_scores[group]["points"] += points

    overall_score = 0
    for group, data in group_scores.items():
        group_score = data["points"] / 100
        group_scores[group]["score"] = group_score
        overall_score += group_score

    overall_score = overall_score / len(group_scores) if group_scores else 0
    scores["groups"] = group_scores
    scores["overall_score"] = overall_score

    return scores


def calculate_overall_scores(results_data):
    """
    Calculate overall scores of results_data. It will include an average of all overall_score values.
    And the count of items with a score of 1.0 out of the total number of items.

    :param results_data: List of results. Each result will have a "score" field with "overall_score" and "groups" with group scores. Each group will have a "points" and "score"
    """

    score_total = 0
    correct_count = 0
    group_aggregates = {}

    for item in results_data:
        item_score = item.get("score", {})
        overall_score = item_score.get("overall_score", 0)
        score_total += overall_score
        if overall_score == 1.0:
            correct_count += 1

        groups = item_score.get("groups", {})
        for group_name, group_data in groups.items():
            if group_name not in group_aggregates:
                group_aggregates[group_name] = {
                    "score_total": 0,
                    "correct_count": 0,
                    "count": 0,
                }

            group_aggregate = group_aggregates[group_name]

            group_aggregate["score_total"] += group_data.get("score", 0)
            if group_data.get("score", 0) == 1.0:
                group_aggregate["correct_count"] += 1
            group_aggregate["count"] += 1

    overall_score = score_total / len(results_data) if results_data else 0
    percent_correct = correct_count / len(results_data) if results_data else 0

    # Calculate group aggregates
    for group_name, group_data in group_aggregates.items():
        group_data["overall_score"] = (
            group_data["score_total"] / group_data["count"]
            if group_data["count"]
            else 0
        )
        group_data["percent_correct"] = (
            group_data["correct_count"] / group_data["count"]
            if group_data["count"]
            else 0
        )

        del group_data["score_total"]
        group_data["total_count"] = group_data["count"]
        del group_data["count"]

    return {
        "overall_score": overall_score,
        "correct_percent": percent_correct,
        "correct_count": correct_count,
        "total_count": len(results_data),
        "group_aggregates": group_aggregates,
    }


if __name__ == "__main__":
    # run_benchmark('./data/benchmark.json')
    parser = argparse.ArgumentParser(description="Run UDI Agent benchmark or analysis.")
    parser.add_argument(
        "-p",
        "--path",
        default="./data/benchmark.json",
        help="Path to benchmark JSON file (default: ./data/benchmark.json)",
    )
    parser.add_argument(
        "-a",
        "--action",
        choices=["full", "collect", "analyze"],
        default="full",
        help="Action to perform: 'full' (run benchmark then analyze), 'collect' (run benchmark only), or 'analyze' (run analysis only).",
    )

    # add option to bypass orchestrator step.
    parser.add_argument(
        "--no-orchestrator",
        action="store_true",
        help="If set, the benchmark will bypass the orchestrator step and directly call the correct tools.",
    )

    # parser.add_argument(
    #     "--out",
    #     default="./data/benchmark_results.json",
    #     help="Output path to write collected results (defaults: ./data/benchmark_results.json)."
    # )

    args = parser.parse_args()

    if args.action == "full":
        # run_benchmark will run collection and analysis internally
        run_benchmark(args.path, args.no_orchestrator)
        sys.exit(0)

    if args.action == "collect":
        try:
            with open(args.path, "r") as f:
                benchmark_data = json.load(f)
        except Exception as e:
            print(f"Failed to read benchmark file {args.path}: {e}")
            sys.exit(1)
        collect_results(benchmark_data, args.no_orchestrator)

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
