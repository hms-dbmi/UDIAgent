import argparse
import json
import os
from datasets import load_dataset, load_from_disk
from huggingface_hub import hf_hub_download

REPO_ID = "HIDIVE/DQVis"
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'data_domains')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'benchmark_dqvis.json')

# Map dataset_schema -> HuggingFace filename for data package (schema) files
DATA_PACKAGE_HF_PATHS = {
    "4DN": "data_packages/4DN/datapackage_udi.json",
    "hubmap_2025-05-05": "data_packages/hubmap_2025-05-05/datapackage_udi.json",
    "MetabolomicsWorkbench": "data_packages/MetabolomicsWorkbench/C2M2_datapackage_udi.json",
    "MoTrPAC": "data_packages/MoTrPAC/C2M2_datapackage_udi.json",
    "SenNet": "data_packages/SenNet/C2M2_datapackage_udi.json",
}

DATA_DOMAIN_PATHS = {
    "4DN": os.path.join(DATA_DIR, "4DN_domains.json"),
    "hubmap_2025-05-05": os.path.join(DATA_DIR, "hubmap_domains.json"),
    "MetabolomicsWorkbench": os.path.join(DATA_DIR, "MetabolomicsWorkbench_domains.json"),
    "MoTrPAC": os.path.join(DATA_DIR, "MoTrPac_domains.json"),
    "SenNet": os.path.join(DATA_DIR, "SenNet_domains.json"),
}


def load_data_packages():
    data_package_map = {}
    for key, hf_path in DATA_PACKAGE_HF_PATHS.items():
        local_path = hf_hub_download(repo_id=REPO_ID, filename=hf_path, repo_type="dataset")
        with open(local_path, 'r') as f:
            data_package_map[key] = json.dumps(json.load(f))
    return data_package_map


def load_data_domains():
    data_domain_map = {}
    for key, path in DATA_DOMAIN_PATHS.items():
        with open(path, 'r') as f:
            data_domain_map[key] = f.read()
    return data_domain_map


def build_test_case(query, spec, dqvis_index, dataset_schema, data_package_map, data_domain_map):
    return {
        "input": {
            "model": "agenticx/UDI-VIS-Beta-v2-Llama-3.1-8B",
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "dqvis_index": dqvis_index,
            "dataSchema": data_package_map.get(dataset_schema, ""),
            "dataDomains": data_domain_map.get(dataset_schema, ""),
        },
        "expected": {
            "tool_calls": [
                {
                    "name": "RenderVisualization",
                    "arguments": {
                        "spec": spec
                    }
                }
            ],
            "orchestrator_choice": "render-visualization"
        },
    }


def sample_from_dqvis(data_package_map, data_domain_map):
    """Sample one row per (query_template, spec_template, dataset_schema) group from DQVis."""
    dataset = load_dataset(REPO_ID)
    df = dataset['train'].to_pandas()

    # filter df to only include dataset_schema of hubmap_2025-05-05
    df = df[df['dataset_schema'].isin(['hubmap_2025-05-05'])]
    # TODO: get other dataset_schemas working

    # group by the combination of query_template and spec_template
    grouped = df.groupby(['query_template', 'spec_template', 'dataset_schema'])

    # randomly sample one row from each group
    df['_original_idx'] = df.index
    sampled_df = grouped.apply(lambda x: x.sample(1, random_state=97930), include_groups=False).reset_index()
    print(sampled_df)

    test_case_list = []
    for _, row in sampled_df.iterrows():
        test_case_list.append(build_test_case(
            query=row['query'],
            spec=row['spec'],
            dqvis_index=int(row['_original_idx']),
            dataset_schema=row['dataset_schema'],
            data_package_map=data_package_map,
            data_domain_map=data_domain_map,
        ))
    return test_case_list


def from_training_test(training_data_path, indices, data_package_map, data_domain_map):
    """Build benchmark cases from specific test indices in dqvis_training_full."""
    data = load_from_disk(training_data_path)
    dqvis = load_dataset(REPO_ID)['train']

    test_case_list = []
    for i in indices:
        test_row = data['test'][i]
        dqvis_index = test_row['original_dqvis_index']
        dqvis_row = dqvis[dqvis_index]

        test_case_list.append(build_test_case(
            query=dqvis_row['query'],
            spec=dqvis_row['spec'],
            dqvis_index=dqvis_index,
            dataset_schema=dqvis_row['dataset_schema'],
            data_package_map=data_package_map,
            data_domain_map=data_domain_map,
        ))
    return test_case_list


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark test cases from DQVis data")
    parser.add_argument('--from-test', metavar='PATH',
                        help='Path to dqvis_training_full dataset. Use with --indices to select specific test rows.')
    parser.add_argument('--indices', type=int, nargs='+',
                        help='Test set indices to include (used with --from-test)')
    parser.add_argument('-o', '--output', default=OUTPUT_PATH,
                        help=f'Output path (default: {OUTPUT_PATH})')
    args = parser.parse_args()

    data_package_map = load_data_packages()
    data_domain_map = load_data_domains()

    if args.from_test:
        if not args.indices:
            parser.error('--indices is required when using --from-test')
        test_case_list = from_training_test(args.from_test, args.indices, data_package_map, data_domain_map)
    else:
        test_case_list = sample_from_dqvis(data_package_map, data_domain_map)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(test_case_list, f, indent=4)
    print(f"Saved {len(test_case_list)} test cases to {args.output}")


if __name__ == '__main__':
    main()
