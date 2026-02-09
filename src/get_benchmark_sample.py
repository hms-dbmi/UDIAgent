import json
import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ID = "HIDIVE/DQVis"

dataset = load_dataset(REPO_ID)
df = dataset['train'].to_pandas()

# fitler df to only include dataset_schema of hubmap_2025-05-05
df = df[df['dataset_schema'].isin(['hubmap_2025-05-05'])]
# TODO: get other dataset_schemas working

# group by the combination of query_template and spec_template
grouped = df.groupby(['query_template', 'spec_template', 'dataset_schema'])

# randomly sample one row from each group
# set random_state for reproducibility
df['_original_idx'] = df.index
sampled_df = grouped.apply(lambda x: x.sample(1, random_state=97930), include_groups=False).reset_index()
print(sampled_df)

# Map dataset_schema -> HuggingFace filename for data package (schema) files
data_package_hf_paths = {
    "4DN": "data_packages/4DN/datapackage_udi.json",
    "hubmap_2025-05-05": "data_packages/hubmap_2025-05-05/datapackage_udi.json",
    "MetabolomicsWorkbench": "data_packages/MetabolomicsWorkbench/C2M2_datapackage_udi.json",
    "MoTrPAC": "data_packages/MoTrPAC/C2M2_datapackage_udi.json",
    "SenNet": "data_packages/SenNet/C2M2_datapackage_udi.json",
}

# Download data packages from HuggingFace and load as JSON strings
data_package_map = {}
for key, hf_path in data_package_hf_paths.items():
    local_path = hf_hub_download(repo_id=REPO_ID, filename=hf_path, repo_type="dataset")
    with open(local_path, 'r') as f:
        data_package_map[key] = json.dumps(json.load(f))

# Load data domain files from local data/data_domains/ directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_domains')
data_domain_paths = {
    "4DN": os.path.join(DATA_DIR, "4DN_domains.json"),
    "hubmap_2025-05-05": os.path.join(DATA_DIR, "hubmap_domains.json"),
    "MetabolomicsWorkbench": os.path.join(DATA_DIR, "MetabolomicsWorkbench_domains.json"),
    "MoTrPAC": os.path.join(DATA_DIR, "MoTrPac_domains.json"),
    "SenNet": os.path.join(DATA_DIR, "SenNet_domains.json"),
}

data_domain_map = {}
for key, path in data_domain_paths.items():
    with open(path, 'r') as f:
        data_domain_map[key] = f.read()

test_case_list = []
for index, row in sampled_df.iterrows():
    test_case = {
        "input": {
            "model": "agenticx/UDI-VIS-Beta-v2-Llama-3.1-8B",
            "messages": [
                {
                    "role": "user",
                    "content": row['query']
                }
            ],
            "dqvis_index": int(row['_original_idx']),
            "dataSchema": data_package_map.get(row['dataset_schema'], ""),
            "dataDomains":  data_domain_map.get(row['dataset_schema'], ""),
        },
        "expected": {
            "tool_calls": [
                {
                    "name": 'RenderVisualization',
                    "arguments": {
                        "spec": row['spec']
                    }
                }
            ],
            "orchestrator_choice": 'render-visualization'
        },
    }
    test_case_list.append(test_case)

# save to json file
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_dqvis.json')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(test_case_list, f, indent=4)
print(f"Saved {len(test_case_list)} test cases to {output_path}")
