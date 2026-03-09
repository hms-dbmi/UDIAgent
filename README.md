# The backend code for YAC

To run this code I use [vllm](https://docs.vllm.ai/en/latest/cli/index.html#serve) to run the main finetuned model:

`vllm serve HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B --port 8080 --host 127.0.0.1`

To run the entrypoint into the multi-agent system I run a simple python api.

`fastapi run ./src/udi_api.py`

This is the endpoint that is called by the YAC frontend. The `udi_api.py` script makes calls to openai and the finetuned model running with vllm.

### set environment variables

# Set Environment Variables

| Item                | Command / Value                                        | Description                                                                                                                                                                                          |
| ------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OPEN_API_KEY        | `OPEN_API_KEY=your-key-goes-here`                      | Optional. The multi-agent system currently makes calls to open ai. If it is not provided as an environment variable, it must be provided in the API calls made by the front-end.                     |
| JWT_SECRET_KEY      | `JWT_SECRET_KEY=your-key-goes-here`                    | Required. Secret key for JWT generation.                                                                                                                                                             |
| UDI_MODEL_NAME      | `UDI_MODEL_NAME=HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B`   | Required. Path to local or public model name, depending on how the model is served via vllm. Note, this is the model name for the fine-tuned visualization generation model, not a foundation model. |
| VLLM_SERVER_URL     | `VLLM_SERVER_URL=http://localhost`                     | Optional. Hostname of the vllm server. Defaults to `http://localhost`.                                                                                                                               |
| VLLM_SERVER_PORT    | `VLLM_SERVER_PORT=55001`                               | Optional. Port of the vllm server. Defaults to `55001`.                                                                                                                                              |
| LANGFUSE_SECRET_KEY | `LANGFUSE_SECRET_KEY=sk-your-key-goes-here`            | Optional, used for integration with LangFuse observability/tracing.                                                                                                                                  |
| LANGFUSE_PUBLIC_KEY | `LANGFUSE_PUBLIC_KEY=pk-your-key-goes-here`            | Optional, used for integration with LangFuse observability/tracing.                                                                                                                                  |
| LANGFUSE_BASE_URL   | `LANGFUSE_BASE_URL=https://your-langfuse-instance.com` | Optional, used for integration with LangFuse observability/tracing.                                                                                                                                  |

# benchmarking process

## Step 0: Start the API server

Start the UDI API server in the background on port 8007, then wait a few seconds for it to be ready:

```bash
uv run fastapi dev ./src/udi_api.py --port 8007 &
```

## Step 1: Run tiny benchmark (1 example)

Run the benchmark on the tiny dataset to validate the pipeline works end-to-end:

```bash
uv run python ./src/benchmark.py --no-orchestrator --path ./data/benchmark_dqvis/tiny.jsonl
```

Check the output in the latest `./out/` timestamped directory. Review both `benchmark_results.json` and `benchmark_analysis.json`.

## Step 2: Run small benchmark (100 examples)

Once tiny passes cleanly, run the small benchmark:

```bash
uv run python ./src/benchmark.py --no-orchestrator  --path ./data/benchmark_dqvis/small.jsonl --workers 5
```

If a run fails partway through, it can be resumed with:

```bash
uv run python ./src/benchmark.py --path ./data/benchmark_dqvis/small.jsonl --workers 5 --resume ./out/<TIMESTAMP>/benchmark_results.json
```
