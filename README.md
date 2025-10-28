# The backend code for YAC

To run this code I use [vllm](https://docs.vllm.ai/en/latest/cli/index.html#serve) to run the main finetuned model:

`vllm serve HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B --port 8080 --host 127.0.0.1`

To run the entrypoint into the multi-agent system I run a simple python api.

`fastapi dev ./src/udi_api.py`


This is the endpoint that is called by the YAC frontend. The `udi_api.py` script makes calls to openai and the finetuned model running with vllm.

The multi-agent system currently makes calls to open ai and requires an open api key. This must be set in a .env file.

`OPEN_API_KEY=your-key-goes-here`
