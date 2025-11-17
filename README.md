# The backend code for YAC

To run this code I use [vllm](https://docs.vllm.ai/en/latest/cli/index.html#serve) to run the main finetuned model:

`vllm serve HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B --port 8080 --host 127.0.0.1`

To run the entrypoint into the multi-agent system I run a simple python api.

`fastapi dev ./src/udi_api.py`


This is the endpoint that is called by the YAC frontend. The `udi_api.py` script makes calls to openai and the finetuned model running with vllm.


### set environment variables

`OPEN_API_KEY=your-key-goes-here`
The multi-agent system currently makes calls to open ai and requires an open api key. This must be set in a .env file.



`JWT_SECRET_KEY=your-key-goes-here`
TODO DESCRIPTION

`UDI_MODEL_NAME=your-key-goes-here`
E.G. HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B


# Set Environment Variables

| Item | Command / Value | Description |
|---|---|---|
| OPEN_API_KEY | `OPEN_API_KEY=your-key-goes-here` | Required. The multi-agent system currently maeks calls to open ai. |
| JWT_SECRET_KEY | `JWT_SECRET_KEY=your-key-goes-here` | Required. Secret key for JWT generation. |
| UDI_MODEL_NAME | `UDI_MODEL_NAME=HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B` | Required. Path to local or public model name, depending on how the model is served via vllm. |
