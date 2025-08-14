from openai import OpenAI
from jinja2 import Template
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# import os

# Use multiprocess backend for workers
# os.environ["VLLM_WORKER_MULTIPROC"] = "1"

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



class UDIAgent:
    """UDIAgent for requesting UDI grammar given a prompt."""

    def __init__(self,
                 model_name: str,
                 vllm_server_url = None,
                 vllm_server_port = None,
                ):
        self.model_name = model_name
        if vllm_server_port is not None and vllm_server_url is not None:
            self.vllm_server_url = vllm_server_url
            self.vllm_server_port = vllm_server_port
            self.init_model_connection()
        # else:
        #     self.init_internal_models()

    def init_model_connection(self):
        base_url = f"{self.vllm_server_url}:{self.vllm_server_port}/v1"
        self.model = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )
    
    # def init_internal_models(self):
    #     self.llm = LLM(
    #         model=self.model_name,
    #         gpu_memory_utilization=0.85,
    #         # dtype="auto",                 # what serve typically auto-picks on A100
    #         # max_model_len=4096,               # ↓ KV cache pressure; raise later if needed
    #         # max_num_seqs=8,                  # cap concurrent sequences/batch size
    #         # swap_space=0,                     # GiB CPU RAM to spill KV blocks
    #         # enforce_eager=False,                # helps fragmentation on some stacks
    #         # kv_cache_dtype="fp8",           # if your vLLM build+model support it, huge win
    #         )

    # def basic_test(prompt):
    #     params = SamplingParams(
    #         temperature=0.0,
    #         n=3)
    #     response = self.llm.generate(prompt, sampling_params=params)
    #     return response

    def chat_completions(self, messages: list[dict]):
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # max_tokens=40960,
            max_tokens=120_000,
            temperature=0.0,
            # top_p=1.0,
        )
        return response

    def completions_guided_choice(self, messages: list[dict], tools: list[dict], choices: list[str]):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        chat_template = Template(tokenizer.chat_template)

        prompt = chat_template.render(
            messages=messages, tools=tools, add_generation_prompt=True
        )

        # todo, add prompt engineering here?
        response = self.model.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=100,
            # temperature=0.0,
            # n=3,
            # logprobs=3,
            extra_body={
                "guided_choice": choices,
            }

        )
        return response

    def completions_guided_json(self, messages: list[dict], tools: list[dict], json_schema: str, n=1):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        chat_template = Template(tokenizer.chat_template)

        prompt = chat_template.render(
        messages=messages, tools=tools, add_generation_prompt=True)

        # todo, add prompt engineering here?
        response = self.model.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=16_384,
            temperature=0.7,
            n=n,
            extra_body={
                "guided_json": json_schema,
            }
        )
        return response

    # def test_completions_with_more_params(self):
    #     tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    #     chat_template = Template(tokenizer.chat_template)

    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "What is the address of the white house?"},
    #     ]
    #     tools = []

    #     print(f"Messages: {messages}")

    #     # Debugging, remove all the tool_calls from the messages
    #     # TODO: try reformatting as strings.
    #     for message in messages:
    #         if 'tool_calls' in message:
    #             del message['tool_calls']

    #     prompt = chat_template.render(
    #         messages=messages, tools=tools, add_generation_prompt=True)


    #     print(f"Prompt: {prompt}")

    #     guided_json = '''
    # {
    #   "$id": "https://example.com/address.schema.json",
    #   "$schema": "https://json-schema.org/draft/2020-12/schema",
    #   "description": "An address similar to http://microformats.org/wiki/h-card",
    #   "type": "object",
    #   "properties": {
    #     "postOfficeBox": {
    #       "type": "string"
    #     },
    #     "extendedAddress": {
    #       "type": "string"
    #     },
    #     "streetAddress": {
    #       "type": "string"
    #     },
    #     "thingy77": {
    #       "type": "string"
    #     },
    #     "region": {
    #       "type": "string"
    #     },
    #     "postalCode": {
    #       "type": "string"
    #     },
    #     "countryName": {
    #       "type": "string"
    #     }
    #   },
    #   "required": [ "thingy77", "region", "countryName" ],
    #   "dependentRequired": {
    #     "postOfficeBox": [ "streetAddress" ],
    #     "extendedAddress": [ "streetAddress" ]
    #   }
    # }
    # '''
    #     response = self.model.completions.create(
    #         model=self.model_name,
    #         prompt=prompt,
    #         max_tokens=16_384,
    #         # n=3,
    #         # logprobs=3,
    #         temperature=0.7,
    #         extra_body={
    #             "guided_json": guided_json,
    #             # "use_beam_search": True
    #         }
    #         # sampling_params= {
    #         #     'use_beam_search':True,
    #         # }
    #         # best_of=3,
    #         # top_p=1.0,
    #         # use_beam_search=True,
    #         # guided_json=guided_json
    #     )
    #     return response


    def completions(self, messages: list[dict], tools: list[dict]):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        chat_template = Template(tokenizer.chat_template)

        print(f"Messages: {messages}")

        # Debugging, remove all the tool_calls from the messages
        # TODO: try reformatting as strings.
        for message in messages:
            if 'tool_calls' in message:
                del message['tool_calls']

        prompt = chat_template.render(
            messages=messages, tools=tools, add_generation_prompt=True)


        print(f"Prompt: {prompt}")

        response = self.model.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=16_384,
            temperature=0.0,
            # top_p=1.0,
        )
        return response