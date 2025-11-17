import os
import json
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from udi_agent import UDIAgent
from fastapi.middleware.cors import CORSMiddleware
import copy


app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init agent
agent = UDIAgent(
    # model_name="agenticx/UDI-VIS-Beta-v0-Llama-3.1-8B",
    model_name="/n/netscratch/mzitnik_lab/Lab/dlange/data/vis-v2/data/hidive/UDI-VIS-Beta-v2-Llama-3.1-8B",
    gpt_model_name="gpt-4.1",
    vllm_server_url="http://localhost",
    vllm_server_port=8080
)

# @app.get("/test1")
# def basic_test():
#     prompt = "Generate a UDI grammar for a bar chart showing sales data."
#     response = agent.basic_test(prompt)
#     return {"response": response}


# @app.get("/testfancycompletions")
# def basic_test():
#     response = agent.test_completions_with_more_params()
#     return { "response": response }

@app.get("/")
def read_root():
    return {
        "service": "UDIAgent API",
        "status": "running",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API status and info"},
        ]
    }

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]

# match openai api endpoints
# https://platform.openai.com/docs/api-reference/chat
@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    response = agent.chat_completions(messages=request.messages)
    return { "response": response } 


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    tools: list[dict]

@app.post("/v1/completions")
def completions(request: CompletionRequest):
    response = agent.completions(messages=request.messages, tools=request.tools)
    return { "response": response }

class YACCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    dataSchema: str
    dataDomains: str


@app.post("/v1/yac/completions")
def yac_completions(request: YACCompletionRequest):
    split_tool_calls(request)
    calls_to_make = determine_function_calls(request)
    print('calls_to_make:', calls_to_make)
    tool_calls = []
    if calls_to_make == "both" or calls_to_make == "get-subset-of-data":
        tool_calls.extend(function_call_filter(request))
    if calls_to_make == "both" or calls_to_make == "render-visualization":
        tool_calls.append(function_call_render_visualization(request))
    print('tool_calls:', tool_calls)
    return tool_calls


@app.post("/v1/yac/benchmark")
def yac_benchmark(request: YACCompletionRequest):
    # copy of yac/completions, but also exports orchestrator choice for easier benchmarking
    split_tool_calls(request)
    calls_to_make = determine_function_calls(request)
    tool_calls = []
    if calls_to_make == "both" or calls_to_make == "get-subset-of-data":
        tool_calls.extend(function_call_filter(request))
    if calls_to_make == "both" or calls_to_make == "render-visualization":
        tool_calls.append(function_call_render_visualization(request))
    return { "tool_calls": tool_calls, "orchestrator_choice": calls_to_make }

@app.get("/v1/yac/benchmark_analysis")
def yac_benchmark_nalysis():
    RESULT_FILENAME = "./out/benchmark_analysis.json"
    if not os.path.exists(RESULT_FILENAME):
        return JSONResponse(
            content={"error": f"File {RESULT_FILENAME} not found."},
            status_code=404
        )
    
    with open(RESULT_FILENAME, 'r') as f:
        data = json.load(f)

    return JSONResponse(content=data)


def split_tool_calls(request: YACCompletionRequest):
    # for each message in the request if there are multiple tool calls, split them into separate messages.
    # Note: this was needed because jinja template used cannot handle multiple tool calls in a single message.
    new_messages = []
    for message in request.messages:
        if 'tool_calls' in message:
            tool_calls = message['tool_calls']
            if isinstance(tool_calls, list) and len(tool_calls) > 1:
                # split into multiple messages
                for i, tool_call in enumerate(tool_calls):
                    new_message = message.copy()
                    new_message['tool_calls'] = [tool_call]
                    new_messages.append(new_message)
        else:
            new_messages.append(message)

    request.messages = new_messages
    return


def strip_tool_calls(messages: list[dict]):
    # remove tool calls from messages
    for message in messages:
        if 'tool_calls' in message:
            del message['tool_calls']
    return messages

def determine_function_calls(request: YACCompletionRequest):
    choices = [ 'render-visualization', 'get-subset-of-data', 'both']
    # add some prompt engineering into the messages.
    messages = copy.deepcopy(request.messages)  # make a copy to avoid mutating the original
    strip_tool_calls(messages)
    messages.append({
        "role": "system",
        "content": """
        The user wants to investigate data. Based on the question determine if it can be answered with a visualization (respond 'render-visualization'), or if the question requires the data to be first filtered, then can be visualized ('both').
        Finally, if the user just asks for the data to be filtered respond with 'get-subset-of-data'.
        
        Assume that past tool calls in the history carry over to the current state:
            - visualizations rendered still appear to the user
            - data filter state still carries over
        Take this into account when deciding. For instance, if the user asks for existing views to be filtered you just need to call 'get-subset-of-data', no need to render the visualization again.
        Similarly, if the user asks for a new visualization, you do not need to filter the data again.
        """})
    response = agent.completions_guided_choice(
        messages=messages,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "determine_function_call",
                    "description": "Decompose the user request into multiple function calls. The choice is determined if the user's request requires filtering data, visualizing data, or both.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "choice": {
                                "type": "string",
                                "enum": choices,
                                "description": "The plan for answering the users's query."
                            }
                        },
                        "required": ["choice"]
                    }
                }
            }
        ],
        choices=choices)
    print('determine choice response:', response)
    return response
    # return 'render-visualization'  # TODO: testing, remove this later.
    # return 'both'  # TODO: testing, remove this later.
    # return 'get-subset-of-data'  # TODO: testing, remove this later.
    # return response.choices[0].text


def function_call_filter(request: YACCompletionRequest):
    messages = copy.deepcopy(request.messages)  # make a copy to avoid mutating the original
    strip_tool_calls(messages)
    interstitialMessage = {
        "role": "system",
        "content": f"You are a helpful assistant that will explore, and analyze datasets. The following defines the available dataset entities, fields, and their domains:\n{request.dataDomains}\nRight now you need to filter the data based on the users request."
    }
    messages.insert(len(messages) - 1, interstitialMessage)


# This schema is not as constrained as it could be:
#   ideally only one of intervalRange and pointValues would be populated
#   the filtertype would match which of them was populated.
# These are possible with json schema, but was having trouble getting chat gpt to
# to work well with anyof / oneof thingies.
    response = agent.gpt_completions_guided_json(
        messages=messages,
        json_schema = json.dumps(
            
            
            
            {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "properties": {
                        "filterType": { "enum": ["point", "interval"] },
                        "intervalRange": {
                            "type": "object",
                            "properties": {
                                "min": {
                                    "type": "number",
                                    "description": "The minimum for the filter."
                                },
                                "max": {
                                    "type": "number",
                                    "description": "The maximum for the filter."
                                }
                            },
                            "required": ["min", "max"],
                            "additionalProperties": False
                        },
                        "pointValues": {
                            "type": "array",
                            "items": { "type": "string" },
                            "minItems": 1,
                            "description": "The values to filter for categorical fields."
                        }
                    },
                    "required": ["filterType","intervalRange","pointValues"],
                    "additionalProperties": False
                    },
                    "entity": {
                        "type": "string",
                        "description": "The entity to filter based on the current dataset schema."
                    },
                    "field": {
                        "type": "string",
                        "description": "The field to filter. Must be a quantitative field from the selected entity."
                    }
                },
                "required": ["entity", "field", "filter"],
                "additionalProperties": False
            }

        



            # {
            #     "type": "object",
            #     "properties": {
            #         "filter": {
            #             "type": "object",
            #             "properties": {
            #                 "filterType": {"enum": ["point", "interval"]},
            #                 "range": {
            #                     "type": "object",
            #                     "properties": {
            #                         "min": {
            #                             "type": "number",
            #                             "description": "The minimum for the filter."
            #                         },
            #                         "max": {
            #                             "type": "number",
            #                             "description": "The maximum for the filter."
            #                         },
            #                     }.
            #                 },
            #                 "values": {
            #                     "type": "array",
            #                     "items": { "type": "string" },
            #                     "minItems": 1,
            #                     "uniqueItems": True,
            #                     "description": "The values to filter for categorical fields."
            #                 },
            #                 "required": ["filterType"],
            #                 "anyOf": [
            #                     {"required": ["values"]},
            #                     {"required": ["range"]}
            #                 ],
            #             }
            #         },
            #         "entity": {
            #             "type": "string",
            #             "description": "The entity to filter based on the current dataset schema."
            #         },
            #         "field": {
            #             "type": "string",
            #             "description": "The field to filter. Must be a quantitative field from the selected entity."
            #         }
            #     },
            #     "required": ["entity", "field", "filter"],
            #     "additionalProperties": False,
            # }
        ))
    print(response)
    tool_calls = [{"name": "FilterData", "arguments": args} for args in response]
    return tool_calls
    # return json.loads(response.choices[0].text)
    # filterArgs = response.choices[0].text
    # return {"name": "FilterData", "arguments": response}

def function_call_render_visualization(request: YACCompletionRequest):
    f = open('./src/UDIGrammarSchema.json', 'r')
    udi_grammar_dict = json.load(f)
    f.close()
    f = open('./src/UDIGrammarSchema_spec_string.json', 'r')
    udi_grammar_string = f.read()
    f.close()
    messages = copy.deepcopy(request.messages)  # make a copy to avoid mutating the original
    firstMessage = {
        "role": "system",
        "content": f"You are a helpful assistant that will explore, and analyze datasets with visualizations. The following defines the available datasets:\n{request.dataSchema}\nTypically, your actions will use the provided functions. You have access to the following functions."
    }
    messages.insert(0, firstMessage)
    response = agent.completions_guided_json(
        messages=messages,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "RenderVisualization",
                    "description": "Render a visualization with a provided visualization grammar of graphics style specification.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spec": udi_grammar_dict,
                        },
                        "required": ["spec"],
                    },
                },
            }
        ],
        json_schema=udi_grammar_string)
    return json.loads(response.choices[0].text)
    # return {"name": "RenderVisualization", "arguments": {"spec": spec}}

class UDICompletionRequest(BaseModel):
    model: str
    messages: list[dict]

@app.post("/v1/udi/completions")
def udi_completions(request: UDICompletionRequest):
    # Assumes the available tools and that a udi spec is requested as a response.
    tools = [{'name': 'RenderVisualization', 'description': 'Render a visualization with a provided visualization grammar of graphics style specification.', 'parameter': {'type': 'object', 'properties': {'spec': {'type': 'object', 'description': 'The UDI Grammar Specification for the visualization.', 'required': True, 'properties': {'$ref': '#/definitions/UDIGrammar', '$schema': 'http://json-schema.org/draft-07/schema#', 'definitions': {'AggregateFunction': {'additionalProperties': False, 'description': 'An aggregate function for summarizing data.', 'properties': {'field': {'description': 'The field to aggregate. Optional.', 'type': 'string'}, 'op': {'description': 'The operation to apply (e.g., count, sum, mean, etc.).', 'enum': ['count', 'sum', 'mean', 'min', 'max', 'median', 'frequency'], 'type': 'string'}}, 'required': ['op'], 'type': 'object'}, 'ArcEncodingOptions': {'description': 'Encoding options for arc marks.', 'enum': ['theta', 'theta2', 'radius', 'radius2', 'color'], 'type': 'string'}, 'ArcFieldMapping': {'$ref': '#/definitions/GenericFieldMapping%3CArcEncodingOptions%3E', 'description': 'A field mapping for arc marks.'}, 'ArcLayer': {'$ref': '#/definitions/GenericLayer%3C%22arc%22%2CArcMapping%3E', 'description': 'A layer for rendering arc marks.'}, 'ArcMapping': {'anyOf': [{'$ref': '#/definitions/ArcFieldMapping'}, {'$ref': '#/definitions/ArcValueMapping'}], 'description': 'The mapping for arc marks.'}, 'ArcValueMapping': {'$ref': '#/definitions/GenericValueMapping%3CArcEncodingOptions%3E', 'description': 'A value mapping for arc marks.'}, 'AreaEncodingOptions': {'description': 'Encoding options for area marks.', 'enum': ['x', 'y', 'y2', 'color', 'stroke', 'opacity'], 'type': 'string'}, 'AreaFieldMapping': {'$ref': '#/definitions/GenericFieldMapping%3CAreaEncodingOptions%3E', 'description': 'A field mapping for area marks.'}, 'AreaLayer': {'$ref': '#/definitions/GenericLayer%3C%22area%22%2CAreaMapping%3E', 'description': 'A layer for rendering area marks.'}, 'AreaMapping': {'anyOf': [{'$ref': '#/definitions/AreaFieldMapping'}, {'$ref': '#/definitions/AreaValueMapping'}], 'description': 'The mapping for area marks.'}, 'AreaValueMapping': {'$ref': '#/definitions/GenericValueMapping%3CAreaEncodingOptions%3E', 'description': 'A value mapping for area marks.'}, 'BarEncodingOptions': {'description': 'Encoding options for bar marks.', 'enum': ['x', 'x2', 'y', 'y2', 'xOffset', 'yOffset', 'color'], 'type': 'string'}, 'BarFieldMapping': {'$ref': '#/definitions/GenericFieldMapping%3CBarEncodingOptions%3E', 'description': 'A field mapping for bar marks.'}, 'BarLayer': {'$ref': '#/definitions/GenericLayer%3C%22bar%22%2CBarMapping%3E', 'description': 'A layer for rendering bar marks.'}, 'BarMapping': {'anyOf': [{'$ref': '#/definitions/BarFieldMapping'}, {'$ref': '#/definitions/BarValueMapping'}], 'description': 'The mapping for bar marks.'}, 'BarValueMapping': {'$ref': '#/definitions/GenericValueMapping%3CBarEncodingOptions%3E', 'description': 'A value mapping for bar marks.'}, 'BinBy': {'additionalProperties': False, 'description': 'Bins data into intervals for a specified field.', 'properties': {'binby': {'additionalProperties': False, 'description': 'Configuration for binning the data.', 'properties': {'bins': {'description': 'The number of bins to create. Optional.', 'type': 'number'}, 'field': {'description': 'The field to bin.', 'type': 'string'}, 'nice': {'description': 'Whether to use "nice" bin boundaries. Optional.', 'type': 'boolean'}, 'output': {'additionalProperties': False, 'description': 'Output field names for the bin start and end. Optional.', 'properties': {'bin_end': {'type': 'string'}, 'bin_start': {'type': 'string'}}, 'type': 'object'}}, 'required': ['field'], 'type': 'object'}, 'in': {'description': 'The name of the input table. If not specified, it assumes the output of the previous operation.', 'type': 'string'}, 'out': {'description': 'The name of the output table. If not specified, it overwrites the name of the previous table.', 'type': 'string'}}, 'required': ['binby'], 'type': 'object'}, 'DataSelection': {'additionalProperties': False, 'description': 'Configuration for data selection in visualizations or tables.', 'properties': {'fields': {'anyOf': [{'type': 'string'}, {'items': {'type': 'string'}, 'type': 'array'}], 'description': 'The fields selected from the data points. If not specified, all fields are selected.'}, 'how': {'anyOf': [{'$ref': '#/definitions/DataSelectionInterval'}, {'$ref': '#/definitions/DataSelectionPoint'}], 'description': 'How the data is selected in the visualization / table.'}, 'name': {'description': 'The name of the data selection.', 'type': 'string'}}, 'required': ['name', 'how'], 'type': 'object'}, 'DataSelectionInterval': {'additionalProperties': False, 'description': 'Configuration for interval-based data selection.', 'properties': {'on': {'description': "The axis or axes to apply the selection on ('x', 'y', or both ('xy')).", 'enum': ['x', 'y', 'xy'], 'type': 'string'}, 'type': {'const': 'interval', 'description': 'The type of selection (interval).', 'type': 'string'}}, 'required': ['type', 'on'], 'type': 'object'}, 'DataSelectionPoint': {'additionalProperties': False, 'description': 'Configuration for point-based data selection.', 'properties': {'type': {'const': 'point', 'description': 'The type of selection (point).', 'type': 'string'}}, 'required': ['type'], 'type': 'object'}, 'DataSource': {'additionalProperties': False, 'description': 'A single tabular data source. Currently, only CSV files are supported.', 'properties': {'name': {'description': 'The unique name of the data source.', 'type': 'string'}, 'source': {'description': 'The URL of the CSV file.', 'type': 'string'}}, 'required': ['name', 'source'], 'type': 'object'}, 'DataTransformation': {'anyOf': [{'$ref': '#/definitions/GroupBy'}, {'$ref': '#/definitions/BinBy'}, {'$ref': '#/definitions/RollUp'}, {'$ref': '#/definitions/Join'}, {'$ref': '#/definitions/OrderBy'}, {'$ref': '#/definitions/Derive'}, {'$ref': '#/definitions/Filter'}, {'$ref': '#/definitions/KDE'}], 'description': 'The possible data transformations. These include operations like grouping, filtering, joining, and more.'}, 'DataTypes': {'description': 'The possible data types for fields.', 'enum': ['quantitative', 'ordinal', 'nominal'], 'type': 'string'}, 'Derive': {'additionalProperties': False, 'description': 'Derives new fields based on expressions.', 'properties': {'derive': {'additionalProperties': {'$ref': '#/definitions/DeriveExpression'}, 'description': 'A mapping of output field names to derive expressions.', 'type': 'object'}, 'in': {'description': 'The name of the input table. If not specified, it assumes the output of the previous operation.', 'type': 'string'}, 'out': {'description': 'The name of the output table. If not specified, it overwrites the name of the previous table.', 'type': 'string'}}, 'required': ['derive'], 'type': 'object'}, 'DeriveExpression': {'anyOf': [{'type': 'string'}, {'$ref': '#/definitions/RollingDeriveExpression'}], 'description': 'A derive expression, which can be a string or a rolling derive expression.'}, 'Filter': {'additionalProperties': False, 'description': 'Filters data based on a specified condition or selection.', 'properties': {'filter': {'$ref': '#/definitions/FilterExpression', 'description': 'The filter condition or selection.'}, 'in': {'description': 'The name of the input table. If not specified, it assumes the output of the previous operation.', 'type': 'string'}, 'out': {'description': 'The name of the output table. If not specified, it overwrites the name of the previous table.', 'type': 'string'}}, 'required': ['filter'], 'type': 'object'}, 'FilterDataSelection': {'additionalProperties': False, 'description': 'A data selection used for filtering.', 'properties': {'match': {'description': "Specifies whether to use 'all' or 'any' of the selected data in a 1-to-many mapping. Default is 'any'.", 'enum': ['all', 'any'], 'type': 'string'}, 'name': {'description': 'The name of the selection.', 'type': 'string'}}, 'required': ['name'], 'type': 'object'}, 'FilterExpression': {'anyOf': [{'type': 'string'}, {'$ref': '#/definitions/FilterDataSelection'}], 'description': 'A filter expression, which can be a string or a data selection.'}, 'GenericFieldMapping<ArcEncodingOptions>': {'additionalProperties': False, 'description': 'A generic field mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/ArcEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'field': {'description': 'The field to map to the encoding.', 'type': 'string'}, 'type': {'$ref': '#/definitions/DataTypes', 'description': 'The data type of the field (e.g., quantitative, ordinal, nominal).'}}, 'required': ['encoding', 'field', 'type'], 'type': 'object'}, 'GenericFieldMapping<AreaEncodingOptions>': {'additionalProperties': False, 'description': 'A generic field mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/AreaEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'field': {'description': 'The field to map to the encoding.', 'type': 'string'}, 'type': {'$ref': '#/definitions/DataTypes', 'description': 'The data type of the field (e.g., quantitative, ordinal, nominal).'}}, 'required': ['encoding', 'field', 'type'], 'type': 'object'}, 'GenericFieldMapping<BarEncodingOptions>': {'additionalProperties': False, 'description': 'A generic field mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/BarEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'field': {'description': 'The field to map to the encoding.', 'type': 'string'}, 'type': {'$ref': '#/definitions/DataTypes', 'description': 'The data type of the field (e.g., quantitative, ordinal, nominal).'}}, 'required': ['encoding', 'field', 'type'], 'type': 'object'}, 'GenericFieldMapping<GeometryEncodingOptions>': {'additionalProperties': False, 'description': 'A generic field mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/GeometryEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'field': {'description': 'The field to map to the encoding.', 'type': 'string'}, 'type': {'$ref': '#/definitions/DataTypes', 'description': 'The data type of the field (e.g., quantitative, ordinal, nominal).'}}, 'required': ['encoding', 'field', 'type'], 'type': 'object'}, 'GenericFieldMapping<LineEncodingOptions>': {'additionalProperties': False, 'description': 'A generic field mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/LineEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'field': {'description': 'The field to map to the encoding.', 'type': 'string'}, 'type': {'$ref': '#/definitions/DataTypes', 'description': 'The data type of the field (e.g., quantitative, ordinal, nominal).'}}, 'required': ['encoding', 'field', 'type'], 'type': 'object'}, 'GenericFieldMapping<PointEncodingOptions>': {'additionalProperties': False, 'description': 'A generic field mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/PointEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'field': {'description': 'The field to map to the encoding.', 'type': 'string'}, 'type': {'$ref': '#/definitions/DataTypes', 'description': 'The data type of the field (e.g., quantitative, ordinal, nominal).'}}, 'required': ['encoding', 'field', 'type'], 'type': 'object'}, 'GenericFieldMapping<RectEncodingOptions>': {'additionalProperties': False, 'description': 'A generic field mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/RectEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'field': {'description': 'The field to map to the encoding.', 'type': 'string'}, 'type': {'$ref': '#/definitions/DataTypes', 'description': 'The data type of the field (e.g., quantitative, ordinal, nominal).'}}, 'required': ['encoding', 'field', 'type'], 'type': 'object'}, 'GenericFieldMapping<TextEncodingOptions>': {'additionalProperties': False, 'description': 'A generic field mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/TextEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'field': {'description': 'The field to map to the encoding.', 'type': 'string'}, 'type': {'$ref': '#/definitions/DataTypes', 'description': 'The data type of the field (e.g., quantitative, ordinal, nominal).'}}, 'required': ['encoding', 'field', 'type'], 'type': 'object'}, 'GenericLayer<"arc",ArcMapping>': {'additionalProperties': False, 'description': 'A generic layer for visualizing data.', 'properties': {'mapping': {'anyOf': [{'$ref': '#/definitions/ArcMapping'}, {'items': {'$ref': '#/definitions/ArcMapping'}, 'type': 'array'}], 'description': 'The mapping of data fields or values to visual encodings.'}, 'mark': {'const': 'arc', 'description': 'The type of mark used in the layer.', 'type': 'string'}, 'select': {'$ref': '#/definitions/DataSelection', 'description': 'The data selection configuration for the layer. Optional.'}}, 'required': ['mark', 'mapping'], 'type': 'object'}, 'GenericLayer<"area",AreaMapping>': {'additionalProperties': False, 'description': 'A generic layer for visualizing data.', 'properties': {'mapping': {'anyOf': [{'$ref': '#/definitions/AreaMapping'}, {'items': {'$ref': '#/definitions/AreaMapping'}, 'type': 'array'}], 'description': 'The mapping of data fields or values to visual encodings.'}, 'mark': {'const': 'area', 'description': 'The type of mark used in the layer.', 'type': 'string'}, 'select': {'$ref': '#/definitions/DataSelection', 'description': 'The data selection configuration for the layer. Optional.'}}, 'required': ['mark', 'mapping'], 'type': 'object'}, 'GenericLayer<"bar",BarMapping>': {'additionalProperties': False, 'description': 'A generic layer for visualizing data.', 'properties': {'mapping': {'anyOf': [{'$ref': '#/definitions/BarMapping'}, {'items': {'$ref': '#/definitions/BarMapping'}, 'type': 'array'}], 'description': 'The mapping of data fields or values to visual encodings.'}, 'mark': {'const': 'bar', 'description': 'The type of mark used in the layer.', 'type': 'string'}, 'select': {'$ref': '#/definitions/DataSelection', 'description': 'The data selection configuration for the layer. Optional.'}}, 'required': ['mark', 'mapping'], 'type': 'object'}, 'GenericLayer<"geometry",GeometryMapping>': {'additionalProperties': False, 'description': 'A generic layer for visualizing data.', 'properties': {'mapping': {'anyOf': [{'$ref': '#/definitions/GeometryMapping'}, {'items': {'$ref': '#/definitions/GeometryMapping'}, 'type': 'array'}], 'description': 'The mapping of data fields or values to visual encodings.'}, 'mark': {'const': 'geometry', 'description': 'The type of mark used in the layer.', 'type': 'string'}, 'select': {'$ref': '#/definitions/DataSelection', 'description': 'The data selection configuration for the layer. Optional.'}}, 'required': ['mark', 'mapping'], 'type': 'object'}, 'GenericLayer<"line",LineMapping>': {'additionalProperties': False, 'description': 'A generic layer for visualizing data.', 'properties': {'mapping': {'anyOf': [{'$ref': '#/definitions/LineMapping'}, {'items': {'$ref': '#/definitions/LineMapping'}, 'type': 'array'}], 'description': 'The mapping of data fields or values to visual encodings.'}, 'mark': {'const': 'line', 'description': 'The type of mark used in the layer.', 'type': 'string'}, 'select': {'$ref': '#/definitions/DataSelection', 'description': 'The data selection configuration for the layer. Optional.'}}, 'required': ['mark', 'mapping'], 'type': 'object'}, 'GenericLayer<"point",PointMapping>': {'additionalProperties': False, 'description': 'A generic layer for visualizing data.', 'properties': {'mapping': {'anyOf': [{'$ref': '#/definitions/PointMapping'}, {'items': {'$ref': '#/definitions/PointMapping'}, 'type': 'array'}], 'description': 'The mapping of data fields or values to visual encodings.'}, 'mark': {'const': 'point', 'description': 'The type of mark used in the layer.', 'type': 'string'}, 'select': {'$ref': '#/definitions/DataSelection', 'description': 'The data selection configuration for the layer. Optional.'}}, 'required': ['mark', 'mapping'], 'type': 'object'}, 'GenericLayer<"rect",RectMapping>': {'additionalProperties': False, 'description': 'A generic layer for visualizing data.', 'properties': {'mapping': {'anyOf': [{'$ref': '#/definitions/RectMapping'}, {'items': {'$ref': '#/definitions/RectMapping'}, 'type': 'array'}], 'description': 'The mapping of data fields or values to visual encodings.'}, 'mark': {'const': 'rect', 'description': 'The type of mark used in the layer.', 'type': 'string'}, 'select': {'$ref': '#/definitions/DataSelection', 'description': 'The data selection configuration for the layer. Optional.'}}, 'required': ['mark', 'mapping'], 'type': 'object'}, 'GenericLayer<"row",RowMapping>': {'additionalProperties': False, 'description': 'A generic layer for visualizing data.', 'properties': {'mapping': {'anyOf': [{'$ref': '#/definitions/RowMapping'}, {'items': {'$ref': '#/definitions/RowMapping'}, 'type': 'array'}], 'description': 'The mapping of data fields or values to visual encodings.'}, 'mark': {'const': 'row', 'description': 'The type of mark used in the layer.', 'type': 'string'}, 'select': {'$ref': '#/definitions/DataSelection', 'description': 'The data selection configuration for the layer. Optional.'}}, 'required': ['mark', 'mapping'], 'type': 'object'}, 'GenericLayer<"text",TextMapping>': {'additionalProperties': False, 'description': 'A generic layer for visualizing data.', 'properties': {'mapping': {'anyOf': [{'$ref': '#/definitions/TextMapping'}, {'items': {'$ref': '#/definitions/TextMapping'}, 'type': 'array'}], 'description': 'The mapping of data fields or values to visual encodings.'}, 'mark': {'const': 'text', 'description': 'The type of mark used in the layer.', 'type': 'string'}, 'select': {'$ref': '#/definitions/DataSelection', 'description': 'The data selection configuration for the layer. Optional.'}}, 'required': ['mark', 'mapping'], 'type': 'object'}, 'GenericValueMapping<ArcEncodingOptions>': {'additionalProperties': False, 'description': 'A generic value mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/ArcEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'value': {'description': 'The value to map to the encoding.', 'type': ['string', 'number']}}, 'required': ['encoding', 'value'], 'type': 'object'}, 'GenericValueMapping<AreaEncodingOptions>': {'additionalProperties': False, 'description': 'A generic value mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/AreaEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'value': {'description': 'The value to map to the encoding.', 'type': ['string', 'number']}}, 'required': ['encoding', 'value'], 'type': 'object'}, 'GenericValueMapping<BarEncodingOptions>': {'additionalProperties': False, 'description': 'A generic value mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/BarEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'value': {'description': 'The value to map to the encoding.', 'type': ['string', 'number']}}, 'required': ['encoding', 'value'], 'type': 'object'}, 'GenericValueMapping<GeometryEncodingOptions>': {'additionalProperties': False, 'description': 'A generic value mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/GeometryEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'value': {'description': 'The value to map to the encoding.', 'type': ['string', 'number']}}, 'required': ['encoding', 'value'], 'type': 'object'}, 'GenericValueMapping<LineEncodingOptions>': {'additionalProperties': False, 'description': 'A generic value mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/LineEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'value': {'description': 'The value to map to the encoding.', 'type': ['string', 'number']}}, 'required': ['encoding', 'value'], 'type': 'object'}, 'GenericValueMapping<PointEncodingOptions>': {'additionalProperties': False, 'description': 'A generic value mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/PointEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'value': {'description': 'The value to map to the encoding.', 'type': ['string', 'number']}}, 'required': ['encoding', 'value'], 'type': 'object'}, 'GenericValueMapping<RectEncodingOptions>': {'additionalProperties': False, 'description': 'A generic value mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/RectEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'value': {'description': 'The value to map to the encoding.', 'type': ['string', 'number']}}, 'required': ['encoding', 'value'], 'type': 'object'}, 'GenericValueMapping<TextEncodingOptions>': {'additionalProperties': False, 'description': 'A generic value mapping for visual encodings.', 'properties': {'encoding': {'$ref': '#/definitions/TextEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'value': {'description': 'The value to map to the encoding.', 'type': ['string', 'number']}}, 'required': ['encoding', 'value'], 'type': 'object'}, 'GeometryEncodingOptions': {'description': 'Encoding options for geometry marks.', 'enum': ['color', 'stroke', 'strokeWidth'], 'type': 'string'}, 'GeometryFieldMapping': {'$ref': '#/definitions/GenericFieldMapping%3CGeometryEncodingOptions%3E', 'description': 'A field mapping for geometry marks.'}, 'GeometryLayer': {'$ref': '#/definitions/GenericLayer%3C%22geometry%22%2CGeometryMapping%3E', 'description': 'A layer for rendering geometry marks.'}, 'GeometryMapping': {'anyOf': [{'$ref': '#/definitions/GeometryFieldMapping'}, {'$ref': '#/definitions/GeometryValueMapping'}], 'description': 'The mapping for geometry marks.'}, 'GeometryValueMapping': {'$ref': '#/definitions/GenericValueMapping%3CGeometryEncodingOptions%3E', 'description': 'A value mapping for geometry marks.'}, 'GroupBy': {'additionalProperties': False, 'description': 'Groups data by specified fields.', 'properties': {'groupby': {'anyOf': [{'type': 'string'}, {'items': {'type': 'string'}, 'type': 'array'}], 'description': 'The field(s) to group by.'}, 'in': {'description': 'The name of the input table. If not specified, it assumes the output of the previous operation.', 'type': 'string'}, 'out': {'description': 'The name of the output table. If not specified, it overwrites the name of the previous table.', 'type': 'string'}}, 'required': ['groupby'], 'type': 'object'}, 'Join': {'additionalProperties': False, 'description': 'Joins two tables based on a specified condition.', 'properties': {'in': {'description': 'The names of the two input tables to join.', 'items': {'type': 'string'}, 'maxItems': 2, 'minItems': 2, 'type': 'array'}, 'join': {'additionalProperties': False, 'description': 'Configuration for the join operation.', 'properties': {'on': {'anyOf': [{'type': 'string'}, {'items': {'type': 'string'}, 'maxItems': 2, 'minItems': 2, 'type': 'array'}], 'description': "The field(s) to join on. If one field is specified, it's assumed to be the same in both tables. If two fields are specified, the first field is from the first table and the second field is from the second table."}}, 'required': ['on'], 'type': 'object'}, 'out': {'description': 'The name of the output table. If not specified, it overwrites the name of the previous table.', 'type': 'string'}}, 'required': ['in', 'join'], 'type': 'object'}, 'KDE': {'additionalProperties': False, 'description': 'Applies Kernel Density Estimation (KDE) to a specified field.', 'properties': {'in': {'description': 'The name of the input table. If not specified, it assumes the output of the previous operation.', 'type': 'string'}, 'kde': {'additionalProperties': False, 'description': 'Configuration for the KDE operation.', 'properties': {'bandwidth': {'description': 'The bandwidth for the KDE. Optional.', 'type': 'number'}, 'field': {'description': 'The field to apply KDE to.', 'type': 'string'}, 'output': {'additionalProperties': False, 'description': 'Output field names for the KDE results. Optional.', 'properties': {'density': {'type': 'string'}, 'sample': {'type': 'string'}}, 'type': 'object'}, 'samples': {'description': 'The number of samples to generate. Optional.', 'type': 'number'}}, 'required': ['field'], 'type': 'object'}, 'out': {'description': 'The name of the output table. If not specified, it overwrites the name of the previous table.', 'type': 'string'}}, 'required': ['kde'], 'type': 'object'}, 'LineEncodingOptions': {'description': 'Encoding options for line marks.', 'enum': ['x', 'y', 'color'], 'type': 'string'}, 'LineFieldMapping': {'$ref': '#/definitions/GenericFieldMapping%3CLineEncodingOptions%3E', 'description': 'A field mapping for line marks.'}, 'LineLayer': {'$ref': '#/definitions/GenericLayer%3C%22line%22%2CLineMapping%3E', 'description': 'A layer for rendering line marks.'}, 'LineMapping': {'anyOf': [{'$ref': '#/definitions/LineFieldMapping'}, {'$ref': '#/definitions/LineValueMapping'}], 'description': 'The mapping for line marks.'}, 'LineValueMapping': {'$ref': '#/definitions/GenericValueMapping%3CLineEncodingOptions%3E', 'description': 'A value mapping for line marks.'}, 'OrderBy': {'additionalProperties': False, 'description': 'Orders data by a specified field or fields.', 'properties': {'in': {'description': 'The name of the input table. If not specified, it assumes the output of the previous operation.', 'type': 'string'}, 'orderby': {'description': 'The field to order by.', 'type': 'string'}, 'out': {'description': 'The name of the output table. If not specified, it overwrites the name of the previous table.', 'type': 'string'}}, 'required': ['orderby'], 'type': 'object'}, 'PointEncodingOptions': {'description': 'Encoding options for point marks.', 'enum': ['x', 'y', 'xOffset', 'yOffset', 'color', 'size', 'shape'], 'type': 'string'}, 'PointFieldMapping': {'$ref': '#/definitions/GenericFieldMapping%3CPointEncodingOptions%3E', 'description': 'A field mapping for point marks.'}, 'PointLayer': {'$ref': '#/definitions/GenericLayer%3C%22point%22%2CPointMapping%3E', 'description': 'A layer for rendering point marks.'}, 'PointMapping': {'anyOf': [{'$ref': '#/definitions/PointFieldMapping'}, {'$ref': '#/definitions/PointValueMapping'}], 'description': 'The mapping for point marks.'}, 'PointValueMapping': {'$ref': '#/definitions/GenericValueMapping%3CPointEncodingOptions%3E', 'description': 'A value mapping for point marks.'}, 'RectEncodingOptions': {'description': 'Encoding options for rect marks.', 'enum': ['x', 'x2', 'y', 'y2', 'color'], 'type': 'string'}, 'RectFieldMapping': {'$ref': '#/definitions/GenericFieldMapping%3CRectEncodingOptions%3E', 'description': 'A field mapping for rect marks.'}, 'RectLayer': {'$ref': '#/definitions/GenericLayer%3C%22rect%22%2CRectMapping%3E', 'description': 'A layer for rendering rect marks.'}, 'RectMapping': {'anyOf': [{'$ref': '#/definitions/RectFieldMapping'}, {'$ref': '#/definitions/RectValueMapping'}], 'description': 'The mapping for rect marks.'}, 'RectValueMapping': {'$ref': '#/definitions/GenericValueMapping%3CRectEncodingOptions%3E', 'description': 'A value mapping for rect marks.'}, 'Representation': {'anyOf': [{'$ref': '#/definitions/VisualizationLayer'}, {'$ref': '#/definitions/RowLayer'}], 'description': 'A visual representation of the data, such as a chart or table.'}, 'Representations': {'anyOf': [{'items': {'$ref': '#/definitions/VisualizationLayer'}, 'type': 'array'}, {'items': {'$ref': '#/definitions/RowLayer'}, 'type': 'array'}], 'description': 'A list of visual representations. Charts and tables cannot be intermixed.'}, 'RollUp': {'additionalProperties': False, 'description': 'Aggregates data by applying specified functions to groups.', 'properties': {'in': {'description': 'The name of the input table. If not specified, it assumes the output of the previous operation.', 'type': 'string'}, 'out': {'description': 'The name of the output table. If not specified, it overwrites the name of the previous table.', 'type': 'string'}, 'rollup': {'additionalProperties': {'$ref': '#/definitions/AggregateFunction'}, 'description': 'A mapping of output field names to aggregate functions.', 'type': 'object'}}, 'required': ['rollup'], 'type': 'object'}, 'RollingDeriveExpression': {'additionalProperties': False, 'description': 'A rolling derive expression for creating new fields based on a rolling window.', 'properties': {'rolling': {'additionalProperties': False, 'description': 'Configuration for the rolling derive expression.', 'properties': {'expression': {'description': 'The expression to apply.', 'type': 'string'}, 'window': {'description': 'The rolling window size. Optional.', 'items': {'type': 'number'}, 'maxItems': 2, 'minItems': 2, 'type': 'array'}}, 'required': ['expression'], 'type': 'object'}}, 'required': ['rolling'], 'type': 'object'}, 'RowEncodingOptions': {'description': 'Encoding options for row marks.', 'enum': ['text', 'x', 'y', 'xOffset', 'yOffset', 'color', 'size', 'shape'], 'type': 'string'}, 'RowLayer': {'$ref': '#/definitions/GenericLayer%3C%22row%22%2CRowMapping%3E', 'description': 'A layer for rendering row marks.'}, 'RowMapping': {'additionalProperties': False, 'description': 'The mapping for row marks.', 'properties': {'encoding': {'$ref': '#/definitions/RowEncodingOptions', 'description': 'The encoding type (e.g., x, y, color).'}, 'field': {'description': 'The field to map to the encoding.', 'type': 'string'}, 'mark': {'$ref': '#/definitions/RowMarkOptions', 'description': 'The type of mark used in the row layer.'}, 'type': {'$ref': '#/definitions/DataTypes', 'description': 'The data type of the field (e.g., quantitative, ordinal, nominal).'}}, 'required': ['encoding', 'field', 'mark', 'type'], 'type': 'object'}, 'RowMarkOptions': {'description': 'Mark options for row layers.', 'enum': ['select', 'text', 'geometry', 'point', 'bar', 'rect', 'line'], 'type': 'string'}, 'TextEncodingOptions': {'description': 'Encoding options for text marks.', 'enum': ['x', 'y', 'color', 'text', 'size', 'theta', 'radius'], 'type': 'string'}, 'TextFieldMapping': {'$ref': '#/definitions/GenericFieldMapping%3CTextEncodingOptions%3E', 'description': 'A field mapping for text marks.'}, 'TextLayer': {'$ref': '#/definitions/GenericLayer%3C%22text%22%2CTextMapping%3E', 'description': 'A layer for rendering text marks.'}, 'TextMapping': {'anyOf': [{'$ref': '#/definitions/TextFieldMapping'}, {'$ref': '#/definitions/TextValueMapping'}], 'description': 'The mapping for text marks.'}, 'TextValueMapping': {'$ref': '#/definitions/GenericValueMapping%3CTextEncodingOptions%3E', 'description': 'A value mapping for text marks.'}, 'UDIGrammar': {'additionalProperties': False, 'description': 'The Universal Discovery Interface (UDI) Grammar.', 'properties': {'representation': {'anyOf': [{'$ref': '#/definitions/Representation'}, {'$ref': '#/definitions/Representations'}], 'description': 'The visual representation of the data as either a visualization or table.'}, 'source': {'anyOf': [{'$ref': '#/definitions/DataSource'}, {'items': {'$ref': '#/definitions/DataSource'}, 'type': 'array'}], 'description': 'The data source or data sources. This can be a single CSV file or a list of CSV files.'}, 'transformation': {'description': 'The data transformations applied to the source data before displaying the data.', 'items': {'$ref': '#/definitions/DataTransformation'}, 'type': 'array'}}, 'required': ['source', 'representation'], 'type': 'object'}, 'VisualizationLayer': {'anyOf': [{'$ref': '#/definitions/LineLayer'}, {'$ref': '#/definitions/AreaLayer'}, {'$ref': '#/definitions/GeometryLayer'}, {'$ref': '#/definitions/RectLayer'}, {'$ref': '#/definitions/BarLayer'}, {'$ref': '#/definitions/PointLayer'}, {'$ref': '#/definitions/TextLayer'}, {'$ref': '#/definitions/ArcLayer'}], 'description': 'A visualization layer for rendering data. This can include various types of marks such as lines, bars, points, etc.'}}}}}}}]
    raw_result = completions(request=CompletionRequest(
        model=request.model,
        messages=request.messages,
        tools=tools
    ))

    print(f"Raw result: {raw_result}")

    output = raw_result.get('response').choices[0].text
    # should containt the text "[TOOL_CALLS][...]"
    # if "[TOOL_CALLS]" not in output:
    #     raise ValueError(f"Output does not contain tool calls: {output}")

    # remove the "[TOOL_CALLS]" part
    # tool_calls_str = output.split("[TOOL_CALLS]")[1].strip()
    tool_calls_str = output.strip()
    if tool_calls_str.startswith("<|python_tag|>"):
        tool_calls_str = output.split("<|python_tag|>")[1].strip()
    # parse the tool calls
    try:
        tool_calls = json.loads(tool_calls_str)
    except json.JSONDecodeError as e:
        print(json)
        raise ValueError(f"Failed to parse tool calls: {e}")
    
    # find the tool call with name "RenderVisualization"
    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]
        
    for tool_call in tool_calls:
        if tool_call.get("name") == "RenderVisualization":
            # return the arguments of the tool call
            args = tool_call.get("arguments", {})
            spec = args.get("spec", {})
            return spec

    raise ValueError("No tool call with name 'RenderVisualization' found in output")