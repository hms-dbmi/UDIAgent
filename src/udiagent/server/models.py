"""Pydantic request/response models for the server."""

from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    tools: list[dict]


class YACCompletionRequest(BaseModel):
    messages: list[dict]
    dataSchema: str
    dataDomains: str


class YACBenchmarkCompletionRequest(BaseModel):
    messages: list[dict]
    dataSchema: str
    dataDomains: str
    orchestrator_choice: str | None = None


class UDICompletionRequest(BaseModel):
    model: str
    messages: list[dict]
