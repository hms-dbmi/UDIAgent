"""Pydantic request/response models for the server."""

from pydantic import BaseModel, Field


class YACCompletionRequest(BaseModel):
    messages: list[dict]
    dataSchema: str
    dataDomains: str
    currentFilters: list[dict] = Field(default_factory=list)


class YACBenchmarkCompletionRequest(BaseModel):
    messages: list[dict]
    dataSchema: str
    dataDomains: str
    currentFilters: list[dict] = Field(default_factory=list)
    orchestrator_choice: str | None = None
