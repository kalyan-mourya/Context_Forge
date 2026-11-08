from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


class ChatInput(BaseModel):
    query: str
    last_3_turn: list[dict[str, Any]]

    @model_validator(mode="wrap")
    @classmethod
    def validate_chat_input(cls, values: Any, handler):
        validated = handler(values)

        if not isinstance(validated.query, str) or not validated.query.strip():
            raise ValueError("query must be a non-empty string")

        if len(validated.last_3_turn) > 6:
            raise ValueError("last_3_turn must contain fewer than 6 items")

        if len(validated.query.strip()) > 200:
            raise ValueError("query length exceeds 200 characters")

        if not all(isinstance(item, dict) for item in validated.last_3_turn):
            raise ValueError("Each item in last_3_turn must be a dictionary")

        return validated


class ChatOutput(BaseModel):
    response: str

    @model_validator(mode="before")
    @classmethod
    def check_response(cls, values: Any):
        if not isinstance(values, dict):
            raise ValueError("Chat output must be a dictionary.")

        response = values.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("response must be a non-empty string")

        return values


class HistoryItem(BaseModel):
    human: str
    ai: str
