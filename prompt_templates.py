from __future__ import annotations

import os
from typing import Any

import yaml
from langchain_core.prompts import PromptTemplate


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROMPT_TEMPLATE_PATH = os.path.join(ROOT_PATH, "prompt_templates.yaml")


def load_prompt_config() -> dict[str, Any]:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as template_file:
        config = yaml.safe_load(template_file) or {}
    if not isinstance(config, dict):
        raise ValueError("Prompt template config must be a dictionary.")
    return config


PROMPT_CONFIG = load_prompt_config()


def get_template(name: str) -> dict[str, PromptTemplate]:
    """Return a named prompt template from the YAML configuration."""
    template_config = PROMPT_CONFIG.get(name)
    if template_config is None:
        raise ValueError(f"Unknown template: {name}")

    prompt_text = template_config.get("prompt")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError(f"Template '{name}' is missing a valid prompt string.")

    return {name: PromptTemplate.from_template(prompt_text)}

