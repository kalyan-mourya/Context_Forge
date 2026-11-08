from __future__ import annotations

from typing import Any

import yaml
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .generation.prompt_templates import get_template
from .settings import PROMPT_TEMPLATES_PATH


with open(PROMPT_TEMPLATES_PATH, "r", encoding="utf-8") as template_file:
    PROMPT_CONFIG = yaml.safe_load(template_file) or {}


TRANSLATE_PROMPT = get_template("translator")["translator"]
HISTORY_AWARE_PROMPT = PROMPT_CONFIG["context_aware_prompt"]["prompt"]
DETAILS_AGENT_PROMPT = PROMPT_CONFIG["details_agent_prompt"]["prompt"]
TRANSLATE_TO_HINDI = get_template("translator_english_hindi")["translator_english_hindi"]


class ChatManager:
    def __init__(self, llm: Any, vector_db: Any) -> None:
        self.llm = llm
        self.vector_db = vector_db
        self.fallback_mode = llm is None or vector_db is None
        self.rag_chain = None if self.fallback_mode else self._create_rag_chain()

    def _fallback_response(self, query: str, chat_history: list[dict] | None = None) -> str:
        chat_history = chat_history or []
        recent_context = " ".join(
            item.get("content", "")
            for item in chat_history[-3:]
            if isinstance(item, dict) and item.get("content")
        )

        if recent_context:
            return (
                "Demo mode is active because the AI backend is not configured. "
                f"Recent context: {recent_context}. Query: {query}"
            )

        return (
            "Demo mode is active because the AI backend is not configured. "
            f"Please set NVIDIA_LLM_API to enable the full RAG experience. Query: {query}"
        )

    def _create_rag_chain(self):
        return create_retrieval_chain(
            self.history_aware_retriever_chain(),
            self.q_a_chain(),
        )

    def history_aware_retriever_chain(self):
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", HISTORY_AWARE_PROMPT),
                ("assistant", "Understood."),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )

        return create_history_aware_retriever(
            self.llm,
            self.vector_db,
            contextualize_q_prompt,
        )

    def q_a_chain(self):
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", DETAILS_AGENT_PROMPT),
                ("assistant", "Understood."),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )

        return create_stuff_documents_chain(self.llm, qa_prompt)

    def _format_chat_history(self, chat_history: list[dict] | None) -> list:
        if not chat_history:
            return []

        formatted_history: list[Any] = []
        for message in chat_history:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "user":
                formatted_history.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted_history.append(AIMessage(content=content))

        merged_history: list[Any] = []
        for message in formatted_history:
            if merged_history and type(message) is type(merged_history[-1]):
                merged_history[-1].content += " " + message.content
            else:
                merged_history.append(message)

        return merged_history

    async def run(self, query: str, chat_history: list[dict] | None) -> str:
        if self.fallback_mode:
            return self._fallback_response(query, chat_history)

        formatted_history = self._format_chat_history(chat_history)
        result = self.rag_chain.invoke({
            "input": query,
            "chat_history": formatted_history,
        })

        return result["answer"].strip()
