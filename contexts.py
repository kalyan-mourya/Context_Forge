import re
from langchain.prompts import PromptTemplate

affirmative_pattern = r"\b(?:ok|okay|okk|yes|yeah|yup|yah|yeh|ya|sure|fine|cool|alright|done|ji|haan|haan ji|yes please|sure thing|certainly|absolutely|of course|ofc|got it|noted|understood|i see|makes sense|okay then|okay done|yuppie|yesss|yeh bro|let’s go|theek hai|sahi hai|chalo theek|done bhai|haan bhai|ji haan|i want to|i’d like to|sounds good|i’m interested|let’s do it|go ahead)\b"


def is_context_continue(current_query, memory, chain):
    past_query = memory.load_memory_variables({})["history"]
    response = chain.run({
        "context": past_query,
        "query": current_query,
    })
    return response.strip()


def recommendation_agent(past_recommendations, past_conversations, chain, context):
    prompt = PromptTemplate.from_template("""
    You are an academic instructor assistant.
    Use the provided course context and research material to guide the learner.
    Recommend the most relevant concept, summary, or follow-up study direction.

    Past recommendations: {past_recommendations}
    Conversations: {past_conversations}
    Context: {context}
    """)
    chain.prompt = prompt
    response = chain.run({
        "conversations": past_conversations,
        "context": context,
        "past_recommendations": past_recommendations,
    })
    return response


class dialogue_optimization:
    def __init__(self, user_response, bot_response):
        self.user_response = user_response
        self.bot_response = bot_response

    def is_affirmative(self):
        return bool(re.search(affirmative_pattern, self.user_response.lower()))


def get_main_prompt_template():
    return PromptTemplate.from_template("""
    You are an academic instructor and lecturer assistant.
    Your role is to explain course concepts, summarize lecture notes, interpret research papers,
    compare academic ideas, and guide students with clear, accurate explanations.

    Ground your answer in the retrieved context and your academic knowledge.
    If the user asks about something unsupported or uncertain, respond with:
    "I'm not confident enough to answer that."

    Context: {context}
    Query: {query}
    Answer:
    """)



