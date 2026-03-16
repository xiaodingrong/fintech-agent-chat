import os
from openai import OpenAI

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =====================================================
# Helper: build conversation context
# =====================================================

def build_contextual_prompt(question, conversation_history):
    """
    Convert conversation history into context so the agent
    can resolve follow-up references like 'that' or 'the two'.
    """

    history_lines = []

    for msg in conversation_history[-6:]:  # up to 3 exchanges
        role = msg["role"].capitalize()
        history_lines.append(f"{role}: {msg['content']}")

    history_text = "\n".join(history_lines)

    prompt = f"""
You are a financial assistant that answers questions about companies and markets.

Use the conversation history to understand follow-up questions.
Resolve references such as:
- "that"
- "it"
- "the two"
- "them"

Conversation history:
{history_text}

Current question:
{question}
"""

    return prompt


# =====================================================
# SINGLE AGENT
# =====================================================

def run_single_agent(question, model_name, conversation_history):

    prompt = build_contextual_prompt(question, conversation_history)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a financial analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return resp.choices[0].message.content


# =====================================================
# MULTI AGENT
# =====================================================

def run_multi_agent(question, model_name, conversation_history):

    prompt = build_contextual_prompt(question, conversation_history)

    # Agent 1 — retrieval
    retrieval = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You retrieve financial data and relevant facts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    ).choices[0].message.content


    # Agent 2 — analysis
    analysis = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You analyze financial metrics and comparisons."},
            {"role": "user", "content": f"""
User question:
{question}

Retrieved information:
{retrieval}

Analyze the information and explain financial insights.
"""}
        ],
        temperature=0
    ).choices[0].message.content


    # Agent 3 — synthesis
    final = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You produce a clear final financial answer."},
            {"role": "user", "content": f"""
User question:
{question}

Retrieved info:
{retrieval}

Analysis:
{analysis}

Produce the final answer clearly.
"""}
        ],
        temperature=0
    ).choices[0].message.content


    return final