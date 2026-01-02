import streamlit as st
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv


# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()
MODEL = "meta-llama/llama-3.2-3b-instruct" # or OpenRouter model
OPENROUTER_API_KEY = api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# -----------------------------
# MASTER PROMPT
# -----------------------------
MASTER_SYSTEM_PROMPT = """You are SIA, a warm, respectful, purpose-driven conversational agent for Sunbird SERVE.

Your role is to onboard volunteers through a single, natural WhatsApp conversation.

You must sound human, encouraging, and calm — never procedural or robotic.

Core principles:

- Start with purpose before asking for details.

- Convert intent → interest through clarity, not pressure.

- Never mention internal concepts like onboarding, registration, FSM, states, or selection.

- Keep messages short (1-2 liners), WhatsApp-friendly.

- Ask only one question at a time, based on the Current state that is passed to you.

- Be honest and transparent about non-negotiables.

- If a volunteer cannot proceed, exit gracefully and share the SERVE community link.

Non-negotiables:

- Eligibility (18+, device + internet, voluntary role) must be met.

- Phone number and email are required to proceed to classroom volunteering.

- If a volunteer refuses required information, do not persuade beyond one gentle explanation.

Tone:

- Warm, respectful, optimistic

- Never salesy or pushy

- Emojis are allowed but minimal

Context you will receive:

- Current state

- Known volunteer details (if any)

- Previous messages (summary)

- SERVE community link

Never invent facts.

Never assume consent.

Never store or repeat sensitive information unnecessarily.

You are guiding a human, not completing a form."""

STATE_PROMPTS = {

    "KNOWING_VOLUNTEER": """
You are SIA, the Sunbird SERVE onboarding guide.

Current state: KNOWING_VOLUNTEER.

Context:
- Start light and friendly.
- Learn a bit about the volunteer's background, motivation, or experience, teaching experience, working with children
- Beginners are welcome.

Goal:
Classify the user's reply and return intent + tone_reply.

Allowed intents:
- BACKGROUND_SHARED
- MOTIVATION_SHARED
- EXPERIENCE_SHARED
- NO_EXPERIENCE
- QUERY
- AMBIGUOUS
- STOP

Rules:
- NO_EXPERIENCE is acceptable.
- Be reassuring.
- Do NOT invent details.
- Do NOT ask more than 5 questions and do not repeat questions
- 

Please generate your response ONLY as JSON in this format:

{
  "text": "<friendly assistant reply to volunteer>",
  "intent": "<optional intent label>",
  "confidence": <optional confidence score between 0.0–1.0>
}

Do not add any extra text outside the JSON.

""",

    "WEEKLY_COMMITMENT": """
You are SIA, the Sunbird SERVE onboarding guide.

Current state: WEEKLY_COMMITMENT.

Context:
- Checking comfort with ~2 hours per week.

Allowed intents:
- TIME_YES
- TIME_MAYBE
- TIME_NO
- QUERY
- AMBIGUOUS
- STOP

Rules:
- Less than 2 hours → TIME_NO
- Hesitant → TIME_MAYBE
""",

    "ORIENTATION": """
You are SIA, the Sunbird SERVE onboarding guide.

Current state: ORIENTATION.

Context:
- Explain how sessions work (30–40 min, lesson plans, coordinator support).
- Then ask if this feels comfortable.

Allowed intents:
- OK
- NOT_OK
- QUERY
- STOP

Tone:
- Warm, reassuring.

"""
}

STATE_ORDER = ["KNOWING_VOLUNTEER", "WEEKLY_COMMITMENT", "ORIENTATION", "CLOSE"]

# -----------------------------
# QUESTIONS PER STATE
# -----------------------------
STATE_QUESTIONS = {
    "KNOWING_VOLUNTEER": [
        "Before we begin, tell me a little about yourself 😊",
        "What made you interested in volunteering with SERVE?",
        "Have you ever taught or helped someone learn before?"
    ],
    "WEEKLY_COMMITMENT": [
        "Would you be comfortable spending about 2 hours a week with students?"
    ],
    "ORIENTATION": [
        "Our sessions are 30–40 minutes long. You’ll get lesson plans, subject content, and full coordinator support.\n\nDoes this feel comfortable for you?"
    ]
}

# -----------------------------
# STREAMLIT STATE INIT
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "state_index" not in st.session_state:
    st.session_state.state_index = 0

if "question_index" not in st.session_state:
    st.session_state.question_index = 0

# -----------------------------
# HELPERS
# -----------------------------
def current_state():
    return STATE_ORDER[st.session_state.state_index]

def next_question():
    state = current_state()
    questions = STATE_QUESTIONS.get(state, [])
    idx = st.session_state.question_index
    if idx < len(questions):
        st.write("next question "+questions[idx])
        return questions[idx]
    return None

import re

def infer_intent_rule_based(text: str) -> str:
    """
    Infer high-level intent from free-form text.
    Always returns a valid intent string.
    """
    st.write("Intent classification text :" + text)
    if not text or not isinstance(text, str):
        return "AMBIGUOUS"

    t = text.lower().strip()

    # Stop / unsubscribe
    if any(x in t for x in [
        "stop", "unsubscribe", "exit", "leave", "quit", "cancel"
    ]):
        return "STOP"

    # Clear yes / confirmation
    if re.search(r"\b(yes|yeah|yep|sure|ok|okay|fine|i do|i am)\b", t):
        return "AFFIRM"

    # Clear no / rejection
    if re.search(r"\b(no|nope|not really|can't|cannot|won't|don’t|do not)\b", t):
        return "NEGATE"

    # Question / clarification
    if "?" in t or any(x in t for x in [
        "how", "what", "when", "where", "why", "can you", "is it", "do i"
    ]):
        return "QUERY"

    # Experience / background sharing
    if any(x in t for x in [
        "teacher", "teaching", "student", "engineer", "working", "experience",
        "background", "volunteer", "profession"
    ]):
        return "INFO"

    return "AMBIGUOUS"

def compute_confidence(text: str) -> float:
    """
    Returns a confidence score between 0.0 and 1.0
    based on clarity and decisiveness of the text.
    """

    if not text or not isinstance(text, str):
        return 0.2

    t = text.lower().strip()

    # Very short / vague
    if len(t) < 3:
        return 0.2

    # Strong confirmation
    if any(x in t for x in [
        "yes", "sure", "absolutely", "definitely", "i can", "i will"
    ]):
        return 0.9

    # Clear rejection
    if any(x in t for x in [
        "no", "can't", "cannot", "not possible"
    ]):
        return 0.9

    # Question → moderate confidence
    if "?" in t:
        return 0.6

    # Contains concrete details
    if any(x in t for x in [
        "years", "hours", "week", "experience", "background"
    ]):
        return 0.7

    # Default neutral response
    return 0.5

def call_llm(user_text):
    st.write("current state:"+ current_state())
    messages = [
        {"role": "system", "content": MASTER_SYSTEM_PROMPT},
        {"role": "system", "content": STATE_PROMPTS.get(current_state(), "")}
    ]

    for m in st.session_state.messages[-6:]:
        messages.append(m)

    messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4
    )
    llm_text = response.choices[0].message.content
    #st.write(response.choices[0].message.content)
    result = {
    "raw_text": llm_text,
    "intent": infer_intent_rule_based(user_text),
    "confidence": compute_confidence(user_text),
    "tone_reply": llm_text
    }
    st.write(result)
    return result

def advance_flow(intent, confidence=1.0):
    # Exit intents — do not advance
    if intent in ["STOP"]:
        st.session_state.state_index = len(STATE_ORDER) - 1  # CLOSE
        st.session_state.question_index = 0
        return

    # Repair intents — repeat question
    if intent in ["QUERY", "AMBIGUOUS"] or confidence < 0.4:
        # Do NOT advance question or state
        return

    # Progress intent — move to next question
    st.session_state.question_index += 1

    # If questions exhausted → move to next state
    questions = STATE_QUESTIONS.get(current_state(), [])
    if st.session_state.question_index >= len(questions):
        st.session_state.state_index += 1
        st.session_state.question_index = 0

# -----------------------------
# UI
# -----------------------------
st.title("SIA – SERVE Volunteer Assistant 🤝")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# First message
if len(st.session_state.messages) == 0:
    first_q = next_question()
    st.session_state.messages.append(
        {"role": "assistant", "content": first_q}
    )
    st.chat_message("assistant").markdown(first_q)

# User input
user_input = st.chat_input("Type your reply...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    st.chat_message("user").markdown(user_input)

    # LLM classification + acknowledgement
    result = call_llm(user_input)
    #tone_reply = result.get("tone_reply", "").strip()

    #if tone_reply:
    #    st.session_state.messages.append(
    #        {"role": "assistant", "content": tone_reply}
    #    )
    #    st.chat_message("assistant").markdown(tone_reply)

    # Advance flow
    advance_flow(result.get("intent", ""))

    # Ask next question or close
    #nq = next_question()
    nq = result.get("raw_text")
    if nq:
        st.session_state.messages.append(
            {"role": "assistant", "content": nq}
        )
        st.chat_message("assistant").markdown(nq)
    else:
        closing = (
            "Thank you so much for sharing 😊\n\n"
            "Based on this conversation, our team will get in touch with you shortly. "
            "If you have any questions meanwhile, feel free to ask!"
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": closing}
        )
        st.chat_message("assistant").markdown(closing)
