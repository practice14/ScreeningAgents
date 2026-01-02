import os
import json
import logging
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv


# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()



# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------
MODEL = "meta-llama/llama-3.2-3b-instruct" # or OpenRouter model
OPENROUTER_API_KEY = api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ---------------------------------------------------------
# STATES & QUESTIONS (SOURCE OF TRUTH)
# ---------------------------------------------------------
STATE_SEQUENCE = [
    "KNOWING_VOLUNTEER",
    "WEEKLY_COMMITMENT",
    "ORIENTATION",
    "CLOSING"
]

STATE_QUESTIONS = {
    "KNOWING_VOLUNTEER": [
        "Nice to meet you 🙂 Could you briefly tell me a little about yourself?",
        "What motivated you to explore volunteering with SERVE?",
        "Have you had any teaching or mentoring experience before?"
    ],
    "WEEKLY_COMMITMENT": [
        "Would you be comfortable taking one short session per week, consistently?",
        "Do you see any challenges with keeping this weekly commitment?"
    ],
    "ORIENTATION": [
        "Just to share — sessions run for about 30–40 minutes. You’ll get lesson plans in advance, coordinator support, and help with tech if needed.\n\nDoes this feel comfortable for you?"
    ],
    "CLOSING": []
}

# ---------------------------------------------------------
# MASTER SYSTEM PROMPT (SPEAKER)
# ---------------------------------------------------------
MASTER_SYSTEM_PROMPT = """
You are SIA, a warm, respectful, purpose-driven conversational agent for Sunbird SERVE.

You speak to volunteers in simple, friendly Indian English.
Messages should be short (1–3 lines), WhatsApp-friendly.
Ask ONLY the question provided to you by the system.
Do NOT invent new questions.
Do NOT mention internal states or processes.

If a tone_reply is provided, acknowledge it briefly before asking the next question.
"""

# ---------------------------------------------------------
# STATE PROMPTS (CLASSIFIERS)
# ---------------------------------------------------------
STATE_PROMPTS = {
    "KNOWING_VOLUNTEER": """
You are silently evaluating the volunteer's response.

Current state: KNOWING_VOLUNTEER.

Your task:
- Infer intent from the user's reply.
- Return intent + confidence + optional short acknowledgement.

Allowed intents:
- BACKGROUND_SHARED
- MOTIVATION_SHARED
- EXPERIENCE_SHARED
- QUERY
- AMBIGUOUS
- STOP

Rules:
- Do NOT ask questions.
- Do NOT move the conversation forward.

Output JSON ONLY:
{
  "intent": "<label>",
  "confidence": 0.0,
  "tone_reply": "<optional short acknowledgement>"
}
""",

    "WEEKLY_COMMITMENT": """
Current state: WEEKLY_COMMITMENT.

Classify comfort with weekly commitment.

Allowed intents:
- COMMIT_YES
- COMMIT_MAYBE
- COMMIT_NO
- QUERY
- AMBIGUOUS

Output JSON ONLY:
{
  "intent": "<label>",
  "confidence": 0.0,
  "tone_reply": "<optional reassurance>"
}
""",

    "ORIENTATION": """
Current state: ORIENTATION.

Classify comfort after orientation explanation.

Allowed intents:
- ORIENT_OK
- ORIENT_CONCERN
- QUERY
- AMBIGUOUS

Output JSON ONLY:
{
  "intent": "<label>",
  "confidence": 0.0,
  "tone_reply": "<optional reassurance>"
}
"""
}

# ---------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------
if "state_index" not in st.session_state:
    st.session_state.state_index = 0
    st.session_state.question_index = 0
    st.session_state.messages = []

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def current_state():
    return STATE_SEQUENCE[st.session_state.state_index]


def next_question():
    state = current_state()
    qs = STATE_QUESTIONS.get(state, [])
    if st.session_state.question_index < len(qs):
        return qs[st.session_state.question_index]
    return None


def advance_flow():
    st.session_state.question_index += 1
    if st.session_state.question_index >= len(
        STATE_QUESTIONS.get(current_state(), [])
    ):
        st.session_state.state_index += 1
        st.session_state.question_index = 0
        st.write("question index "+st.session_state.question_index)


def call_classifier(state_prompt, user_text):
    messages = [
        {"role": "system", "content": state_prompt},
        {"role": "user", "content": user_text}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    raw = response.choices[0].message.content
    logger.info(f"Classifier raw output: {raw}")

    try:
        return json.loads(raw)
    except Exception:
        return {
            "intent": "AMBIGUOUS",
            "confidence": 0.3,
            "tone_reply": ""
        }


def call_speaker(tone_reply, question):
    content = ""
    if tone_reply:
        content += tone_reply.strip() + "\n\n"
    if question:
        content += question

    messages = [
        {"role": "system", "content": MASTER_SYSTEM_PROMPT},
        {"role": "assistant", "content": content}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("SIA – SERVE Volunteer Assistant")

# Show chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Start conversation
if not st.session_state.messages:
    first_q = next_question()
    st.session_state.messages.append(
        {"role": "assistant", "content": first_q}
    )
    st.chat_message("assistant").write(first_q)

# User input
user_input = st.chat_input("Type your reply…")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    st.chat_message("user").write(user_input)

    state = current_state()
    classifier_prompt = STATE_PROMPTS.get(state)

    tone_reply = ""

    if classifier_prompt:
        result = call_classifier(classifier_prompt, user_input)
        tone_reply = result.get("tone_reply", "")

    advance_flow()

    question = next_question()
    st.write("current state:"+current_state())

    if question:
        reply = call_speaker(tone_reply, question)
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        st.chat_message("assistant").write(reply)
    else:
        closing = "Thank you so much for your time 🙏 We’ll review this and get back to you shortly."
        st.session_state.messages.append(
            {"role": "assistant", "content": closing}
        )
        st.chat_message("assistant").write(closing)
