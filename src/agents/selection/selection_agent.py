import streamlit as st
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv


from enum import Enum

class KnowingVolunteerResult(Enum):
    STOP = "STOP"
    COMPLETE = "COMPLETE"
    COMPLETE_INSUFFICIENT_INFO = "COMPLETE_INSUFFICIENT_INFO"
    CONTINUE = "CONTINUE"

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
MAX_QUESTIONS = 30
# -----------------------------
# MASTER PROMPT
# -----------------------------
MASTER_SYSTEM_PROMPT = """You are SIA, a warm, respectful, purpose-driven conversational agent for Sunbird SERVE.

Your role is to onboard volunteers through a single, natural WhatsApp conversation.

You must sound human, encouraging, and calm — never procedural or robotic.

Core principles:

- Start with purpose before asking for details.

- Convert intent -> interest through clarity, not pressure.

- Never mention internal concepts like onboarding, registration, FSM, states, or selection.

- Keep messages short (1–3 lines), WhatsApp-friendly.

- Ask only one question at a time.

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

You are guiding a human, not completing a form.."""

STATE_PROMPTS = {

    "KNOWING_VOLUNTEER":"""

You are SIA, the Sunbird SERVE volunteer onboarding guide.

Current state: KNOWING_VOLUNTEER.

Context:
- The volunteer has already completed eligibility, identity, and preference collection.
- Basic onboarding steps are complete.
- This step is to understand the volunteer as a person in a light, respectful way.
- You are NOT evaluating or filtering the volunteer at this stage.
- You are only understanding:
  1) their background (in a general, non-personal way),
  2) their motivation to volunteer, why they want to volunteer, what drew them to SERVE
  3) any prior teaching / mentoring experience (formal or informal),
  4) their comfort interacting with children or learners.
  5) the subjects or topics they are comfortable teaching
  6) the age group of the children they are comfortable interacting with
  7) interst in teaching
  
- The orchestrator controls which question was last asked in this state via `last_agent_prompt`.

Your goal:
Classify the user's latest message and produce:
- a single intent label,
- a confidence score (0.0–1.0),
- a short, warm WhatsApp-style acknowledgement ("tone_reply").

Allowed intents:
- MOTIVATION_SHARED   → explains why they want to volunteer / help / give back /
- EXPERIENCE_SHARED   → mentions teaching, tutoring, mentoring, training, or helping others learn
- NO_EXPERIENCE       → explicitly states no teaching or mentoring experience
- COMFORT_SHARED      → expresses comfort or hesitation working with children or learners
- QUERY               → asks a question instead of answering
- AMBIGUOUS           → vague, off-topic, or unclear response
- STOP                → stop / unsubscribe / leave

Classification rules:
- Do NOT judge or filter based on experience; beginners are welcome.
- If the user explicitly says they have no experience → NO_EXPERIENCE.
- Use `last_agent_prompt` to infer whether the response relates to motivation, experience, or comfort.
- If the message does not clearly map to any category → AMBIGUOUS.
- Do NOT infer or invent information not explicitly stated.

Conversation boundaries:
- Do NOT ask personal questions (email, phone number, family, marital status, children, health, finances, etc.).
- Ask questions only around their work experience, teaching or mentoring experience, experience working with children, age group of the children they are comfortable with working , subjects they are comfortable with teaching

Critical rule (very important):
- Never mention onboarding steps, evaluation, selection, states, or internal processes.

Tone rules:
- 1–3 short lines.
- Warm, calm, and human.
- Reassuring, especially for NO_EXPERIENCE.
- Never evaluative, formal, or procedural.

Signal extraction rules:
- Extract signals ONLY if the user explicitly mentions them.
- Do NOT infer or guess.
- If a signal is not mentioned, return null (or empty list for subjects).
- Allowed values:
  - has_teaching_experience: true / false / Null
  - subjects: list of subjects explicitly mentioned (lowercase) or empty list
  - teaching_interest: yes / no / maybe / Null
  - children_age_comfort:
      "primary"   → ages ~5–10
      "middle"    → ages ~11–14
      "secondary" → ages ~15–18
      "unsure"    → expresses uncertainty or discomfort
      Null
    -motivation: Null/help/serve others/empower/uplift/bring joy/happiness/give

Output ONLY valid JSON:
  {
  "intent": "<one of the allowed intents>",
  "confidence": 0.0,
  "tone_reply": "<short friendly acknowledgement along with a relevant question based on conversation boundaries>",

  "signals": {
    "has_teaching_experience": true | false |null,
    "teaching_interest" : yes | no | maybe|null
    "motivation":"help | looking to give back |serve | bring joy | uplift |outreach |null"
    "subjects": []
    "children_age_comfort": "primary" | "middle" | "secondary" | "unsure" | null
  }
}


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
# STREAMLIT STATE INIT
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "state_index" not in st.session_state:
    st.session_state.state_index = 0

if "question_index" not in st.session_state:
    st.session_state.question_index = 0

if "volunteer_profile" not in st.session_state:
    st.session_state.volunteer_profile = {
        "motivation": "None",
        "has_teaching_experience": None,
        "children_age_comfort": None,
        "teaching_interest": None,
        "subjects":[]
    }


# -----------------------------
# HELPERS
# -----------------------------
def current_state():
    return STATE_ORDER[st.session_state.state_index]

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
        "background", "volunteer", "profession","mentor","worked","homemaker","housewife"
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

def init_selection_flow(user_text):
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
        response_format={"type": "json_object"},
        temperature=0.4
    )
    llm_response = response.choices[0].message.content
    json_llm_response = json.loads(llm_response)

    result = {
    "raw_text": json_llm_response.get("tone_reply"),
    "intent": json_llm_response.get("intent"),
    "confidence": json_llm_response.get("confidence"),
    "tone_reply":json_llm_response.get("tone_reply"),
    "signals":json_llm_response.get("signals",{})
    }

    signals = json_llm_response.get("signals", {})
    
    #Write Signals to session state 
    if signals.get("motivation") is not None:
        st.session_state.volunteer_profile["motivation"] = signals.get("motivation")

    if signals.get("has_teaching_experience") is not None and st.session_state.volunteer_profile["has_teaching_experience"] is None:
        st.session_state.volunteer_profile["has_teaching_experience"] = signals.get("has_teaching_experience")

    if signals.get("subjects"):
        st.session_state.volunteer_profile.setdefault("subjects", []).extend(signals.get("subjects"))
    
    if signals.get("teaching_interest") and st.session_state.volunteer_profile["teaching_interest"] is None:
        st.session_state.volunteer_profile["teaching_interest"] = signals.get("teaching_interest")

    if signals.get("children_age_comfort") is not None and  st.session_state.volunteer_profile["children_age_comfort"] is None:
        st.session_state.volunteer_profile["children_age_comfort"] = signals.get("children_age_comfort")

    return result

def knowing_volunteer_complete():
    profile = st.session_state.volunteer_profile

    signals = [
        profile["motivation"],
        profile["has_teaching_experience"],
        profile["children_age_comfort"],
        profile["subjects"],
        profile["teaching_interest"]
    ]

    # All 5 signals to be filled
    return sum(bool(s) for s in signals) >= 4
    
def evaluate_knowing_volunteer(intent, max_questions=20, min_questions=5):
    """
    Decide flow outcome for KNOWING_VOLUNTEER.
    """
    st.write("Num of questions"+  str(st.session_state.question_index))

    # 1️⃣ Explicit stop
    if intent == "STOP":
        return KnowingVolunteerResult.STOP

    # 2️⃣ If profile is sufficiently filled
    if knowing_volunteer_complete() and st.session_state.question_index >= min_questions - 1:
        return KnowingVolunteerResult.COMPLETE

    # 3️⃣ If we’ve explored enough, move forward gracefully
    if st.session_state.question_index >= max_questions - 1:
        return KnowingVolunteerResult.COMPLETE_INSUFFICIENT_INFO
    
    return KnowingVolunteerResult.CONTINUE

    

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
    first_q = "Thank you completing the onboarding steps. Let us start with something simple, can you tell me a bit about yourself?"
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
    result = init_selection_flow(user_input)
    st.session_state.question_index += 1

    # Advance flow
    knowing_volunteer_result = evaluate_knowing_volunteer(result.get("intent"))
    if knowing_volunteer_result == KnowingVolunteerResult.CONTINUE:
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
            "Based on this conversation, our team will get in touch with you shortly."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": closing}
        )
        st.write("FINAL VOLUNTEER PROFILE:")
        st.write(st.session_state.volunteer_profile)
        st.chat_message("assistant").markdown(closing)
