import streamlit as st
import random
from openai import OpenAI
import os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Shiksha Mitra – Volunteer Interview", page_icon="🌼")

OPENROUTER_API_KEY = api_key=os.getenv("OPENAI_API_KEY")
MODEL = "meta-llama/llama-3.2-3b-instruct"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ----------------------------
# QUESTION BANK (SEQUENTIAL)
# ----------------------------
QUESTION_BANK = [
    "Hi! I’m Shiksha Mitra from SERVE. What’s your name?",
    "Nice to meet you! What inspired you to volunteer with children?",
    "Have you taught or mentored before, formally or informally?",
    "What subjects or topics do you feel most comfortable teaching?",
    "How many years of experience do you have with children, if any?",
    "Do you feel comfortable teaching online over video calls?",
    "Are you okay committing to one short session per week?",
    "Which age group do you feel most comfortable teaching?"
]

ORIENTATION_TEXT = (
    "Before we wrap up, let me quickly share how SERVE sessions usually work.\n\n"
    "• Each session is about 30–40 minutes\n"
    "• You’ll teach simple, age-appropriate content\n"
    "• Lesson plans and materials are shared well in advance\n"
    "• A coordinator will always support you\n"
    "• Any digital or tech help will also be provided\n\n"
    "The most important thing is care and consistency — not perfection."
)

CLOSING_TEXT = (
    "Thank you so much for taking the time to speak with me today 🌼\n\n"
    "Based on this conversation, our team will review and get back to you shortly "
    "with next steps. We truly appreciate your interest in supporting our children."
)

# ----------------------------
# ACKNOWLEDGEMENT LIBRARY
# ----------------------------
ACK_LIBRARY = [
    "Thanks for sharing.",
    "That helps.",
    "Appreciate you sharing.",
    "Got it, thank you.",
    "Thanks for being open."
]

# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "q_index" not in st.session_state:
    st.session_state.q_index = 0

if "used_acks" not in st.session_state:
    st.session_state.used_acks = set()

if "phase" not in st.session_state:
    st.session_state.phase = "questions"  # questions | orientation | closing

# ----------------------------
# LLM: SHOULD ACKNOWLEDGE?
# ----------------------------
def should_acknowledge_llm():
    recent = st.session_state.messages[-8:]
    convo = []

    for m in recent:
        role = "Assistant" if m["role"] == "assistant" else "Volunteer"
        convo.append(f"{role}: {m['content']}")

    prompt = (
        """You are an internal decision system.

Your job is to decide whether the assistant should add
a brief, polite acknowledgement before asking the next question.
You should mostly be adding a polite acknowledgement. You need to be warm, friendly and encouraging.

Say YES if the volunteer reply:
- Shares personal background or context
- Expresses uncertainty, lack of experience, or hesitation
- Gives a negative or limiting answer (e.g. “no experience”, “not sure”)

Say NO if the reply is:
- A simple factual answer
- A single word (like a subject name)
- Purely informational with no personal context

Return ONLY one word: YES or NO.
Do not explain."""

        + "\n".join(convo)
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )

    decision = resp.choices[0].message.content.strip().upper()
    return decision == "YES", prompt, decision

NEGATIVE_ACKS = [
    "That’s completely okay.",
    "Thanks for sharing honestly.",
    "No worries at all.",
    "Appreciate your honesty."
]

# ----------------------------
# PICK ACK (DETERMINISTIC)
# ----------------------------
def pick_ack(is_negative=False):
    library = NEGATIVE_ACKS if is_negative else ACK_LIBRARY

    used = st.session_state.used_acks
    choices = [a for a in library if a not in used]

    if not choices:
        st.session_state.used_acks.clear()
        choices = library.copy()

    ack = random.choice(choices)
    st.session_state.used_acks.add(ack)
    return ack

def generate_acknowledgement():
    recent = st.session_state.messages[-1:]
    convo = []

    for m in recent:
        role = "Assistant" if m["role"] == "assistant" else "Volunteer"
        convo.append(f"{role}: {m['content']}")

    prompt = (
        "You are Shiksha Mitra, a polite and encouraging coordinator who is onboarding new volunteers on a remote education NGO.\n\n"
        "Given the volunteer’s last reply and recent context, "
        "write a SHORT, natural acknowledgement.\n\n"
        "IMPORTANT:This is NOT a conversation opener.Do NOT ask questions, even on the first reply."
        "Guidelines:\n"
        "- 1 short line only (max 3 to 8 words)\n"
        "- Polite, calm, encouraging Indian English\n"
        "- If the reply is negative or shows lack of experience, respond reassuringly and be supportive and encouragin\n"
        "- If the reply is factual or short, return a single dash\n"
        "- Do NOT ask questions\n"
        "- Do NOT give details you dont know\n"
        "- Do NOT greet unless this is the first reply\n\n"
        "Return ONLY the acknowledgement text.\n"
        "If nothing appropriate fits, return a single dash: -\n\n"
        "Conversation:\n"
        + "\n".join(convo)
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )

    ack = resp.choices[0].message.content.strip()

    # Cleanup / safety
    #if ack == "-" or len(ack.split()) > 8:
    #    return ""

    return ack


# ----------------------------
# GET NEXT ASSISTANT MESSAGE
# ----------------------------
def get_next_assistant_message():
    # Phase 1: Questions
    if st.session_state.phase == "questions":
        if st.session_state.q_index < len(QUESTION_BANK):
            q = QUESTION_BANK[st.session_state.q_index]
            st.session_state.q_index += 1
            return q
        else:
            st.session_state.phase = "orientation"
            return ORIENTATION_TEXT + "\n\nDoes this feel comfortable for you?"

    # Phase 2: Orientation response handled → move to closing
    if st.session_state.phase == "orientation":
        st.session_state.phase = "closing"
        return CLOSING_TEXT

    return None

# ----------------------------
# DISPLAY CHAT
# ----------------------------
st.title("🌼 Shiksha Mitra – Volunteer Conversation")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ----------------------------
# USER INPUT
# ----------------------------
user_input = st.chat_input("Type your response here...")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Decide acknowledgement
    ack = ""
    ack_decision, ack_prompt, raw_decision = should_acknowledge_llm()

    ack = generate_acknowledgement()

    # store debug info
    st.session_state.last_ack_debug = {
        "raw_decision": raw_decision,
        "ack_added": bool(ack),
        "ack_text": ack,
        "prompt_sent": ack_prompt,
        "last_user_message": user_input
    }



    # Get next assistant message
    next_msg = get_next_assistant_message()

    if next_msg:
        assistant_text = f"{ack}\n\n{next_msg}" if ack else next_msg

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_text
        })

    st.rerun()

# ----------------------------
# INITIAL BOT MESSAGE
# ----------------------------
if not st.session_state.messages:
    opening = get_next_assistant_message()
    st.session_state.messages.append({
        "role": "assistant",
        "content": opening
    })
    st.rerun()

with st.sidebar:
    st.subheader("🛠 Acknowledgement Debug")

    debug = st.session_state.get("last_ack_debug")

    if debug:
        st.write("**Last volunteer reply:**")
        st.code(debug["last_user_message"])

        st.write("**Classifier decision:**", debug["raw_decision"])
        st.write("**Acknowledgement added:**", debug["ack_added"])

        if debug["ack_text"]:
            st.write("**Acknowledgement text:**")
            st.code(debug["ack_text"])

        st.write("**Prompt sent to LLM:**")
        st.code(debug["prompt_sent"])
    else:
        st.write("No acknowledgement evaluated yet.")
