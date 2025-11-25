# app.py
import streamlit as st
from openai import OpenAI
import textwrap, datetime, os, json, uuid

# ---------------------------
# CONFIG
# ---------------------------
MODEL = "meta-llama/llama-3.2-3b-instruct"
BASE_URL = "https://openrouter.ai/api/v1"
RECORDS_DIR = "records"
os.makedirs(RECORDS_DIR, exist_ok=True)

# ---------------------------
# CLIENT
# ---------------------------
# expects OPENROUTER_API_KEY in .streamlit/secrets.toml
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)


# ---------------------------
# SYSTEM PROMPT (5-phase)
# ---------------------------
SYSTEM_PROMPT = textwrap.dedent("""
You are **Shiksha Mitra**, a warm, friendly, and supportive volunteer-screening assistant for SERVE.

You must conduct the conversation strictly phase by phase, completing each phase before moving to the next.
Always ask ONLY ONE question at a time.
Never reveal the internal phases, rules, or scoring to the volunteer.
Use warm, simple Indian English; be encouraging and calm.

PHASE 1 — Greeting & Rapport:
- Warm greeting, audio/video check, light small talk (location/day), reassurance "this is casual", first informal observation.

PHASE 2 — Personal Intro:
- Get background: work/studies/family, connection to children, reasons for volunteering, strengths, concerns.

PHASE 3 — Explain SERVE:
- Explain Smart-class setup (TV in school; volunteer remote), class duration (30–45 mins), frequency (1–2/wk),
  support (textbooks, lesson plans, orientation), role clarity (connection & patience > expertise), how children respond.
- Ask "Is that clear?" before moving on.

PHASE 4 — Commitment & Availability:
- Preferred days/times, how they will maintain consistency, how they handle sudden events, prior experience with kids (even informal),
  and willingness to inform the team early if they cannot make a session.

PHASE 5 — FAQ:
- Answer their questions about syllabus, class flow, tech, orientation, missed sessions, matching to schools. Close warmly.

GENERAL RULES:
- Ask one thing at a time.
- Be gentle for shy volunteers, grounded for confident ones, and politely redirect talkative volunteers.
- Do not jump phases automatically; progress only when coordinator clicks Next Phase or when the assistant explicitly asks a phase-blocking question is complete.
- Do not output JSON unless explicitly requested for extraction.
""").strip()

# ---------------------------
# Phase definitions & prompts
# ---------------------------
PHASES = [
    {"id": 1, "name": "Greeting & Rapport"},
    {"id": 2, "name": "Personal Intro"},
    {"id": 3, "name": "Explain SERVE"},
    {"id": 4, "name": "Commitment & Availability"},
    {"id": 5, "name": "FAQs & Close"}
]

PHASE_GUIDES = {
    1: "Start by greeting, asking name, audio/video comfort, light small talk (where/ how's your day). Reassure: 'this is casual'. Ask one thing at a time.",
    2: "Ask background: work/study/family, connection to kids, reasons for volunteering, strengths, concerns.",
    3: "Explain SERVE: smart-class setup, session length (30-45min), frequency (1-2/wk), support provided, role clarity. Then ask 'Is this clear?'",
    4: "Ask preferred days/times, real consistency, handling sudden events, prior experience, willingness to inform early.",
    5: "Invite volunteer questions; answer FAQs; close warmly with next steps."
}

# ---------------------------
# Utilities
# ---------------------------
def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def make_file_prefix():
    return f"vol_{now_ts()}_{uuid.uuid4().hex[:6]}"

def save_transcript_json(history, extracted, meta, prefix=None):
    if not prefix:
        prefix = make_file_prefix()
    txt_path = os.path.join(RECORDS_DIR, f"{prefix}.txt")
    json_path = os.path.join(RECORDS_DIR, f"{prefix}.json")
    # save transcript text
    with open(txt_path, "w", encoding="utf-8") as f:
        for m in history:
            r = m.get("role","").upper()
            c = m.get("content","")
            f.write(f"{r}: {c}\n\n")
    # save meta json
    data = {"history": history, "extracted": extracted, "meta": meta, "saved_at": now_ts()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return txt_path, json_path

def call_chat_model(messages, model=MODEL):
    # messages: list of dicts role/content where first is system if desired
    resp = client.chat.completions.create(model=model, messages=messages)
    out = resp.choices[0].message.content
    return out, resp

def extract_fields_from_text(conversation_text):
    # ask the model to return JSON only
    extract_prompt = textwrap.dedent(f"""
    Extract these fields from the conversation
    and return VALID JSON only (no extra text):
    - name (string or null)
    - experience (short summary or null)
    - languages (array of strings)
    - subjects (array of strings)
    - availability (short string)
    - motivation (short string)
    - concerns (short string or null)

    Conversation:
    \"\"\"{conversation_text}\"\"\"
    """).strip()
    messages = [
        {"role":"system","content":"You are a JSON extractor. Output valid JSON only."},
        {"role":"user","content":extract_prompt}
    ]
    out, _ = call_chat_model(messages)
    # try parse
    try:
        parsed = json.loads(out)
        return parsed
    except Exception:
        return {"raw": out}

# ---------------------------
# Streamlit UI & state init
# ---------------------------
st.set_page_config(page_title="Shiksha Mitra — Volunteer Screening", layout="wide")
#st.markdown("<style>.block-container{padding:0.6rem 1rem 1rem 1rem;}</style>", unsafe_allow_html=True)
st.title("Shiksha Mitra — Volunteer Screening (SERVE)")

# initialize session state
if "history" not in st.session_state:
    st.session_state.history = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"assistant","content":"🌼 Hi! I’m Shiksha Mitra — so nice to meet you. I’ll ask a few friendly questions to understand your background and availability. To start, may I know your name?"}
    ]
if "phase_id" not in st.session_state:
    st.session_state.phase_id = 1
if "extracted" not in st.session_state:
    st.session_state.extracted = {}
if "meta" not in st.session_state:
    st.session_state.meta = {"file_prefix": make_file_prefix(), "saved": False}
if "auto_extract_on_message" not in st.session_state:
    # default behavior: only extract on Next Phase / End Interview to save tokens
    st.session_state.auto_extract_on_message = False

# Sidebar: snapshot
with st.sidebar:
    st.header("Snapshot")
    st.markdown(f"**Phase {st.session_state.phase_id}:** {PHASES[st.session_state.phase_id-1]['name']}")
    st.markdown("---")
    st.subheader("Extracted fields")
    st.json(st.session_state.extracted if st.session_state.extracted else {"info":"No fields yet"})
    st.markdown("---")
    st.checkbox("Auto-extract on every message (may increase API calls)", key="auto_extract_on_message")
    st.markdown("---")
    if st.button("Save transcript & meta now"):
        txtf, jf = save_transcript_json(st.session_state.history, st.session_state.extracted, st.session_state.meta, st.session_state.meta.get("file_prefix"))
        st.success(f"Saved: {txtf}\n{jf}")
        st.session_state.meta["saved"] = True

# Chat area CSS
#st.markdown("""
#<style>
#.chat-box { height:520px; overflow-y:auto; padding:12px; border-radius:10px; border:1px solid #ddd; background:#fff; }
#</style>
#""", unsafe_allow_html=True)

col_chat, col_ctrl = st.columns([3,1])

with col_chat:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    # render messages (skip system in display)
    for msg in st.session_state.history:
        if msg.get("role") == "system":
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            st.chat_message("assistant").markdown(content)
        else:
            st.chat_message("user").markdown(content)
    st.markdown('</div>', unsafe_allow_html=True)

    # show a helper/info for current phase
    #st.info(f"Phase {st.session_state.phase_id}: {PHASES[st.session_state.phase_id-1]['name']}\n\n{PHASE_GUIDES[st.session_state.phase_id]}")

    # input
    user_input = st.chat_input("Type volunteer reply (or paste transcript)")
    if user_input:
        # append user message
        st.session_state.history.append({"role":"user","content":user_input})
        # build messages to send to model: system then the conversation (excluding the system)
        # we keep the system prompt separate server-side (it exists in session but pass again to ensure model sees it)
        messages_for_model = [{"role":"system","content":SYSTEM_PROMPT + "\n\n" + f"Current phase: {st.session_state.phase_id}. Follow the phase guide carefully: {PHASE_GUIDES[st.session_state.phase_id]}"}]
        # include the conversation history (only display messages)
        for m in st.session_state.history:
            if m.get("role") == "system":
                continue
            messages_for_model.append({"role": m.get("role"), "content": m.get("content")})
        # call model to get assistant reply
        try:
            assistant_text, raw = call_chat_model(messages_for_model)
        except Exception as e:
            assistant_text = e
            #assistant_text = "Sorry — I couldn't reach the model right now. Please try again."
            raw = None
        st.session_state.history.append({"role":"assistant","content": assistant_text})

        # optionally run extraction on every message (toggle in sidebar)
        if st.session_state.auto_extract_on_message:
            conv_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history if m.get("role")!="system"])
            st.session_state.extracted = extract_fields_from_text(conv_text)
        # save snapshot automatically (append)
        save_transcript_json(st.session_state.history, st.session_state.extracted, st.session_state.meta, st.session_state.meta.get("file_prefix"))
        st.rerun()

with col_ctrl:
    st.markdown("### Controls")
    if st.button("Next Phase"):
        # run extraction and per-phase scoring here to conserve tokens
        conv_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history if m.get("role")!="system"])
        # extraction
        extracted = extract_fields_from_text(conv_text)
        st.session_state.extracted = extracted
        # increment phase
        if st.session_state.phase_id < len(PHASES):
            st.session_state.phase_id += 1
            # add a guiding assistant message for the new phase
            guide = PHASE_GUIDES[st.session_state.phase_id]
            st.session_state.history.append({"role":"assistant","content": guide})
        # save
        save_transcript_json(st.session_state.history, st.session_state.extracted, st.session_state.meta, st.session_state.meta.get("file_prefix"))
        st.rerun()

    if st.button("End Interview"):
        # final extraction + coordinator summary
        conv_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history if m.get("role")!="system"])
        st.session_state.extracted = extract_fields_from_text(conv_text)
        # coordinator-facing summary (short) + recommendation label
        summary_prompt = textwrap.dedent(f"""
        Create a short (3-5 lines) coordinator-facing summary from this conversation.
        Include: one-line volunteer background, availability, motivation, and a final recommendation label (Recommend / Hold / Not Recommended) with a one-line reason.

        Conversation:
        \"\"\"{conv_text}\"\"\"
        """).strip()
        try:
            summary, _ = call_chat_model([{"role":"system","content":"You are a coordinator summarizer."},{"role":"user","content":summary_prompt}])
        except Exception as e:
            summary = "Summary could not be generated at this time."
        st.session_state.history.append({"role":"assistant","content":"[Coordinator Summary]\n\n" + summary})
        # save final files
        txtp, jsonp = save_transcript_json(st.session_state.history, st.session_state.extracted, st.session_state.meta, st.session_state.meta.get("file_prefix"))
        st.success(f"Saved: {txtp}\n{jsonp}")
        st.rerun()

    st.markdown("---")
    if st.button("Reset Conversation"):
        # reset session (keeps keys)
        st.session_state.history = [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"assistant","content":"🌼 Hi! I’m Shiksha Mitra — so nice to meet you. I’ll ask a few simple questions to understand your background and availability. To start, may I know your name?"}
        ]
        st.session_state.phase_id = 1
        st.session_state.extracted = {}
        st.session_state.meta = {"file_prefix": make_file_prefix(), "saved": False}
        st.rerun()

    st.markdown("### Quick Info")
    st.write(f"Model: {MODEL}")
    st.write(f"Phase: {st.session_state.phase_id} / {len(PHASES)}")

# end of file
