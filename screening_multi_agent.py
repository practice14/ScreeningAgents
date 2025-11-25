# app.py
import streamlit as st
from openai import OpenAI
import textwrap, datetime, os, json, uuid
from dotenv import load_dotenv

# ---------------------------
# Configuration
# ---------------------------
MODEL = "meta-llama/llama-3.2-3b-instruct"
BASE_URL = "https://openrouter.ai/api/v1"
RECORDS_DIR = "records"
os.makedirs(RECORDS_DIR, exist_ok=True)

load_dotenv()
# ---------------------------
# CLIENT
# ---------------------------
# expects OPENROUTER_API_KEY in .streamlit/secrets.toml
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENAI_API_KEY"]
)

# ---------------------------
# System-level structured prompts for each phase (short)
# ---------------------------
PHASES = {
    1: {"name": "Greeting & Rapport",
        "guide": "Goalis to develop a rapport with the volunteer getting screened, start with a Warm greeting, understan if they are comfortable,have a light small talk (location/how their day is going/are they comfortable). Reassure them that 'this is a casual'. Ask one thing at a time, try to wrap up the conversation with 4 or 5 questions."},
    2: {"name": "Personal Intro",
        "guide": "Learn background (work/study), connection to children, motivation, strengths, concerns. One question at a time."},
    3: {"name": "Explain SERVE",
        "guide": "Explain how the organization runs Smart-classes (there is a TV in schools, mostly rural), classes are usually 30-45min sessions, 1-2 classes per week, support is given to the volunteer teacher by the org through lesson plans, orientation, what is needed by the volunteer teacher: connection to children & patience more than teaching expertise. Ask 'Is this clear?'. One thing at a time."},
    4: {"name": "Commitment & Availability",
        "guide": "Ask preferred days/times, how they'll maintain consistency, handling sudden events, prior experience with kids, and communication responsibility. One question at a time."},
    5: {"name": "FAQs & Close",
        "guide": "Invite volunteer questions and answer succinctly. Close warmly and explain next steps."}
}

SYSTEM_BASE = textwrap.dedent("""
You are Shiksha Mitra — a warm, kind, Indian-English volunteer screening assistant for SERVE. 
You will be having a friendly conversation with potential volunteers 
who will remotely teach school students from rural areas through digital tools. 
The goal is to have a conversation with the volunteers to assess them and 
see if they are a good fit. Always be friendly and conversational. Ask ONE question at a time.
Do not reveal internal scoring or phase logic to the volunteer.
Adjust tone gently for shy, grounded for confident, and politely redirect talkative volunteers.
""").strip()

# ---------------------------
# Helpers
# ---------------------------
def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def make_prefix():
    return f"vol_{now_ts()}_{uuid.uuid4().hex[:6]}"

def save_records(history, extracted, meta, prefix=None):
    if not prefix:
        prefix = meta.get("file_prefix", make_prefix())
    txt_path = os.path.join(RECORDS_DIR, f"{prefix}.txt")
    json_path = os.path.join(RECORDS_DIR, f"{prefix}.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        for m in history:
            f.write(f"{m['role'].upper()}: {m['content']}\n\n")
    payload = {"history": history, "extracted": extracted, "meta": meta, "saved_at": now_ts()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return txt_path, json_path

def call_model(messages, model=MODEL):
    """
    messages: list of dicts {role, content}
    returns: assistant_text (str)
    """
    resp = client.chat.completions.create(model=model, messages=messages)
    text = resp.choices[0].message.content
    return text

# Extraction prompt (returns JSON only)
def extract_fields(conversation_text):
    prompt = textwrap.dedent(f"""
    Extract the following fields from the conversation below and return VALID JSON ONLY:
    - name (string or null)
    - experience (short string or null)
    - languages (list of strings)
    - subjects (list of strings)
    - availability (short string)
    - motivation (short string)
    - concerns (short string or null)

    Conversation:
    \"\"\"{conversation_text}\"\"\"
    """).strip()
    messages = [
        {"role":"system","content":"You are a JSON extractor. Output VALID JSON only."},
        {"role":"user","content":prompt}
    ]
    out = call_model(messages)
    try:
        parsed = json.loads(out)
        return parsed
    except Exception:
        # fallback: return raw under "raw"
        return {"raw": out}

# Phase scoring (1-5) using a lightweight rubric per phase, returns JSON
def score_phase(phase_id, conversation_text):
    rubric_lines = {
        1: "Rate comfort, clarity, and engagement (1-5).",
        2: "Rate motivation, empathy, and stability (1-5).",
        3: "Rate understanding of program and comfort with idea of teaching (1-5).",
        4: "Rate availability consistency, reliability, and communication responsibility (1-5).",
        5: "Rate clarity of questions and comfort asking doubts (1-5)."
    }
    prompt = textwrap.dedent(f"""
    Using the rubric: {rubric_lines[phase_id]}
    Evaluate the volunteer's responses in this conversation section and return VALID JSON ONLY:
    {{
      "score": <number between 1 and 5>,
      "notes": "<one-sentence explanation>"
    }}

    Conversation:
    \"\"\"{conversation_text}\"\"\"
    """).strip()
    messages = [
        {"role":"system","content":"You are an evaluator following the given rubric. Output VALID JSON only."},
        {"role":"user","content":prompt}
    ]
    out = call_model(messages)
    try:
        parsed = json.loads(out)
        # normalize score numeric if string
        parsed['score'] = float(parsed.get('score', 0))
        return parsed
    except Exception:
        return {"raw": out}

# ---------------------------
# Phase agent functions (simple: they get history and produce one assistant reply)
# Each agent uses the SYSTEM_BASE + PHASE guide to generate the next assistant message.
# ---------------------------
def run_phase_agent(phase_id, history):
    phase = PHASES[phase_id]
    system_prompt = SYSTEM_BASE + "\n\n" + f"Phase {phase_id}: {phase['name']} — {phase['guide']}"
    messages = [{"role":"system","content":system_prompt}]
    # pass a truncated history (last 20 messages) to keep context manageable
    for m in history[-20:]:
        # include only assistant/user entries (system not repeated)
        messages.append({"role": m["role"], "content": m["content"]})
    assistant_text = call_model(messages)
    return assistant_text

# ---------------------------
# Streamlit UI + state
# ---------------------------
st.set_page_config(page_title="Shiksha Mitra — Volunteer Screening", layout="wide")
#st.markdown("<style>.block-container{padding:0.6rem 1rem 1rem 1rem;}.chat-box{height:520px;overflow:auto;padding:12px;border-radius:10px;border:1px solid #ddd;background:#fff}</style>", unsafe_allow_html=True)
st.title("Shiksha Mitra — Volunteer Screening")

# init session state
if "history" not in st.session_state:
    st.session_state.history = [
        {"role":"assistant","content":"🌼 Hi! I’m Shiksha Mitra — nice to meet you. I’ll ask a few friendly questions to help you onboard to SERVE. To start, may I have your name?"}
    ]
if "phase_id" not in st.session_state:
    st.session_state.phase_id = 1
if "meta" not in st.session_state:
    st.session_state.meta = {"file_prefix": make_prefix(), "scores": {}}
if "extracted" not in st.session_state:
    st.session_state.extracted = {}

# Sidebar
with st.sidebar:
    st.header("Controls")
    #st.write(f"Phase: {st.session_state.phase_id} / {len(PHASES)}")
    st.write(f"Current: {PHASES[st.session_state.phase_id]['name']}")
    st.markdown("---")
    if st.button("Next Phase (run extraction+scoring)"):
        # run extraction and per-phase scoring on the conversation so far
        conv_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history])
        st.session_state.extracted = extract_fields(conv_text)
        # score current phase
        sc = score_phase(st.session_state.phase_id, conv_text)
        st.session_state.meta["scores"][st.session_state.phase_id] = sc
        # advance phase (manual)
        if st.session_state.phase_id < len(PHASES):
            st.session_state.phase_id += 1
            # append a guiding assistant message for the new phase (without calling LLM here)
            guide = PHASES[st.session_state.phase_id]["guide"]
            st.session_state.history.append({"role":"assistant","content": f"(Guide) {PHASES[st.session_state.phase_id]['name']}: {guide}"})
        # save snapshot
        save_records(st.session_state.history, st.session_state.extracted, st.session_state.meta, st.session_state.meta.get("file_prefix"))
        st.rerun()
    if st.button("End Interview (final extract & save)"):
        conv_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history])
        st.session_state.extracted = extract_fields(conv_text)
        # score missing phases if any
        for pid in range(1, len(PHASES)+1):
            if pid not in st.session_state.meta["scores"]:
                st.session_state.meta["scores"][pid] = score_phase(pid, conv_text)
        # final coordinator summary (3 lines + recommendation)
        summary_prompt = textwrap.dedent(f"""
            Create a short (3-4 line) coordinator-facing summary and a final recommendation label (Recommend / Hold / Not Recommended) with a one-line reason.
            Conversation:
            \"\"\"{conv_text}\"\"\"
        """).strip()
        try:
            summary = call_model([{"role":"system","content":"You are a coordinator summarizer."},{"role":"user","content":summary_prompt}])
        except Exception:
            summary = "Summary generation failed."
        st.session_state.history.append({"role":"assistant","content":"[Coordinator Summary]\n\n" + summary})
        txt, js = save_records(st.session_state.history, st.session_state.extracted, st.session_state.meta, st.session_state.meta.get("file_prefix"))
        st.success(f"Saved: {txt}\n{js}")
        st.rerun()
    if st.button("Reset Conversation"):
        st.session_state.history = [{"role":"assistant","content":"🌼 Hi! I’m Shiksha Mitra — nice to meet you. I’ll ask a few friendly questions to understand your background and availability. To start, may I have your name?"}]
        st.session_state.phase_id = 1
        st.session_state.meta = {"file_prefix": make_prefix(), "scores": {}}
        st.session_state.extracted = {}
        st.rerun()
    st.markdown("---")
    st.subheader("Extracted (live after Next Phase)")
    st.json(st.session_state.extracted if st.session_state.extracted else {"info":"No extract yet"})
    st.markdown("---")
    st.subheader("Per-phase scores")
    if st.session_state.meta.get("scores"):
        st.json(st.session_state.meta["scores"])
    else:
        st.write("No scores yet")

# Main chat area
st.markdown('<div class="chat-box">', unsafe_allow_html=True)
for m in st.session_state.history:
    if m["role"] == "assistant":
        st.chat_message("assistant").markdown(m["content"])
    else:
        st.chat_message("user").markdown(m["content"])
st.markdown('</div>', unsafe_allow_html=True)

# show phase guide
st.info(f"Phase {st.session_state.phase_id}: {PHASES[st.session_state.phase_id]['name']}\n\n{PHASES[st.session_state.phase_id]['guide']}")

# input
user_text = st.chat_input("Type volunteer reply (or paste transcript)...")
if user_text:
    # store user message
    st.session_state.history.append({"role":"user","content":user_text})
    # run the phase agent to get a single assistant reply
    try:
        assistant_reply = run_phase_agent(st.session_state.phase_id, st.session_state.history)
    except Exception as e:
        assistant_reply = "Sorry — couldn't call the model just now. Please try again."
    st.session_state.history.append({"role":"assistant","content":assistant_reply})
    # autosave a snapshot (append)
    save_records(st.session_state.history, st.session_state.extracted, st.session_state.meta, st.session_state.meta.get("file_prefix"))
    st.rerun()

# end of file
