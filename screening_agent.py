# app.py
import streamlit as st
from openai import OpenAI
import datetime, os, json, uuid, textwrap

# ---------------------------
# CONFIG
# ---------------------------
MODEL = "meta-llama/llama-3.2-3b-instruct"   # free / light choice (change if desired)
BASE_URL = "https://openrouter.ai/api/v1"

# create records folder
os.makedirs("records", exist_ok=True)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ---------------------------------------
# System prompt (driving the agent)
# ---------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
You are **Shiksha Mitra**, a warm and friendly volunteer-screening assistant for SERVE.

Follow the conversation strictly **phase-wise**.  
You MUST complete each phase before moving to the next.  
Always ask ONLY ONE question at a time.
                                
###PHASE RULES:
PHASE 1 - aim is to set tone and develop rapport and covers the following -
Basic greetings
Sound/video check
Small talk (location, day, etc.)
Reassurance about the nature of the conversation
First informal observation (confidence, comfort, clarity)
                                
PHASE 2 - aim is to get to know them and covers the following -
Personal intro: work, student life, family
Their connection to children
Reasons for volunteering
Their strengths/comfort areas
Their concerns or fears (if any)
                                
PHASE 3 - aim to explain about how SERVE platform works, build enthusiasm and clarity, reduce fear of teaching
covers the following - 
Explain that it follows a Smart class setup (TV in rural school, where volunteer teacher is connected remote)
Class timing (30 - 45mins, once/twice a week)
Support given to volunteer teachers: textbooks, lesson plans, orientation
Volunteer role: clarity, patience, connection more important than “teaching expertise”
Need to connect with children and see how they respond
                                
PHASE 4 - aim is to understand the Commitment, Availability & Prior Experience of volunteer and Assess reliability
Confirm realistic time availability, Understand routine and constraints, Check any past volunteering/teaching exposure, Understand how they handle scheduling changes
Covers the following questions -
Preferred days/times
How they plan to maintain consistency
How they handle sudden work/personal events
Prior experience with kids (even informal)
Communication responsibility (informing early)
                                
PHASE 5 - covers any questions asked by the volunteer
                                





""").strip()

# ---------------------------------------
# Phase definitions (ordered)
# ---------------------------------------
PHASES = [
    {"id": 1, "name": "Welcome & Rapport"},
    {"id": 2, "name": "Getting to Know (Background & Motivation)"},
    {"id": 3, "name": "Program Explanation"},
    {"id": 4, "name": "Commitment, Availability & Experience"},
    {"id": 5, "name": "FAQs"},
    {"id": 6, "name": "Closing & Internal Decision"}
]

# prompts per phase (agent will ask these as the main guide)
PHASE_PROMPTS = {
    1: "Hi! 👋 It’s so nice to meet you. I’m here to help you get started with your volunteer journey. Could you please tell me your name and how you are today?",
    2: "Thanks! Could you tell me a little about yourself — work/study, and what brought you to volunteering with children?",
    3: "Quickly, let me explain SERVE so you know what to expect: we connect volunteers to classrooms via a smart TV; sessions are short (30–45 min) and we provide lesson plans and orientation. Does that sound good?",
    4: "What days/times usually work for you for a 30-minute session? Also, have you taught or volunteered before (even informal experiences)?",
    5: "Do you have any questions for me about the program, technology, or the classroom setup?",
    6: "Thank you so much. I’ll share next steps with you soon. Any last thing you want to tell me before we finish?"
}

# scoring rubrics mapping (simple)
PHASE_SCORE_PROMPTS = {
    1: "Score comfort, clarity, and engagement in this phase on 1–5 where 5 excellent. Return JSON: {\"score\": <num>, \"notes\": \"...\"}",
    2: "Score motivation, empathy, and stability on 1–5. JSON output: {\"score\": <num>, \"notes\":\"...\"}",
    3: "Score understanding of program and comfort with idea of teaching on 1–5. JSON output.",
    4: "Score availability consistency, reliability, and communication responsibility on 1–5. JSON output.",
    5: "Score clarity of questions and comfort asking doubts on 1–5. JSON output.",
    6: "Combine prior phase signals and give an overall recommendation score 1–5 and a short final note. JSON output."
}

# ---------------------------------------
# Utility functions
# ---------------------------------------
def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_transcript_and_meta(history, extracted, scores, filename_prefix=None):
    if not filename_prefix:
        filename_prefix = f"vol_{now_ts()}_{uuid.uuid4().hex[:6]}"
    txt_file = os.path.join("records", f"{filename_prefix}.txt")
    json_file = os.path.join("records", f"{filename_prefix}.json")
    # write readable transcript
    with open(txt_file, "w", encoding="utf-8") as f:
        for m in history:
            role = m.get("role")
            content = m.get("content","")
            f.write(f"{role.upper()}: {content}\n\n")
    # write structured json
    meta = {
        "history": history,
        "extracted": extracted,
        "scores": scores,
        "timestamp": now_ts()
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return txt_file, json_file

def llm_chat_call(messages, model=MODEL, max_tokens=512):
    # messages is list of dicts with role/content
    # return assistant text (string) and the raw response
    resp = client.chat.completions.create(model=model, messages=messages)
    text = resp.choices[0].message.content
    return text, resp

def extract_key_fields(text_block):
    # Ask the model to extract the canonical fields and return JSON
    prompt = textwrap.dedent(f"""
    Extract the following fields from the conversation text below and return valid JSON ONLY:
    - name (string or null)
    - experience (short string summarizing prior teaching/volunteering)
    - languages (list of language strings)
    - subjects (list of strings)
    - availability (short string)
    - motivation (short string)
    - concerns (short string or null)

    Conversation:
    \"\"\"{text_block}\"\"\"
    """).strip()
    messages = [
        {"role": "system", "content": "You are an extraction assistant. Output valid JSON only."},
        {"role": "user", "content": prompt}
    ]
    out, _ = llm_chat_call(messages)
    # try to parse JSON from the response; if fails, return raw text under 'raw'
    try:
        parsed = json.loads(out)
        return parsed
    except Exception:
        return {"raw": out}

def score_phase(phase_id, text_block):
    # Ask the model to produce a numeric score (1-5) and short notes in JSON
    system = "You are an evaluator. Use the rubric provided. Output JSON only."
    user = f"Phase {phase_id} evaluation. Text:\n'''{text_block}'''\n\n{PHASE_SCORE_PROMPTS[phase_id]}"
    messages = [{"role":"system","content":system}, {"role":"user","content":user}]
    out, _ = llm_chat_call(messages)
    try:
        parsed = json.loads(out)
        # normalize numeric
        parsed['score'] = float(parsed.get('score', 0))
        return parsed
    except Exception:
        # fallback: ask model to give score inline if parsing failed
        return {"raw": out}

def compute_overall_recommendation(scores):
    # compute average numeric if available
    numeric_scores = []
    for p in PHASES:
        pid = p['id']
        sc = scores.get(pid)
        if isinstance(sc, dict) and isinstance(sc.get("score"), (int,float)):
            numeric_scores.append(float(sc["score"]))
    if not numeric_scores:
        return {"recommendation": "Hold", "reason": "Insufficient numeric scores"}
    avg = sum(numeric_scores)/len(numeric_scores)
    if avg >= 4.0:
        rec = "Recommend"
    elif avg >= 2.5:
        rec = "Hold / Re-screen"
    else:
        rec = "Not Recommended"
    return {"avg": round(avg,2), "recommendation": rec}

# ---------------------------------------
# Streamlit UI + State init
# ---------------------------------------
st.set_page_config(page_title="Volunteer Screening – Shiksha Mitra", layout="wide")
st.markdown("<style>.block-container{padding:0.6rem 1rem 1rem 1rem;}</style>", unsafe_allow_html=True)
st.title("Shiksha Mitra — Volunteer Screening (Phase 1)")

if "history" not in st.session_state:
    st.session_state.history = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role":"assistant", "content": "🌼 Hi! I’m Shiksha Mitra — so nice to meet you. I’ll ask a few simple questions to understand your background and availability so we can find the best volunteering match. Ready to begin? Can I have your name?"}
    ]
if "phase" not in st.session_state:
    st.session_state.phase = 1
if "scores" not in st.session_state:
    st.session_state.scores = {}   # {phase_id: {score:..., notes:...}, ...}
if "extracted" not in st.session_state:
    st.session_state.extracted = {}
if "auto_save_name" not in st.session_state:
    st.session_state.auto_save_name = f"vol_{now_ts()}"

# Sidebar: extracted + scores
with st.sidebar:
    st.header("Snapshot")
    st.markdown(f"**Phase:** {st.session_state.phase} — {PHASES[st.session_state.phase-1]['name']}")
    st.subheader("Key extracted fields")
    st.json(st.session_state.extracted if st.session_state.extracted else {"info":"No fields yet"})
    st.subheader("Phase scores")
    if st.session_state.scores:
        for pid, val in st.session_state.scores.items():
            if isinstance(val, dict):
                st.markdown(f"**Phase {pid}** — {PHASES[pid-1]['name']}: {val.get('score','-')}")
                st.write(val.get("notes",""))
    else:
        st.write("No scores yet")
    st.markdown("---")
    st.button("Save snapshot now", key="save_snapshot")
    if st.session_state.get("save_snapshot"):
        txtf, jf = save_transcript_and_meta(st.session_state.history, st.session_state.extracted, st.session_state.scores, st.session_state.auto_save_name)
        st.success(f"Saved to {txtf} and {jf}")

# Main chat container
st.markdown("""
<style>
.chat-box { height: 520px; overflow-y: auto; padding: 12px; border-radius:10px; border:1px solid #ddd; background:#fff; }
</style>
""", unsafe_allow_html=True)

chat_col, control_col = st.columns([3,1])

with chat_col:
    #st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    # render messages (skip system)
    for msg in st.session_state.history[1:]:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            st.chat_message("assistant").markdown(content)
        else:
            st.chat_message("user").markdown(content)
    #st.markdown('</div>', unsafe_allow_html=True)

    # current phase prompt anchor (shows the guideline)
    st.info(f"Phase {st.session_state.phase}: {PHASES[st.session_state.phase-1]['name']}\n\nTip: {PHASE_PROMPTS[st.session_state.phase]}")

    # user input
    user_input = st.chat_input("Type volunteer reply (or paste transcript/clipped audio text):")
    if user_input:
        # 1) store user message
        st.session_state.history.append({"role":"user","content":user_input})
        # 2) call model to generate assistant reply (follow-up or friendly next Q)
        messages = st.session_state.history.copy()
        assistant_text, _ = llm_chat_call(messages)
        st.session_state.history.append({"role":"assistant","content":assistant_text})
        # 3) update extraction from conversation so far (optional: only from last X messages)
        conversation_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history if m['role']!='system'])
        extracted = extract_key_fields(conversation_text)
        st.session_state.extracted = extracted
        # 4) run phase scoring for current phase
        score = score_phase(st.session_state.phase, conversation_text)
        st.session_state.scores[st.session_state.phase] = score
        # 5) auto-save snapshot after each message (append)
        save_transcript_and_meta(st.session_state.history, st.session_state.extracted, st.session_state.scores, st.session_state.auto_save_name)
        # 6) refresh UI (rerun)
        st.rerun()

with control_col:
    st.markdown("### Controls")
    if st.button("Next Phase"):
        if st.session_state.phase < len(PHASES):
            st.session_state.phase += 1
            # push a guiding a ssistant message for the new phase
            guide = PHASE_PROMPTS[st.session_state.phase]
            st.session_state.history.append({"role":"assistant","content": guide})
            # score the previous phase one last time using accumulated conversation
            conv = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history if m['role']!='system'])
            st.session_state.scores[st.session_state.phase-1] = score_phase(st.session_state.phase-1, conv)
            save_transcript_and_meta(st.session_state.history, st.session_state.extracted, st.session_state.scores, st.session_state.auto_save_name)
            st.rerun()
    if st.button("End Interview"):
        # final scoring and recommendation
        conv = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history if m['role']!='system'])
        # ensure all phases scored
        for pid in range(1, len(PHASES)+1):
            if pid not in st.session_state.scores:
                st.session_state.scores[pid] = score_phase(pid, conv)
        overall = compute_overall_recommendation(st.session_state.scores)
        st.session_state.scores['overall'] = overall
        # final assistant closing summary (not revealing internal tag)
        summary_prompt = f"Create a short coordinator-facing summary (3-6 lines) and next steps from this conversation. Conversation:\n\n{conv}\n\nAlso include a final recommendation label (Recommend / Hold / Not Recommended) and a one-line reason."
        summary, _ = llm_chat_call([{"role":"system","content":"You are a coordinator summarizer."},{"role":"user","content":summary_prompt}])
        st.session_state.history.append({"role":"assistant","content": "Interview complete. Summary (for coordinator):\n\n" + summary})
        save_transcript_and_meta(st.session_state.history, st.session_state.extracted, st.session_state.scores, st.session_state.auto_save_name)
        st.experimental_rerun()

    st.markdown("---")
    if st.button("Export latest transcript & meta now"):
        txtf, jf = save_transcript_and_meta(st.session_state.history, st.session_state.extracted, st.session_state.scores, st.session_state.auto_save_name)
        st.success(f"Saved: {txtf}\n{jf}")

    st.markdown("### Recommendation (live)")
    # live compute
    rec = compute_overall_recommendation(st.session_state.scores)
    st.write(rec)

# end of app
