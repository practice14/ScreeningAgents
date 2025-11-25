# ----------------------------
# File: agents/phase1_greeting.py
# ----------------------------
PHASE_ID = 1
PHASE_NAME = "Greeting & Rapport"
PHASE_PROMPT = (
    "You are the Phase-1 agent (Greeting & Rapport) for Shiksha Mitra."
    " Always be warm and friendly. Goal: establish rapport, check audio/video comfort,"
    " do light small talk (location/day), reassure this is casual. Ask ONE question at a time."
    " End phase only when you have at least one clear volunteer response and the volunteer seems comfortable."
)

def run_phase(client, model, history):
    # history: list of {'role':..,'content':..}
    messages = [{"role":"system","content":PHASE_PROMPT}] + history
    reply, raw = chat_completion(client, model, messages)
    # simple heuristic: include phase_complete tag text for controller to parse
    # we'll format an answer asking next question; controller will decide to advance
    return {"reply": reply, "phase_complete": False, "raw": raw}