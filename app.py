import streamlit as st
from openai import OpenAI

# ---------------------------------------
# OpenRouter Client
# ---------------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# ---------------------------------------
# System prompt for the bot
# ---------------------------------------
SYSTEM_PROMPT = """
You are a Volunteer Screening Assistant. Your job is to conduct
a structured, warm, and friendly interview with volunteers.
Steps:
1. Build rapport and welcome them.
2. Ask about background, motivation, personality.
3. Explain the SERVE program and how classes work.
4. Check availability and commitment.
5. Ask about prior volunteer experience.
6. Answer FAQs.
7. Internally think at the end: Recommend or Hold.
"""

# ---------------------------------------
# Initialize chat history
# ---------------------------------------
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]

st.title("Volunteer Screening Bot (Llama 3.2 3B Instruct)")
# Initialize messages with default welcome message


with st.chat_message("ai"):
    st.write("Hi! 👋 It’s so nice to meet you. I’m here to help you get started with your volunteer journey. I’ll ask a few simple questions about your background, interests, and availability so we can find the best match for you. Take your time — and feel free to share anything you’re comfortable with. Ready whenever you are!")

# ---------------------------------------
# Chat container CSS
# ---------------------------------------
st.markdown("""
<style>
.chat-box {
    height: 5px;
    overflow-y: scroll;
    padding: 2px
    border: 1px solid #ddd;
    border-radius: 12px;
    background-color: #fafafa;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# CHAT DISPLAY
# ---------------------------------------
chat_container = st.container()

with chat_container:
    #st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    
    # Show all messages except system
    
    for msg in st.session_state.history[1:]:
        st.chat_message(msg["role"]).markdown(msg["content"])

    #st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------
# ---------------------------------------
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message
    st.session_state.history.append({"role": "user", "content": user_input})

    # Call LLM
    response = client.chat.completions.create(
        model="meta-llama/llama-3.2-3b-instruct",
        messages=st.session_state.history
    )

    bot_reply = response.choices[0].message.content

    # Add bot message
    st.session_state.history.append(
        {"role": "assistant", "content": bot_reply}
    )

    # Refresh UI to show latest messages
    st.rerun()
