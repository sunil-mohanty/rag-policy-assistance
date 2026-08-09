"""
policy_mobile_app.py — MOBILE version.
Dedicated app for mobile screens, hardcoded compact sizing throughout —
no responsive breakpoints or clamp() needed since this file is ALWAYS
viewed on a phone. For desktop/browser, use policy_app.py (port 8501).

IMPORTANT: this uses a CSS background-image for the logo, NOT a
positioned <img> element. An earlier version used <img> with
position:absolute (matching policy_app.py's desktop layout) and hit
serious cross-browser rendering bugs specifically on mobile (logo
rendering above/outside the visible viewport, or vanishing entirely) —
see project history. Backgrounds are naturally clipped to their
container and don't have that failure mode, which is exactly why this
file's logo rendering deliberately differs from policy_app.py's.

Run with:
    streamlit run policy_mobile_app.py --server.port 8503
(alongside: ollama serve, knowledge_service.py, admin_app.py)
"""

import base64
import random
import time

import requests
import streamlit as st

import chat_logic as logic

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="assets/inb-logo.png",
    layout="centered",
    initial_sidebar_state="collapsed",  # mobile: start collapsed, toggle stays available
)

LOGO_PATH = "assets/inb-logo.png"
with open(LOGO_PATH, "rb") as f:
    _logo_b64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Montserrat', sans-serif;
}}

.block-container {{
    padding-top: 1rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
}}

.hero-banner {{
    background-color: #041153;
    background-image:
        url('data:image/png;base64,{_logo_b64}'),
        linear-gradient(135deg, #041153 0%, #0C1F6B 100%);
    background-position: center 14px, center;
    background-repeat: no-repeat, no-repeat;
    background-size: 70px auto, cover;
    border-radius: 16px;
    padding: 60px 16px 16px 16px;
    margin-bottom: 16px;
    color: #FFFFFF;
    text-align: center;
}}
.hero-banner h1 {{
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0 0 2px 0;
    color: #FFFFFF;
}}
.hero-banner p {{
    font-size: 0.75rem;
    opacity: 0.85;
    margin: 0;
}}

.stButton > button {{
    border-radius: 9999px !important;
    padding: 8px 20px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border: none !important;
    background-color: #B02435 !important;
    color: #FFFFFF !important;
}}

[data-testid="stChatMessage"] {{
    border-radius: 20px !important;
    padding: 2px 4px;
    font-size: 0.92rem;
}}
[data-testid="stChatMessage"] img {{
    width: 28px !important;
    height: 28px !important;
}}

/* Mobile: sidebar toggle stays available (never hidden) — no room for
   a permanently-open sidebar alongside content on a phone screen. */
</style>

<div class="hero-banner">
    <h1>HR Policy Assistant</h1>
    <p>For better answers, please ask one policy topic at a time</p>
</div>
""", unsafe_allow_html=True)

try:
    _health = requests.get(f"{logic.KNOWLEDGE_SERVICE_URL}/health", timeout=5).json()
except requests.exceptions.RequestException as e:
    st.error(
        "Could not reach the knowledge service. Make sure it's running, "
        "or ingest at least one document via the Admin app first, which "
        "starts it automatically."
    )
    st.exception(e)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_email_topic" not in st.session_state:
    st.session_state.pending_email_topic = None
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False

# Same empty-chat scroll fix as the desktop app — Streamlit's chat UI
# auto-scrolls to the bottom even on a fresh, empty page, which can push
# the banner above the fold before a single message has been sent.
if not st.session_state.messages:
    st.markdown("""
    <script>
    setTimeout(function() {
        window.scrollTo(0, 0);
        const scrollContainer = window.parent.document.querySelector(
            '[data-testid="stAppScrollToBottomContainer"]'
        );
        if (scrollContainer) { scrollContainer.scrollTop = 0; }
    }, 150);
    </script>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    avatar = logic.BOT_AVATAR if msg["role"] == "assistant" else logic.USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if st.session_state.conversation_ended:
    if st.button("🔄 Start New Conversation", type="primary"):
        st.session_state.messages = []
        st.session_state.pending_email_topic = None
        st.session_state.conversation_ended = False
        st.rerun()
    st.stop()

question = st.chat_input("Ask about leave policy, benefits, code of conduct...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=logic.USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=logic.BOT_AVATAR):
        stripped = logic.normalize_for_intent_match(question)

        if logic.GREETING_PATTERN.match(stripped):
            reply_text = random.choice(logic.GREETING_REPLIES)
            answer = st.write_stream(logic.stream_text(reply_text))

        elif st.session_state.pending_email_topic and logic.AFFIRMATIVE_PATTERN.match(stripped):
            topic = st.session_state.pending_email_topic
            email_draft = logic.build_hr_email(topic)
            intro = "Here's a draft you can copy into Outlook and send:\n\n"
            st.write_stream(logic.stream_text(intro))
            st.markdown(email_draft)
            closing = "\n\n" + random.choice(logic.CLOSING_REPLIES)
            st.write_stream(logic.stream_text(closing))
            answer = intro + email_draft + closing
            st.session_state.pending_email_topic = None
            st.session_state.conversation_ended = True

        elif st.session_state.pending_email_topic and logic.NEGATIVE_PATTERN.match(stripped):
            answer = ("No problem! Let me know if there's anything else "
                       "I can help with.")
            st.write_stream(logic.stream_text(answer))
            st.session_state.pending_email_topic = None

        else:
            st.session_state.pending_email_topic = None

            try:
                with st.spinner(""):
                    t0 = time.perf_counter()
                    scored_chunks = logic.retrieve_scored_chunks(question)
                    t1 = time.perf_counter()
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            top_score = scored_chunks[0]["score"] if scored_chunks else 0.0
            confident_chunks = [c for c in scored_chunks if c["score"] >= logic.CONFIDENCE_THRESHOLD]

            print(f"[TIMING] Retrieval took {t1 - t0:.2f}s | top_score={top_score:.3f} | "
                  f"{len(confident_chunks)} chunk(s) above {logic.CONFIDENCE_THRESHOLD}")

            if confident_chunks:
                context = logic.format_context(confident_chunks)
                try:
                    answer = st.write_stream(logic.stream_generate(context, question))
                except requests.exceptions.RequestException:
                    st.error("Couldn't reach the knowledge service for generation. "
                              "Please try again in a moment.")
                    st.stop()

                with st.expander("📄 Retrieved policy excerpts"):
                    for c in confident_chunks:
                        st.markdown(f"**{c['source']}** · confidence: `{c['score']:.2f}`")
                        st.text(c["text"][:400] + "...")
                        st.divider()

            elif top_score >= logic.OFF_TOPIC_THRESHOLD:
                answer = ("As per my knowledge from the current HR policies "
                          "I will not be able to answer this. Do you want "
                          "me to draft email to HR on your behalf?")
                st.markdown(answer)
                st.session_state.pending_email_topic = question

            else:
                answer = random.choice(logic.OFF_TOPIC_REPLIES)
                st.write_stream(logic.stream_text(answer))

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.markdown("### 📋 Popular Topics")
    st.markdown("""
- Annual & sick leave entitlements
- Maternity & unpaid leave
- Code of conduct
- Remuneration & benefits
- Health & disability insurance
- Grievance & disciplinary process
- Relocation benefits
    """)
    st.divider()
    st.markdown(f"### 📧 Still need help?\nContact HR at **{logic.HR_EMAIL_PRIMARY}**")
    st.divider()
    if st.button("🔄 Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_email_topic = None
        st.session_state.conversation_ended = False
        st.rerun()