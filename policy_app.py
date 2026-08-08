"""
policy_app.py
Streamlit chat UI for the HR Policy RAG assistant. This process NEVER
needs restarting — it talks to knowledge_service.py over HTTP for
everything data-related (search + generation), and that service is what
gets restarted after every ingest instead. See knowledge_service.py's
docstring for why.

Run with:
    streamlit run policy_app.py
(alongside: ollama serve, knowledge_service.py, and admin_app.py)
"""

import os
import re
import time
import random
import base64
import requests
import streamlit as st

# Local dev default stays 127.0.0.1. On Railway, set this env var to the
# backend service's private networking address, e.g.
# http://backend.railway.internal:8600 — see railway/RAILWAY_DEPLOY.md
KNOWLEDGE_SERVICE_URL = os.environ.get("KNOWLEDGE_SERVICE_URL", "http://127.0.0.1:8600")

BOT_AVATAR = "assets/bot-avatar.png"
USER_AVATAR = "assets/user-avatar.png"

# --- CHANGE THESE to your actual HR mailboxes before real use ---------
HR_EMAIL_PRIMARY = "hr@yourcompany.com"
HR_EMAIL_SECONDARY = "hr-support@yourcompany.com"

TOP_K = 4

# --- Confidence thresholds for retrieval -------------------------------
# This business logic stays here in the frontend, not in
# knowledge_service.py — that service is a deliberately "dumb", fast,
# freely-restartable data layer; deciding what to DO with the scores
# belongs with the UI/UX logic.
CONFIDENCE_THRESHOLD = 0.50
OFF_TOPIC_THRESHOLD = 0.30

# --- Greeting / small-talk short-circuit -------------------------------
_GREETING_ATOM = (
    r"(hi+(\s*there)?|hello+(\s*there)?|hey+(\s*there)?|yo|"
    r"good\s*(morning|afternoon|evening)|"
    r"how\s*are\s*you(\s*doing)?|how('?s|\s*is)\s*it\s*going|"
    r"how('?s|\s*is)\s*your\s*day|"
    r"what'?s\s*up|thanks?(\s*you)?|thank\s*you|"
    r"ok(ay)?|cool|great|nice|bye|goodbye|see\s*you(\s*later)?)"
)
GREETING_PATTERN = re.compile(
    rf"^(\s*{_GREETING_ATOM}\s*[,!.?]*\s*(and\s*)?)+$",
    re.IGNORECASE,
)

GREETING_REPLIES = [
    "Hey there! Hope you're having a good day so far. I'm the HR Policy "
    "Assistant — ask me anything about leave, benefits, code of conduct, "
    "and more.",
    "Hi! Hope things are going well on your end. I can help with "
    "questions about company HR policies — what would you like to know?",
    "Hello! Good to see you. I'm here to help with HR policy questions — "
    "leave, benefits, conduct, you name it. What's on your mind?",
    "Hey! Hope your day's going smoothly. Feel free to ask me anything "
    "about our HR policies whenever you're ready.",
    "Hi there! Hope you're doing well today. I'm your HR Policy "
    "Assistant — happy to help with any policy questions you've got.",
    "Hello! Hope it's been a good one so far. Ask away if you've got "
    "any HR policy questions — leave, benefits, conduct, etc.",
]

OFF_TOPIC_REPLIES = [
    "I'm really sorry, but I can only help with HR-related questions — "
    "things like leave, benefits, or company policies. Is there something "
    "HR-related I can help you with?",
    "Apologies, that's outside what I can help with — I'm focused on HR "
    "policy questions only. Happy to help if you've got one of those!",
    "Sorry about that, but I'm limited to HR policy topics — leave, "
    "benefits, conduct, and similar. Anything in that area I can help with?",
]

# --- Affirmative / negative replies to the "draft an email?" offer -----
_AFFIRM_ATOM = (
    r"(yes+|yeah+|yep|yup|sure|of\s*course|ok(ay)?|please|"
    r"go\s*ahead|do\s*it|draft\s*it|send\s*it|sounds\s*good)"
)
AFFIRMATIVE_PATTERN = re.compile(
    rf"^(\s*{_AFFIRM_ATOM}\s*[,!.]*\s*)+$",
    re.IGNORECASE,
)

_NEGATIVE_ATOM = r"(no+|nope|nah|not\s*now|no\s*thanks?|never\s*mind|nvm)"
NEGATIVE_PATTERN = re.compile(
    rf"^(\s*{_NEGATIVE_ATOM}\s*[,!.]*\s*)+$",
    re.IGNORECASE,
)

CLOSING_REPLIES = [
    "Thanks so much for reaching out — have a great day!",
    "Thank you for stopping by — hope the rest of your day goes well!",
    "Glad I could help point you in the right direction — have a great day!",
]


def normalize_for_intent_match(text):
    """Strip stray leading/trailing punctuation before checking
    greeting/yes/no intent."""
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text.strip())


def stream_text(text, delay=0.02):
    """Yield word-by-word with a small delay so canned replies get the
    same 'typing' feel as real streamed LLM answers."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def show_typing_indicator(min_seconds=0.7):
    """Animated typing-dots shown BEFORE any response text appears."""
    placeholder = st.empty()
    frames = ["typing", "typing.", "typing..", "typing..."]
    start = time.time()
    i = 0
    while time.time() - start < min_seconds:
        placeholder.markdown(f"*{frames[i % len(frames)]}*")
        time.sleep(0.25)
        i += 1
    placeholder.empty()


_FILLER_PREFIXES = [
    r"^what about\s+",
    r"^tell me about\s+",
    r"^can you tell me about\s+",
    r"^can you tell me\s+",
    r"^i want to know about\s+",
    r"^i wanted to know about\s+",
    r"^what is the policy on\s+",
    r"^what'?s the policy on\s+",
    r"^how do i\s+",
    r"^how can i\s+",
    r"^could you tell me\s+",
    r"^please tell me about\s+",
    r"^do you know\s+",
]
_FILLER_PATTERN = re.compile("|".join(_FILLER_PREFIXES), re.IGNORECASE)


def clean_query_for_retrieval(question):
    """Strip filler phrasing for the EMBEDDING/search step only."""
    cleaned = _FILLER_PATTERN.sub("", question.strip()).strip()
    return cleaned if cleaned else question


def build_hr_email(topic):
    return f"""**To:** {HR_EMAIL_PRIMARY}; {HR_EMAIL_SECONDARY}
**Subject:** HR Policy Query — Clarification Needed

Dear HR Team,

I recently used the HR Policy Assistant to look up the following question, but couldn't find a clear answer in the current policy documents:

"{topic}"

Could you please help clarify the applicable policy?

Thank you,
[Your Name]
"""


def retrieve_scored_chunks(question, k=TOP_K):
    """Calls knowledge_service's /search endpoint. Returns a list of
    dicts: {text, source, score}. Raises RuntimeError with a friendly
    message if the service is unreachable (e.g. mid-restart)."""
    search_query = clean_query_for_retrieval(question)
    try:
        resp = requests.post(
            f"{KNOWLEDGE_SERVICE_URL}/search",
            json={"question": search_query, "k": k},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "Couldn't reach the knowledge service. It may be restarting "
            "after a recent document update — please wait a few seconds "
            "and try again."
        ) from e

    for r in results:
        print(f"[SCORE] {r['score']:.3f}  ({r['source']}) — {r['text'][:60]!r}")
    return results


def format_context(chunks):
    return "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)


def stream_generate(context, question):
    """Streams the answer from knowledge_service's /generate endpoint,
    yielding plain text chunks suitable for st.write_stream — same
    live-typing feel as before, just sourced over HTTP now."""
    with requests.post(
        f"{KNOWLEDGE_SERVICE_URL}/generate",
        json={"context": context, "question": question},
        stream=True,
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="assets/inb-logo.png",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Invest Bank theme, exact colors sampled from the uploaded logo ------
LOGO_PATH = "assets/inb-logo.png"
with open(LOGO_PATH, "rb") as f:
    _logo_b64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Montserrat', sans-serif;
}}

.hero-banner {{
    background: linear-gradient(135deg, #041153 0%, #0C1F6B 100%);
    border-radius: 24px;
    padding: 24px 32px;
    margin-bottom: 28px;
    color: #FFFFFF;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 90px;
}}
.hero-banner img {{
    position: absolute;
    left: 32px;
    top: 50%;
    transform: translateY(-50%);
    height: 56px;
    background: #FFFFFF;
    border-radius: 12px;
    padding: 8px 12px;
}}
.hero-banner .hero-text {{
    text-align: left;
}}
.hero-banner .hero-text h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    color: #FFFFFF;
}}
.hero-banner .hero-text p {{
    font-size: 0.9rem;
    opacity: 0.85;
    margin: 0;
}}

.stButton > button {{
    border-radius: 9999px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    border: none !important;
    background-color: #B02435 !important;
    color: #FFFFFF !important;
    transition: transform 0.2s ease, opacity 0.2s ease;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    opacity: 0.9;
}}

[data-testid="stChatMessage"] {{
    border-radius: 20px !important;
    padding: 4px 8px;
}}

/* Sidebar fixed/no-minimize — DESKTOP ONLY. On mobile there's no room
   for a permanently-open sidebar alongside content, so the toggle needs
   to stay available there (see media query below, which overrides this
   back to visible under 640px). */
[data-testid="collapsedControl"] {{
    display: none !important;
}}
[data-testid="stSidebarCollapseButton"] {{
    display: none !important;
}}
button[kind="header"] {{
    display: none !important;
}}

/* --- Mobile fixes (screens under 640px) --------------------------- */
@media (max-width: 640px) {{
    /* Stack the banner instead of pinning the logo to the left edge —
       on a narrow screen the absolutely-positioned logo and the
       centered title text collide/overlap otherwise. */
    .hero-banner {{
        flex-direction: column;
        gap: 12px;
        padding: 20px;
    }}
    .hero-banner img {{
        position: static;
        transform: none;
    }}
    .hero-banner .hero-text {{
        text-align: center;
    }}

    /* Restore the sidebar's own toggle on mobile — without it there's
       no way to open/close the drawer on a screen too narrow for the
       sidebar and main content to coexist. */
    [data-testid="collapsedControl"] {{
        display: flex !important;
    }}
    [data-testid="stSidebarCollapseButton"] {{
        display: flex !important;
    }}
}}
</style>

<div class="hero-banner">
    <img src="data:image/png;base64,{_logo_b64}" alt="INB logo" />
    <div class="hero-text">
        <h1>HR Policy Assistant</h1>
        <p>For better answers, please ask one policy topic at a time</p>
    </div>
</div>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = """You are an HR Policy Assistant for the company.

Answer the employee's question using the policy context provided below.
The context has already been filtered to only include content judged
relevant to the question, so treat it as trustworthy — you do not need
an exact word-for-word match; a reasonable, well-supported inference from
the context counts as answerable.

Answering rules:
1. Give a concise, synthesized answer in your own words — 2 to 4
   sentences for most questions. NEVER paste or quote large blocks of the
   source text verbatim. Summarize; don't reproduce the document.
2. Always end your answer with the source document name(s), like:
   (Source: Leave_Policy.pdf)

Context:
{context}
"""
# NOTE: SYSTEM_PROMPT is defined here for reference/consistency but the
# actual prompt used is the one baked into knowledge_service.py at its
# own startup — keep them in sync if you edit either one.

try:
    _health = requests.get(f"{KNOWLEDGE_SERVICE_URL}/health", timeout=5).json()
except requests.exceptions.RequestException as e:
    st.error(
        "Could not reach the knowledge service. Make sure it's running: "
        "`uvicorn knowledge_service:app --host 127.0.0.1 --port 8600` "
        "— or ingest at least one document via the Admin app first, "
        "which starts it automatically."
    )
    st.exception(e)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_email_topic" not in st.session_state:
    st.session_state.pending_email_topic = None
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False

for msg in st.session_state.messages:
    avatar = BOT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
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
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        stripped = normalize_for_intent_match(question)
        show_typing_indicator()

        if GREETING_PATTERN.match(stripped):
            reply_text = random.choice(GREETING_REPLIES)
            answer = st.write_stream(stream_text(reply_text))

        elif st.session_state.pending_email_topic and AFFIRMATIVE_PATTERN.match(stripped):
            topic = st.session_state.pending_email_topic
            email_draft = build_hr_email(topic)
            intro = "Here's a draft you can copy into Outlook and send:\n\n"
            st.write_stream(stream_text(intro))
            st.markdown(email_draft)
            closing = "\n\n" + random.choice(CLOSING_REPLIES)
            st.write_stream(stream_text(closing))
            answer = intro + email_draft + closing
            st.session_state.pending_email_topic = None
            st.session_state.conversation_ended = True

        elif st.session_state.pending_email_topic and NEGATIVE_PATTERN.match(stripped):
            answer = ("No problem! Let me know if there's anything else "
                       "I can help with.")
            st.write_stream(stream_text(answer))
            st.session_state.pending_email_topic = None

        else:
            st.session_state.pending_email_topic = None

            try:
                with st.spinner(""):
                    t0 = time.perf_counter()
                    scored_chunks = retrieve_scored_chunks(question)
                    t1 = time.perf_counter()
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            top_score = scored_chunks[0]["score"] if scored_chunks else 0.0
            confident_chunks = [c for c in scored_chunks if c["score"] >= CONFIDENCE_THRESHOLD]

            print(f"[TIMING] Retrieval took {t1 - t0:.2f}s | top_score={top_score:.3f} | "
                  f"{len(confident_chunks)} chunk(s) above {CONFIDENCE_THRESHOLD}")

            if confident_chunks:
                context = format_context(confident_chunks)
                try:
                    answer = st.write_stream(stream_generate(context, question))
                except requests.exceptions.RequestException as e:
                    st.error("Couldn't reach the knowledge service for generation. "
                              "Please try again in a moment.")
                    st.stop()

                with st.expander("📄 Retrieved policy excerpts (confidence scores)"):
                    for c in confident_chunks:
                        st.markdown(f"**{c['source']}** · confidence: `{c['score']:.2f}`")
                        st.text(c["text"][:400] + "...")
                        st.divider()

            elif top_score >= OFF_TOPIC_THRESHOLD:
                answer = ("As per my knowledge from the current HR policies "
                          "I will not be able to answer this. Do you want "
                          "me to draft email to HR on your behalf?")
                st.write_stream(stream_text(answer))
                st.session_state.pending_email_topic = question

            else:
                answer = random.choice(OFF_TOPIC_REPLIES)
                st.write_stream(stream_text(answer))

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

    st.markdown("### 💡 Tips for better answers")
    st.markdown("""
- Ask one policy topic at a time
- Be specific — e.g. *"How many annual leave days do I get?"*
  works better than *"tell me about leave"*
    """)

    st.divider()

    st.markdown("### 📧 Still need help?")
    st.markdown(f"Contact HR directly at **{HR_EMAIL_PRIMARY}**")

    st.divider()

    if st.button("🔄 Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_email_topic = None
        st.session_state.conversation_ended = False
        st.rerun()

    st.divider()

    st.markdown("### ⚙️ Knowledge service")
    try:
        h = requests.get(f"{KNOWLEDGE_SERVICE_URL}/health", timeout=2).json()
        st.caption(f"Serving version: `{h['version']}`")
    except requests.exceptions.RequestException:
        st.caption("⚠️ Not reachable right now")