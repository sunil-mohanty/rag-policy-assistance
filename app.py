"""
app.py
Streamlit chat UI for the HR Policy RAG assistant.
Fully self-hosted: Ollama for LLM + embeddings, Chroma for vector search.

Document upload/ingestion now happens on the Admin page (see pages/) —
no terminal, no manual `python ingest.py`, and no restart needed here
afterward. See rag_core.py for how that's wired.

Run with:
    streamlit run app.py
"""

import re
import time
import random
import base64
import streamlit as st
import rag_core
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BOT_AVATAR = "assets/bot-avatar.png"
USER_AVATAR = "assets/user-avatar.png"

# --- CHANGE THESE to your actual HR mailboxes before real use ---------
HR_EMAIL_PRIMARY = "hr@yourcompany.com"
HR_EMAIL_SECONDARY = "hr-support@yourcompany.com"

TOP_K = 4

# --- Confidence thresholds for retrieval -------------------------------
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
    """Strip stray leading/trailing punctuation (typos like a trailing
    backslash, extra periods, etc.) before checking greeting/yes/no
    intent — handles cases like 'Hello\\' that a fixed punctuation list
    would otherwise miss, without needing to enumerate every possible
    stray character by hand."""
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text.strip())


def stream_text(text, delay=0.02):
    """Yield word-by-word with a small delay so canned replies get the
    same 'typing' feel as real streamed LLM answers, via st.write_stream."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def show_typing_indicator(min_seconds=0.7):
    """Animated typing-dots shown BEFORE any response text appears, for
    every response path — mimics the pause before a human starts typing
    back, rather than replies appearing instantly. Cleared automatically
    once done."""
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
    """Strip filler phrasing for the EMBEDDING/search step only. The LLM
    still sees the original, unmodified question — this only affects
    which policy chunks get retrieved."""
    cleaned = _FILLER_PATTERN.sub("", question.strip()).strip()
    return cleaned if cleaned else question


def build_hr_email(topic):
    """Return a ready-to-copy email draft for an uncovered policy question."""
    return f"""**To:** {HR_EMAIL_PRIMARY}; {HR_EMAIL_SECONDARY}
**Subject:** HR Policy Query — Clarification Needed

Dear HR Team,

I recently used the HR Policy Assistant to look up the following question, but couldn't find a clear answer in the current policy documents:

"{topic}"

Could you please help clarify the applicable policy?

Thank you,
[Your Name]
"""


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

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="assets/inb-logo.png",
    layout="centered",
    initial_sidebar_state="expanded",  # always visible, no minimize
)

# --- Invest Bank theme, using EXACT colors sampled from the uploaded logo ---
# Navy #041153 and red #B02435 were read directly from the logo's pixels
# via PIL — not approximated. See assets/inb-logo.png.
LOGO_PATH = "assets/inb-logo.png"
with open(LOGO_PATH, "rb") as f:
    _logo_b64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Montserrat', sans-serif;
}}

/* Hero-style header banner — logo pinned left, text centered independently */
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

/* Pill-shaped buttons, matching the site's rounded-full CTA style */
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

/* Rounded chat bubbles, matching the site's generous card radii */
[data-testid="stChatMessage"] {{
    border-radius: 20px !important;
    padding: 4px 8px;
}}

/* Sidebar is fixed — hide the collapse/expand controls entirely */
[data-testid="collapsedControl"] {{
    display: none !important;
}}
[data-testid="stSidebarCollapseButton"] {{
    display: none !important;
}}
button[kind="header"] {{
    display: none !important;
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


@st.cache_resource
def get_generation_chain():
    """Cached prompt|llm|parser chain. Built from rag_core.get_llm(),
    which is independent of document ingestion — never needs clearing."""
    llm = rag_core.get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


def format_docs(docs):
    formatted = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        formatted.append(f"[Source: {src}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


def retrieve_scored_chunks(vectordb, question, k=TOP_K):
    """Return every candidate chunk with its raw relevance score.
    Searches using a filler-stripped version of the question to avoid
    diluting the embedding vector with non-search-relevant wording."""
    search_query = clean_query_for_retrieval(question)
    if search_query != question.strip():
        print(f"[QUERY] original={question!r} -> cleaned={search_query!r}")
    results = vectordb.similarity_search_with_relevance_scores(search_query, k=k)
    for doc, score in results:
        src = doc.metadata.get("source", "unknown")
        print(f"[SCORE] {score:.3f}  ({src}) — {doc.page_content[:60]!r}")
    return results


try:
    chain = get_generation_chain()
    vectordb = rag_core.get_vectordb()
except Exception as e:
    st.error(
        "Could not load the vector store. Use the separate Admin app "
        "(admin_app.py) to upload and ingest HR policy documents first, "
        "and make sure Ollama is running."
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

# --- Ended-conversation state: hide input, show restart button ---------
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
        show_typing_indicator()  # brief pause before ANY response text appears

        # --- 1. Greeting short-circuit --------------------------------
        if GREETING_PATTERN.match(stripped):
            reply_text = random.choice(GREETING_REPLIES)
            answer = st.write_stream(stream_text(reply_text))

        # --- 2. Answering the "draft an email?" yes/no offer -----------
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

        # --- 3. Normal RAG flow -----------------------------------------
        else:
            st.session_state.pending_email_topic = None  # any prior offer is moot now

            with st.spinner(""):
                t0 = time.perf_counter()
                scored_chunks = retrieve_scored_chunks(vectordb, question)
                t1 = time.perf_counter()

            top_score = scored_chunks[0][1] if scored_chunks else 0.0
            confident_chunks = [(d, s) for d, s in scored_chunks if s >= CONFIDENCE_THRESHOLD]

            print(f"[TIMING] Retrieval took {t1 - t0:.2f}s | top_score={top_score:.3f} | "
                  f"{len(confident_chunks)} chunk(s) above {CONFIDENCE_THRESHOLD}")

            if confident_chunks:
                # Found it — answer normally.
                retrieved_docs = [doc for doc, _score in confident_chunks]
                context = format_docs(retrieved_docs)
                answer = st.write_stream(
                    chain.stream({"context": context, "question": question})
                )
                with st.expander("📄 Retrieved policy excerpts (confidence scores)"):
                    for doc, score in confident_chunks:
                        src = doc.metadata.get("source", "unknown")
                        st.markdown(f"**{src}** · confidence: `{score:.2f}`")
                        st.text(doc.page_content[:400] + "...")
                        st.divider()

            elif top_score >= OFF_TOPIC_THRESHOLD:
                # Plausibly HR-related, but not covered by the documents.
                answer = ("As per my knowledge from the current HR policies "
                          "I will not be able to answer this. Do you want "
                          "me to draft email to HR on your behalf?")
                st.write_stream(stream_text(answer))
                st.session_state.pending_email_topic = question

            else:
                # Doesn't look HR-related at all.
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

    st.markdown("### 🔄 Knowledge base")
    current_version = rag_core._current_version()
    if current_version:
        st.caption(f"Active version: `{current_version}`")

    if st.session_state.get("just_refreshed"):
        st.success("Refreshed — now using the latest ingested documents.")
        st.session_state.just_refreshed = False

    if st.button("Refresh knowledge base"):
        st.session_state.just_refreshed = True
        st.rerun()  # forces get_vectordb() to re-read the pointer file fresh