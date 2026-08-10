"""
chat_logic.py
Shared business logic used by BOTH policy_app.py (browser/desktop) and
policy_mobile_app.py (mobile) — retrieval, intent-matching patterns,
canned replies, email drafting. Deliberately contains NO Streamlit UI
rendering code; each app file owns its own visual layout independently,
but both share this so patterns/replies never drift out of sync.
"""

import os
import re
import time
import requests

# Local dev default stays 127.0.0.1. On Railway, set this env var to the
# backend service's private networking address — see
# railway/RAILWAY_DEPLOY.md
KNOWLEDGE_SERVICE_URL = os.environ.get("KNOWLEDGE_SERVICE_URL", "http://127.0.0.1:8600")

BOT_AVATAR = "assets/bot-avatar.png"
USER_AVATAR = "assets/user-avatar.png"

# --- CHANGE THESE to your actual HR mailboxes before real use ---------
HR_EMAIL_PRIMARY = "hr@yourcompany.com"
HR_EMAIL_SECONDARY = "hr-support@yourcompany.com"

TOP_K = 4

CONFIDENCE_THRESHOLD = 0.50
OFF_TOPIC_THRESHOLD = 0.30

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
# NOTE: reference only — the actual prompt used is the one baked into
# knowledge_service.py at its own startup; keep them in sync if edited.


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
    """Streams the answer from knowledge_service's /generate endpoint."""
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
