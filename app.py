"""
app.py
-------
Streamlit chatbot for the 4-domain hybrid TF-IDF + SBERT QA engine.

Login is OPTIONAL (like any normal AI tool): the chatbot works immediately
as a guest, no account needed. Logging in (via the popover at the bottom of
the sidebar) only adds saved chat history. A browser refresh continues where you
were; opening the site again starts a fresh guest session. If a
guest logs in mid-conversation, their current chat is automatically adopted
into the new account. Chat history (when logged in) only ever shows
non-empty chats, created lazily on the first real message.

Run with:  streamlit run app.py
"""

import json
import re
import secrets
import streamlit as st
import db
import smalltalk

DOMAIN_LABELS = {
    "cs": "💻 Computer Science",
    "cooking": "🍳 Cooking",
    "gk": "🌍 General Knowledge",
    "medical": "🩺 Medical",
}
DOMAIN_NAMES = {"cs": "Computer Science", "cooking": "Cooking", "gk": "General Knowledge", "medical": "Medical"}
DOMAIN_ALIASES = {
    "cs": ["computer science", "computer", "cs", "coding", "programming"],
    "cooking": ["cooking", "cook", "food", "recipe", "recipes"],
    "gk": ["general knowledge", "general", "gk"],
    "medical": ["medical", "medicine", "health", "doctor"],
}
SWITCH_COMMANDS = ("change domain", "switch domain", "change")
MAX_MEDICAL_CHARS = 1200

st.set_page_config(page_title="Multi-Domain QA Chatbot", page_icon="🤖", layout="centered")

# ---- Chat bubble styling (replaces Streamlit's default chat_message look) ----
st.markdown("""
<style>
.chat-row { display: flex; margin: 6px 0; }
.chat-row.user { justify-content: flex-end; }
.chat-row.assistant { justify-content: flex-start; }
.bubble {
    max-width: 75%;
    padding: 10px 16px;
    border-radius: 18px;
    line-height: 1.45;
    font-size: 0.95rem;
}
.bubble.user {
    background: #2563eb;
    color: white;
    border-bottom-right-radius: 4px;
}
.bubble.assistant {
    background: #f1f3f5;
    color: #111;
    border-bottom-left-radius: 4px;
}
@media (prefers-color-scheme: dark) {
    .bubble.assistant { background: #2a2a2a; color: #eee; }
}
/* "typing..." indicator: three bouncing dots shown while the answer is being prepared */
.typing { display: inline-flex; gap: 5px; align-items: center; height: 18px; }
.typing span {
    width: 8px; height: 8px; border-radius: 50%; background: #9aa0a6;
    animation: typing-bounce 1.2s infinite ease-in-out;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing-bounce {
    0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
    40% { opacity: 1; transform: translateY(-4px); }
}
</style>
""", unsafe_allow_html=True)


def render_bubble(role, content):
    css_role = "user" if role == "user" else "assistant"
    import html as _html
    # allow the markdown we generate ourselves (bold/newlines) but escape raw user text
    safe = _html.escape(content).replace("\n", "<br>")
    # re-enable the **bold** we generate ourselves (domain names, disclaimers)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"\*(.+?)\*", r"<i>\1</i>", safe)
    st.markdown(
        f'<div class="chat-row {css_role}"><div class="bubble {css_role}">{safe}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading the AI model and knowledge base (first time only)...")
def load_engine(domain):
    from matcher import get_engine
    return get_engine(domain)


def shorten(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("\n"))
    cut = cut[: end + 1] if end > limit * 0.5 else cut
    return cut.strip() + " …(answer shortened)"


FALLBACK = ("Hmm, I couldn't find a confident answer to that in this domain. "
            "Try rephrasing, or type 'change domain' if you meant a different topic.")

TYPING_HTML = ('<div class="chat-row assistant"><div class="bubble assistant">'
               '<div class="typing"><span></span><span></span><span></span></div></div></div>')


def format_answer(result, domain):
    """The stored answer, shortened for the Medical domain, plus the medical disclaimer."""
    answer = result["answer"]
    if domain == "medical":
        answer = shorten(answer, MAX_MEDICAL_CHARS)
    parts = [answer]
    if domain == "medical":
        parts.append("⚠️ General information only, not medical advice. "
                     "Please consult a doctor for medical concerns.")
    return "\n\n".join(parts)


def format_reply(result, domain):
    return format_answer(result, domain) if result["confident"] else FALLBACK


@st.cache_resource(show_spinner=False)
def get_spellfixer():
    """Spelling corrector built from our own datasets. Returns None if pyspellchecker isn't installed
    (the chatbot then simply works without typo correction)."""
    try:
        from textfix import SpellFixer
        return SpellFixer()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def start_preload():
    """Load the other domains' indexes quietly in the background (only those already cached on disk)."""
    try:
        from matcher import preload_cached_engines
        return preload_cached_engines()
    except Exception:
        return None


def answer_question(text, domain):
    """
    Spelling fix -> match in the chosen domain -> if no confident match, look in the other domains.

    A correction is only USED when the match depends on it (see matcher.correction_helps), so names
    and new terms the dictionary doesn't know are left alone.
    The reply always lists which words were changed.
    """
    from matcher import best_other_domain, correction_helps, correction_supported
    fixer = get_spellfixer()
    fixed, changes = fixer.fix_detailed(text) if fixer else (text, [])
    fixes = ", ".join(f"{a} → {b}" for a, b in changes)

    engine = load_engine(domain)
    result, note = engine.query(text), ""
    if changes:
        corrected = engine.query(fixed)
        if correction_helps(result, corrected, changes):
            result, note = corrected, f"Showing results for: *{fixed}* (corrected {fixes})\n\n"

    if result["confident"]:
        return note + format_answer(result, domain)

    # not answerable here: is it a question for another domain?
    other = best_other_domain(text, domain)
    if changes:
        other_fixed = best_other_domain(fixed, domain)
        if other_fixed and correction_supported(other_fixed[1], changes) and \
                (other is None or other_fixed[1]["sbert_score"] >= other[1]["sbert_score"]):
            other, note = other_fixed, f"Showing results for: *{fixed}* (corrected {fixes})\n\n"
    if other:
        other_domain, other_result = other
        return (note
                + f"This looks like a **{DOMAIN_NAMES[other_domain]}** question, but you're in "
                  f"**{DOMAIN_NAMES[domain]}**. Type 'change domain' to switch for quicker answers. "
                  "Here is the answer anyway:\n\n"
                + format_answer(other_result, other_domain))
    return FALLBACK


def detect_domain(text):
    text = text.strip().lower()
    for key, aliases in DOMAIN_ALIASES.items():
        for a in aliases:
            if re.search(r"\b" + re.escape(a) + r"\b", text):
                return key
    return None


# =================== Model warmup (BEFORE login, shared resource) ===================
with st.spinner("Starting up (one-time setup, ~10-30s)..."):
    from matcher import get_sbert_model
    get_sbert_model()
    from preprocess import preprocess_text
    preprocess_text("warm up")   # loads NLTK data now instead of during the first question
    get_spellfixer()    # loads the word list from disk (built once by build_cache.py)
    start_preload()     # other domains load quietly in the background


# =================== Tab id: a refresh continues, opening the site again starts fresh ===================
# The address gets a random "?s=..." id. A browser refresh keeps the address, so the same tab loads its
# own saved state (login + chat). Opening the site again (no "?s=") gets a new id = a new guest, new chat.
# Two tabs or two browsers therefore never share a login or a chat.
if "tab_id" not in st.session_state:
    st.session_state.tab_id = st.query_params.get("s") or secrets.token_hex(16)
if st.query_params.get("s") != st.session_state.tab_id:
    st.query_params["s"] = st.session_state.tab_id


# =================== LOGIN IS OPTIONAL ===================
# No forced gate: the chatbot works immediately without an account, like any
# normal AI tool. Logging in is only needed to SAVE chat history across
# visits. Guest conversations stay in this browser tab only and are lost on
# refresh - that's the expected, normal trade-off of not having an account.
if "user" not in st.session_state:
    st.session_state.user = None

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "domain" not in st.session_state:
    st.session_state.domain = None
if "title_set" not in st.session_state:
    st.session_state.title_set = True
if "pending_messages" not in st.session_state:
    st.session_state.pending_messages = []


# =================== Continue where the user stopped (page refresh) ===================
def save_state():
    s = st.session_state
    data = {"user": s.user, "domain": s.domain, "chat_id": s.chat_id, "title_set": s.title_set}
    if s.chat_id is None:                      # nothing in the database yet: keep the messages here
        data["messages"] = s.messages
        data["pending"] = s.pending_messages
    try:
        db.save_ui_state(s.tab_id, json.dumps(data))
    except Exception:
        pass


if "restored" not in st.session_state:
    st.session_state.restored = True
    raw = db.load_ui_state(st.session_state.tab_id)
    if raw:
        try:
            saved = json.loads(raw)
            st.session_state.user = saved.get("user")
            cid = saved.get("chat_id")
            if cid is not None and st.session_state.user and db.chat_belongs_to(cid, st.session_state.user):
                st.session_state.chat_id = cid
                st.session_state.messages = db.get_chat_messages(cid)
            elif cid is None:
                st.session_state.messages = saved.get("messages", [])
                st.session_state.pending_messages = saved.get("pending", [])
            st.session_state.domain = saved.get("domain")
            st.session_state.title_set = saved.get("title_set", True)
        except Exception:
            pass

username = st.session_state.user  # None while browsing as a guest


def do_login(user):
    st.session_state.user = user
    save_state()
    if "page" in st.query_params:
        del st.query_params["page"]
    st.rerun()


# =================== Full-page Log in / Sign up ===================
if st.query_params.get("page") in ("login", "signup") and st.session_state.user is None:
    st.markdown("""<style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none; }
    .block-container { max-width: 460px; padding-top: 6vh; }
    </style>""", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center'>🤖 Multi-Domain QA Chatbot</h2>"
                "<p style='text-align:center;opacity:.7'>Log in to save your chats and continue them later.</p>",
                unsafe_allow_html=True)
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            if db.verify_user(u, p):
                do_login(u.strip())
            else:
                st.error("Incorrect username or password.")
    with tab_signup:
        with st.form("signup_form"):
            u2 = st.text_input("Choose a username", key="signup_u")
            p2 = st.text_input("Choose a password", type="password", key="signup_p")
            submitted2 = st.form_submit_button("Create account", use_container_width=True)
        if submitted2:
            ok, msg = db.create_user(u2, p2)
            if ok:
                st.success("Account created successfully! Now open the **Log in** tab and log in to start chatting.")
            else:
                st.error(msg)
    if st.button("← Back to chat", use_container_width=True):
        del st.query_params["page"]
        st.rerun()
    st.stop()


def start_new_chat():
    # Lazy creation: NO database row yet. Only created once the first real
    # message is sent, so an untouched "New chat" never clutters the sidebar.
    st.session_state.chat_id = None
    st.session_state.messages = []
    st.session_state.domain = None
    st.session_state.title_set = True
    st.session_state.pending_messages = []
    save_state()


def open_chat(chat_id, domain):
    st.session_state.chat_id = chat_id
    st.session_state.messages = db.get_chat_messages(chat_id)
    st.session_state.domain = domain
    st.session_state.title_set = True
    st.session_state.pending_messages = []
    save_state()


def add_message(role, content, persist=True):
    """
    persist=False: show the message, but DON'T create a chat for it yet
    (used for domain-pick messages, since picking a domain alone shouldn't
    make a chat appear in the sidebar). Queued and saved retroactively once
    a real chat actually gets created.

    Guests (not logged in): everything is queued in memory only - there's no
    username to save a chat under. If they log in LATER in the same session,
    the next real message flushes this entire queued conversation into their
    new account automatically, so nothing is lost by logging in partway through.
    """
    st.session_state.messages.append({"role": role, "content": content})
    current_user = st.session_state.user
    if not persist or current_user is None:
        st.session_state.pending_messages.append({"role": role, "content": content})
        save_state()
        return
    if st.session_state.chat_id is None:
        st.session_state.chat_id = db.create_chat(current_user, title="New chat", domain=st.session_state.domain)
    # write anything queued earlier (domain pick, small talk, guest messages) first, in order
    for m in st.session_state.pending_messages:
        db.add_message(st.session_state.chat_id, m["role"], m["content"])
    st.session_state.pending_messages = []
    db.add_message(st.session_state.chat_id, role, content)
    save_state()


# ---- Sidebar ----
with st.sidebar:
    # --- top: chat controls ---
    if st.button("➕ New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    if username:
        chats = db.get_user_chats(username)
        if chats:
            st.caption("Your chats")
        for c in chats:
            label = c["title"] if c["title"] else "New chat"
            is_active = c["chat_id"] == st.session_state.chat_id
            if st.button(("🟢 " if is_active else "") + label, key=f"chat_{c['chat_id']}", use_container_width=True):
                open_chat(c["chat_id"], c["domain"])
                st.rerun()
    else:
        st.caption("💡 Log in (below) to save your chats across visits.")

    # --- bottom: account panel ---
    st.markdown("<div style='flex-grow:1'></div>", unsafe_allow_html=True)  # pushes account panel down
    st.divider()

    if username:
        st.markdown(f"👤 **{username}**")
        if st.button("Log out", use_container_width=True):
            tab = st.session_state.tab_id
            db.save_ui_state(tab, "{}")
            st.session_state.clear()
            st.session_state.tab_id = tab
            st.rerun()

        with st.popover("🗑️ Delete account", use_container_width=True):
            st.warning("This permanently deletes your account and ALL your saved chats.")
            pw = st.text_input("Enter your password to confirm", type="password", key="delete_pw")
            if st.button("Yes, delete my account", type="primary", use_container_width=True):
                if db.verify_user(username, pw):
                    db.delete_user(username)
                    tab = st.session_state.tab_id
                    db.save_ui_state(tab, "{}")
                    st.session_state.clear()
                    st.session_state.tab_id = tab
                    st.session_state.notice = "Your account was deleted."
                    st.rerun()
                else:
                    st.error("Incorrect password.")
    else:
        if st.button("👤 Log in / Sign up", use_container_width=True):
            st.query_params["page"] = "login"
            st.rerun()

st.title("🤖 Multi-Domain QA Chatbot")
if st.session_state.get("notice"):
    st.success(st.session_state.pop("notice"))

if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": ("Hi! 👋 Which domain do you want help with today?\n\n"
                    "Pick one below, or just type it: Computer Science, Cooking, "
                    "General Knowledge, or Medical."),
    })

for msg in st.session_state.messages:
    render_bubble(msg["role"], msg["content"])


def choose_domain(key):
    st.session_state.domain = key
    if st.session_state.chat_id is not None:
        db.set_chat_domain(st.session_state.chat_id, key)
    load_engine(key)
    add_message("assistant",
                f"Great, you're now in **{DOMAIN_NAMES[key]}**. Ask me anything! "
                f"(Type 'change domain' any time to switch.)", persist=False)
    st.session_state.title_set = False
    save_state()


# ---- Domain buttons ----
if st.session_state.domain is None:
    cols = st.columns(len(DOMAIN_LABELS))
    for col, (key, label) in zip(cols, DOMAIN_LABELS.items()):
        if col.button(label, use_container_width=True):
            add_message("user", label, persist=False)
            choose_domain(key)
            st.rerun()

# ---- Chat input ----
user_input = st.chat_input("Type your question here...")

if user_input:
    text = user_input.strip()
    domain = st.session_state.domain

    if text.lower() in SWITCH_COMMANDS:
        add_message("user", text)
        st.session_state.domain = None
        add_message("assistant", "Sure — which domain do you want now?")
        save_state()
        st.rerun()

    elif domain is None:
        key = detect_domain(text)
        chat_reply = None if key else smalltalk.reply(text, None)
        if key:
            add_message("user", text, persist=False)
            choose_domain(key)
        elif chat_reply:
            add_message("user", text, persist=False)
            add_message("assistant", chat_reply, persist=False)
        else:
            add_message("user", text, persist=False)
            add_message("assistant", "Please choose a domain first: Computer Science, "
                                     "Cooking, General Knowledge, or Medical.", persist=False)
        st.rerun()

    else:
        chat_reply = smalltalk.reply(text, DOMAIN_NAMES[domain])
        if chat_reply:                      # "hi", "thanks", "who are you"... normal conversation
            keep = st.session_state.chat_id is not None
            add_message("user", text, persist=keep)
            add_message("assistant", chat_reply, persist=keep)
            st.rerun()
        else:
            render_bubble("user", text)
            typing = st.empty()
            typing.markdown(TYPING_HTML, unsafe_allow_html=True)   # three dots while we work
            reply = answer_question(text, domain)
            typing.empty()
            render_bubble("assistant", reply)
            add_message("user", text)
            add_message("assistant", reply)

            if not st.session_state.title_set and st.session_state.chat_id is not None:
                db.set_chat_title(st.session_state.chat_id, text)
                st.session_state.title_set = True