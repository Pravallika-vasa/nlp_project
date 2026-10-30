"""
smalltalk.py
-------------
Friendly replies for normal conversation ("hi", "thanks", "how are you", ...),
so the bot doesn't answer every non-question with "I couldn't find an answer".

Only a message that is *entirely* small talk is caught (whole-message match), so a
real question such as "history of india" or "help me with python" is never
swallowed by mistake.
"""

import re
import random

GREET = r"(hi+|hello+|hey+|hii+|hai|hola|yo|namaste|greetings|good\s+(morning|afternoon|evening|day))"
HOW = r"(how\s+are\s+(you|u)|how\s+r\s+(you|u)|how\s+do\s+you\s+do|how(s|\s+is)\s+it\s+going|what'?s\s+up|whats\s+up|sup|how\s+have\s+you\s+been)"
THANKS = r"(thanks|thank\s+you|thank\s+u|thx|ty|thanks\s+a\s+lot|thank\s+you\s+so\s+much|many\s+thanks|thanks\s+a\s+ton|appreciate\s+it)"
BYE = r"(bye|goodbye|good\s+bye|see\s+you|see\s+ya|see\s+you\s+later|good\s*night|take\s+care|cya|talk\s+to\s+you\s+later)"
WHO = r"(who\s+are\s+(you|u)|what\s+are\s+(you|u)|what\s+is\s+your\s+name|what'?s\s+your\s+name|whats\s+your\s+name|your\s+name|tell\s+me\s+about\s+yourself|introduce\s+yourself)"
CAN = r"(what\s+can\s+you\s+do|what\s+do\s+you\s+do|help|help\s+me|can\s+you\s+help(\s+me)?|how\s+can\s+you\s+help(\s+me)?|how\s+do\s+you\s+work|how\s+to\s+use\s+(this|you)|what\s+can\s+i\s+ask)"
OKAY = r"(ok(ay)?|k|cool|nice|great|good|awesome|fine|alright|got\s+it|sure|understood|i\s+see|oh|wow|hmm+)"
PRAISE = r"(good\s+(bot|job)|well\s+done|great\s+job|nice\s+(bot|one|work)|you(\s+are|'?re)\s+(good|great|smart|awesome|helpful|amazing)|love\s+it)"
SORRY = r"(sorry|my\s+bad|oops|excuse\s+me)"
BOTQ = r"(are\s+you\s+(a\s+)?(bot|robot|human|real|ai|machine))"
YESNO = r"(yes|yeah|yep|yup|no|nope|nah)"

DOMAIN_LIST = "Computer Science, Cooking, General Knowledge and Medical"


def _clean(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _full(pattern: str, t: str) -> bool:
    return re.fullmatch(pattern, t) is not None


def _where(domain_label):
    if domain_label:
        return f"You're in **{domain_label}**. Ask me a question, or type 'change domain' to switch."
    return f"I can answer questions in {DOMAIN_LIST}. Which domain would you like to start with?"


def _core(t: str, domain_label):
    """Reply for non-greeting small talk, or None."""
    if _full(HOW, t):
        return "I'm doing great, thank you for asking! 😊 " + _where(domain_label)
    if _full(THANKS, t) or _full(PRAISE, t):
        return random.choice(["You're welcome! 😊", "Glad I could help! 😊", "Anytime!"]) + " Ask me anything else whenever you like."
    if _full(BYE, t):
        return "Goodbye! 👋 Come back any time you have a question."
    if _full(WHO, t):
        return ("I'm a question answering chatbot. I answer questions from curated knowledge in "
                f"{DOMAIN_LIST}. " + _where(domain_label))
    if _full(CAN, t):
        return (f"I can answer questions in {DOMAIN_LIST}. Pick a domain, then ask in your own words "
                "(small spelling mistakes are fine). Type 'change domain' any time to switch. "
                + ("" if not domain_label else f"Right now you're in **{domain_label}**."))
    if _full(BOTQ, t):
        return "I'm an AI chatbot, not a human. 🤖 " + _where(domain_label)
    if _full(SORRY, t):
        return "No problem at all! 😊 What would you like to know?"
    if _full(OKAY, t) or _full(YESNO, t):
        return "Okay! 😊 " + ("Ask me your next question whenever you're ready." if domain_label
                              else "Please choose a domain to get started.")
    return None


def reply(text: str, domain_label: str | None = None):
    """Returns a friendly reply string, or None if the message is not small talk."""
    t = _clean(text)
    if not t or len(t.split()) > 8:
        return None
    hello = random.choice(["Hello! 👋", "Hi there! 👋", "Hey! 😊"])

    if _full(GREET, t):
        return hello + " " + _where(domain_label)

    # a greeting followed by something else: "hi how are you", "hello there", "hey thanks"
    words = t.split()
    for n in (2, 1):
        head, rest = " ".join(words[:n]), " ".join(words[n:])
        if rest and _full(GREET, head):
            rest = re.sub(r"^(there|bot|buddy|bro|friend)\s*", "", rest).strip()
            if not rest:
                return hello + " " + _where(domain_label)
            sub = _core(rest, domain_label)
            return (hello + " " + sub) if sub else None
    return _core(t, domain_label)
