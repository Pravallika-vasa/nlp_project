"""
db.py
------
Local SQLite database for user accounts + chat history.

Tables:
  users    (username PRIMARY KEY, password_hash, created_at)
  chats    (chat_id, username, title, domain, created_at)
  messages (id, chat_id, role, content, created_at)

All functions are plain Python + sqlite3 (no Streamlit dependency), so this
file can be tested and reused on its own.
"""

import sqlite3
import bcrypt
from datetime import datetime, timezone

DB_PATH = "qa_system.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chats (
            chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            title TEXT NOT NULL,
            domain TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(chat_id)
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
        );
        CREATE TABLE IF NOT EXISTS ui_state (
            token TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


# ---------------- users ----------------

def create_user(username: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password can't be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    conn = get_conn()
    try:
        existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return False, "That username is already taken."
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, pw_hash, _now()),
        )
        conn.commit()
        return True, "Account created."
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    conn = get_conn()
    try:
        row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username.strip(),)).fetchone()
        if not row:
            return False
        return bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8"))
    finally:
        conn.close()


# ---------------- chats ----------------

def create_chat(username: str, title: str = "New chat", domain: str | None = None) -> int:
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO chats (username, title, domain, created_at) VALUES (?, ?, ?, ?)",
            (username, title, domain, _now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_user_chats(username: str):
    """Newest first. Returns list of sqlite3.Row with chat_id, title, domain, created_at."""
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT chat_id, title, domain, created_at FROM chats "
            "WHERE username = ? ORDER BY chat_id DESC",
            (username,),
        ).fetchall()
    finally:
        conn.close()


def set_chat_domain(chat_id: int, domain: str):
    conn = get_conn()
    try:
        conn.execute("UPDATE chats SET domain = ? WHERE chat_id = ?", (domain, chat_id))
        conn.commit()
    finally:
        conn.close()


def set_chat_title(chat_id: int, title: str):
    """Title is capped at 60 chars so the sidebar stays tidy."""
    conn = get_conn()
    try:
        conn.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (title[:60], chat_id))
        conn.commit()
    finally:
        conn.close()


def delete_chat(chat_id: int):
    conn = get_conn()
    try:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------- messages ----------------

def add_message(chat_id: int, role: str, content: str):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def get_chat_messages(chat_id: int):
    """Oldest first. Returns list of {'role':..., 'content':...}."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    finally:
        conn.close()


# ---------------- sessions (persistent "remember me" login) ----------------

def create_session(username: str) -> str:
    """Creates a new random session token for this user and returns it."""
    import secrets
    token = secrets.token_hex(32)
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO sessions (token, username, created_at) VALUES (?, ?, ?)",
            (token, username, _now()),
        )
        conn.commit()
        return token
    finally:
        conn.close()


def get_session_user(token: str):
    """Returns the username for a valid session token, or None."""
    if not token:
        return None
    conn = get_conn()
    try:
        row = conn.execute("SELECT username FROM sessions WHERE token = ?", (token,)).fetchone()
        return row["username"] if row else None
    finally:
        conn.close()


def delete_session(token: str):
    conn = get_conn()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


# ---------------- UI state (so a page refresh continues where the user was) ----------------

def save_ui_state(token: str, data: str):
    conn = get_conn()
    try:
        conn.execute("INSERT OR REPLACE INTO ui_state (token, data, updated_at) VALUES (?, ?, ?)",
                     (token, data, _now()))
        conn.commit()
    finally:
        conn.close()


def load_ui_state(token: str):
    if not token:
        return None
    conn = get_conn()
    try:
        row = conn.execute("SELECT data FROM ui_state WHERE token = ?", (token,)).fetchone()
        return row["data"] if row else None
    finally:
        conn.close()


def chat_belongs_to(chat_id, username: str) -> bool:
    conn = get_conn()
    try:
        row = conn.execute("SELECT 1 FROM chats WHERE chat_id = ? AND username = ?", (chat_id, username)).fetchone()
        return row is not None
    finally:
        conn.close()


def delete_user(username: str) -> bool:
    """Delete an account with all its chats, messages, login sessions and saved page state."""
    conn = get_conn()
    try:
        conn.execute("DELETE FROM messages WHERE chat_id IN (SELECT chat_id FROM chats WHERE username = ?)", (username,))
        conn.execute("DELETE FROM chats WHERE username = ?", (username,))
        conn.execute("DELETE FROM sessions WHERE username = ?", (username,))
        conn.execute("DELETE FROM ui_state WHERE data LIKE ?", ('%"user": ' + __import__("json").dumps(username) + '%',))
        deleted = conn.execute("DELETE FROM users WHERE username = ?", (username,)).rowcount
        conn.commit()
        return deleted > 0
    finally:
        conn.close()


init_db()  # safe to call every import: CREATE TABLE IF NOT EXISTS