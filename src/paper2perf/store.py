"""Small SQLite catalog. Every query is scoped to an explicit session/document set."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid


def identifier() -> str:
    return uuid.uuid4().hex


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "documents").mkdir(exist_ok=True)
        (root / "runs").mkdir(exist_ok=True)
        self.db = root / "catalog.sqlite"
        with self.connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, created TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_options (
                    session TEXT PRIMARY KEY, assumptions TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, name TEXT NOT NULL,
                    sha256 TEXT NOT NULL, suffix TEXT NOT NULL, pages INTEGER NOT NULL,
                    created TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document TEXT NOT NULL, page INTEGER NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
                    chunk_id UNINDEXED, text, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL, run_id TEXT,
                    created TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, status TEXT NOT NULL,
                    request TEXT NOT NULL, result TEXT, created TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    stage TEXT NOT NULL, detail TEXT NOT NULL, created TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS doc_session ON documents(session);
                CREATE INDEX IF NOT EXISTS run_session ON runs(session);
            """)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def new_session(self):
        sid = identifier()
        with self.connect() as c:
            c.execute("INSERT INTO sessions VALUES (?,?)", (sid, now()))
        return sid

    def session_exists(self, sid):
        with self.connect() as c:
            return bool(c.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone())

    def sessions(self):
        with self.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT sessions.*, (SELECT COUNT(*) FROM documents WHERE session=sessions.id) "
                "AS document_count, (SELECT COUNT(*) FROM runs WHERE session=sessions.id) "
                "AS run_count FROM sessions ORDER BY created"
            )]

    def session_for_run(self, rid):
        with self.connect() as c:
            row = c.execute("SELECT session FROM runs WHERE id=?", (rid,)).fetchone()
            return row["session"] if row else None

    def session_assumptions(self, sid):
        with self.connect() as c:
            row = c.execute("SELECT assumptions FROM session_options WHERE session=?", (sid,)).fetchone()
            return json.loads(row["assumptions"]) if row else None

    def set_session_assumptions(self, sid, assumptions):
        with self.connect() as c:
            c.execute("INSERT INTO session_options VALUES (?,?) "
                      "ON CONFLICT(session) DO UPDATE SET assumptions=excluded.assumptions",
                      (sid, json.dumps(assumptions)))

    def documents(self, sid):
        with self.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM documents WHERE session=? ORDER BY created", (sid,)
            )]

    def add_document(self, sid, name, suffix, raw, pages):
        digest = hashlib.sha256(raw).hexdigest()
        with self.connect() as c:
            old = c.execute("SELECT id FROM documents WHERE session=? AND sha256=?",
                            (sid, digest)).fetchone()
            if old:
                return old["id"]
            did = identifier()
            path = self.root / "documents" / (did + suffix)
            path.write_bytes(raw)
            c.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
                      (did, sid, name, digest, suffix, len(pages), now()))
            for page, text in enumerate(pages, 1):
                # Overlap preserves sentences crossing chunk boundaries.
                for start in range(0, len(text), 2000):
                    chunk = text[start:start + 2400]
                    if not chunk.strip():
                        continue
                    cid = f"{did}-p{page}-c{start // 2000}"
                    c.execute("INSERT INTO chunks VALUES (?,?,?,?)", (cid, did, page, chunk))
                    c.execute("INSERT INTO search VALUES (?,?)", (cid, chunk))
        return did

    def chunks(self, document_ids):
        if not document_ids:
            return []
        slots = ",".join("?" for _ in document_ids)
        with self.connect() as c:
            return [dict(r) for r in c.execute(
                f"SELECT chunks.*, documents.name FROM chunks JOIN documents "
                f"ON chunks.document=documents.id WHERE document IN ({slots}) "
                "ORDER BY document,page,id", document_ids
            )]

    def search(self, document_ids, query, limit=12):
        import re
        words = list(dict.fromkeys(re.findall(r"[\w]+", query.lower())))[:30]
        words = [w for w in words if len(w) > 1]
        if not document_ids or not words:
            return []
        match = " OR ".join('"' + w.replace('"', '') + '"' for w in words)
        slots = ",".join("?" for _ in document_ids)
        with self.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT chunks.*, documents.name FROM search JOIN chunks "
                "ON search.chunk_id=chunks.id JOIN documents ON chunks.document=documents.id "
                f"WHERE search MATCH ? AND document IN ({slots}) ORDER BY rank LIMIT ?",
                [match, *document_ids, limit]
            )]

    def history(self, sid, limit=20):
        with self.connect() as c:
            rows = c.execute("SELECT * FROM messages WHERE session=? ORDER BY seq DESC LIMIT ?",
                             (sid, limit)).fetchall()
            return [dict(r) for r in reversed(rows)]

    def message(self, sid, role, content, run_id=None):
        with self.connect() as c:
            c.execute("INSERT INTO messages(session,role,content,run_id,created) VALUES (?,?,?,?,?)",
                      (sid, role, content, run_id, now()))

    def start_run(self, sid, request):
        rid = identifier()
        with self.connect() as c:
            c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)",
                      (rid, sid, "queued", json.dumps(request), None, now()))
        return rid

    def set_run(self, rid, status, result=None):
        with self.connect() as c:
            c.execute("UPDATE runs SET status=?,result=? WHERE id=?",
                      (status, json.dumps(result) if result is not None else None, rid))

    def run(self, sid, rid):
        with self.connect() as c:
            row = c.execute("SELECT * FROM runs WHERE id=? AND session=?", (rid, sid)).fetchone()
            if row is None:
                return None
            data = dict(row)
            data["request"] = json.loads(data["request"])
            data["result"] = json.loads(data["result"]) if data["result"] else None
            data["events"] = [dict(r) for r in c.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY seq", (rid,)
            )]
            return data

    def runs(self, sid):
        with self.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT id,status,created FROM runs WHERE session=? ORDER BY created", (sid,)
            )]

    def event(self, rid, stage, detail):
        with self.connect() as c:
            c.execute("INSERT INTO events(run_id,stage,detail,created) VALUES (?,?,?,?)",
                      (rid, stage, detail, now()))

    def artifact(self, rid, name, content):
        if Path(name).name != name or not name:
            raise ValueError("artifact name must be a basename")
        folder = self.root / "runs" / rid
        folder.mkdir(exist_ok=True)
        path = folder / name
        data = content if isinstance(content, str) else json.dumps(content, indent=2)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(data)
        temp.replace(path)
        return path
