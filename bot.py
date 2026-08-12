#!/usr/bin/env python3
"""
AI README Maintenance Bot
--------------------------
Runs ONCE per day (scheduled via launchd/cron).
- Picks ONE project (oldest unmaintained)
- Reads README.md only — touches NOTHING else
- Calls Groq AI to generate improvements
- Makes 3-4 small atomic commits directly to main
- Pushes to GitHub to maintain streak
- Exits. Done.
"""

import os
import sys
import json
import sqlite3
import hashlib
import difflib
import re
import math
import subprocess
import tempfile
import shutil
import httpx
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

# ─── Paths ─────────────────────────────────────────────────────────────────────
BOT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BOT_DIR, "data")
DB_PATH      = os.path.join(DATA_DIR, "bot.db")
VECTOR_FILE  = os.path.join(DATA_DIR, "vectors.json")
LOG_FILE     = os.path.join(DATA_DIR, "bot.log")
ENV_FILE     = os.path.join(BOT_DIR, ".env")

os.makedirs(DATA_DIR, exist_ok=True)

# Directories to scan for local projects
LOCAL_SCAN_PATHS = [
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Projects"),
    os.path.expanduser("~/Developer"),
]

# Groq settings
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"

# Safety: reject patches that alter more than this fraction of README lines
MAX_CHANGE_RATIO = 0.35

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("bot")

# ─── Load .env ─────────────────────────────────────────────────────────────────
def load_env():
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_env()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ─── Database ──────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            type            TEXT NOT NULL,
            path_or_url     TEXT NOT NULL,
            last_maintained TEXT
        );
        CREATE TABLE IF NOT EXISTS runs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            date       TEXT NOT NULL,
            project_id TEXT NOT NULL,
            status     TEXT NOT NULL,
            summary    TEXT,
            commits    TEXT,
            timestamp  TEXT NOT NULL
        );
        """)

def get_projects() -> List[Dict]:
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(
            "SELECT * FROM projects ORDER BY last_maintained ASC NULLS FIRST, name ASC"
        ).fetchall()]

def upsert_project(pid, name, ptype, path):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
        INSERT INTO projects (id, name, type, path_or_url)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name, path_or_url=excluded.path_or_url
        """, (pid, name, ptype, path))

def mark_maintained(pid):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("UPDATE projects SET last_maintained=? WHERE id=?",
                  (datetime.now(timezone.utc).isoformat(), pid))

def already_ran_today() -> Optional[Dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        row = c.execute(
            "SELECT * FROM runs WHERE date=? AND status='success'", (today,)
        ).fetchone()
        return dict(row) if row else None

def log_run(pid, status, summary, commits_json):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
        INSERT INTO runs (date, project_id, status, summary, commits, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (today, pid, status, summary, commits_json,
              datetime.now(timezone.utc).isoformat()))

# ─── Offline Vector Store (TF-IDF cosine) ─────────────────────────────────────
class VectorStore:
    def __init__(self):
        self.data: Dict[str, List[Dict]] = {}
        self._load()

    def _load(self):
        if os.path.exists(VECTOR_FILE):
            try:
                with open(VECTOR_FILE, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def _save(self):
        with open(VECTOR_FILE, "w") as f:
            json.dump(self.data, f)

    def _vec(self, text: str) -> Dict[str, float]:
        words = re.findall(r"\w+", text.lower())
        n = len(words) or 1
        freq: Dict[str, float] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return {w: c / n for w, c in freq.items()}

    def _cos(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        common = set(a) & set(b)
        num = sum(a[k] * b[k] for k in common)
        dA = math.sqrt(sum(v**2 for v in a.values()))
        dB = math.sqrt(sum(v**2 for v in b.values()))
        return num / (dA * dB) if dA and dB else 0.0

    def index(self, pid: str, docs: List[Dict[str, str]]):
        key = f"p_{pid.replace('-','_')}"
        self.data[key] = [
            {"content": d["content"][:800], "vector": self._vec(d["content"])}
            for d in docs if d.get("content", "").strip()
        ]
        self._save()

    def search(self, pid: str, query: str, top_k: int = 4) -> List[str]:
        key = f"p_{pid.replace('-','_')}"
        docs = self.data.get(key, [])
        if not docs:
            return []
        qv = self._vec(query)
        ranked = sorted(docs, key=lambda d: self._cos(qv, d["vector"]), reverse=True)
        return [d["content"] for d in ranked[:top_k]]

vector_store = VectorStore()

# ─── Project Discovery ─────────────────────────────────────────────────────────
def scan_local() -> List[Dict]:
    seen, projects = set(), []
    for base in LOCAL_SCAN_PATHS:
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            if entry.startswith(".") or entry in ("venv", ".venv", "node_modules", "Library", "Applications", "bot"):
                continue
            path = os.path.join(base, entry)
            if not os.path.isdir(path) or path in seen:
                continue
            seen.add(path)
            is_git     = os.path.exists(os.path.join(path, ".git"))
            has_marker = any(os.path.exists(os.path.join(path, f))
                             for f in ["package.json", "requirements.txt",
                                       "pyproject.toml", "Cargo.toml",
                                       "go.mod", "Makefile", "README.md"])
            if is_git or has_marker:
                pid = "local-" + hashlib.md5(path.encode()).hexdigest()[:12]
                projects.append({"id": pid, "name": entry, "type": "local", "path": path})
    return projects

def scan_github() -> List[Dict]:
    projects = []
    try:
        res = subprocess.run(
            ["gh", "repo", "list", "--limit", "100",
             "--json", "name,nameWithOwner,url"],
            capture_output=True, text=True, timeout=15
        )
        if res.returncode == 0 and res.stdout.strip():
            for repo in json.loads(res.stdout):
                nwo = repo.get("nameWithOwner") or repo["name"]
                pid = "github-" + hashlib.md5(nwo.encode()).hexdigest()[:12]
                https_url = repo.get("url", "")
                projects.append({
                    "id": pid, "name": repo["name"],
                    "type": "github",
                    "path": https_url,  # Always use HTTPS
                })
    except Exception as e:
        log.warning(f"GitHub scan skipped: {e}")
    return projects

def discover_projects():
    log.info("Scanning local desktop + GitHub repos...")
    all_projs = scan_local() + scan_github()
    for p in all_projs:
        upsert_project(p["id"], p["name"], p["type"], p["path"])
    log.info(f"Inventory: {len(all_projs)} projects")

# ─── README Utilities ──────────────────────────────────────────────────────────
def find_readme(folder: str) -> Optional[str]:
    """Returns path to README.md — and ONLY README.md. Touches nothing else."""
    for name in ["README.md", "readme.md"]:
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            return p
    return None

def read_project_context(folder: str) -> List[Dict[str, str]]:
    """Reads manifest files as context ONLY — never modifies them."""
    docs = []
    for fname in ["package.json", "requirements.txt", "pyproject.toml",
                  "Cargo.toml", "go.mod", "setup.py"]:
        fp = os.path.join(folder, fname)
        if os.path.isfile(fp):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    docs.append({"content": f.read()[:1500]})
            except Exception:
                pass
    # File tree for context
    try:
        tree = []
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and d not in ("node_modules", "venv", ".venv", "dist", "build")]
            rel = os.path.relpath(root, folder)
            for f in files:
                tree.append(os.path.join("" if rel == "." else rel, f))
            if len(tree) > 100:
                break
        if tree:
            docs.append({"content": "Project files:\n" + "\n".join(tree[:100])})
    except Exception:
        pass
    return docs

# ─── Safety Guardrails ─────────────────────────────────────────────────────────
def validate_patch(old: str, new: str) -> Tuple[bool, str]:
    """Strict safety checks — ensures patch is surgical, not destructive."""
    if not new.strip():
        return False, "New content is empty."

    old_lines = old.splitlines()
    new_lines = new.splitlines()

    # Must retain at least 75% of original lines
    if len(old_lines) > 5 and len(new_lines) < len(old_lines) * 0.75:
        return False, f"Too many lines removed ({len(old_lines)} → {len(new_lines)})."

    # Change ratio cap
    ratio = difflib.SequenceMatcher(None, old_lines, new_lines).ratio()
    if ratio < (1 - MAX_CHANGE_RATIO):
        return False, f"Patch alters {round((1-ratio)*100,1)}% — exceeds {int(MAX_CHANGE_RATIO*100)}% daily cap."

    # Unclosed code fences
    if new.count("```") % 2 != 0:
        return False, "Patch introduced unclosed code fences."

    # Primary title preserved
    h1s = [l for l in old_lines if l.startswith("# ")]
    if h1s and h1s[0].strip() not in new:
        return False, f"Primary title '{h1s[0].strip()}' was removed."

    return True, "OK"

def compute_diff(old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="a/README.md", tofile="b/README.md", n=3,
    ))

# ─── Patch Splitter (3-4 atomic commits for streak) ───────────────────────────
def split_into_commits(old: str, new: str) -> List[Dict[str, str]]:
    """
    Splits the improved README into 3-4 atomic incremental commits.
    Each commit adds one meaningful improvement section, building toward the final version.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = [op for op in matcher.get_opcodes() if op[0] != 'equal']
    
    if not opcodes:
        return []
    
    # Split opcodes into 3-4 groups
    target_commits = min(4, max(3, len(opcodes)))
    chunk_size = max(1, len(opcodes) // target_commits)
    
    commits = []
    current = old_lines[:]
    
    labels = [
        "fix typos and formatting",
        "improve structure and sections",
        "add missing documentation",
        "finalize badges and links",
    ]
    
    for i in range(0, len(opcodes), chunk_size):
        group = opcodes[i:i + chunk_size]
        # Apply this group of changes to current
        temp = new_lines[:]  # We'll build incrementally using sequence matcher
        
        # For simplicity: stage 1 = 25%, 2 = 50%, 3 = 75%, 4 = 100%
        stage_idx = i // chunk_size
        progress = min(1.0, (stage_idx + 1) / target_commits)
        
        # Interpolate between old and new
        n_new_lines = int(len(new_lines) * progress + len(old_lines) * (1 - progress))
        staged_content = "".join(new_lines)  # Just use final for last, intermediate for others
        
        label = labels[stage_idx % len(labels)]
        commits.append({
            "message": f"docs(readme): {label} [AI bot - {stage_idx+1}/{target_commits}]",
            "is_final": (progress >= 1.0 or i + chunk_size >= len(opcodes))
        })
        
        if commits[-1]["is_final"]:
            break
    
    # Always ensure we have 3-4 commits pointing to the final content
    if len(commits) < 3:
        commits = [
            {"message": "docs(readme): fix typos and improve readability [AI bot]",          "is_final": False},
            {"message": "docs(readme): add missing sections and badges [AI bot]",             "is_final": False},
            {"message": "docs(readme): finalize formatting and documentation [AI bot]",       "is_final": True},
        ]
    
    return commits

# ─── Groq AI Patch Generator ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an elite automated README.md maintenance bot with 99.999% accuracy.

YOUR STRICT RULES:
1. ONLY return improved README.md content — raw markdown, no explanations.
2. Do NOT rewrite from scratch. Make surgical improvements only.
3. Fix typos, broken links, missing badges, improve formatting.
4. Add missing Installation/Usage sections ONLY if manifest files confirm them.
5. Keep ALL existing project information intact.
6. Preserve the primary # Title header exactly.
7. Do NOT wrap output in code fences."""

def call_groq(proj_name: str, readme: str, context: str) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    try:
        resp = httpx.post(
            GROQ_ENDPOINT,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                        f"Project: {proj_name}\n\n"
                        f"Codebase context:\n{context[:2000]}\n\n"
                        f"Current README.md:\n{readme}"}
                ],
                "temperature": 0.1,
                "max_tokens": 4096,
            },
            timeout=30.0,
        )
        if resp.status_code == 200:
            result = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip any accidental wrapping fences
            result = re.sub(r"^```\w*\n?", "", result)
            result = re.sub(r"\n?```$", "", result)
            return result.strip() or None
        else:
            log.warning(f"Groq error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        log.warning(f"Groq call failed: {e}")
    return None

# ─── Rule-Based Fallback Patcher ───────────────────────────────────────────────
def rule_based_patch(proj_name: str, readme: str, context: str) -> Tuple[str, str]:
    lines = readme.splitlines()
    changed = []

    # Fix raw URLs → markdown links
    new_lines = []
    for line in lines:
        if re.search(r"(?<!\()(https?://[^\s)>]+)", line) and "[" not in line:
            line = re.sub(r"(?<!\()(https?://[^\s)>]+)", r"[\1](\1)", line)
            changed.append("formatted raw URLs as markdown links")
        new_lines.append(line)
    lines = new_lines

    # Add missing Installation section based on manifest files found
    has_install = any("install" in l.lower() or "setup" in l.lower() for l in lines)
    if not has_install:
        if "requirements.txt" in context or "pyproject.toml" in context:
            lines += ["", "## 🚀 Installation", "", "```bash", "pip install -r requirements.txt", "```", ""]
            changed.append("added Python installation section")
        elif "package.json" in context:
            lines += ["", "## 🚀 Installation", "", "```bash", "npm install", "```", ""]
            changed.append("added Node.js installation section")

    # Add AI maintenance badge if missing
    badge = "![AI Maintained](https://img.shields.io/badge/readme-AI%20maintained-blue)"
    if lines and not any("AI" in l and "maintained" in l.lower() for l in lines[:6]):
        lines.insert(1, "")
        lines.insert(2, badge)
        changed.append("added AI maintenance badge")

    new_readme = "\n".join(lines)
    summary    = "; ".join(changed) if changed else "validated README structure"
    return new_readme, summary

# ─── Git: 3-4 Commits Directly to Main ────────────────────────────────────────
def git_commit_and_push(local_path: str, proj_name: str,
                        old_readme: str, new_readme: str,
                        summary: str) -> Dict[str, Any]:
    """
    Makes 3-4 atomic commits directly to main branch.
    Only modifies README.md. Pushes once at the end.
    """
    readme_path = os.path.join(local_path, "README.md")
    
    # Verify git repo
    if not os.path.exists(os.path.join(local_path, ".git")):
        return {"success": False, "error": "Not a git repository"}

    try:
        # Ensure we're on main (or master)
        branch_res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=local_path, capture_output=True, text=True
        )
        current_branch = branch_res.stdout.strip()
        if current_branch not in ("main", "master"):
            # Try switching to main
            subprocess.run(["git", "checkout", "main"], cwd=local_path, capture_output=True)
            subprocess.run(["git", "pull", "origin", "main", "--rebase"],
                           cwd=local_path, capture_output=True)

        # Configure git user if not set (for CI/bot environments)
        subprocess.run(["git", "config", "user.email", "ai-readme-bot@local"],
                       cwd=local_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "AI README Bot"],
                       cwd=local_path, capture_output=True)

        # Build 3-4 intermediate README states for streak commits
        old_lines = old_readme.splitlines()
        new_lines = new_readme.splitlines()
        
        commit_stages = [
            (0.40, "docs(readme): fix typos and improve readability [AI bot]"),
            (0.70, "docs(readme): improve structure and add missing sections [AI bot]"),
            (0.90, "docs(readme): refine formatting and links [AI bot]"),
            (1.00, f"docs(readme): finalize daily AI maintenance update [AI bot]\n\nSummary: {summary}"),
        ]
        
        commits_made = []
        
        for fraction, commit_msg in commit_stages:
            if fraction < 1.0:
                # Interpolate between old and new README for incremental commits
                n = int(len(new_lines) * fraction)
                staged_lines = new_lines[:n] + old_lines[n:] if n < len(old_lines) else new_lines
                staged_content = "\n".join(staged_lines)
            else:
                staged_content = new_readme
            
            # Write README (only file we touch)
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(staged_content)
            
            # Stage README.md ONLY
            subprocess.run(["git", "add", "README.md"],
                           cwd=local_path, check=True, capture_output=True)
            
            # Check if there's actually a diff to commit
            diff_check = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=local_path, capture_output=True
            )
            if diff_check.returncode == 0:
                continue  # Nothing staged, skip this commit
            
            commit_res = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=local_path, capture_output=True, text=True
            )
            if commit_res.returncode == 0:
                sha = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=local_path, capture_output=True, text=True
                ).stdout.strip()
                commits_made.append(sha)
                log.info(f"  ✔ Committed: {sha} — {commit_msg.splitlines()[0]}")

        if not commits_made:
            return {"success": False, "error": "No changes committed (README already up to date)"}

        # Push all commits to main in one shot
        push_res = subprocess.run(
            ["git", "push", "origin", "HEAD"],
            cwd=local_path, capture_output=True, text=True
        )
        if push_res.returncode != 0:
            log.warning(f"Push stderr: {push_res.stderr.strip()}")
            return {"success": False, "error": f"Push failed: {push_res.stderr.strip()}"}

        log.info(f"  🚀 Pushed {len(commits_made)} commits to origin")
        return {"success": True, "commits": commits_made}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── Main Bot Logic ────────────────────────────────────────────────────────────
def run():
    log.info("=" * 60)
    log.info("🤖 AI README Bot — daily run starting")

    if not GROQ_API_KEY:
        log.warning("GROQ_API_KEY not set — rule-based patcher will be used.")

    init_db()

    # 1. Check if already ran today
    done = already_ran_today()
    if done:
        log.info(f"Already ran today for project '{done['project_id']}'. Exiting.")
        return

    # 2. Discover all projects
    discover_projects()

    # 3. Pick the project with oldest last_maintained (round-robin)
    projects = get_projects()
    if not projects:
        log.warning("No projects found. Exiting.")
        return

    target   = projects[0]
    pid      = target["id"]
    name     = target["name"]
    ptype    = target["type"]
    path     = target["path_or_url"]

    log.info(f"Selected: [{ptype.upper()}] {name} ({pid})")

    # 4. Resolve local path — clone GitHub repos temporarily if needed
    local_path = path
    temp_dir   = None

    if ptype == "github" and not os.path.isdir(path):
        desktop_clone = os.path.expanduser(f"~/Desktop/{name}")
        if os.path.isdir(desktop_clone):
            local_path = desktop_clone
        else:
            log.info(f"Cloning {name} from GitHub (HTTPS)...")
            temp_dir = tempfile.mkdtemp(prefix=f"bot_{name}_")
            # Use HTTPS clone to avoid SSH key/timeout issues
            clone_url = path if path.startswith("https://") else f"https://github.com/santusht06/{name}.git"
            res = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, temp_dir],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            )
            if res.returncode != 0:
                log.warning(f"Clone failed: {res.stderr.strip()[:200]}")
                log_run(pid, "skipped", "Clone failed", "[]")
                return
            local_path = temp_dir

    try:
        # 5. Find README.md — ONLY file we care about
        readme_path = find_readme(local_path)
        if not readme_path:
            log.info(f"No README.md found in {name}. Creating one...")
            readme_path = os.path.join(local_path, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(f"# {name}\n\n")

        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            current_readme = f.read()

        log.info(f"README loaded: {len(current_readme.splitlines())} lines")

        # 6. Build codebase context for Vector DB (read-only)
        context_docs = read_project_context(local_path)
        vector_store.index(pid, context_docs)
        context_chunks = vector_store.search(pid, "install usage dependencies setup", top_k=4)
        context = "\n---\n".join(context_chunks)

        # 7. Generate improved README via Groq AI
        log.info("Calling Groq AI for surgical README improvements...")
        new_readme = call_groq(name, current_readme, context)

        if new_readme:
            source = f"Groq AI ({GROQ_MODEL})"
            log.info(f"Groq returned improved README ({len(new_readme.splitlines())} lines)")
        else:
            log.info("Groq unavailable — using rule-based patcher")
            new_readme, _ = rule_based_patch(name, current_readme, context)
            source = "Rule-based patcher"

        # 8. Safety guardrails — reject destructive patches
        ok, reason = validate_patch(current_readme, new_readme)
        if not ok:
            log.warning(f"Patch rejected: {reason}")
            log_run(pid, "rejected", reason, "[]")
            return

        if new_readme.strip() == current_readme.strip():
            log.info("README already perfect — no changes needed.")
            log_run(pid, "no_change", "Already up to date", "[]")
            mark_maintained(pid)
            return

        diff = compute_diff(current_readme, new_readme)
        log.info(f"Diff: {len([l for l in diff.splitlines() if l.startswith(('+','-'))])} changed lines")

        added = len([l for l in diff.splitlines() if l.startswith('+') and not l.startswith('+++')])
        summary = f"{source}: {added} lines improved"

        # 9. Commit 3-4 times to main, push once (streak!)
        result = git_commit_and_push(local_path, name, current_readme, new_readme, source)

        if result["success"]:
            commits = result["commits"]
            log.info(f"✅ Done! {len(commits)} commits pushed for '{name}'")
            mark_maintained(pid)
            log_run(pid, "success", source, json.dumps(commits))
        else:
            log.error(f"❌ Git failed: {result['error']}")
            log_run(pid, "failed", result["error"], "[]")

    finally:
        # Clean up temp clone — only README was modified inside it
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    log.info("Bot run complete. Exiting.")
    log.info("=" * 60)

if __name__ == "__main__":
    run()
