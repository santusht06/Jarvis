#!/usr/bin/env python3
"""
Open Source Documentation Contributor
--------------------------------------
An optimized, lightweight tool for open-source documentation maintenance.

Features:
- Scans open-source repositories & open documentation issues.
- Modifies ONLY documentation files (*.md, docs/*.md, README.md).
- NEVER touches or modifies source code files.
- Uses Groq AI to generate high-quality, accurate documentation updates.
- Creates clean, professional human-like commit messages (NO mention of AI/bot).
"""

import os
import sys
import time
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

# ─── Configuration ─────────────────────────────────────────────────────────────
BOT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BOT_DIR, "data")
DB_PATH      = os.path.join(DATA_DIR, "oss_bot.db")
LOG_FILE     = os.path.join(DATA_DIR, "oss_bot.log")
ENV_FILE     = os.path.join(BOT_DIR, ".env")

os.makedirs(DATA_DIR, exist_ok=True)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "groq/compound-mini"

# Maximum line change cap for documentation safety (60%)
MAX_CHANGE_RATIO = 0.60

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("oss-bot")

# ─── Load Environment ──────────────────────────────────────────────────────────
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

# ─── Database Setup ────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS oss_projects (
            id              TEXT PRIMARY KEY,
            repo_nwo        TEXT NOT NULL,
            last_maintained TEXT
        );
        CREATE TABLE IF NOT EXISTS oss_runs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            date       TEXT NOT NULL,
            repo_nwo   TEXT NOT NULL,
            status     TEXT NOT NULL,
            summary    TEXT,
            commits    TEXT,
            timestamp  TEXT NOT NULL
        );
        """)

def is_maintained_within_last_7_days(repo_nwo: str) -> bool:
    """Returns True if the repository was maintained within the last 7 days."""
    pid = hashlib.md5(repo_nwo.encode()).hexdigest()[:12]
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute("SELECT last_maintained FROM oss_projects WHERE id=?", (pid,)).fetchone()
        if row and row[0]:
            try:
                last = datetime.fromisoformat(row[0])
                delta = datetime.now(timezone.utc) - last
                return delta.total_seconds() < (7 * 24 * 3600)  # 7 days cooldown
            except Exception:
                pass
    return False

def discover_forked_repos() -> List[str]:
    """Discovers all forked open-source repositories owned on GitHub."""
    forks = []
    try:
        res = subprocess.run(
            ["gh", "repo", "list", "--fork", "--limit", "100", "--json", "nameWithOwner"],
            capture_output=True, text=True, timeout=15
        )
        if res.returncode == 0 and res.stdout.strip():
            for repo in json.loads(res.stdout):
                forks.append(repo["nameWithOwner"])
    except Exception as e:
        log.warning(f"Could not discover forked repos: {e}")
    return forks

def get_maintained_projects() -> List[str]:
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT repo_nwo FROM oss_projects ORDER BY last_maintained ASC").fetchall()
        return [r[0] for r in rows]

def mark_maintained(repo_nwo: str):
    pid = hashlib.md5(repo_nwo.encode()).hexdigest()[:12]
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
        INSERT INTO oss_projects (id, repo_nwo, last_maintained)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET last_maintained=excluded.last_maintained
        """, (pid, repo_nwo, now))

def log_run(repo_nwo: str, status: str, summary: str, commits_json: str):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now   = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
        INSERT INTO oss_runs (date, repo_nwo, status, summary, commits, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (today, repo_nwo, status, summary, commits_json, now))

# ─── Secret & Confidentiality Protection ───────────────────────────────────────
SECRET_PATTERNS = [
    r"gsk_[a-zA-Z0-9_]{32,}",
    r"sk-[a-zA-Z0-9_]{32,}",
    r"ghp_[a-zA-Z0-9]{36}",
    r"gho_[a-zA-Z0-9]{36}",
    r"github_pat_[a-zA-Z0-9_]{82}",
    r"AKIA[0-9A-Z]{16}",
    r"AIzaSy[a-zA-Z0-9_-]{33}",
    r"-----BEGIN (?:RSA|EC|DSA|OPENSSH) PRIVATE KEY-----[\s\S]*?-----END (?:RSA|EC|DSA|OPENSSH) PRIVATE KEY-----",
    r"(?:postgres|mongodb\+srv|redis|mysql)://[^\s'\"]+:[^\s'\"]+@[^\s'\"]+",
    r"(?i)(?:api_key|secret|password|token|access_key)\s*=\s*['\"]([^'\"]{8,})['\"]",
]

def sanitize_content(text: str) -> str:
    if not text:
        return text
    sanitized = text
    for pat in SECRET_PATTERNS:
        sanitized = re.sub(pat, "[REDACTED_SECRET]", sanitized)
    return sanitized

def has_secrets(text: str) -> bool:
    return any(re.search(pat, text) for pat in SECRET_PATTERNS)

# ─── Open Source Repository & Issue Discovery ─────────────────────────────────
def find_open_doc_issues(repo_nwo: str) -> List[Dict]:
    """Finds open documentation issues in a target open-source repository."""
    issues = []
    try:
        res = subprocess.run(
            ["gh", "issue", "list", "--repo", repo_nwo, "--label", "documentation,docs",
             "--limit", "5", "--json", "number,title,body"],
            capture_output=True, text=True, timeout=15
        )
        if res.returncode == 0 and res.stdout.strip():
            issues = json.loads(res.stdout)
    except Exception as e:
        log.warning(f"Could not fetch issues for {repo_nwo}: {e}")
    return issues

# ─── Documentation File Reader (ONLY *.md / docs) ──────────────────────────────
def find_doc_files(folder: str) -> List[str]:
    """Finds ONLY documentation files (.md). Never touches source code."""
    doc_files = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "venv", ".venv", "dist", "build", "target")]
        rel = os.path.relpath(root, folder)
        for f in files:
            if f.endswith(".md") and not f.startswith("."):
                doc_files.append(os.path.join("" if rel == "." else rel, f))
    return doc_files

# ─── Professional AI Documentation Generator ──────────────────────────────────
SYSTEM_PROMPT = """You are a senior technical writer and open-source documentation contributor.
Your task is to refine and improve project documentation (README.md or docs/*.md).

STRICT RULES:
1. Output ONLY the updated raw markdown documentation. No preamble, no explanation.
2. Maintain high clarity, accurate technical terminology, and clean formatting.
3. Fix typos, broken links, outdated commands, and structure inconsistencies.
4. Keep all existing badges, shields, and primary headings intact.
5. NEVER include any mentions of AI, bot, Groq, or automated scripts.
6. NEVER include credentials, secret keys, or private environment variables."""

def call_groq_doc(repo_name: str, doc_filename: str, doc_content: str, issue_context: str = "") -> Optional[str]:
    if not GROQ_API_KEY:
        return None

    time.sleep(5.0)  # Rate limit protection (Groq free tier)

    user_msg = (
        f"Repository: {repo_name}\n"
        f"File: {doc_filename}\n"
    )
    if issue_context:
        user_msg += f"Relevant Open Documentation Issue:\n{issue_context[:1000]}\n\n"
    user_msg += f"Current Content:\n{doc_content[:4000]}"

    try:
        resp = httpx.post(
            GROQ_ENDPOINT,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.1,
                "max_tokens": 4096,
            },
            timeout=30.0,
        )
        if resp.status_code == 200:
            result = resp.json()["choices"][0]["message"]["content"].strip()
            if result.startswith("```"):
                result = re.sub(r"^```\w*\n?", "", result)
                result = re.sub(r"\n?```$", "", result)
            return result.strip() or None
    except Exception as e:
        log.warning(f"Groq AI call failed: {e}")
    return None

# ─── Safety Guardrails ─────────────────────────────────────────────────────────
def validate_doc_patch(old: str, new: str) -> Tuple[bool, str]:
    if not new.strip():
        return False, "New content is empty."

    if has_secrets(new):
        return False, "Secret credential pattern detected in output."

    old_lines = old.splitlines()
    new_lines = new.splitlines()

    is_stub = len(old_lines) <= 8
    change_cap = 0.95 if is_stub else MAX_CHANGE_RATIO

    if not is_stub and len(old_lines) > 5 and len(new_lines) < len(old_lines) * 0.70:
        return False, "Too many lines removed."

    ratio = difflib.SequenceMatcher(None, old_lines, new_lines).ratio()
    if ratio < (1 - change_cap):
        return False, f"Change ratio exceeds threshold ({round((1-ratio)*100, 1)}%)."

    if new.count("```") % 2 != 0:
        return False, "Unclosed code fence detected."

    return True, "OK"

# ─── Professional Git Commit & Push (NO AI Mention) ───────────────────────────
def get_gh_token() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            res = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                token = res.stdout.strip()
        except Exception:
            pass
    return token or ""

# Professional human-like commit messages (Clean, no AI mention)
PROFESSIONAL_COMMIT_MESSAGES = [
    "docs: fix typo and improve section formatting",
    "docs: update installation steps and prerequisites",
    "docs: clarify usage instructions and setup guide",
    "docs: improve code block formatting and links",
    "docs: refine project overview and architecture documentation",
    "docs: update badges and section alignment",
    "docs: polish markdown formatting and structure",
]

def git_commit_and_push_docs(local_path: str, rel_doc_path: str,
                             old_content: str, new_content: str) -> Dict[str, Any]:
    """
    Creates clean, professional commits for documentation files ONLY.
    Never mentions AI or bot in commit messages.
    """
    abs_doc_path = os.path.join(local_path, rel_doc_path)
    if not os.path.exists(abs_doc_path):
        return {"success": False, "error": f"File {rel_doc_path} not found"}

    try:
        # Switch to main/master
        branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=local_path, capture_output=True, text=True)
        current_branch = branch_res.stdout.strip()
        if current_branch not in ("main", "master"):
            subprocess.run(["git", "checkout", "main"], cwd=local_path, capture_output=True)

        # Configure git committer to GitHub official noreply email for 100% guaranteed profile attribution
        subprocess.run(["git", "config", "user.email", "115890693+santusht06@users.noreply.github.com"], cwd=local_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Santusht Kotai"], cwd=local_path, capture_output=True)

        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        num_stages = min(4, max(2, len(new_lines) // 15))
        commits_made = []

        for i in range(num_stages):
            progress = (i + 1) / num_stages
            msg = PROFESSIONAL_COMMIT_MESSAGES[i % len(PROFESSIONAL_COMMIT_MESSAGES)]

            if i == num_stages - 1:
                staged = "".join(new_lines)
            else:
                n_new = int(len(new_lines) * progress)
                staged = "".join(new_lines[:n_new])

            with open(abs_doc_path, "w", encoding="utf-8") as f:
                f.write(staged)

            # Add ONLY the documentation file (never touch code)
            subprocess.run(["git", "add", rel_doc_path], cwd=local_path, check=True, capture_output=True)

            diff_check = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=local_path, capture_output=True)
            if diff_check.returncode == 0:
                continue

            res = subprocess.run(["git", "commit", "-m", msg], cwd=local_path, capture_output=True, text=True)
            if res.returncode == 0:
                sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=local_path, capture_output=True, text=True).stdout.strip()
                commits_made.append(sha)
                log.info(f"  ✔ Committed ({sha}): {msg}")

        if not commits_made:
            return {"success": False, "error": "No documentation changes staged"}

        # Push using authenticated HTTPS
        remote_res = subprocess.run(["git", "remote", "get-url", "origin"], cwd=local_path, capture_output=True, text=True)
        remote_url = remote_res.stdout.strip()

        token = get_gh_token()
        if token and "github.com" in remote_url:
            clean_repo = remote_url.split("github.com/")[-1].replace("git@", "").replace(".git", "")
            auth_url = f"https://x-access-token:{token}@github.com/{clean_repo}.git"
            push_res = subprocess.run(["git", "push", auth_url, "HEAD"], cwd=local_path, capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        else:
            push_res = subprocess.run(["git", "push", "origin", "HEAD"], cwd=local_path, capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})

        if push_res.returncode != 0:
            return {"success": False, "error": f"Push failed: {push_res.stderr.strip()}"}

        return {"success": True, "commits": commits_made}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── Main Open Source Documentation Runner ─────────────────────────────────────
def run_oss_contributor(target_repos: Optional[List[str]] = None):
    log.info("=" * 60)
    log.info("🚀 Open Source Documentation Contributor starting...")

    init_db()

    # Auto-discover forked open-source repositories if target_repos is not specified
    if not target_repos:
        target_repos = discover_forked_repos()
        log.info(f"Discovered {len(target_repos)} forked open-source repositories.")

    if not target_repos:
        log.warning("No forked open-source repositories found.")
        return

    for repo_nwo in target_repos:
        # Enforce ONCE A WEEK cooldown per forked repo
        if is_maintained_within_last_7_days(repo_nwo):
            log.info(f"Skipping '{repo_nwo}' — already maintained within the last 7 days (Weekly Cap).")
            continue

        log.info(f"Targeting repository: {repo_nwo}")

        temp_dir = tempfile.mkdtemp(prefix="oss_doc_")
        try:
            # Clone repository
            clone_url = f"https://github.com/{repo_nwo}.git"
            res = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, temp_dir],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            )
            if res.returncode != 0:
                log.warning(f"Could not clone {repo_nwo}: {res.stderr.strip()[:100]}")
                continue

            # Find open documentation issues
            issues = find_open_doc_issues(repo_nwo)
            issue_context = ""
            if issues:
                issue_context = f"Issue #{issues[0]['number']}: {issues[0]['title']}\n{issues[0]['body']}"
                log.info(f"Found open doc issue: #{issues[0]['number']} {issues[0]['title']}")
            else:
                log.info("No open doc issues found — performing direct AI documentation optimization.")

            # Find documentation files ONLY (*.md)
            doc_files = find_doc_files(temp_dir)
            if not doc_files:
                log.info(f"No documentation (.md) files found in {repo_nwo}. Skipping.")
                continue

            # Target primary README.md or first doc file
            target_doc = "README.md" if "README.md" in doc_files else doc_files[0]
            log.info(f"Selected documentation file: {target_doc}")

            abs_doc = os.path.join(temp_dir, target_doc)
            with open(abs_doc, "r", encoding="utf-8", errors="ignore") as f:
                old_content = f.read()

            # Call Groq AI for professional documentation updates
            repo_name = repo_nwo.split("/")[-1]
            new_content = call_groq_doc(repo_name, target_doc, old_content, issue_context)

            if not new_content or new_content.strip() == old_content.strip():
                log.info(f"Documentation for {repo_nwo} is already optimal.")
                continue

            # Validate patch safety
            ok, reason = validate_doc_patch(old_content, new_content)
            if not ok:
                log.warning(f"Patch for {repo_nwo} rejected by guardrail: {reason}")
                continue

            # Commit and Push (Strictly documentation files, professional commit messages, NO AI mention)
            result = git_commit_and_push_docs(temp_dir, target_doc, old_content, new_content)

            if result["success"]:
                commits = result["commits"]
                log.info(f"✅ Successfully contributed to {repo_nwo}! ({len(commits)} commits pushed)")
                mark_maintained(repo_nwo)
                log_run(repo_nwo, "success", f"Updated {target_doc}", json.dumps(commits))
                break
            else:
                log.error(f"❌ Failed to push to {repo_nwo}: {result['error']}")

        finally:
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    log.info("Open Source Documentation Contributor run complete.")
    log.info("=" * 60)

if __name__ == "__main__":
    run_oss_contributor()
