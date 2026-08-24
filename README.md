# AI README Maintenance Bot

An autonomous AI agent that automatically improves your project README files daily, powered by **Groq AI** and a **Vector Database**.

[![AI Maintained](https://img.shields.io/badge/readme-AI%20maintained-blue)](https://github.com/santusht06/bot)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🤖 What it does

Runs **once per day** automatically (via macOS LaunchAgent). On each run it:

1. Scans all **local desktop repos** + **GitHub repositories** (via `gh` CLI)
2. Picks the **oldest unmaintained project** (round‑robin)
3. Reads only `README.md` — touches nothing else
4. Calls **Groq AI** (`llama-3.3-70b-versatile`) for surgical README improvements
5. Validates the patch with safety guardrails (≤35% change cap, no broken fences, title preserved)
6. Makes **3–4 atomic commits** directly to `main` — for GitHub streak activity
7. Pushes, exits. Done.

## 🚀 Setup

```bash
git clone https://github.com/santusht06/bot.git
cd bot
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=gsk_your_key_here
```

Make sure you're authenticated with GitHub CLI:

```bash
gh auth login
```

## ▶️ Run manually

```bash
./venv/bin/python bot.py
```

## ⏰ Auto-start on macOS login (LaunchAgent)

```bash
cp com.santusht.ai-readme-bot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.santusht.ai-readme-bot.plist
```

The bot will run automatically every day at midnight.

## 📋 View logs

```bash
tail -f data/bot.log
```

## 🛡️ Safety Guardrails

- **README.md only** — no other files touched
- **≤35% line change cap** per day
- Rejects patches that remove the primary `# Title`
- Rejects unclosed code fences
- **1 project per day** enforced via SQLite

## 🗂️ Project Structure

```
bot/
├── bot.py                          # Main autonomous daemon
├── run.sh                          # Simple shell launcher
├── com.santusht.ai-readme-bot.plist  # macOS LaunchAgent
├── requirements.txt
├── .env                            # Your Groq API key (not committed)
└── data/
    ├── bot.db                      # SQLite: project inventory + run history
    ├── vectors.json                # Offline vector embeddings
    └── bot.log                     # Daily run logs
```

## 📦 Requirements

- Python 3.12+
- `gh` CLI (authenticated)
- `git`
- Groq API key (free tier works)
