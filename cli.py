#!/usr/bin/env python3
import sys
import argparse
import json
from config import settings
from db import db
from project_scanner import project_scanner
from vector_db import vector_db
from scheduler import scheduler_manager

def cmd_scan(args):
    print("[AI Maintainer Tool] Scanning local desktop and GitHub CLI repositories...")
    projects = project_scanner.scan_all()
    print(f"✅ Discovered & Indexed {len(projects)} total projects.")
    for p in projects[:15]:
        print(f"  • [{p['type'].upper()}] {p['name']} ({p['id']}) -> {p['path_or_url']}")
    if len(projects) > 15:
        print(f"  ... and {len(projects) - 15} more projects.")

def cmd_list(args):
    projects = db.list_projects()
    print(f"\n📦 Tracked Maintainer Inventory ({len(projects)} projects):")
    print("-" * 80)
    print(f"{'ID':<18} {'TYPE':<8} {'NAME':<25} {'LAST MAINTAINED'}")
    print("-" * 80)
    for p in projects:
        last_m = p['last_maintained_at'] or "Never"
        print(f"{p['id']:<18} {p['type']:<8} {p['name']:<25} {last_m}")
    print("-" * 80)

def cmd_run(args):
    print("[AI Maintainer Tool] Running 1-project-per-day maintenance routine...")
    res = scheduler_manager.run_daily_job(
        force=args.force,
        target_project_id=args.project_id,
        dry_run=args.dry_run
    )
    
    if not res.get('executed'):
        print(f"⚠️ Maintenance Job Not Executed: {res.get('reason')}")
        sys.exit(0)

    proj = res['project']
    exec_info = res.get('execution_details', {})
    print(f"✅ Successfully processed project: {proj['name']} ({proj['id']})")
    print(f"📝 Summary: {res.get('patch_summary')}")
    if exec_info.get('commit_sha'):
        print(f"📌 Git Commit SHA: {exec_info.get('commit_sha')}")
    if exec_info.get('pr_url'):
        print(f"🔗 GitHub PR: {exec_info.get('pr_url')}")
    print("\n--- Diff Preview ---")
    print(res.get('diff', 'No diff changes'))

def cmd_status(args):
    status = db.get_daily_project_executed_today()
    projects = db.list_projects()
    print(f"\n🤖 AI Agent Maintainer Status:")
    print(f"  • Total Tracked Projects: {len(projects)}")
    print(f"  • Daily Quota Policy: 1 project per 24 hours")
    if status:
        print(f"  • Today's Status: EXECUTED today for project '{status['project_id']}' at {status['updated_at']}")
    else:
        print(f"  • Today's Status: READY (No maintenance executed yet today)")

def cmd_logs(args):
    logs = db.list_logs(limit=args.limit)
    print(f"\n📋 Audit Logs (Last {len(logs)} entries):")
    for log in logs:
        print(f"[{log['timestamp']}] ID:{log['id']} Proj:{log['project_name']} Status:{log['status'].upper()}")
        print(f"  Summary: {log['patch_summary']}")
        if log.get('commit_sha'):
            print(f"  Commit: {log['commit_sha']}")
        if log.get('pr_url'):
            print(f"  PR: {log['pr_url']}")
        print("-" * 60)

def cmd_vector_search(args):
    results = vector_db.search_context(args.project_id, args.query, top_k=args.top_k)
    print(f"\n🔍 Vector DB Results for Project '{args.project_id}' (Query: '{args.query}'):")
    for idx, r in enumerate(results, 1):
        print(f"\n--- Result #{idx} ({r['metadata'].get('path')}) ---")
        print(r['content'][:400])

def cmd_server(args):
    import uvicorn
    print(f"[AI Maintainer Tool] Starting FastAPI Server on {args.host}:{args.port}...")
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)

def main():
    parser = argparse.ArgumentParser(description="AI Smart Agent CLI Tool - Automated 1-Project-a-Day README Maintainer")
    subparsers = parser.add_subparsers(dest="command", help="Available tool commands")

    # scan
    subparsers.add_parser("scan", help="Scan local desktop and GitHub CLI repositories & index vector DB")
    
    # list
    subparsers.add_parser("list", help="List all tracked projects and maintenance status")

    # run-daily
    run_parser = subparsers.add_parser("run-daily", help="Run daily maintenance routine (1 project per day)")
    run_parser.add_argument("--force", action="store_true", help="Bypass 1-project-per-day rate limit constraint")
    run_parser.add_argument("--dry-run", action="store_true", help="Generate patch & diff without committing/pushing")
    run_parser.add_argument("--project-id", type=str, default=None, help="Target specific project ID")

    # status
    subparsers.add_parser("status", help="Check today's maintenance status and daily quota")

    # logs
    logs_parser = subparsers.add_parser("logs", help="View maintenance audit logs")
    logs_parser.add_argument("--limit", type=int, default=10, help="Number of log entries")

    # vector-search
    vec_parser = subparsers.add_parser("vector-search", help="Query vector DB codebase context for a project")
    vec_parser.add_argument("project_id", type=str, help="Target project ID")
    vec_parser.add_argument("query", type=str, help="Search query string")
    vec_parser.add_argument("--top-k", type=int, default=3, help="Number of results")

    # server
    srv_parser = subparsers.add_parser("server", help="Launch FastAPI HTTP API backend server daemon")
    srv_parser.add_argument("--host", type=str, default="127.0.0.1")
    srv_parser.add_argument("--port", type=int, default=8000)
    srv_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "scan": cmd_scan,
        "list": cmd_list,
        "run-daily": cmd_run,
        "status": cmd_status,
        "logs": cmd_logs,
        "vector-search": cmd_vector_search,
        "server": cmd_server
    }
    cmds[args.command](args)

if __name__ == "__main__":
    main()
