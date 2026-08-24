import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from config import settings
from db import db
from project_scanner import project_scanner
from patch_engine import patch_engine
from git_automator import git_automator
from vector_db import vector_db

class MaintenanceScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.is_running = False

    def start(self):
        if not self.is_running:
            # Schedule daily job at midnight / 00:05
            self.scheduler.add_job(
                func=self.run_daily_job,
                trigger="cron",
                hour=0,
                minute=5,
                id="daily_readme_maintenance",
                replace_existing=True
            )
            self.scheduler.start()
            self.is_running = True
            print("[Scheduler] Daily README maintenance scheduler started.")

    def run_daily_job(self, force: bool = False, target_project_id: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Executes maintenance for exactly ONE project per day.
        Enforces rate limiting: Max 1 project per 24h unless force=True.
        """
        # Step 1: Check daily rate limit
        today_run = db.get_daily_project_executed_today()
        if today_run and not force and not target_project_id:
            return {
                "executed": False,
                "reason": f"Daily limit reached. Project '{today_run['project_id']}' was already maintained today. (Enforcing 1 project per day policy)",
                "today_run": today_run
            }

        # Step 2: Refresh project inventory
        all_projects = project_scanner.scan_all()
        if not all_projects:
            return {"executed": False, "reason": "No projects found locally or on GitHub CLI."}

        # Step 3: Pick the target project
        target_project = None
        if target_project_id:
            target_project = db.get_project_by_id(target_project_id)
        
        if not target_project:
            # Pick project with oldest last_maintained_at (round-robin rotation)
            target_project = all_projects[0]

        proj_id = target_project['id']
        proj_name = target_project['name']
        path_or_url = target_project['path_or_url']

        print(f"[Scheduler] Processing project for maintenance: {proj_name} ({proj_id})")

        # Step 4: Read and index codebase context into Vector DB
        indexed_chunks = project_scanner.read_and_index_project(target_project)

        # Step 5: Read current README content
        current_readme = ""
        local_dir = path_or_url
        if not os.path.exists(local_dir):
            possible_local = os.path.expanduser(f"~/Desktop/{proj_name}")
            if os.path.exists(possible_local):
                local_dir = possible_local

        readme_file = project_scanner.find_readme(local_dir) if os.path.exists(local_dir) else None
        if readme_file and os.path.exists(readme_file):
            try:
                with open(readme_file, 'r', encoding='utf-8', errors='ignore') as f:
                    current_readme = f.read()
            except Exception:
                pass

        if not current_readme:
            # Query vector DB for indexed README chunk
            v_context = vector_db.search_context(proj_id, "README project title overview", top_k=3)
            for ctx in v_context:
                if 'README' in ctx.get('metadata', {}).get('path', ''):
                    current_readme = ctx.get('content', '')
                    break

        # Step 6: Generate surgical patch via Patch Engine
        patch_result = patch_engine.generate_and_validate_patch(
            project_id=proj_id,
            project_name=proj_name,
            current_readme=current_readme,
            context_docs=[]
        )

        if not patch_result.get('success'):
            db.record_daily_run(proj_id, "failed")
            return {
                "executed": False,
                "project": target_project,
                "reason": patch_result.get('summary', 'Patch validation failed'),
                "patch_result": patch_result
            }

        # Step 7: Apply patch and push to Git / GitHub
        exec_res = git_automator.execute_maintenance(
            project=target_project,
            new_readme_content=patch_result['new_content'],
            patch_summary=patch_result['summary'],
            diff_str=patch_result['diff'],
            dry_run=dry_run
        )

        # Record execution for today's quota
        if not dry_run:
            db.record_daily_run(proj_id, "completed")

        return {
            "executed": True,
            "project": target_project,
            "indexed_chunks": indexed_chunks,
            "patch_summary": patch_result['summary'],
            "diff": patch_result['diff'],
            "execution_details": exec_res
        }

scheduler_manager = MaintenanceScheduler()
