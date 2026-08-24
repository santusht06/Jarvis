import os
import subprocess
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Any, Tuple
from config import settings
from db import db

class GitAutomator:
    def execute_maintenance(
        self,
        project: Dict[str, Any],
        new_readme_content: str,
        patch_summary: str,
        diff_str: str,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Applies patch, commits, and pushes to GitHub / local git repository."""
        proj_id = project['id']
        proj_name = project['name']
        proj_type = project['type']
        path_or_url = project['path_or_url']

        if dry_run:
            log_id = db.add_log(
                project_id=proj_id,
                project_name=proj_name,
                patch_summary=f"[DRY-RUN] {patch_summary}",
                diff_content=diff_str,
                status="dry_run"
            )
            return {"success": True, "status": "dry_run", "log_id": log_id, "message": "Dry run completed."}

        target_dir = path_or_url
        is_temp = False

        # If GitHub URL and not local folder, clone temp repository
        if proj_type == 'github' and (not os.path.exists(path_or_url) or not os.path.isdir(path_or_url)):
            temp_dir = tempfile.mkdtemp(prefix=f"maint_{proj_name}_")
            clone_cmd = ["gh", "repo", "clone", path_or_url, temp_dir]
            res = subprocess.run(clone_cmd, capture_output=True, text=True)
            if res.returncode != 0:
                # Try standard git clone
                clone_cmd = ["git", "clone", path_or_url, temp_dir]
                res = subprocess.run(clone_cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    err_msg = f"Failed to clone repository: {res.stderr}"
                    db.add_log(proj_id, proj_name, patch_summary, diff_str, "failed", error_message=err_msg)
                    return {"success": False, "error": err_msg}
            target_dir = temp_dir
            is_temp = True

        try:
            # Step 1: Ensure git repository exists
            git_dir = os.path.join(target_dir, '.git')
            if not os.path.exists(git_dir):
                subprocess.run(["git", "init"], cwd=target_dir, check=True)
                subprocess.run(["git", "branch", "-M", "main"], cwd=target_dir, check=True)

            # Step 2: Create a daily maintenance branch
            today_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            branch_name = f"{settings.DEFAULT_BRANCH_PREFIX}{today_stamp}"
            
            # Write new README.md
            readme_path = os.path.join(target_dir, "README.md")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(new_readme_content)

            # Step 3: Git checkout branch & commit
            subprocess.run(["git", "config", "user.email", settings.GIT_USER_EMAIL], cwd=target_dir, capture_output=True)
            subprocess.run(["git", "config", "user.name", settings.GIT_USER_NAME], cwd=target_dir, capture_output=True)
            subprocess.run(["git", "checkout", "-b", branch_name], cwd=target_dir, capture_output=True)
            subprocess.run(["git", "add", "README.md"], cwd=target_dir, check=True)
            commit_msg = f"docs(readme): automated daily maintenance update [AI Agent]\n\nChanges:\n- {patch_summary}"
            commit_res = subprocess.run(["git", "commit", "-m", commit_msg], cwd=target_dir, capture_output=True, text=True)

            commit_sha = ""
            sha_res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=target_dir, capture_output=True, text=True)
            if sha_res.returncode == 0:
                commit_sha = sha_res.stdout.strip()[:8]

            # Step 4: Push to GitHub / Remote if origin exists
            pr_url = ""
            has_remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=target_dir, capture_output=True).returncode == 0
            
            if has_remote:
                # Push branch
                push_res = subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=target_dir, capture_output=True, text=True)
                if push_res.returncode == 0:
                    # Create PR via GH CLI if configured or available
                    pr_cmd = ["gh", "pr", "create", "--title", f"docs: AI README maintenance ({proj_name})", "--body", f"Automated README Patch Summary:\n{patch_summary}\n\nDiff preview:\n```diff\n{diff_str[:1500]}\n```"]
                    pr_res = subprocess.run(pr_cmd, cwd=target_dir, capture_output=True, text=True)
                    if pr_res.returncode == 0 and "http" in pr_res.stdout:
                        pr_url = pr_res.stdout.strip()

            # Record success log & update project status
            db.mark_project_maintained(proj_id)
            log_id = db.add_log(
                project_id=proj_id,
                project_name=proj_name,
                patch_summary=patch_summary,
                diff_content=diff_str,
                status="success",
                commit_sha=commit_sha,
                pr_url=pr_url
            )

            return {
                "success": True,
                "status": "success",
                "log_id": log_id,
                "commit_sha": commit_sha,
                "pr_url": pr_url,
                "message": f"Successfully updated README for {proj_name}"
            }
        except Exception as e:
            err_msg = str(e)
            db.add_log(proj_id, proj_name, patch_summary, diff_str, "failed", error_message=err_msg)
            return {"success": False, "error": err_msg}
        finally:
            if is_temp:
                shutil.rmtree(target_dir, ignore_errors=True)

git_automator = GitAutomator()
