import os
import json
import subprocess
import hashlib
from typing import List, Dict, Any, Optional
from config import settings
from db import db
from vector_db import vector_db

class ProjectScanner:
    def scan_all(self) -> List[Dict[str, Any]]:
        local_projects = self.scan_local_projects()
        github_projects = self.scan_github_projects()
        
        all_projs = local_projects + github_projects
        for proj in all_projs:
            db.upsert_project(
                proj_id=proj['id'],
                name=proj['name'],
                p_type=proj['type'],
                path_or_url=proj['path_or_url'],
                description=proj.get('description', ''),
                has_readme=proj.get('has_readme', False)
            )
            # Index codebase into Vector DB
            self.read_and_index_project(proj)
        return db.list_projects()

    def scan_local_projects(self) -> List[Dict[str, Any]]:
        projects = []
        visited = set()

        for base_path in settings.LOCAL_SCAN_PATHS:
            if not os.path.exists(base_path):
                continue
            
            try:
                entries = os.listdir(base_path)
            except Exception:
                continue

            for entry in entries:
                if entry.startswith('.') or entry in ('node_modules', 'venv', '.venv', 'Library', 'Applications'):
                    continue

                full_path = os.path.join(base_path, entry)
                if not os.path.isdir(full_path) or full_path in visited:
                    continue
                visited.add(full_path)

                # Check if it looks like a project
                is_git = os.path.exists(os.path.join(full_path, '.git'))
                has_manifest = any(os.path.exists(os.path.join(full_path, f)) for f in [
                    'package.json', 'pyproject.toml', 'requirements.txt', 'Cargo.toml', 'go.mod', 'CMakeLists.txt', 'Makefile', 'README.md', 'readme.md'
                ])

                if is_git or has_manifest:
                    has_readme = self.find_readme(full_path) is not None
                    proj_id = "local-" + hashlib.md5(full_path.encode()).hexdigest()[:12]
                    projects.append({
                        "id": proj_id,
                        "name": entry,
                        "type": "local",
                        "path_or_url": full_path,
                        "description": f"Local repository at {full_path}",
                        "has_readme": has_readme
                    })
        return projects

    def scan_github_projects(self) -> List[Dict[str, Any]]:
        projects = []
        try:
            cmd = ["gh", "repo", "list", "--limit", "100", "--json", "name,nameWithOwner,url,sshUrl,description,isPrivate,defaultBranchRef"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and res.stdout.strip():
                repos = json.loads(res.stdout)
                for repo in repos:
                    name_with_owner = repo.get('nameWithOwner') or repo.get('name')
                    proj_id = "github-" + hashlib.md5(name_with_owner.encode()).hexdigest()[:12]
                    projects.append({
                        "id": proj_id,
                        "name": repo.get('name', ''),
                        "type": "github",
                        "path_or_url": repo.get('url', ''),
                        "description": repo.get('description') or f"GitHub repository {name_with_owner}",
                        "has_readme": True  # Default assumed, verified on clone/fetch
                    })
        except Exception as e:
            print(f"[ProjectScanner] GitHub CLI scan notice: {e}")
        return projects

    def find_readme(self, folder_path: str) -> Optional[str]:
        if not os.path.exists(folder_path):
            return None
        for name in ['README.md', 'readme.md', 'README.rst', 'README.txt', 'README']:
            p = os.path.join(folder_path, name)
            if os.path.exists(p) and os.path.isfile(p):
                return p
        return None

    def read_and_index_project(self, project: Dict[str, Any]) -> int:
        """Reads key project codebase files & embeds into vector DB."""
        proj_type = project['type']
        path_or_url = project['path_or_url']
        proj_id = project['id']
        proj_name = project['name']

        files_data = []
        root_dir = path_or_url

        # If GitHub repo and not local path, check if we can inspect locally or clone temp
        if proj_type == 'github' and not os.path.exists(path_or_url):
            # Check if cached locally or in desktop
            possible_local = os.path.expanduser(f"~/Desktop/{proj_name}")
            if os.path.exists(possible_local):
                root_dir = possible_local

        if os.path.exists(root_dir):
            readme_path = self.find_readme(root_dir)
            if readme_path:
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        files_data.append({'path': 'README.md', 'content': f.read()})
                except Exception:
                    pass

            # Read key metadata & manifest files
            manifest_files = ['package.json', 'pyproject.toml', 'requirements.txt', 'Cargo.toml', 'go.mod', 'Makefile', 'setup.py']
            for mf in manifest_files:
                p = os.path.join(root_dir, mf)
                if os.path.exists(p) and os.path.isfile(p):
                    try:
                        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                            files_data.append({'path': mf, 'content': f.read()[:2000]})
                    except Exception:
                        pass

            # Scan top-level code files and structure
            try:
                tree_lines = []
                for root, dirs, files in os.walk(root_dir):
                    # Skip heavy directories
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'venv', '.venv', 'dist', 'build')]
                    rel_root = os.path.relpath(root, root_dir)
                    if rel_root == '.':
                        rel_root = ''
                    for f in files:
                        if not f.startswith('.'):
                            tree_lines.append(os.path.join(rel_root, f))
                
                files_data.append({
                    'path': 'FILE_TREE.txt',
                    'content': "Project File Structure:\n" + "\n".join(tree_lines[:200])
                })
            except Exception:
                pass

        if files_data:
            indexed_count = vector_db.index_project(proj_id, proj_name, files_data)
            db.mark_project_indexed(proj_id)
            return indexed_count
        return 0

project_scanner = ProjectScanner()
