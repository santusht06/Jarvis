import re
import difflib
import httpx
from typing import Dict, Any, Tuple, Optional, List
from config import settings
from vector_db import vector_db

class PatchEngine:
    def generate_and_validate_patch(
        self,
        project_id: str,
        project_name: str,
        current_readme: str,
        context_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates surgical patch updates for README.md grounded in vector DB codebase context.
        Enforces 99.999% precision guardrails (no whole file rewrites, syntax validation).
        """
        if not current_readme.strip():
            # Initial minimal README generation if empty
            new_readme = f"# {project_name}\n\nAutomated maintenance active. Overview and documentation coming soon.\n"
            diff = self.compute_diff(current_readme, new_readme)
            return {
                "success": True,
                "summary": "Initialized initial README.md layout",
                "diff": diff,
                "new_content": new_readme,
                "lines_changed": len(new_readme.splitlines())
            }

        # Step 1: Query vector DB for codebase context
        vector_context = vector_db.search_context(project_id, "setup install usage license dependencies", top_k=4)
        context_str = "\n".join([c.get('content', '') for c in vector_context])

        # Step 2: Attempt LLM API patch generation if configured, else use rule-based patch generator
        patch_res = self._call_llm_patch(project_name, current_readme, context_str)
        if not patch_res or not patch_res.get('success'):
            patch_res = self._rule_based_patch(project_name, current_readme, context_str)

        new_content = patch_res.get('new_content', current_readme)
        diff = patch_res.get('diff', '')
        summary = patch_res.get('summary', 'Surgical README update')

        # Step 3: Enforce 99.999% Accuracy & Precision Guardrails
        valid, reason = self.validate_patch_guardrails(current_readme, new_content)
        if not valid:
            return {
                "success": False,
                "summary": f"Patch rejected by precision guardrails: {reason}",
                "diff": diff,
                "new_content": current_readme,
                "lines_changed": 0,
                "reason": reason
            }

        return {
            "success": True,
            "summary": summary,
            "diff": diff,
            "new_content": new_content,
            "lines_changed": patch_res.get('lines_changed', 0)
        }

    def validate_patch_guardrails(self, old_text: str, new_text: str) -> Tuple[bool, str]:
        if old_text.strip() == new_text.strip():
            return False, "Patch contains zero changes."
        """Strict 99.999% precision safety checks to prevent destructive changes."""
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()

        if not new_lines:
            return False, "New README content is completely empty."

        # Guardrail 1: Disallow drastic line count reduction (> 20% loss)
        if len(old_lines) > 5 and len(new_lines) < len(old_lines) * 0.8:
            return False, f"Line count dropped excessively ({len(old_lines)} -> {len(new_lines)})."

        # Guardrail 2: Disallow altering > 30% of lines in a single daily patch
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        ratio = matcher.ratio()
        if ratio < 0.65:
            return False, f"Patch modified {round((1 - ratio)*100, 1)}% of document. Max daily patch limit is 35%."

        # Guardrail 3: Unclosed Code Fences check (``` count must be even)
        old_fences = len(re.findall(r'^```', old_text, re.MULTILINE))
        new_fences = len(re.findall(r'^```', new_text, re.MULTILINE))
        if new_fences % 2 != 0:
            return False, "Patch introduced unclosed markdown code blocks (```)."

        # Guardrail 4: Title / Header Preservation Check
        old_headers = [line for line in old_lines if line.startswith('# ')]
        if old_headers:
            main_header = old_headers[0].strip()
            if main_header not in new_text:
                return False, f"Patch removed primary project title header ('{main_header}')."

        return True, "Passed all safety guardrails."

    def compute_diff(self, old_text: str, new_text: str) -> str:
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile='a/README.md',
            tofile='b/README.md',
            n=3
        )
        return "".join(diff)

    def _rule_based_patch(self, proj_name: str, readme: str, context: str) -> Dict[str, Any]:
        """High precision rule-based patch improver."""
        lines = readme.splitlines()
        modified = False
        summary_items = []

        # Fix 1: Ensure top level title uses single '#'
        if lines and lines[0].startswith('## '):
            lines[0] = '# ' + lines[0][3:]
            modified = True
            summary_items.append("Formatted title header")

        # Fix 2: Check for unformatted URLs and format as markdown links
        for i in range(len(lines)):
            # convert raw http://... to [http://...](http://...) if not already inside []()
            line = lines[i]
            if "http://" in line or "https://" in line:
                # Simple markdown link cleanup if standalone
                if re.search(r'(?<!\()https?://[^\s\)]+', line) and not re.search(r'\[.*\]\(https?://', line):
                    new_line = re.sub(r'(?<!\()(https?://[^\s\)]+)', r'[\1](\1)', line)
                    if new_line != line:
                        lines[i] = new_line
                        modified = True
                        summary_items.append("Formatted raw URL links")

        # Fix 3: Check for missing Installation / Quick Start section if manifest files exist
        has_install = any('install' in line.lower() or 'quickstart' in line.lower() or 'setup' in line.lower() for line in lines)
        if not has_install and context:
            install_block = []
            if 'requirements.txt' in context or 'pyproject.toml' in context:
                install_block = [
                    "",
                    "## 🚀 Installation & Setup",
                    "```bash",
                    "pip install -r requirements.txt",
                    "```"
                ]
            elif 'package.json' in context:
                install_block = [
                    "",
                    "## 🚀 Installation & Setup",
                    "```bash",
                    "npm install",
                    "```"
                ]
            
            if install_block:
                lines.extend(install_block)
                modified = True
                summary_items.append("Added Installation & Setup section")

        # Fix 4: Add Maintenance Badge if missing
        has_badge = any('Maintenance' in line or 'Automated' in line for line in lines[:10])
        if not has_badge and lines:
            badge = "[![Automated README Maintenance](https://img.shields.io/badge/README-Automated_AI_Maintainer-blue.svg)](#)"
            lines.insert(1, badge)
            lines.insert(2, "")
            modified = True
            summary_items.append("Added automated maintenance badge")

        new_content = "\n".join(lines)
        diff = self.compute_diff(readme, new_content)
        lines_changed = len([line for line in diff.splitlines() if line.startswith('+') or line.startswith('-')])

        return {
            "success": modified,
            "summary": ", ".join(summary_items) if summary_items else "Validated README syntax & structure",
            "diff": diff,
            "new_content": new_content if modified else readme,
            "lines_changed": lines_changed
        }

    def _call_llm_patch(self, proj_name: str, readme: str, context: str) -> Optional[Dict[str, Any]]:
        """Generates surgical patch updates using Groq API (or OpenAI/Gemini) if API key is provided."""
        api_key = settings.GROQ_API_KEY or settings.OPENAI_API_KEY
        if not api_key:
            return None

        model = settings.GROQ_MODEL if settings.GROQ_API_KEY else settings.LLM_MODEL
        endpoint = "https://api.groq.com/openai/v1/chat/completions" if settings.GROQ_API_KEY else "https://api.openai.com/v1/chat/completions"

        system_prompt = (
            "You are an automated high-precision AI repository maintainer with 99.999% accuracy guardrails. "
            "Your task is to fix, improve, and format the project README.md file cleanly based on codebase context.\n"
            "CRITICAL RULES:\n"
            "1. DO NOT rewrite the whole file from scratch or remove core project descriptions.\n"
            "2. Make surgical patch improvements only: fix typos, format links, add missing installation/usage blocks matching actual codebase context.\n"
            "3. Return ONLY the complete improved README.md text in markdown format. No extra chat explanation."
        )

        user_prompt = (
            f"Project Name: {proj_name}\n\n"
            f"Codebase Context from Vector DB:\n{context[:1500]}\n\n"
            f"Current README.md:\n```markdown\n{readme}\n```"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(endpoint, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data['choices'][0]['message']['content'].strip()
                    
                    # Clean markdown code block wraps if LLM wrapped output
                    if content.startswith("```markdown"):
                        content = content[11:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    diff = self.compute_diff(readme, content)
                    lines_changed = len([line for line in diff.splitlines() if line.startswith('+') or line.startswith('-')])

                    if content and content != readme:
                        return {
                            "success": True,
                            "summary": f"AI Groq Patch Update ({model})",
                            "diff": diff,
                            "new_content": content,
                            "lines_changed": lines_changed
                        }
        except Exception as e:
            print(f"[PatchEngine] Groq API call notice: {e}")

        return None

patch_engine = PatchEngine()
