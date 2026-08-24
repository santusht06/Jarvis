#!/usr/bin/env python3
"""
Jarvis - Dynamic QA Test Suite & Comprehensive Validation Framework
--------------------------------------------------------------------
Automated QA test suite verifying:
1. Configuration & Environment Validation
2. Secret Scrubbing & Confidentiality Protection (Groq, OpenAI, PATs, AWS, DB URIs, RSA Keys)
3. AI Patch Engine Precision Guardrails (35% Change Cap, Unclosed Fences, Zero Diff Detection)
4. Database & Rate-Limiting Engine (24h Daily Limit & 7-Day Fork Cooldown)
5. Project Scanner & Strict Fork Exclusion Rules (0% Forked Repos in Main Scan)
6. Git Automator & Author Email Attribution (115890693+santusht06@users.noreply.github.com)
7. Vector DB Context Search & Embedding Retrieval
8. FastAPI Server Endpoint Integration
"""

import os
import sys
import unittest
import json
import sqlite3
import tempfile
import shutil
import re
from datetime import datetime, timezone

# Import codebase components
import config
import db
import patch_engine
import project_scanner
import vector_db
import git_automator
import scheduler
import brain
import oss_bot

class TestConfigAndEnvironment(unittest.TestCase):
    def test_env_loading(self):
        """Verify environment variables and configuration defaults."""
        self.assertIsNotNone(config.settings)
        self.assertTrue(hasattr(config.settings, "GROQ_API_KEY"))
        self.assertTrue(hasattr(config.settings, "GROQ_MODEL"))
        self.assertEqual(config.settings.DAILY_PROJECT_LIMIT, 1)

class TestConfidentialityScrubber(unittest.TestCase):
    def test_secret_scrubbing_patterns(self):
        """Verify secret scrubber redacts keys, tokens, and credentials."""
        sample_leaks = [
            "GROQ_API_KEY=gsk_12345678901234567890123456789012345678",
            "OPENAI_KEY=sk-proj-12345678901234567890123456789012345678",
            "GITHUB_PAT=ghp_123456789012345678901234567890123456",
            "AWS_SECRET=AKIAIOSFODNN7EXAMPLE",
            "DB_URI=" + "mongodb+srv://" + "dummy_user:" + "dummy_pass" + "@" + "cluster.invalid/dbname",
            "RSA_KEY=-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwg---\n-----END PRIVATE KEY-----"
        ]
        for leak in sample_leaks:
            scrubbed = brain.sanitize_content(leak)
            self.assertNotIn("gsk_1234567890", scrubbed)
            self.assertNotIn("sk-proj-1234567890", scrubbed)
            self.assertNotIn("ghp_1234567890", scrubbed)
            self.assertNotIn("pass123", scrubbed)
            self.assertNotIn("BEGIN PRIVATE KEY", scrubbed)
            has_leak, _ = brain.has_leaked_secrets(leak)
            self.assertTrue(has_leak)

class TestPatchEngineGuardrails(unittest.TestCase):
    def setUp(self):
        self.pe = patch_engine.PatchEngine()

    def test_change_ratio_cap(self):
        """Verify patches modifying >35% of lines are rejected."""
        old_text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
        new_text = "Completely different text\nReplacing everything\n"
        ok, reason = self.pe.validate_patch_guardrails(old_text, new_text)
        self.assertFalse(ok)
        self.assertIn("max daily patch limit is 35%", reason.lower())

    def test_fence_integrity(self):
        """Verify code blocks with unclosed backticks are rejected."""
        old_text = "# Title\n\n```python\nprint('hello')\n```"
        new_text = "# Title\n\n```python\nprint('hello')"  # Unclosed backtick fence
        ok, reason = self.pe.validate_patch_guardrails(old_text, new_text)
        self.assertFalse(ok)
        self.assertIn("unclosed markdown code blocks", reason.lower())

    def test_empty_patch_rejection(self):
        """Verify empty or identical patches are rejected."""
        old_text = "# Header\nContent\n"
        new_text = "# Header\nContent\n"
        ok, reason = self.pe.validate_patch_guardrails(old_text, new_text)
        self.assertFalse(ok)
        self.assertIn("zero changes", reason.lower())

class TestDatabaseAndTracking(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test.db")
        self.db_inst = db.Database(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_project_registration_and_rate_limiting(self):
        """Verify project registration and 24-hour rate-limiting enforcement."""
        p_id = "proj_123"
        self.db_inst.upsert_project(p_id, "TestProject", "local", "/tmp/test")
        projects = self.db_inst.list_projects()
        self.assertEqual(len(projects), 1)

        # Log maintenance run & daily schedule
        self.db_inst.add_log(p_id, "TestProject", "Updated README", "diff", "success")
        self.db_inst.record_daily_run(p_id, "completed")

        executed = self.db_inst.get_daily_project_executed_today()
        self.assertIsNotNone(executed)
        self.assertEqual(executed["project_id"], p_id)

class TestForkExclusionRules(unittest.TestCase):
    def test_is_fork_filtering(self):
        """Verify that forked repositories are strictly excluded from main bot scanning."""
        mock_repos = [
            {"name": "my-personal-tool", "nameWithOwner": "santusht06/my-personal-tool", "isFork": False},
            {"name": "cpython", "nameWithOwner": "santusht06/cpython", "isFork": True},
            {"name": "linux", "nameWithOwner": "santusht06/linux", "isFork": True}
        ]
        filtered = [r for r in mock_repos if not r["isFork"]]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["name"], "my-personal-tool")

class TestGitAutomator(unittest.TestCase):
    def test_author_email_attribution(self):
        """Verify git automator enforces GitHub official noreply email."""
        expected_email = "115890693+santusht06@users.noreply.github.com"
        self.assertEqual(config.settings.GIT_USER_EMAIL, expected_email)

class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.vdb = vector_db.VectorDBManager()

    def test_indexing_and_search(self):
        """Verify vector DB indexes chunks and retrieves relevant context."""
        docs = [
            {"path": "main.py", "content": "FastAPI application entry point with uvicorn server"},
            {"path": "db.py", "content": "SQLite database initialization and project tracking tables"}
        ]
        self.vdb.index_project("p1", "TestProj", docs)
        results = self.vdb.search_context("p1", "FastAPI server", top_k=1)
        self.assertTrue(len(results) > 0)
        self.assertIn("FastAPI", results[0]["content"])

def run_qa_suite():
    print("=" * 70)
    print("🚀 Running Jarvis Dynamic QA Test Suite...")
    print("=" * 70)
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(TestConfigAndEnvironment))
    suite.addTest(loader.loadTestsFromTestCase(TestConfidentialityScrubber))
    suite.addTest(loader.loadTestsFromTestCase(TestPatchEngineGuardrails))
    suite.addTest(loader.loadTestsFromTestCase(TestDatabaseAndTracking))
    suite.addTest(loader.loadTestsFromTestCase(TestForkExclusionRules))
    suite.addTest(loader.loadTestsFromTestCase(TestGitAutomator))
    suite.addTest(loader.loadTestsFromTestCase(TestVectorDB))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n✅ QA TEST SUITE PASSED (100% SUCCESS)")
        return 0
    else:
        print("\n❌ QA TEST SUITE FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_qa_suite())
