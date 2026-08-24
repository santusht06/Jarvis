import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from config import settings
from db import db
from project_scanner import project_scanner
from vector_db import vector_db
from scheduler import scheduler_manager

app = FastAPI(
    title="AI Agent Project Maintainer Tool",
    description="Backend tool for automated 1-project-per-day README maintenance with Vector DB",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    try:
        project_scanner.scan_all()
    except Exception as e:
        print(f"[Startup] Initial scan notice: {e}")
    scheduler_manager.start()

@app.get("/")
def get_root():
    return {
        "tool": "AI Smart Agent Project Maintainer",
        "status": "online",
        "policy": "1 project per 24 hours rate-limited maintenance",
        "endpoints": [
            "GET /api/status",
            "GET /api/projects",
            "POST /api/scan",
            "POST /api/run-daily",
            "GET /api/logs",
            "GET /api/vector-context/{project_id}"
        ]
    }

@app.get("/api/status")
def get_system_status():
    projects = db.list_projects()
    today_run = db.get_daily_project_executed_today()
    logs = db.list_logs(limit=10)
    
    local_count = sum(1 for p in projects if p['type'] == 'local')
    github_count = sum(1 for p in projects if p['type'] == 'github')
    
    next_proj = projects[0] if projects else None
    
    return {
        "status": "online",
        "total_projects": len(projects),
        "local_projects": local_count,
        "github_projects": github_count,
        "today_executed": today_run,
        "next_project_in_queue": next_proj,
        "recent_logs_count": len(logs),
        "daily_limit": settings.DAILY_PROJECT_LIMIT
    }

@app.get("/api/projects")
def list_projects():
    return db.list_projects()

@app.post("/api/scan")
def trigger_scan():
    projects = project_scanner.scan_all()
    return {"success": True, "count": len(projects), "projects": projects}

@app.get("/api/logs")
def list_logs(limit: int = 50):
    return db.list_logs(limit=limit)

class TriggerRequest(BaseModel):
    force: bool = False
    dry_run: bool = False
    project_id: Optional[str] = None

@app.post("/api/run-daily")
def trigger_daily_run(req: TriggerRequest):
    res = scheduler_manager.run_daily_job(
        force=req.force,
        target_project_id=req.project_id,
        dry_run=req.dry_run
    )
    return res

@app.get("/api/vector-context/{project_id}")
def get_vector_context(project_id: str, query: str = "overview setup usage"):
    results = vector_db.search_context(project_id, query, top_k=5)
    return {"project_id": project_id, "query": query, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
