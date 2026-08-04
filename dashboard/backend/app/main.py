"""QuantSilico Generals research dashboard API.

Binds to 127.0.0.1 only. Jobs and paths are allowlisted.
Runs from .venv-training. CLI remains independent.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dashboard.backend.app import path_setup  # noqa: F401
from dashboard.backend.app.paths import REPO_ROOT
from dashboard.backend.app.routes.api import router as api_router

app = FastAPI(title="QuantSilico Generals Research Console", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(api_router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
FRONTEND_DIST = REPO_ROOT / "dashboard" / "frontend" / "dist"
ASSETS_DIR = FRONTEND_DIST / "assets"

if ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
def spa_fallback(full_path: str):
    """Explicit SPA fallback — never captures /api/** (registered above)."""
    if full_path.startswith("api/") or full_path == "api":
        return JSONResponse(status_code=404, content={"detail": "Not Found", "schema_version": 1})

    dist = FRONTEND_DIST if FRONTEND_DIST.is_dir() else STATIC_DIR
    if full_path:
        candidate = dist / full_path
        if candidate.is_file():
            return FileResponse(candidate)

    index = dist / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="frontend not built")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "dashboard.backend.app.main:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
