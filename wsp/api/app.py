"""FastAPI 应用入口。

同时托管单页前端（``wsp/web/index.html``）与 API。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .routes import router

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="WSP Studio",
    version="0.1.0",
    description="小模型权重编辑工作台",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>WSP Studio</h1><p>前端文件缺失</p>", status_code=500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


def main() -> None:
    """启动开发服务器。"""
    import uvicorn

    host = os.environ.get("WSP_HOST", "127.0.0.1")
    port = int(os.environ.get("WSP_PORT", "8765"))
    uvicorn.run("wsp.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
