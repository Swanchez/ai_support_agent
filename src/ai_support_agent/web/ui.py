"""Same-origin browser interface for the support API.

The module serves presentation assets only. Authentication, ownership checks,
CSRF validation and order mutations remain in the API layer.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


STATIC_DIRECTORY = Path(__file__).parent / "static"


def install_browser_ui(app: FastAPI) -> None:
    """Expose the single-page browser client and its immutable local assets."""

    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    @app.get("/", include_in_schema=False)
    def browser_index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get("/login", include_in_schema=False)
    def browser_login() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "login.html")
