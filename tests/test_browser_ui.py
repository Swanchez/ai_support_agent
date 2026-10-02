from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_support_agent.web.ui import install_browser_ui


def test_browser_ui_serves_the_entry_page_and_static_assets() -> None:
    app = FastAPI()
    install_browser_ui(app)
    client = TestClient(app)

    index = client.get("/")
    login = client.get("/login")
    stylesheet = client.get("/static/app.css")
    script = client.get("/static/app.js")

    assert index.status_code == 200
    assert "<title>AI Support Agent</title>" in index.text
    assert login.status_code == 200
    assert "Вход · AI Support Agent" in login.text
    assert stylesheet.status_code == 200
    assert ".auth-card" in stylesheet.text
    assert script.status_code == 200
    assert "/api/v1/conversations" in script.text
    assert "new-chat-button" in index.text
