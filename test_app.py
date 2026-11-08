from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"health": "ok"}


def test_chat_rejects_invalid_payload() -> None:
    payload = {"query": "", "last_3_turn": []}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422

