from fastapi.testclient import TestClient

from blackgeorge.adapters.base import ModelResponse
from tests.utils import FakeAdapter


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_create_worker(client: TestClient, desk):
    response = client.post(
        "/api/v1/workers",
        json={"name": "Researcher", "model": "gpt-4", "instructions": "You are a researcher"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Researcher"
    assert data["model"] == "gpt-4"
    assert data["instructions"] == "You are a researcher"


def test_create_worker_duplicate(client: TestClient, registered_worker):
    response = client.post(
        "/api/v1/workers",
        json={"name": "TestWorker"},
    )
    assert response.status_code == 409


def test_list_workers(client: TestClient, registered_worker):
    response = client.get("/api/v1/workers")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "TestWorker"


def test_get_worker(client: TestClient, registered_worker):
    response = client.get("/api/v1/workers/TestWorker")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "TestWorker"


def test_get_worker_not_found(client: TestClient):
    response = client.get("/api/v1/workers/NonExistent")
    assert response.status_code == 404


def test_delete_worker(client: TestClient, registered_worker):
    response = client.delete("/api/v1/workers/TestWorker")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    response = client.get("/api/v1/workers/TestWorker")
    assert response.status_code == 404


def test_delete_worker_not_found(client: TestClient):
    response = client.delete("/api/v1/workers/NonExistent")
    assert response.status_code == 404
