from fastapi.testclient import TestClient

from blackgeorge.adapters.base import ModelResponse
from blackgeorge.worker import Worker
from tests.utils import FakeAdapter


def test_run_worker_success(client: TestClient, desk, registered_worker):
    responses = [ModelResponse(content="Done", tool_calls=[], usage={}, raw={})]
    desk.adapter = FakeAdapter(responses)

    response = client.post(
        "/api/v1/runs/worker/TestWorker",
        json={"input": "Test task", "expected_output": "Test result"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["content"] == "Done"
    assert "run_id" in data


def test_run_worker_not_found(client: TestClient):
    response = client.post(
        "/api/v1/runs/worker/NonExistent",
        json={"input": "Test task"},
    )
    assert response.status_code == 404


def test_run_workforce_success(client: TestClient, desk):
    worker = Worker(name="Worker1", model="test-model")
    desk.register_worker(worker)

    from blackgeorge.workforce import Workforce
    from pydantic import BaseModel

    class WorkerSelection(BaseModel):
        worker: str

    workforce = Workforce(workers=[worker], name="MyWorkforce")
    desk.register_workforce(workforce)

    responses = [
        ModelResponse(content=None, tool_calls=[], usage={}, raw={}),
    ]
    desk.adapter = FakeAdapter(responses)

    response = client.post(
        "/api/v1/runs/workforce/MyWorkforce",
        json={"input": "Test task"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["completed", "failed"]


def test_run_workforce_not_found(client: TestClient):
    response = client.post(
        "/api/v1/runs/workforce/NonExistent",
        json={"input": "Test task"},
    )
    assert response.status_code == 404


def test_resume_run_not_found(client: TestClient):
    response = client.post(
        "/api/v1/runs/nonexistent-run-id/resume",
        json={"decision": True},
    )
    assert response.status_code == 404


def test_resume_run_not_paused(client: TestClient, desk, registered_worker):
    responses = [ModelResponse(content="Done", tool_calls=[], usage={}, raw={})]
    desk.adapter = FakeAdapter(responses)

    run_response = client.post(
        "/api/v1/runs/worker/TestWorker",
        json={"input": "Test task"},
    )
    run_data = run_response.json()
    run_id = run_data["run_id"]

    response = client.post(
        f"/api/v1/runs/{run_id}/resume",
        json={"decision": True},
    )
    assert response.status_code == 400
