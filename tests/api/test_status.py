from fastapi.testclient import TestClient

from blackgeorge.adapters.base import ModelResponse
from tests.utils import FakeAdapter


def test_get_run(client: TestClient, desk, registered_worker):
    responses = [ModelResponse(content="Done", tool_calls=[], usage={}, raw={})]
    desk.adapter = FakeAdapter(responses)

    run_response = client.post(
        "/api/v1/runs/worker/TestWorker",
        json={"input": "Test task"},
    )
    run_data = run_response.json()
    run_id = run_data["run_id"]

    response = client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == run_id
    assert data["status"] == "completed"


def test_get_run_not_found(client: TestClient):
    response = client.get("/api/v1/runs/nonexistent-run-id")
    assert response.status_code == 404


def test_list_runs_empty(client: TestClient):
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 0


def test_list_runs_with_data(client: TestClient, desk, registered_worker):
    responses = [ModelResponse(content="Done", tool_calls=[], usage={}, raw={})]
    desk.adapter = FakeAdapter(responses)

    client.post(
        "/api/v1/runs/worker/TestWorker",
        json={"input": "Test task"},
    )

    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["status"] == "completed"


def test_list_runs_with_status_filter(client: TestClient, desk, registered_worker):
    responses = [ModelResponse(content="Done", tool_calls=[], usage={}, raw={})]
    desk.adapter = FakeAdapter(responses)

    client.post(
        "/api/v1/runs/worker/TestWorker",
        json={"input": "Test task"},
    )

    response = client.get("/api/v1/runs?status=completed")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1

    response = client.get("/api/v1/runs?status=paused")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 0


def test_list_runs_with_limit(client: TestClient, desk, registered_worker):
    responses = [ModelResponse(content="Done", tool_calls=[], usage={}, raw={})]
    desk.adapter = FakeAdapter(responses)

    for i in range(5):
        client.post(
            "/api/v1/runs/worker/TestWorker",
            json={"input": f"Test task {i}"},
        )

    response = client.get("/api/v1/runs?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_list_runs_with_offset(client: TestClient, desk, registered_worker):
    responses = [ModelResponse(content="Done", tool_calls=[], usage={}, raw={})]
    desk.adapter = FakeAdapter(responses)

    for i in range(5):
        client.post(
            "/api/v1/runs/worker/TestWorker",
            json={"input": f"Test task {i}"},
        )

    response = client.get("/api/v1/runs?offset=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


def test_get_run_events(client: TestClient, desk, registered_worker):
    responses = [ModelResponse(content="Done", tool_calls=[], usage={}, raw={})]
    desk.adapter = FakeAdapter(responses)

    run_response = client.post(
        "/api/v1/runs/worker/TestWorker",
        json={"input": "Test task"},
    )
    run_data = run_response.json()
    run_id = run_data["run_id"]

    response = client.get(f"/api/v1/runs/{run_id}/events")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0


def test_get_run_events_not_found(client: TestClient):
    response = client.get("/api/v1/runs/nonexistent-run-id/events")
    assert response.status_code == 404
