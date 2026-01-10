from fastapi.testclient import TestClient

from blackgeorge.worker import Worker


def test_create_workforce(client: TestClient, desk):
    worker1 = Worker(name="Worker1", model="test-model")
    worker2 = Worker(name="Worker2", model="test-model")
    desk.register_worker(worker1)
    desk.register_worker(worker2)

    response = client.post(
        "/api/v1/workforces",
        json={"name": "MyWorkforce", "workers": ["Worker1", "Worker2"], "mode": "managed"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "MyWorkforce"
    assert data["workers"] == ["Worker1", "Worker2"]
    assert data["mode"] == "managed"


def test_create_workforce_duplicate(client: TestClient, desk):
    worker = Worker(name="Worker1", model="test-model")
    desk.register_worker(worker)

    from blackgeorge.workforce import Workforce
    workforce = Workforce(workers=[worker], name="ExistingWorkforce")
    desk.register_workforce(workforce)

    response = client.post(
        "/api/v1/workforces",
        json={"name": "ExistingWorkforce", "workers": ["Worker1"]},
    )
    assert response.status_code == 409


def test_create_workforce_worker_not_found(client: TestClient):
    response = client.post(
        "/api/v1/workforces",
        json={"name": "MyWorkforce", "workers": ["NonExistent"]},
    )
    assert response.status_code == 404


def test_create_workforce_manager_not_found(client: TestClient, desk):
    worker = Worker(name="Worker1", model="test-model")
    desk.register_worker(worker)

    response = client.post(
        "/api/v1/workforces",
        json={"name": "MyWorkforce", "workers": ["Worker1"], "mode": "managed", "manager": "NonExistent"},
    )
    assert response.status_code == 404


def test_list_workforces(client: TestClient, desk):
    worker = Worker(name="Worker1", model="test-model")
    desk.register_worker(worker)

    from blackgeorge.workforce import Workforce
    workforce = Workforce(workers=[worker], name="MyWorkforce")
    desk.register_workforce(workforce)

    response = client.get("/api/v1/workforces")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "MyWorkforce"


def test_get_workforce(client: TestClient, desk):
    worker = Worker(name="Worker1", model="test-model")
    desk.register_worker(worker)

    from blackgeorge.workforce import Workforce
    workforce = Workforce(workers=[worker], name="MyWorkforce")
    desk.register_workforce(workforce)

    response = client.get("/api/v1/workforces/MyWorkforce")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "MyWorkforce"


def test_get_workforce_not_found(client: TestClient):
    response = client.get("/api/v1/workforces/NonExistent")
    assert response.status_code == 404


def test_delete_workforce(client: TestClient, desk):
    worker = Worker(name="Worker1", model="test-model")
    desk.register_worker(worker)

    from blackgeorge.workforce import Workforce
    workforce = Workforce(workers=[worker], name="MyWorkforce")
    desk.register_workforce(workforce)

    response = client.delete("/api/v1/workforces/MyWorkforce")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    response = client.get("/api/v1/workforces/MyWorkforce")
    assert response.status_code == 404


def test_delete_workforce_not_found(client: TestClient):
    response = client.delete("/api/v1/workforces/NonExistent")
    assert response.status_code == 404
