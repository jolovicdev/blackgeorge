import pytest
from fastapi.testclient import TestClient

from blackgeorge.api.dependencies import get_desk
from blackgeorge.api.main import create_app
from blackgeorge.desk import Desk
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.worker import Worker


@pytest.fixture
def desk():
    return Desk(
        model="test-model",
        run_store=InMemoryRunStore(),
    )


@pytest.fixture
def app(desk):
    app = create_app()

    async def override_get_desk():
        yield desk

    app.dependency_overrides[get_desk] = override_get_desk
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def sample_worker():
    return Worker(name="TestWorker", model="test-model")


@pytest.fixture
def registered_worker(desk, sample_worker):
    desk.register_worker(sample_worker)
    return sample_worker
