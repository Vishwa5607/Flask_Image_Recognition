"""Shared pytest fixtures for the Flask app tests."""
from io import BytesIO

import pytest

from app import app


@pytest.fixture(name="client")
def client_fixture():
    """Provide a Flask test client."""
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture(name="upload")
def upload_fixture(client):
    """Provide a helper that posts a file to the /prediction route."""
    def _upload(content, filename):
        data = {"file": (BytesIO(content), filename)}
        return client.post(
            "/prediction",
            data=data,
            content_type="multipart/form-data"
        )
    return _upload
