import io
import pytest
from PIL import Image
import numpy as np

from app import app


@pytest.fixture
def client():
    """
    Create Flask test client
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    """
    Test index route
    """
    response = client.get("/")
    assert response.status_code == 200


def test_health_check(client):
    """
    Test health endpoint
    """
    response = client.get("/health")

    assert response.status_code in [200, 503]

    data = response.get_json()
    assert "status" in data


def test_predict_no_file(client):
    """
    Test /predict without file
    """
    response = client.post("/predict")
    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data


def test_predict_empty_filename(client):
    """
    Test /predict with empty filename
    """
    data = {
        "file": (io.BytesIO(b"fake image data"), "")
    }

    response = client.post(
        "/predict",
        data=data,
        content_type="multipart/form-data"
    )

    assert response.status_code == 400


def test_predict_valid_image(client, monkeypatch):
    """
    Test /predict with valid image
    We mock model.predict so test doesn't depend on real model
    """

    # Mock model.predict
    class MockModel:
        def predict(self, x):
            return [[0.8]]  # pretend model predicts Dog with 80%

    monkeypatch.setattr("app.model", MockModel())

    # Create dummy image
    img = Image.fromarray(
        np.uint8(np.random.rand(224, 224, 3) * 255)
    )

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    data = {
        "file": (img_bytes, "test.jpg")
    }

    response = client.post(
        "/predict",
        data=data,
        content_type="multipart/form-data"
    )

    assert response.status_code == 200

    json_data = response.get_json()

    assert "prediction" in json_data
    assert "confidence" in json_data
    assert json_data["prediction"] in ["Cat", "Dog"]