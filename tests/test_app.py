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


def test_feedback_missing_fields(client):
    """Test /feedback with missing fields."""
    response = client.post("/feedback", json={})
    assert response.status_code == 400


def test_feedback_invalid_label(client):
    """Test /feedback with invalid actual_label."""
    response = client.post(
        "/feedback",
        json={"prediction_id": "00000000-0000-0000-0000-000000000001", "actual_label": "Bird", "was_correct": False},
    )
    assert response.status_code == 400


def test_feedback_success(client, monkeypatch):
    """Test /feedback with valid data."""
    monkeypatch.setattr("app.store_feedback", lambda *a, **k: True)
    response = client.post(
        "/feedback",
        json={"prediction_id": "00000000-0000-0000-0000-000000000001", "actual_label": "Cat", "was_correct": True},
    )
    assert response.status_code == 200
    assert response.get_json().get("status") == "ok"


# --- Feedback CRUD tests ---

def test_list_feedbacks(client, monkeypatch):
    """Test GET /feedbacks lists feedbacks."""
    monkeypatch.setattr("app.get_all_feedbacks", lambda **kw: [{"id": 1, "prediction_id": "uuid-1", "actual_label": "Cat", "was_correct": True}])
    response = client.get("/feedbacks")
    assert response.status_code == 200
    data = response.get_json()
    assert "feedbacks" in data
    assert len(data["feedbacks"]) == 1
    assert data["feedbacks"][0]["actual_label"] == "Cat"


def test_get_feedback_not_found(client, monkeypatch):
    """Test GET /feedbacks/<id> returns 404 when not found."""
    monkeypatch.setattr("app.get_feedback", lambda x: None)
    response = client.get("/feedbacks/999")
    assert response.status_code == 404


def test_get_feedback_success(client, monkeypatch):
    """Test GET /feedbacks/<id> returns feedback when found."""
    monkeypatch.setattr("app.get_feedback", lambda x: {"id": 1, "prediction_id": "uuid-1", "actual_label": "Dog", "was_correct": False})
    response = client.get("/feedbacks/1")
    assert response.status_code == 200
    assert response.get_json()["actual_label"] == "Dog"


def test_update_feedback_success(client, monkeypatch):
    """Test PUT /feedbacks/<id> updates feedback."""
    monkeypatch.setattr("app.update_feedback", lambda id, **kw: True)
    response = client.put("/feedbacks/1", json={"was_correct": True})
    assert response.status_code == 200
    assert response.get_json().get("status") == "ok"


def test_update_feedback_not_found(client, monkeypatch):
    """Test PUT /feedbacks/<id> returns 404 when not found."""
    monkeypatch.setattr("app.update_feedback", lambda id, **kw: False)
    response = client.put("/feedbacks/999", json={"was_correct": True})
    assert response.status_code == 404


def test_delete_feedback_success(client, monkeypatch):
    """Test DELETE /feedbacks/<id> deletes feedback."""
    monkeypatch.setattr("app.delete_feedback", lambda x: True)
    response = client.delete("/feedbacks/1")
    assert response.status_code == 200
    assert response.get_json().get("status") == "ok"


def test_delete_feedback_not_found(client, monkeypatch):
    """Test DELETE /feedbacks/<id> returns 404 when not found."""
    monkeypatch.setattr("app.delete_feedback", lambda x: False)
    response = client.delete("/feedbacks/999")
    assert response.status_code == 404