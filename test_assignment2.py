"""Assignment 2 tests: Flask routes and image preprocessing/prediction (model mocked)."""
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import model as model_module
from model import preprocess_img, predict_result

ERROR_TEXT = b"File cannot be processed."


def make_image_bytes(mode="RGB", size=(50, 30)):
    """Build a small in-memory PNG image and return its raw bytes."""
    buffer = BytesIO()
    Image.new(mode, size).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------- home page

def test_home_page_loads(client):
    """GET / returns the upload page successfully."""
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.content_type


# ------------------------------------------------------------ valid request

def test_valid_upload_calls_model_and_shows_result(upload):
    """A real image is preprocessed, sent to the (mocked) model, and no error is shown."""
    with patch("app.predict_result", return_value=7) as mock_predict:
        response = upload(make_image_bytes(), "digit.png")

    assert response.status_code == 200
    assert ERROR_TEXT not in response.data
    mock_predict.assert_called_once()
    model_input = mock_predict.call_args[0][0]
    assert model_input.shape == (1, 224, 224, 3)


def test_valid_upload_passes_prediction_to_template(upload):
    """The predicted digit is handed to result.html as a string."""
    with patch("app.predict_result", return_value=7), \
            patch("app.render_template", return_value="rendered") as mock_render:
        response = upload(make_image_bytes(), "digit.png")

    assert response.status_code == 200
    mock_render.assert_called_once_with("result.html", predictions="7")


# ------------------------------------------------------------ missing file

def test_missing_file_is_handled(client):
    """A POST with no file shows the error message and never calls the model."""
    with patch("app.predict_result") as mock_predict:
        response = client.post(
            "/prediction", data={}, content_type="multipart/form-data"
        )

    assert response.status_code == 200
    assert ERROR_TEXT in response.data
    mock_predict.assert_not_called()


def test_empty_upload_is_handled(upload):
    """An upload with an empty filename and no content shows the error message."""
    with patch("app.predict_result") as mock_predict:
        response = upload(b"", "")

    assert response.status_code == 200
    assert ERROR_TEXT in response.data
    mock_predict.assert_not_called()


# ----------------------------------------------------- invalid/unexpected input

def test_non_image_file_is_handled(upload):
    """A text file uploaded as an image shows the error message."""
    with patch("app.predict_result") as mock_predict:
        response = upload(b"this is not an image", "notes.txt")

    assert response.status_code == 200
    assert ERROR_TEXT in response.data
    mock_predict.assert_not_called()


def test_grayscale_image_is_rejected(upload):
    """A one-channel image cannot be reshaped to 3 channels, so the error is shown."""
    with patch("app.predict_result") as mock_predict:
        response = upload(make_image_bytes(mode="L"), "gray.png")

    assert response.status_code == 200
    assert ERROR_TEXT in response.data
    mock_predict.assert_not_called()


# ------------------------------------------------- preprocessing and prediction

def test_preprocess_img_output_shape_and_range():
    """preprocess_img returns a (1, 224, 224, 3) array scaled to 0-1."""
    result = preprocess_img(BytesIO(make_image_bytes()))

    assert result.shape == (1, 224, 224, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_img_resizes_large_image():
    """A large image is resized down to the 224x224 model input."""
    result = preprocess_img(BytesIO(make_image_bytes(size=(2000, 1000))))

    assert result.shape == (1, 224, 224, 3)


def test_preprocess_img_rejects_invalid_data():
    """Bytes that are not an image raise an OSError (Pillow's UnidentifiedImageError)."""
    with pytest.raises(OSError):
        preprocess_img(BytesIO(b"junk"))


def test_predict_result_returns_most_likely_class():
    """predict_result returns the index of the highest probability from the model."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.05, 0.10, 0.70, 0.15]])

    with patch.object(model_module, "model", fake_model):
        result = predict_result(np.zeros((1, 224, 224, 3)))

    assert result == 2
    fake_model.predict.assert_called_once()
