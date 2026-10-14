"""Integration test for a successful image upload."""


def test_successful_prediction(upload):
    """Test the successful image upload and prediction."""
    # Upload a mock image file to the prediction endpoint
    response = upload(b"fake_image_data", "test.jpg")

    assert response.status_code == 200
    assert b"Prediction" in response.data
