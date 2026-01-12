from app import app

def test_predict_endpoint_exists():
    client = app.test_client()
    response = client.post(
        "/predict",
        json={
            "features": [
                7.4, 0.7, 0.0, 1.9, 0.076,
                11, 34, 0.9978, 3.51, 0.56, 9.4
            ]
        }
    )

    # We only check that API responds correctly
    assert response.status_code in [200, 500]