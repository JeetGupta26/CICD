from app import app
def test_predict():
    client = app.test_client()
    response = client.post(
        "/predict",
        json={"features":[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]}
    )
    assert response.status_code == 200