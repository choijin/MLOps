import requests


def test_predict():
    payload = {
        "records": [
            {
                "VendorID": 2.0,
                "lpep_pickup_datetime": "2025-01-15T08:30:00",
                "lpep_dropoff_datetime": "2025-01-15T08:48:00",
                "passenger_count": 1.0,
                "trip_distance": 2.7,
                "RatecodeID": 1.0,
                "store_and_fwd_flag": "N",
                "PULocationID": 74,
                "DOLocationID": 41,
                "trip_type": 1.0,
            }
        ]
    }

    url = "http://127.0.0.1:9696/predict"

    response = requests.post(url, json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] > 0
