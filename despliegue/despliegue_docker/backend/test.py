from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

def test_predict_one():
    # Test a simple prediction case
    test_file = "_3_test_file.csv"
    with open(test_file, "w") as f:  # Create a dummy file
        f.write("dummy,data")

    try:
        with open(test_file, "rb") as f:
            response = client.post(
                "/predict_one",
                files={"file": (test_file, f, "text/csv")}
            )
        
        assert response.status_code == 200
        assert response.json() == {
            "filename": test_file,
            "prediction": "HIGH GRADE (3)"
        }

    finally:
        if os.path.exists(test_file):
            os.path.exists(test_file)

def test_retrieve_last_predictions():
    # First make a prediction to create some log data
    test_file = "_2_test_file.csv"
    with open(test_file, "w") as f:
        f.write("dummy,data")

    try:
        # Make a prediction to populate the log
        with open(test_file, "rb") as f:
            client.post("/predict_one", files={"file": (test_file, f, "text/csv")})
        
        # Test retrieving predictions
        response = client.get("/retreive_last_predictions")
        assert response.status_code == 200
        predictions = response.json()
        assert len(predictions) > 0
        assert predictions[0]["filename"] == test_file
        assert predictions[0]["tumor_grade"] == "LOW GRADE (2)"
    finally:
        if os.path.exists(test_file):
            os.path.exists(test_file)
        if os.path.exists("predictions_log.json"):
            os.path.exists("predictions_log.json")