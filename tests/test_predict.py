import json
import sys
# sys.path.append("services")
from services import predict


client = predict.app.test_client()

# PORT = os.getenv("PORT", "9696")
# url = f"http://localhost:{PORT}/predict"

def test_predict():
    """Test predict function"""

    payload = {
            "customer_age": 31,
            "dependent_count": 0,
            "gender": "F",
            "education_level": "Post-Graduate",
            "marital_status": "Divorced",
            "income_category": "Less than $40K",
            "card_category": "Blue",
            "months_on_book": 36,
            "total_relationship_count":6,
            "credit_limit": 4871,
            "total_revolving_bal": 0
        }
    resp = client.post(
            "/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
    assert resp.json["is_churned"] == "No"

# test_predict()
