import requests

url = "http://localhost:9696/predict"

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
resp = requests.post(
            url,
            # headers={"Content-Type": "application/json"},
            json=payload
        ).json()

assert resp['is_churned'] in ['No', 'Yes']