"""Smoke test for the live MLP prediction endpoint.

Start the service first with Uvicorn, then run this file with the Approach-2 venv.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen


PAYLOAD = {
    "ClaimAmount": 794.07,
    "ClaimDate": "2024-03-04",
    "PatientAge": 75,
    "PatientGender": "M",
    "ProviderSpecialty": "Orthopedics",
    "PatientIncome": 99260.16,
    "PatientMaritalStatus": "Widowed",
    "PatientEmploymentStatus": "Employed",
    "ClaimType": "Emergency",
    "ClaimSubmissionMethod": "Paper",
    "Cluster": 0,
}


def predict() -> dict[str, object]:
    request = Request(
        "http://127.0.0.1:8000/predict",
        data=json.dumps(PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        assert response.status == 200
        return json.loads(response.read().decode("utf-8"))


first = predict()
second = predict()
assert first == second, "Repeated inference was not deterministic"
assert first["model_name"] == "MLP"
assert first["feature_count"] == 29
assert first["score_is_calibrated"] is False
assert first["prediction"] in {"Fraud", "Legitimate"}
assert 0 <= first["fraud_score"] <= 1
print(json.dumps({"status": "ok", "response": first}, indent=2))
