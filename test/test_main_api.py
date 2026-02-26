from fastapi.testclient import TestClient 
from main import app


# create a reusable test client instance for making requests to the app.
client = TestClient(app) 



def test_health_endpoint_returns_running_status():
    """verify the health endpoint reports running status"""

    # call health endpoint
    response = client.get("/health")

    # ensure response status code is 200 (OK).
    assert response.status_code == 200  

    # parse response body as json.
    payload = response.json() 

    # verify response contains expected status message.
    assert "status" in payload  

    # check that the status message matches the expected value.
    assert payload["status"] == "Service is running!"



def test_loan_eligibility_endpoint_with_valid_payload():
    """verify loan_eligibility returns a prediction and probability for valid payload"""

    # construct a valid input payload for loan eligbility prediction.
    payload = { 
        "applicantIncome": 5000.0,
        "coapplicantIncome": 2000.0,
        "loanAmount": 150.0,
        "creditHistory": 1.0,
    }

    # send a POST request to the loan_eligbility endpoint with the payload as JSON.
    response = client.post("/loan_eligibility", json=payload) 

    # check that the response status code is 200 (OK).
    assert response.status_code == 200 

    # parse the response body as JSON to access the prediction results.
    body = response.json() 

    # verify that the response contains both "prediction" and "probability" fields.
    assert "prediction" in body 

    # verify that the probability value is present.
    assert "probability" in body  

    # verify that the model prediction is binary (either 0 or 1).
    assert body["prediction"] in (0.0, 1.0, 0, 1) 

    # verify that the probability is within the range of 0 and 1.
    assert 0.0 <= float(body["probability"]) <= 1.0


def test_loan_eligibility_endpoint_with_invalid_payload_returns_422():
    """verify invalid payload is rejected with 422"""

    #  construct payload without mandatory loan amount: missing required field "loanamount".
    payload = { 
        "applicantIncome": 5000.0,
        "coapplicantIncome": 2000.0,
        "creditHistory": 1.0,
    }

    # send a POST request to the loan_eligbility endpoint with the invalid payload.
    response = client.post("/loan_eligibility", json=payload)  

    # verify that it expects a validation error for bad payload.
    assert response.status_code == 422 

