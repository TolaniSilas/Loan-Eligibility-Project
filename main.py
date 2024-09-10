from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from model import LoanEligibilityModel

# Instantiate the FastAPI application.
app = FastAPI()

# Define a GET endpoint for the health check.
@app.get("/health")
def health():
    """
    Health check endpoint to confirm the service is running.

    Returns
    -------
    dict
        A dictionary with a single key "Status" and value "Get Started!".
    """
    
    return {"status": "Service is running"}



# Define the input data model for loan eligibility requests.
class LoanEligibilityRequest(BaseModel):
    """
    Data model for handling loan eligibility request inputs.

    Attributes
    ----------
    applicantIncome : float
        The income of the primary loan applicant.
    coapplicantIncome : float
        The income of the coapplicant (if any).
    loanAmount : float
        The total amount of loan applied for.
    creditHistory : float
        The credit history score of the applicant, typically 0.0 or 1.0.

    Notes
    -----
    This model ensures that all necessary information for processing a loan
    eligibility request is provided in the correct format. The attributes
    should align with the expected input fields of the loan eligibility
    prediction system.
    """
    
    applicantIncome: float
    coapplicantIncome: float
    loanAmount: float
    creditHistory: float
    
    
    
# Define a POST endpoint for predicting loan eligibility based on user information.
@app.post("/loan_eligibility")
async def loan_eligibility(request: LoanEligibilityRequest):
    """
    Endpoint to predict loan eligibility based on user-provided information.

    Parameters
    ----------
    request : LoanEligibilityRequest
        The request body containing the applicant's income, coapplicant's income,
        loan amount, and credit history.

    Returns
    -------
    dict
        A dictionary containing:
        - `prediction`: A float value indicating loan eligibility (1.0 for eligible, 0.0 for non-eligible).
        - `probability`: A float value representing the probability of the loan eligibility prediction.

    Raises
    ------
    HTTPException
        If there is an error during the prediction process, a 500 Internal Server Error is raised with the error detail.
    """
    
    try:
        # Extract user information from the request body.
        applicant_income = request.applicantIncome
        coapplicant_income = request.coapplicantIncome
        loan_amount = request.loanAmount
        credit_history = request.creditHistory
        
        # Initialize the loan eligibility model.
        loan_model = LoanEligibilityModel()
        
        # Generate loan eligibility prediction and probability.
        prediction, prediction_probability = loan_model.generate_eligibility(
            applicant_income=applicant_income,
            coapplicant_income=coapplicant_income,
            loan_amount=loan_amount,
            credit_history=credit_history
        )
        
        # Prepare the response in a serializable format.
        response = {
            "prediction": float(prediction),
            "probability": float(prediction_probability)
        }
        
        # Return the result as a JSON response.
        return response
        
    except Exception as e:
        # Raise an HTTPException with status code 500 if an error occurs.
        raise HTTPException(status_code=500, detail=str(e))

                                               
        

if __name__ == "__main__":
    """
    Steps:
    1. Import the Uvicorn ASGI server.
    2. Print a message to indicate that the API endpoint is starting.
    3. Run the Uvicorn server with the specified host, port, and reload settings.
    """
    
    import uvicorn
    
    # Print a message to the console indicating the API server is starting.
    print("Starting Model API Endpoint!")
    
    # Start the Uvicorn ASGI server to serve the FastAPI application.
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    
    