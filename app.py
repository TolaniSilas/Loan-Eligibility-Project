from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from model import LoanEligibilityModel

# Instantiate the FastAPI Class.
app = FastAPI()


# Define the input Data Model.
class UserData(BaseModel):
    applicantIncome: float
    coapplicantIncome: float
    loanAmount: float
    creditHistory: float


@app.get("/health")
def health():
    
    return {"Status": "Get Started!"}

        
@app.post("/loan_eligibility")
def loan_eligibility(user_data: UserData):
    
    
    # Get the user information.
    applicantincome = user_data.applicantIncome
    coapplicantincome = user_data.coapplicantIncome
    loanamount = user_data.loanAmount
    credithistory = user_data.creditHistory
    
    loan_model = LoanEligibilityModel()
    
    #prediction, prediction_probability = loan_model.generate_eligibility(applicant_income=applicantincome, coapplicant_income=coapplicantincome, loan_amount=loanamount, credit_history=credithistory)
    
    return user_data
                                                  
        

if __name__ == "__main__":
    
    import uvicorn
    print("Starting Model Api Endpoint!")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)  