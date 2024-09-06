from flask import Flask, request, render_template, url_for, jsonify
import requests
import numpy as np
from model import LoanModel



app = Flask(__name__)

print(np.__version__)
@app.route('/health', methods=["GET"])
def health():
    return "<p>Hello! EndPoint is active!</p>"


@app.route('/loan_app', methods=["GET"])
def show_loan_form():

    return render_template('index.html')


@app.route('/submit_loan', methods=["POST"])
def submit_loan():
    
    if request.method == "POST":
        user_info = request.form
        name = user_info.get("name")
        email = user_info.get("email")

        # Validate name and email.
        if not name or not email:
            return jsonify({"Error": "Name or Email can't be empty!"})

        # Validate and convert user input to floats.
        try:
            applicant_income = float(user_info.get("applicantIncome"))
            coapplicant_income = float(user_info.get("coapplicantIncome"))
            loan_amount = float(user_info.get("loanAmount"))
            
        except ValueError:
            return jsonify({"error": "Applicant Income, Coapplicant Income, and Loan Amount must be integers."})

        try:
            # Convert credit history to integer
            credit_history = float(user_info.get("creditHistory"))

            if credit_history in (0, 1):
                credit_history = credit_history
                
            else:
                return jsonify({"error": "Credit history must be 0 or 1."})

        except ValueError:
            return jsonify({"error": "Invalid credit history input!"})
        
    
        # Define the user data that willl be sent to the API Endpoint.
        user_data = {
            "applicant_income": applicant_income, 
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "credit_history": credit_history,
            # "credit_history_1": credit_history_1
            }
        loan_model = LoanModel()
        
        df = loan_model.user_info_processing(user_data["applicant_income"], user_data["coapplicant_income"], user_data["loan_amount"], user_data["credit_history"])
    
        # # Call the FastAPI Endpoint 
        # model_endpoint = "https//"
        # response = requests.post(model_endpoint, json=user_data)

     
    
    return jsonify({"message": f"Receievd name: {name}, email: {email}"})




if __name__ == '__main__':
    app.run(debug=True)
