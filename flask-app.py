from flask import Flask, request, render_template, jsonify
import requests
import numpy as np


app = Flask(__name__)


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
            return jsonify({"error": "Name and Email are required!"}), 400

        # Validate and convert user input to floats.
        try:
            applicant_income = float(user_info.get("applicantIncome"))
            coapplicant_income = float(user_info.get("coapplicantIncome"))
            loan_amount = float(user_info.get("loanAmount"))
        except ValueError:
            return jsonify({"error": "Applicant Income, Coapplicant Income, and Loan Amount must be valid numbers."}), 400

        try:
            # Convert credit history to float and validate
            credit_history = float(user_info.get("creditHistory"))
            if credit_history not in (0, 1):
                return jsonify({"error": "Credit history must be 0 or 1."}), 400
        except ValueError:
            return jsonify({"error": "Invalid credit history input!"}), 400
        
        # Define the user data to send to the API Endpoint.
        user_data = {
            "applicantIncome": applicant_income, 
            "coapplicantIncome": coapplicant_income,
            "loanAmount": loan_amount,
            "creditHistory": credit_history,
        }
    
        # Call the FastAPI Endpoint.
        model_endpoint = "https://loan-eligibility-project-1.onrender.com/loan_eligibility"
        try:
            response = requests.post(model_endpoint, json=user_data)
            response.raise_for_status()  # Raise an error for bad responses.
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Error calling the model endpoint: {str(e)}"}), 500
        # Print the full response content for debugging
        print(response.text)
        # Return the response from the FastAPI endpoint.
        return jsonify(response.json()), response.status_code




if __name__ == '__main__':
    app.run(debug=True)
