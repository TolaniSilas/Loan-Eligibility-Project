from flask import Flask, request, render_template, jsonify, url_for, redirect
import requests, json


# Initialize the Flask application.
app = Flask(__name__)


@app.route('/', methods=["GET"])
def health():
    """
    Redirects the root URL to the 'get_loan_form' route.
    """
    
    return redirect(url_for("get_loan_form"))


@app.route('/loan_app', methods=["GET", "POST"])
def get_loan_form():
    """
    Handles GET and POST requests for the loan application form.
    For GET requests, renders the loan application form.
    For POST requests, processes form data, validates it, and redirects to the eligibility result.
    """
    
    if request.method == "POST":
        
        # Extract user information from the form submission.
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
            # Convert credit history to float and validate.
            credit_history = float(user_info.get("creditHistory"))
            if credit_history not in (0, 1):
                return jsonify({"error": "Credit history must be 0 or 1."}), 400
        except ValueError:
            return jsonify({"error": "Invalid credit history input!"}), 400
        
        # Define the user data to send to the model API Endpoint.
        user_data = {
            "applicantIncome": applicant_income, 
            "coapplicantIncome": coapplicant_income,
            "loanAmount": loan_amount,
            "creditHistory": credit_history,
        }
    
        # Call the model FastAPI Endpoint.
        model_endpoint = "https://loan-eligibility-project-1.onrender.com/loan_eligibility"
        
        try:
            response = requests.post(model_endpoint, json=user_data)
            
            # Raise an error for unsuccessful responses.
            response.raise_for_status()  
            
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Error calling the model endpoint: {str(e)}"}), 500
    
        response = response.json()
        # Add the user's name to the response data.
        response["name"] = name 
        
        # Extract prediction result and the username from the response.
        prediction = response["prediction"]
        username = response["name"]
           
        # Redirect to the eligibility result page with the prediction and username.
        return redirect(url_for("eligibility", prediction=prediction, username=username))
        
    # Render the loan application form.
    return render_template('index.html')


@app.route('/eligibility', methods=["GET"])
def eligibility():
    """
    Renders the eligibility result page based on the prediction value.
    Retrieves prediction and username from query parameters and determines the eligibility status.
    """
    
    # Retrieve the prediction and username from query parameters.
    prediction = float(request.args.get('prediction'))
    username = request.args.get('username')
    
    # Determine eligibility status based on the prediction value.
    if prediction == 1:
        status = "Eligible"
    else:
        status = "Non-eligible"
    
    # Render the eligibility.html template and pass the data to it.
    return render_template('eligibility.html', username=username.title(), status=status)

    

if __name__ == '__main__':
    
    # Run the Flask application with debugging disabled (Production Environment)
    app.run()