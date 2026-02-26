# **Loan Eligibility Project**

**Problem Statement:** **Dream Housing Finance** company deals in all home loans. They have a presence across all urban, semi-urban, and rural areas. Customer-first applies for a home loan after that company validates the customer eligibility for a loan.


The company wants to automate the loan eligibility process (real-time) based on customer detail provided while filling the online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others. To automate this process, they have given a problem to identify the **customer's segments**, those who are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set. The **loan dataset** can be found [here](loan-dataset)

## Project Overview

This project implements a complete, production-ready loan eligibility system that:

- **Trains and serves a machine learning model** that predicts whether a customer is eligible for a home loan.
- **Exposes a REST API** (via FastAPI) for programmatic access to the model.
- **Provides a user-friendly web interface** (via Flask and HTML templates) where customers can submit loan applications and view eligibility results.

The solution is designed to be:

- **Modular**: Model logic, API, and web UI are clearly separated.
- **Deployable**: The model API can be deployed as a standalone service (e.g. on Render, as currently configured).
- **Extensible**: Additional features or input fields can be incorporated without major architectural changes.

## High-Level Architecture

The project consists of three main layers:

- **Model & Data Layer (`model.py`)**
  - Loads and preprocesses the loan dataset from `loan-dataset/cleaned-loan-train.csv`.
  - Trains and uses a scikit-learn model for binary loan eligibility prediction.
  - Handles feature scaling, encoding, and inference via the `LoanEligibilityModel` class.
  - Persists the trained model to `loan-model-file/loan_model.pkl` (which is then loaded at runtime).

- **Model API Layer (`main.py`)**
  - Implements a **FastAPI** application that wraps `LoanEligibilityModel` behind REST endpoints.
  - Provides:
    - `GET /health` – health check endpoint.
    - `POST /loan_eligibility` – inference endpoint returning prediction and probability.
  - Runs via Uvicorn (ASGI server), either directly (`python main.py`) or through a process manager such as Gunicorn.

- **Web Application Layer (`flask_app.py` + templates)**
  - Implements a **Flask** web app that:
    - Displays a loan application form (`templates/index.html`).
    - Collects user details and submits them to the model API.
    - Renders an eligibility result page (`templates/eligibility.html`) with a user-friendly status.
  - Uses the `requests` library to call the model API (currently configured to use a deployed endpoint on Render).

### End-to-End Flow

1. A user opens the web application in a browser (Flask app).
2. The root route `/` redirects to `/loan_app`, which displays the loan application form from `index.html`.
3. The user fills in personal and financial information and submits the form (`POST /loan_app`).
4. The Flask app validates the input, constructs a JSON payload, and sends it to the FastAPI model endpoint `/loan_eligibility`.
5. The model API:
   - Uses `LoanEligibilityModel` to preprocess the input.
   - Runs the ML model to generate a prediction (`0` or `1`) and a probability.
   - Returns a JSON response with `prediction` and `probability`.
6. The Flask app reads the prediction, decorates it with the user’s name, and redirects to `/eligibility`.
7. The `/eligibility` route determines a human-readable status (`Eligible` or `Non-eligible`) and renders `eligibility.html`, which displays the result with color-coded feedback.

## Project Structure

At a high level, the repository is organized as follows:

```text
Loan-Eligibility-Project/
├─ README.md                  # project documentation (this file)
├─ requirements.txt           # python dependencies for training and serving
├─ main.py                    # fastAPI application exposing the model as a REST API
├─ flask_app.py               # flask web application (html frontend) that calls the model API
├─ model.py                   # data loading, preprocessing, and inference logic (LoanEligibilityModel)
├─ templates/
│  ├─ index.html              # loan application form and landing page
│  └─ eligibility.html        # result page showing whether the user is eligible or not
├─ loan-dataset/
│  └─ cleaned-loan-train.csv  # cleaned training dataset used for model development
└─ loan-model-file/
   └─ loan_model.pkl          # serialized scikit-learn model used by LoanEligibilityModel (stored as pickle file)
```


## Technology Stack

- **Language**: Python
- **Modeling & Data**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `joblib`
- **API & Web Frameworks**:
  - **FastAPI** (model backend logic API)
  - **Flask** (web frontend application)
- **Serving & Runtime**:
  - `uvicorn` (ASGI server for FastAPI)
  - `gunicorn` (optional production process manager)
- **Visualization & EDA (this was utilized in notebooks/scripts)**:
  - `matplotlib`
  - `seaborn`

All dependencies are pinned in `requirements.txt` for reproducibility.

## Setup and Installation

### 1. Prerequisites

- **Python**: 3.10+ (recommended)
- **Git**: To clone the repository
- A virtual environment tool such as `venv` or `conda` (strongly recommended).

### 2. Clone the Repository

```bash
git clone https://github.com/TolaniSilas/Loan-Eligibility-Project.git

cd Loan-Eligibility-Project
```

### 3. Create and Activate a Virtual Environment

```bash
python -m venv .venv

source .venv/bin/activate        # On macOS/Linux

# .venv\Scripts\activate         # On Windows (PowerShell)
```

### 4. Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### 5. Ensure Data and Model Files Are Present

- The training dataset is expected at:
  - `loan-dataset/cleaned-loan-train.csv`
- The trained model file is expected at:
  - `loan-model-file/loan_model.pkl`

If these are missing, you will need to:

- Place the dataset in the correct directory, or
- Re-train and serialize the model (see **Model Training & Inference** below). Refer to the loan eligibility notebook in the repository's root for guidance on the machine learning workflows considered - for project replication or further improvement in model performance.


## Running the Model API (FastAPI)

The model API is defined in `main.py`. It wraps the `LoanEligibilityModel` class and exposes HTTP endpoints.

### Start the API Locally (Development)

From the project root:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

This will:

- Start Uvicorn on `http://127.0.0.1:8000`
- Serve the FastAPI application defined in `main.py`

### Available API Endpoints

- **`GET /health`**
  - **Description**: Health check endpoint to verify that the service is running.
  - **Response**:
    ```json
    {
      "status": "Service is running!"
    }
    ```

- **`POST /loan_eligibility`**
  - **Description**: Predicts loan eligibility given applicant and loan details.
  - **Request body (JSON)**:
    ```json
    {
      "applicantIncome": 5000.0,
      "coapplicantIncome": 2000.0,
      "loanAmount": 150.0,
      "creditHistory": 1.0
    }
    ```
  - **Response body (JSON)**:
    ```json
    {
      "prediction": 1.0,
      "probability": 0.92
    }
    ```
    - `prediction`: `1.0` for eligible, `0.0` for non-eligible.
    - `probability`: Confidence associated with the predicted class.

### Example cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/loan_eligibility" \
  -H "Content-Type: application/json" \
  -d '{
    "applicantIncome": 5000.0,
    "coapplicantIncome": 2000.0,
    "loanAmount": 150.0,
    "creditHistory": 1.0
  }'
```

## Running the Frontend Web Application (Flask)

The web application is defined in `flask_app.py` and uses the FastAPI model service behind the scenes.

### Start the Flask App Locally

From the project root:

```bash
python flask_app.py
```

By default, this will:

- Start the Flask app on `http://127.0.0.1:5000`
- Redirect `/` to `/loan_app`, which renders `templates/index.html`

### Web Application Flow

1. Navigate to `http://127.0.0.1:5000/`.
2. Click the **“Check for Loan Eligibility!”** button to reveal the form.
3. Fill in:
   - Personal info: name, email, etc.
   - Financial info: applicant income, coapplicant income, loan amount.
   - Credit history (0 or 1).
4. Submit the form.
5. The app sends a JSON payload to the configured model API endpoint.
6. Based on the model’s `prediction`, you are redirected to `/eligibility`, which:
   - Displays your name.
   - Shows your eligibility **status**:
     - `"Eligible"` - styled in green.
     - `"Non-eligible"` - styled in red.

> **Note:** In the current configuration, `flask_app.py` calls a deployed FastAPI endpoint hosted on Render (`https://loan-eligibility-project-1.onrender.com/loan_eligibility`). For purely local operation, you can point this URL to your locally running FastAPI server instead.

## Model Training & Inference Details

### Data Loading and Preprocessing (`model.py`)

- Loads cleaned training data from `loan-dataset/cleaned-loan-train.csv` into a pandas DataFrame.
- Splits the data into:
  - **Features**: All columns except `Loan_Status`.
  - **Labels**: The `Loan_Status` column (binary target).
- Encodes `Credit_History` as one-hot features:
  - `Credit_History_0`
  - `Credit_History_1`
- Scales numerical features using `MinMaxScaler`:
  - `ApplicantIncome`
  - `CoapplicantIncome`
  - `LoanAmount`

### `LoanEligibilityModel` Class

Key responsibilities:

- **Model loading**
  - Loads a pre-trained scikit-learn model from `loan-model-file/loan_model.pkl` using `joblib.load`.
  - Prints a confirmation message when the model is successfully loaded.

- **`user_info_processing(...)`**
  - Accepts raw numeric inputs:
    - `applicant_income`
    - `coapplicant_income`
    - `loan_amount`
    - `credit_history` (0.0 or 1.0)
  - Creates a one-row DataFrame representing the user.
  - Manually encodes `credit_history` into `Credit_History_0` and `Credit_History_1`.
  - Scales income- and loan-related features using the pre-fitted `MinMaxScaler`.
  - Returns a DataFrame with the exact feature schema expected by the trained model.

- **`generate_eligibility(...)`**
  - Calls `user_info_processing` to transform inputs.
  - Uses the loaded model to:
    - Predict eligibility (`0` or `1`).
    - Compute the probability of the positive class.
  - Returns:
    - `(prediction, probability_of_predicted_class)`
    - If the model predicts **eligible (1)**:
      - Returns `(1, P(eligible))`.
    - If the model predicts **non-eligible (0)**:
      - Returns `(0, P(non-eligible))`, computed as `1 - P(eligible)`.

## Configuration

Key configuration points:

- **Model path**
  - Defined in `LoanEligibilityModel.model_path`:
    - `loan-model-file/loan_model.pkl`
  - Ensure this file is present and compatible with the preprocessing pipeline.

- **Model API endpoint used by Flask**
  - Configured in `flask_app.py` as:
    - `https://loan-eligibility-project-1.onrender.com/loan_eligibility`
  - To use a local API instead, update this URL to:
    - `http://127.0.0.1:8000/loan_eligibility` (or another host/port you deploy to).

## Testing and Validation

While there is no dedicated test suite included yet, recommended practices include:

- **Unit tests**
  - Test `LoanEligibilityModel.user_info_processing` for a variety of inputs.
  - Test `LoanEligibilityModel.generate_eligibility` with synthetic/sampled data.

- **API tests**
  - Test `GET /health` to ensure service availability.
  - Test `POST /loan_eligibility` with valid and invalid payloads.

- **End-to-end tests**
  - Automate browser-level tests (e.g. with Playwright or Selenium) to:
    - Fill out the form.
    - Submit.
    - Assert that `/eligibility` renders the correct status.


## Contributing

Contributions to improve the model, UI, or infrastructure are welcome. Suggested contribution areas:

- Enhancing feature engineering and model performance.
- Improving form validation and user experience.
- Adding automated tests and CI/CD pipelines.
- Extending the API to support additional endpoints (e.g. model explanation, batch scoring).

Before opening a pull request:

- Run the application locally (both FastAPI and Flask).
- Ensure that style and type checks (if any) pass.
- Update this documentation if your changes affect usage or behavior.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for the full text and terms.
