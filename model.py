import numpy as np
import pandas as pd 
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


# Read the csv file into a DataFrame.
loan_df = pd.read_csv("loan-dataset/cleaned-loan-train.csv")

# Get the features and corresponding labels from the dataset.
features = loan_df.drop("Loan_Status", axis=1)   
labels = loan_df[["Loan_Status"]]

# Define the columns to be encoded.
columns = ["Credit_History"]

features = pd.get_dummies(features, columns=columns, prefix=columns, dtype=float)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42, test_size=0.2, stratify=labels, shuffle=True)

X_train = X_train[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History_0", "Credit_History_1"]]

# Separate numerical columns from categorical columns
numerical_columns = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
categorical_columns = ["Credit_History_0", "Credit_History_1"]

scaler = MinMaxScaler()

# Fit and transform the numerical columns
X_train_scaled_numerical = scaler.fit_transform(X_train[numerical_columns])

# Convert the scaled data back to a DataFrame
X_train_scaled_numerical = pd.DataFrame(X_train_scaled_numerical, columns=numerical_columns, index=X_train.index)

# Combine the scaled numerical columns with the original categorical columns
X_train_scaled = pd.concat([X_train_scaled_numerical, X_train[categorical_columns]], axis=1)


class LoanEligibilityModel():
    """
    A class to handle the loan eligibility prediction using a pre-trained model.

    Attributes
    ----------
    model_path : str
        The file path where the pre-trained model is stored.
    loaded_model : sklearn.base.BaseEstimator
        The pre-trained model loaded from the file path.

    Methods
    -------
    generate_eligibility(applicant_income, coapplicant_income, loan_amount, credit_history):
        Processes the input data and predicts loan eligibility along with the prediction probability.
    user_info_processing(applicant_income, coapplicant_income, loan_amount, credit_history):
        Transforms and scales user input data for model prediction.
    """
    
    # Define the model path.
    model_path = "loan-model-file/loan_model.pkl"    

    # Load the trained model.
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    
    
    def user_info_processing(self, applicant_income, coapplicant_income, loan_amount, credit_history):
        """
        Processes and transforms user information for model prediction.

        Parameters
        ----------
        applicant_income : float
            The income of the primary applicant.
        coapplicant_income : float
            The income of the coapplicant.
        loan_amount : float
            The amount of the loan applied for.
        credit_history : float
            The credit history score of the applicant, which can be 0.0 or 1.0.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the processed and scaled numerical features along with the
            transformed categorical features, suitable for model prediction.

        Notes
        -----
        - The `Credit_History` column is one-hot encoded into two separate columns:
        `Credit_History_0` and `Credit_History_1`.
        - The numerical columns are scaled using a pre-fitted scaler.
        - The categorical columns are combined with the scaled numerical columns to form the final
        DataFrame for model input.
        """
        
        # Create a DataFrame with the user input data.
        user_df = pd.DataFrame({
            "ApplicantIncome": [applicant_income], 
            "CoapplicantIncome": [coapplicant_income], 
            "LoanAmount": [loan_amount], 
            "Credit_History": [credit_history]
        })

        # One-hot encode the Credit_History column.
        if user_df.loc[0, "Credit_History"] == 0.0:
            user_df["Credit_History_0"] = 1.0
            user_df["Credit_History_1"] = 0.0
            
        elif user_df.loc[0, "Credit_History"] == 1.0:
            user_df["Credit_History_0"] = 0.0
            user_df["Credit_History_1"] = 1.0
        
        # Drop the original Credit_History column.
        user_df.drop(["Credit_History"], axis=1, inplace=True)
        
        # Scale the numerical features using the pre-fitted scaler.
        user_df_scaled_numerical = scaler.transform(user_df[numerical_columns])

        # Convert the scaled numerical data back to a DataFrame.
        user_df_scaled_numerical = pd.DataFrame(user_df_scaled_numerical, columns=numerical_columns)

        # Combine the scaled numerical features with the transformed categorical features.
        user_df_scaled = pd.concat([user_df_scaled_numerical, user_df[categorical_columns]], axis=1)
        
        # Return the processed DataFrame.
        return user_df_scaled

    
    def generate_eligibility(self, applicant_income, coapplicant_income, loan_amount, credit_history):
        """
        Predicts the eligibility of a loan applicant based on their financial information.

        Parameters
        ----------
        applicant_income : float
            The income of the primary applicant.
        coapplicant_income : float
            The income of the coapplicant.
        loan_amount : float
            The amount of the loan applied for.
        credit_history : float
            The credit history score of the applicant.

        Returns
        -------
        tuple
            A tuple where the first element is the prediction (0 or 1) indicating 
            whether the applicant is non-eligible (0) or eligible (1), and the second 
            element is the probability associated with the prediction. If the prediction 
            is 1, the probability is the likelihood of being eligible. If the prediction 
            is 0, the probability is the likelihood of being non-eligible, calculated 
            as `1 - prediction_probability`.

        Notes
        -----
        The `prediction_probability` represents the probability of the applicant being 
        eligible (positive class). For a prediction of 0 (non-eligible), the returned 
        probability is adjusted to reflect the likelihood of non-eligibility.
        """
        
        # Preprocess user information for prediction.
        user_preprocessed_info = self.user_info_processing(applicant_income, coapplicant_income, loan_amount, credit_history)
        
        # Predict the eligibility of the customer and its corresponding probability. 
        prediction = self.loaded_model.predict(user_preprocessed_info)[0]
        prediction_probability = self.loaded_model.predict_proba(user_preprocessed_info)[0][1]
        
        if prediction == 1:
            # If the prediction is 1 (eligible), return the prediction and the probability of being eligible.
            return prediction, prediction_probability
        
        elif prediction == 0:
            # If the prediction is 0 (non-eligible), return the prediction and the probability of being non-eligible.
            return prediction, 1 - prediction_probability
        
   
                                                          
if __name__ == "__main__":
    LoanEligibilityModel()
