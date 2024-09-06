import numpy as np
import pandas as pd 
import joblib
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


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


# df = pd.DataFrame({"ApplicantIncome": , "CoapplicantIncome": , "LoanAmount":, "Credit_History":})

class LoanModel():
    
    # Define the model path.
    model_path = "loan-model-file/loan_model.pkl"

    # Load the trained model.
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully.")
    
    
    
    def user_info_processing(self, applicant_income, coapplicant_income, loan_amount, credit_history):
        
        df = pd.DataFrame({"ApplicantIncome": [applicant_income], "CoapplicantIncome": [coapplicant_income], "LoanAmount": [loan_amount], "Credit_History": [credit_history]})

        return df
        
         
        
        
                             
    
    
    
    # def generate_eligibility(self, ):
        
        
        
        
    #     predictions = self.loaded_model.predict(X_test_scaled)

    #     y_proba = self.loaded_model.predict_proba(X_test_scaled)
        
        
        
    
    
    
    # def  





if __name__ == "__main__":
    LoanModel()
