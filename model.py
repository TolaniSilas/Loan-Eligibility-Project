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


class LoanEligibilityModel():
    
    # Define the model path.
    model_path = "loan-model-file/loan_model.pkl"    

    # Load the trained model.
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    
    
    def user_info_processing(self, applicant_income, coapplicant_income, loan_amount, credit_history):
        
        user_df = pd.DataFrame({"ApplicantIncome": [applicant_income], "CoapplicantIncome": [coapplicant_income], "LoanAmount": [loan_amount], "Credit_History": [credit_history]})

        if user_df.loc[0, "Credit_History"]== 0.0:
            user_df["Credit_History_0"] = 1.0
            user_df["Credit_History_1"] = 0.0
            
        elif user_df.loc[0, "Credit_History"] == 1.0:
            user_df["Credit_History_0"] = 0.0
            user_df["Credit_History_1"] = 1.0
            
        # Drop the Credit_History column.    
        user_df.drop(["Credit_History"], axis=1, inplace=True)
        
        # Fit and transform the numerical columns.
        user_df_scaled_numerical = scaler.transform(user_df[numerical_columns])

        # Convert the scaled data back to a DataFrame.
        user_df_scaled_numerical = pd.DataFrame(user_df_scaled_numerical, columns=numerical_columns)

        # Combine the scaled numerical columns with the original categorical columns
        user_df_scaled = pd.concat([user_df_scaled_numerical, user_df[categorical_columns]], axis=1)

        return user_df_scaled
    
    
    def generate_eligibility(self, applicant_income, coapplicant_income, loan_amount, credit_history):
        
        # Get the user preprocessed information.
        user_preprocessed_info = self.user_info_processing(applicant_income, coapplicant_income, loan_amount, credit_history)
        
        # user_preprocessed_info = user_preprocessed_info.to_numpy()
        
        prediction = self.loaded_model.predict(user_preprocessed_info)[0]

        prediction_probability = self.loaded_model.predict_proba(user_preprocessed_info)[0]
        
        return prediction_probability
        
         
        
# loan_model = LoanEligibilityModel()
# print(loan_model.generate_eligibility(3000, 0, 66, 1))        
                             
                             
                             
if __name__ == "__main__":
    LoanEligibilityModel()
