from model import LoanEligibilityModel 



def test_user_info_processing_produces_expected_schema_and_ranges():
    """
    basic invariant test for loaneligibilitymodel.user_info_processing
    it ensures the output has the right shape, columns, and reasonable value ranges.
    """

    # instantiate the model class.
    model = LoanEligibilityModel() 

    # sample applicant income.
    applicant_income = 5000.0 

    # sample coapplicant income.
    coapplicant_income = 2000.0 

    # sample loan amount.
    loan_amount = 150.0 

    # good credit history flag.
    credit_history = 1.0 

    # run preprocessing on sample inputs
    processed = model.user_info_processing( 
        applicant_income=applicant_income,
        coapplicant_income=coapplicant_income,
        loan_amount=loan_amount,
        credit_history=credit_history,
    )

    # one row of features
    assert processed.shape[0] == 1

    # expected feature columns (order-insensitive).
    expected_cols = {
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Credit_History_0",
        "Credit_History_1",
    }
    
    # verify if the expected columns matches the processed columns (columns used in training the model).
    assert set(processed.columns) == expected_cols

    # numerical features should be scaled into [0, 1].
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]: 

        # get single scaled value.
        value = float(processed[col].iloc[0]) 

        # ensure value is in unit interval.
        assert 0.0 <= value <= 1.0  

    # credit history should be one-hot encoded and mutually exclusive.
    ch0 = float(processed["Credit_History_0"].iloc[0])  # first one hot flag.
    ch1 = float(processed["Credit_History_1"].iloc[0])  # second one hot flag.

    # check valid 0 or 1.
    assert ch0 in (0.0, 1.0) 

    # check valid 0 or 1.
    assert ch1 in (0.0, 1.0)  
    
    # ensure exactly one flag is set.
    assert ch0 + ch1 == 1.0  



def test_user_info_processing_handles_both_credit_history_values():
    """
    ensure credit history one-hot encoding behaves correctly for both 0.0 and 1.0.
    """

    # instantiate the model class.
    model = LoanEligibilityModel()

    # shared numeric inputs for both cases.
    base_kwargs = dict(  
        applicant_income=4000.0,
        coapplicant_income=1500.0,
        loan_amount=120.0,
    )

    # process bad credit.
    processed_ch0 = model.user_info_processing(credit_history=0.0, **base_kwargs) 

    # process good credit.
    processed_ch1 = model.user_info_processing(credit_history=1.0, **base_kwargs) 

    # credit_history_0 should be set.
    assert processed_ch0["Credit_History_0"].iloc[0] == 1.0  

    # credit_history_1 should be cleared.
    assert processed_ch0["Credit_History_1"].iloc[0] == 0.0  

    # credit_history_0 should be cleared.
    assert processed_ch1["Credit_History_0"].iloc[0] == 0.0 

    # credit_history_1 should be set.
    assert processed_ch1["Credit_History_1"].iloc[0] == 1.0  



def test_generate_eligibility_returns_valid_prediction_and_probability():
    """
    smoke test for generate_eligibility
    verifies prediction is 0 or 1 and probability is within [0, 1].
    """

    # instantiate the model class.
    model = LoanEligibilityModel() 

    # run a single prediction call with sample inputs.
    prediction, probability = model.generate_eligibility(
        applicant_income=6000.0,
        coapplicant_income=2500.0,
        loan_amount=180.0,
        credit_history=1.0,
    )

    # ensure binary classification outcome.
    assert prediction in (0, 1)  
    
    # ensure probability is within bounds.
    assert 0.0 <= float(probability) <= 1.0  