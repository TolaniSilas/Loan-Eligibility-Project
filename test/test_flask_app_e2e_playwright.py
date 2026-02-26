import re
from playwright.sync_api import Page, expect 



def test_loan_eligibility_path(page: Page) -> None:
    """end-to-end: fill form, submit, see eligibility result."""

    # navigate to running flask app (flask app runninglocally).
    page.goto("http://127.0.0.1:5000/",  wait_until='domcontentloaded')  

    # reveal loan form.
    page.get_by_role("button", name="Check for Loan Eligibility!").click()

    # fill required name.
    page.fill("#name", "john doe")  

    # fill required email.
    page.fill("#email", "john.doe@example.com")  

    # select gender type.
    page.select_option("#gender", "Male")  

    # select marital status.
    page.select_option("#married", "Yes") 

    # fill the number of dependents.
    page.fill("#dependents", "1") 

    # select education level.
    page.select_option("#education", "Graduate")  

    # select self employed status.
    page.select_option("#selfEmployed", "No") 

    # fill applicant income.
    page.fill("#applicantIncome", "6000")  

    # fill coapplicant income.
    page.fill("#coapplicantIncome", "2500")  

    # fill loan amount.
    page.fill("#loanAmount", "120")  
    
    # fill loan term.
    page.fill("#loanAmountTerm", "20")  

    # select good credit history.
    page.select_option("#creditHistory", "1")  

    # select property area.
    page.select_option("#propertyArea", "Urban")  

    # submit the form.
    page.get_by_role("button", name="Submit").click()  

    # assert we landed on eligibility page.
    expect(page).to_have_url(re.compile(r"/eligibility"))  

    # confirm eligibility heading is visible.
    expect(page.get_by_text("Eligibility Result")).to_be_visible()
    
    # locate the paragraph element that contains the word "status".
    status_paragraph = page.locator("p", has_text="Status")

    # locate the paragraph element that contains the word "status".
    status_value = status_paragraph.locator("strong").nth(1).inner_text()
    
    # ensure the status value is one of the expected outcomes.
    assert status_value in {"Eligible", "Non-eligible"}

    # confirm username appears on page.
    assert "john doe".title() in page.inner_text("body")  