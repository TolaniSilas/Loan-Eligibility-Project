from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def show_loan_form():

    return render_template('index.html')

@app.route('/submit-info', methods=['POST'])
def submit_loan():
    
    name = request.form['name']  
    email = request.form['email']  

    # Process the form data
    return f"Name: {name}, Email: {email}"

if __name__ == '__main__':
    app.run(debug=True)
