from flask import Flask, render_template, request
import pickle
import numpy as np
from flask_mail import Mail, Message
import os
import logging
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# Setup logger
logging.basicConfig(filename='app_errors.log', level=logging.ERROR)

# Load models and transformers
try:
    filename_spam = 'spam-model.pkl'
    classifier_spam = pickle.load(open(filename_spam, 'rb'))
    cv_spam = pickle.load(open('spam-transform.pkl', 'rb'))
    
    filename_rest = 'restaurant-model.pkl'
    classifier_rest = pickle.load(open(filename_rest, 'rb'))
    cv_rest = pickle.load(open('restaurant-transform.pkl', 'rb'))
    
    filename_movie = 'movie-model.pkl'
    classifier_movie = pickle.load(open(filename_movie, 'rb'))
    cv_movie = pickle.load(open('movie-transform.pkl', 'rb'))
    
    filename_dia = 'diabetes-model.pkl'
    classifier_dia = pickle.load(open(filename_dia, 'rb'))
except FileNotFoundError as e:
    logging.error(f"File not found: {str(e)}")
    raise

app = Flask(__name__)

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('EMAIL')
app.config['MAIL_PASSWORD'] = os.getenv('PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('EMAIL')

mail = Mail(app)

@app.route('/')
def start():
    return render_template('index.html')


@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        recipient = request.form.get('email')
        name = request.form.get('name')
        body = request.form.get('message')

        if not recipient or not name or not body:
            raise ValueError("Missing required fields for sending email.")
        
        msg = Message(
            "Message from Minor Project",
            recipients=[recipient],
            body=f"Hi Abhishek, you have a message from {name} \n\n {body}"
        )
        mail.send(msg)
        return render_template('email-success.html', name=name)
    
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        return render_template('email-failed.html', error_message=str(e))


@app.route('/spam')
def home_spam():
    return render_template('home_spam.html')


@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    try:
        if request.method == 'POST':
            message = request.form['message']
            data = [message]
            vect = cv_spam.transform(data).toarray()
            my_prediction = classifier_spam.predict(vect)
            return render_template('result_spam.html', prediction=my_prediction)
    
    except Exception as e:
        logging.error(f"Error predicting spam: {str(e)}")
        return render_template('error.html', error_message="An error occurred while predicting spam.")


@app.route('/rest')
def home_rest():
    return render_template('home_rest.html')


@app.route('/predict_rest', methods=['POST'])
def predict_rest():
    try:
        if request.method == 'POST':
            message = request.form['message']
            data = [message]
            vect = cv_rest.transform(data).toarray()
            my_prediction = classifier_rest.predict(vect)
            return render_template('result_rest.html', prediction=my_prediction)
    
    except Exception as e:
        logging.error(f"Error predicting restaurant review: {str(e)}")
        return render_template('error.html', error_message="An error occurred while predicting the restaurant review.")


@app.route('/movie')
def home_movie():
    return render_template('home_movie.html')


@app.route('/predict_movie', methods=['POST'])
def predict_movie():
    try:
        if request.method == 'POST':
            message = request.form['message']
            data = [message]
            vect = cv_movie.transform(data).toarray()
            my_prediction = classifier_movie.predict(vect)
            return render_template('result_movie.html', prediction=my_prediction)
    
    except Exception as e:
        logging.error(f"Error predicting movie genre: {str(e)}")
        return render_template('error.html', error_message="An error occurred while predicting the movie genre.")


@app.route('/diabetes')
def home_dia():
    return render_template('home_dia.html')


@app.route('/predict_dia', methods=['POST'])
def predict_dia():
    try:
        if request.method == 'POST':
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier_dia.predict(data)
            return render_template('result_dia.html', prediction=my_prediction)
    
    except ValueError as e:
        logging.error(f"Value error in diabetes prediction: {str(e)}")
        return render_template('error.html', error_message="Invalid input values. Please check your entries.")
    except Exception as e:
        logging.error(f"Error predicting diabetes: {str(e)}")
        return render_template('error.html', error_message="An error occurred while predicting diabetes.")


# Error handling for 404 and 500 errors
@app.errorhandler(404)
def page_not_found(e):
    logging.error(f"404 Error: {str(e)}")
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    logging.error(f"500 Error: {str(e)}")
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 8080)))
