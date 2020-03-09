import flask
from flask import request
from predictor_api import make_prediction

from glob import glob
import os


# Initialize the app
app = flask.Flask(__name__, template_folder='templates')


# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!
@app.route("/")
def hello():
    """
        Function for rendering the first page
    """
    form_inputs = {'Campaign Name': '',
                   'Campaign Duration': '',
                   'Campaign Goal Amount': '',
                   'Main Category': '',
                   'Backers': ''}


    return flask.render_template('index.html',
                                 form_inputs=form_inputs,
                                 probability='',
                                 prediction='',
                                 SHAP_force_plot='')


@app.route("/about",methods=["POST", "GET"])
def about():
    return flask.render_template('about.html')


@app.route("/contact",methods=["POST", "GET"])
def contact():
    return flask.render_template('contact.html')



@app.route("/predict", methods=["POST", "GET"])
def predict():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)

    
    for file in glob('static/img/SHAP_force_plot*.png'):
        os.remove(file)

    form_inputs, y_proba, y_pred = make_prediction(request.args)

    img = glob('static/img/SHAP_force_plot*.png')[0]


    return flask.render_template('index.html', 
                                 form_inputs=form_inputs,
                                 #we have to turn this in a string to avoid extra decimal places
                                 #round does not fix the issue, only casting as a string works
                                 probability=str(round(y_proba*100, 1)),
                                 prediction=y_pred,
                                 SHAP_force_plot=img)


# @app.route("/calculator", methods=["POST","GET"])
# def calculate():
#     i_input, matrix, appts, do_nothing, int_cost, cost_w_int, tot_cost_w_int = calculate_int(request.args)
#     return flask.render_template('calculator.html',i_input=i_input,
#                                  input_names=input_names,
#                                  input_defaults=input_defaults,
#                                  matrix = matrix,
#                                  appts = appts,
#                                  do_nothing = do_nothing,
#                                  int_cost = int_cost,
#                                  cost_w_int = cost_w_int,
#                                  tot_cost_w_int = tot_cost_w_int)

# Start the server, continuously listen to requests.
# We'll have a running web app!

if __name__=="__main__":
    # For local development:
    # app.run(debug=True)
    # For public web serving:
    app.run()