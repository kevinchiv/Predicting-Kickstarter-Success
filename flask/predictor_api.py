import os
import pickle

import pandas as pd
import numpy as np
import scipy

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt

import shap

from datetime import datetime




#load MinxMaxScaler
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

#load SGDClassifier
with open("model/sgd_classifier.pkl", "rb") as f:
    sgd_classifier = pickle.load(f)

#load SHAP explainer for SGD
with open("model/sgd_explainer.pkl", "rb") as f:
    sgd_explainer = pickle.load(f)



def make_prediction(feature_dict):
    """
    Input:
    feature_dict: a dictionary of the form {"feature_name": "value"}
    Function makes sure the features are fed to the model in the same order the
    model expects them.
    Output:
    Returns (x_inputs, probs) where
      x_inputs: a list of feature values in the order they appear in the model
      probs: a list of dictionaries with keys 'name', 'prob'
    """

    #form_input_names should match what `name` in each input/select tag says
    form_input_names = ['Campaign Name', 'Campaign Duration', 'Campaign Goal Amount', 
    					'Main Category', 'Backers']

   	#fetch the form inputs so we can render them again in the html
   	#by default, submitting causes form inputs to reset
    form_inputs = {name:feature_dict.get(name) for name in form_input_names}


    #obtain the title, split by spaces to get words, 
    #and use len to get number of words in title
    title_length = len(form_inputs['Campaign Name'].split(' '))

    project_duration = int(form_inputs['Campaign Duration'])
    
    goal = int(form_inputs['Campaign Goal Amount'])

    #these are the main categories of interest
    categories_of_interest = { 'Design': 0, 'Film & Video': 0, 'Games': 0, 'Technology': 0}

    main_category = form_inputs['Main Category']

    if main_category in categories_of_interest.keys():
    	categories_of_interest[main_category] = 1

    #only interested in whether the number of expected backers is
    #greater than or equal to 45
    min_45_backers = 1 if int(form_inputs['Backers']) >= 45 else 0

    #turning the dictionary into an array of values
    category_values = list(categories_of_interest.values())

    #combine input data into one list
    #make sure the data is ordered the same way the model was trained on
    input_data = [title_length, min_45_backers, goal, project_duration] + category_values

    #a dense array is needed for SHAP
    input_data_dense = scipy.sparse.csr_matrix(input_data).toarray()


    #cast input data as np.array to apply scaler
    input_data = np.array(input_data)


    #if scaling was applied on train data, scale user input data as well
    input_data = scaler.transform([input_data])

    y_proba = sgd_classifier.predict_proba(input_data)[:, 1][0]

    #assign a label based on threshold
    #normally there are two class labels, success or failure
    #however, since the model threshold for success is so low,
    #anything greater or equal than the threshold is a success
    #I decided to further break 'success' into two kinds
    #so the results would be more intuitive for the users
    threshold = 0.209

    if y_proba > 0.5:
    	y_pred = 'Likely Successful'

    elif y_proba >= threshold and y_proba <= 0.5:
    	y_pred = 'Highly Unlikely Success'

    else:
    	y_pred = 'Likely Failure'

    shap_values = sgd_explainer.shap_values(input_data)


    shap.force_plot(sgd_explainer.expected_value, shap_values, input_data_dense,
    				feature_names = ['Title Length', 'Minimum 45 Backers', 'Campaign Goal Amount', 
    								'Campaign Duration', 'Design', 'Film & Video', 'Games', 'Technology'], 
    				link = 'logit', matplotlib = True, text_rotation = 10);

    current_date_time = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")


    plt.savefig('static/img/SHAP_force_plot' + current_date_time +'.png', 
    			 transparent = True, bbox_inches = 'tight',  pad_inches=0)

    return (form_inputs, round(y_proba, 3), y_pred)


if __name__ == '__main__':
    from pprint import pprint
    print("Checking function")
    
    #using form inputs for failed Kickstarter campaign
    form_inputs = {'Campaign Name': 'Greeting From Earth: ZGAC Arts Capsule For ET',
    			   'Campaign Duration': 59.80,
    			   'Campaign Goal Amount': 30000, 
    			   'Main Category': 'Film & Video',
    			   'Backers': 15}

    print('Form Data:')
    pprint(form_inputs)

    form_inputs, y_proba, y_pred = make_prediction(form_inputs)
    print('Input values:')
    pprint(form_inputs)
    print('Probability of Success: %s' %y_proba)
    print(y_pred)