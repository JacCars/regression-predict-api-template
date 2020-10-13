"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
import math
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV



def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    #Import data and remove whitespace from columns
    data = pd.read_csv('Model_data/Train.csv')
    data.columns = [col.replace(' ','_') for col in data.columns]
    #data.info()

    #Check for and remove anomalies
    outliers = data[data['Placement_-_Day_of_Month'] != data['Arrival_at_Destination_-_Day_of_Month']]
    data = data.drop(outliers.index, axis=0)
    data = data.fillna(data.mean())

    #Create function that will be applied to time columns (feature engineering) to make them more usable in model
    # takes time e.g. 10:30:00 AM and converts it to datetime object and then formats it as a float 10.50 that model can use
    def converter (column):
        out = []
        for value in column.values:
            work = pd.to_datetime(value)
            hour = str(int(work.hour))
            minute = str(int(work.minute/60*100))
            out.append(float(hour+'.'+minute))
        return pd.Series(out)


    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = feature_vector_df[['Pickup Lat','Pickup Long',
                                        'Destination Lat','Destination Long']]
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
