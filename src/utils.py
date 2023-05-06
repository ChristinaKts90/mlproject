import os
import sys
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
import pickle 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        logging.info("Create the report")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_params=params[list(models.keys())[i]]
            gs = RandomizedSearchCV(model,param_distributions=model_params,cv=3, n_iter = 1, n_jobs=-1)
            #gs = gs = GridSearchCV(model,para,cv=3, n_jobs=-1)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
        
        logging.info("Report has be created successfully")
        return report
    
    except Exception as e:
        CustomException(e,sys)
        
        
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
