import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting train and test data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Splitting train and test data Completed")
            
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Linear Regression":LinearRegression(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "XGBoost Regressor": XGBRegressor()
            }
                
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                               models=models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No good model found")
            logging.info(f"Best found model on both training and testing dataset")
            logging.info("Best model is [{0}] with r2_score [{1}]".format(best_model_name,best_model_score))
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            CustomException(e,sys)
            

    