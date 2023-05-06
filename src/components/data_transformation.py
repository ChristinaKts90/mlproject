import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self,X_train):
        try:       
            num_features = X_train.select_dtypes(exclude="object").columns.to_list()
            cat_features = X_train.select_dtypes(include="object").columns.to_list()
            
            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
                         
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)       
                ]
            )
            return preprocessor
        
        except Exception as e:
            CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path,target):
        
        try:
            logging.info("Try to read the data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test datasets completed")
            
            logging.info("Get X and y for train and test")
            y_train=train_df[target]
            X_train = train_df.drop(columns=[target],axis=1)
            
            y_test=test_df[target]
            X_test = test_df.drop(columns=[target],axis=1)
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(X_train)
            
            X_train_arr=preprocessing_obj.fit_transform(X_train)
            X_test_arr=preprocessing_obj.transform(X_test)
            
            
            train_arr = np.c_[
                X_train_arr, np.array(y_train)
            ]
            test_arr = np.c_[X_test_arr, np.array(y_test)]
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            CustomException(e,sys)
            

    