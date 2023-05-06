import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass
    
    def train_model(self,target):
        try:
            logging.info("Start model training")
            data_ingestion_obj = DataIngestion()
            train_path,test_path = data_ingestion_obj.initiate_data_ingestion()
            #target='math score'
            data_transformation_obj = DataTransformation()
            train_array,test_array,preprocessor_obj_file_path = data_transformation_obj.initiate_data_transformation(train_path,test_path,target)
            model_obj = ModelTrainer()
            model_obj.initiate_model_trainer(train_array,test_array)
            logging.info("Model training completed")
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=TrainPipeline()
    obj.train_model(target='math score')