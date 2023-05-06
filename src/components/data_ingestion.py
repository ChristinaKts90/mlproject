import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer,ModelTrainingConfig

@dataclass
class DataIngestionConfig:
    logging.info("Setting the data paths")
    try:
        train_path: str=os.path.join("artifacts","train.csv")
        test_path: str=os.path.join("artifacts","test.csv")
        raw_path: str=os.path.join("artifacts","raw.csv")
        logging.info("Setting of the data paths completed")
    except Exception as e:
        raise CustomException(e,sys)
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("notebook\data\StudentsPerformance.csv")
            logging.info("Read the dataset as dataframe completed")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_path, index=False, header=True)
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            return(
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
 
"""
#test it    
if __name__=="__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()
    target='math score'
    obj2 = DataTransformation()
    train_array,test_array,preprocessor_obj_file_path = obj2.initiate_data_transformation(train_path,test_path,target)
    model_obj = ModelTrainer()
    model_obj.initiate_model_trainer(train_array,test_array)

"""