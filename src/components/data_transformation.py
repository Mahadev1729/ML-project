import sys
from dataclasses import dataclass
import numpy as np
import os
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")
    
class DataTransformation:
        def __init__(self):
            self.Data_Transformation_Config=DataTransformationConfig()
        def get_data_transformer_object(self):
            '''
            This function is responsible for data transformation 
            '''
            
            try:
                numerical_features=['writing score','reading score']
                cat_features=['gender',
                    'race/ethnicity',
                    'parental level of education',
                    'lunch',
                    'test preparation course'
                    ]
                
                num_pipeline=Pipeline(
                    steps=[
                        ("imputer",SimpleImputer(strategy="median")),
                        ("scaler",StandardScaler())
                    ]
                )
                cat_pipeline=Pipeline(
                    steps=[
                        ("imputer",SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder",OneHotEncoder()),
                        ("scaling",StandardScaler())
                    ]
                )
                logging.info(f"Categerical columns:{cat_features}")
                logging.info(f"numrical columns:{numerical_features}")
                preprocessor=ColumnTransformer(
                    [
                        ("num_pipeline",num_pipeline,numerical_features),
                        ("cat_pipeline",cat_pipeline,cat_features)
                    ]
                )
                return preprocessor
            except Exception as e:
                raise CustomException(e,sys)
            
                
            
    

